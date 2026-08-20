"""The pitch engine: the shift maths, the explanation, and the length contract.

The decision maths and the playback mechanics are covered against fakes so the
whole file runs with no torch and no python-stretch -- exactly what CI has. The
one real-shifter test skips itself when python-stretch is absent.
"""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionParams, VoiceHandle
from svc_engine.backends.separation import StemKind
from svc_engine.pitch.explain import explain_candidate, explain_decision
from svc_engine.pitch.instrumental import PitchedStems, PlaybackStrategy, shift_playback
from svc_engine.pitch.quality_probe import (
    DEFAULT_PROBE_SHIFTS,
    measure_artifacts,
    probe_quality_curve,
)
from svc_engine.pitch.shifter import PythonStretchShifter
from svc_engine.pitch.softening import soften_f0
from svc_engine.pitch.strategy import (
    CostWeights,
    PitchDistribution,
    decide_shift,
    decompose,
    evaluate_shift,
    playback_penalty,
)
from svc_engine.profiles import VoiceProfile

SR = 44100


def _hz_from_midi(midi: np.ndarray) -> np.ndarray:
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


def _voice(name: str, cl: float, ch: float, al: float, ah: float, med: float) -> VoiceProfile:
    return VoiceProfile(name, cl, ch, al, ah, med, "test", 1.0, 100)


# --- decomposition s = 12k + r -------------------------------------------- #

@pytest.mark.parametrize("s", range(-30, 31))
def test_decompose_roundtrips_and_r_in_window(s: int) -> None:
    k, r = decompose(s)
    assert 12 * k + r == s
    assert -6 <= r <= 5


def test_octave_shifts_have_zero_remainder() -> None:
    for s in (-24, -12, 0, 12, 24):
        _, r = decompose(s)
        assert r == 0


def test_playback_penalty_zero_at_octave_and_monotone() -> None:
    assert playback_penalty(0) == 0.0
    assert playback_penalty(6) == pytest.approx(1.0)
    assert playback_penalty(1) < playback_penalty(3) < playback_penalty(6)
    assert playback_penalty(-3) == playback_penalty(3)


# --- pitch distribution ---------------------------------------------------- #

def test_distribution_mass_outside_is_a_weighted_fraction() -> None:
    midi = np.concatenate([np.full(80, 60.0), np.full(20, 80.0)])
    dist = PitchDistribution.from_f0(_hz_from_midi(midi))
    # Unshifted: 20% of frames sit above a ceiling of 70.
    assert dist.mass_outside(0.0, 70.0) == pytest.approx(0.20, abs=1e-6)
    # Shift everything down 12: now nothing is above 70.
    assert dist.mass_outside(0.0, 70.0, shift=-12) == pytest.approx(0.0, abs=1e-6)


def test_distribution_energy_weighting_counts_held_notes() -> None:
    midi = np.concatenate([np.full(50, 60.0), np.full(50, 80.0)])  # equal counts
    unweighted = PitchDistribution.from_f0(_hz_from_midi(midi))
    loud_low = np.concatenate([np.full(50, 10.0), np.full(50, 1.0)])  # low note held louder
    weighted = PitchDistribution.from_f0(_hz_from_midi(midi), loud_low)
    # Equal counts put the plain median mid-way; energy weighting drags it toward
    # the loud, held low note.
    assert unweighted.median == pytest.approx(70.0, abs=1.0)
    assert weighted.median < 63.0


def test_distribution_rejects_silence_and_length_mismatch() -> None:
    with pytest.raises(ValueError):
        PitchDistribution.from_f0(np.zeros(100))
    with pytest.raises(ValueError):
        PitchDistribution.from_f0(_hz_from_midi(np.full(10, 60.0)), np.ones(9))


# --- the decision: predefined profiles, expected recommendation ------------ #
# docs/roadmap.md Phase 4: "unit tests on the maths: for predefined profiles,
# is the recommendation the expected one?"

def _female_song() -> PitchDistribution:
    midi = np.concatenate([np.full(600, 64.0), np.full(250, 69.0), np.full(150, 57.0)])
    energy = np.concatenate([np.full(600, 1.0), np.full(250, 0.8), np.full(150, 0.5)])
    return PitchDistribution.from_f0(_hz_from_midi(midi), energy)


def test_female_to_baritone_recommends_a_clean_octave_down() -> None:
    song = _female_song()
    baritone = _voice("baritone", cl=45, ch=57, al=40, ah=60, med=50)
    decision = decide_shift(song, baritone)
    assert decision.best.semitones == -12
    assert decision.best.remainder == 0          # the playback never moves
    assert not decision.best.needs_playback_shift


def test_female_to_alto_needs_little_or_no_shift() -> None:
    song = _female_song()
    alto = _voice("alto", cl=57, ch=72, al=53, ah=76, med=64)
    decision = decide_shift(song, alto)
    assert abs(decision.best.semitones) <= 2


def test_cost_breakdown_sums_to_total() -> None:
    song = _female_song()
    tenor = _voice("tenor", cl=48, ch=64, al=45, ah=69, med=55)
    cand = evaluate_shift(song, tenor, CostWeights(), -9)
    assert cand.cost == pytest.approx(sum(cand.breakdown.values()))


def test_alternatives_bring_distinct_remainders() -> None:
    song = _female_song()
    baritone = _voice("baritone", cl=45, ch=57, al=40, ah=60, med=50)
    decision = decide_shift(song, baritone, n_alternatives=2)
    seen = {decision.best.remainder}
    for alt in decision.alternatives:
        assert alt.remainder not in seen
        seen.add(alt.remainder)


def test_measured_quality_curve_steers_the_choice_and_sets_the_flag() -> None:
    song = _female_song()
    base = _voice("v", cl=45, ch=69, al=40, ah=74, med=57)
    plain = decide_shift(song, base)
    assert plain.quality_measured is False

    # Punish this voice hard at exactly the plain winner; a neighbour should win.
    winner = plain.best.semitones
    curve = [[float(winner), 5.0], [float(winner - 1), 0.0], [float(winner + 1), 0.0]]
    measured = base.with_quality_curve(curve)
    steered = decide_shift(song, measured)
    assert steered.quality_measured is True
    assert steered.best.semitones != winner


# --- explanation ----------------------------------------------------------- #

def test_explain_octave_case_says_music_is_untouched() -> None:
    song = _female_song()
    baritone = _voice("baritone", cl=45, ch=57, al=40, ah=60, med=50)
    text = explain_candidate(decide_shift(song, baritone).best)
    assert "אוקטבה" in text
    assert "המוזיקה נשארת" in text


def test_explain_non_octave_case_mentions_moving_the_music() -> None:
    song = _female_song()
    tenor = _voice("tenor", cl=48, ch=64, al=45, ah=69, med=55)
    decision = decide_shift(song, tenor)
    if decision.best.remainder != 0:
        text = explain_candidate(decision.best)
        assert "המוזיקה" in text and "צריך" in text


def test_explain_decision_flags_unmeasured_quality() -> None:
    song = _female_song()
    alto = _voice("alto", cl=57, ch=72, al=53, ah=76, med=64)
    block = explain_decision(decide_shift(song, alto))
    assert "מומלץ" in block
    assert "עדיין לא נמדדה" in block  # honest about the missing w5 curve


# --- the shifter's hard contract: exact length ----------------------------- #

def test_shifter_info_reflects_availability() -> None:
    info = PythonStretchShifter().info()
    assert info.available == (importlib.util.find_spec("python_stretch") is not None)
    assert info.supports_gpu is False  # CPU-only by design


def test_zero_shift_is_identity_without_touching_the_library() -> None:
    audio = AudioBuffer(samples=np.ones((1, 1000), dtype=np.float32), sample_rate=SR)
    assert PythonStretchShifter().shift(audio, 0) is audio


@pytest.mark.skipif(
    importlib.util.find_spec("python_stretch") is None,
    reason="python-stretch not installed (as in CI)",
)
def test_real_shift_preserves_length_exactly() -> None:
    for n in (SR, 33333):
        audio = AudioBuffer(
            samples=(0.1 * np.random.default_rng(0).standard_normal((1, n))).astype(np.float32),
            sample_rate=SR,
        )
        out = PythonStretchShifter().shift(audio, -5)
        assert out.frames == n
        assert np.isfinite(out.samples).all()


# --- playback strategies A and B ------------------------------------------- #

class _LengthBreakingShifter:
    """A stand-in that returns the wrong length, to prove fit_length rescues it."""

    def info(self) -> BackendInfo:
        return BackendInfo("fake", "בדיקה", available=True)

    def shift(self, audio: AudioBuffer, semitones: float) -> AudioBuffer:
        s = audio.samples
        longer = np.concatenate([s, s[:, :17]], axis=1)
        return AudioBuffer(samples=longer, sample_rate=audio.sample_rate)


def _noise(n: int, ch: int = 1) -> AudioBuffer:
    data = 0.1 * np.random.default_rng(1).standard_normal((ch, n))
    return AudioBuffer(samples=data.astype(np.float32), sample_rate=SR)


def test_strategy_a_preserves_length_even_if_the_shifter_does_not() -> None:
    inst = _noise(SR)
    out = shift_playback(inst, -2, _LengthBreakingShifter(), PlaybackStrategy.WHOLE)
    assert out.frames == SR


def test_strategy_a_zero_remainder_is_identity() -> None:
    inst = _noise(SR)
    assert shift_playback(inst, 0, _LengthBreakingShifter(), PlaybackStrategy.WHOLE) is inst


def test_strategy_b_shifts_all_but_drums_and_keeps_length() -> None:
    inst = _noise(SR)
    drums = _noise(SR)
    stems = {StemKind.BASS: inst, StemKind.OTHER: inst, StemKind.DRUMS: drums}
    out = shift_playback(stems, -3, _LengthBreakingShifter(), PlaybackStrategy.SPLIT)
    assert out.frames == SR


def test_strategy_b_leaves_drums_untouched_when_r_is_zero() -> None:
    drums = _noise(SR)
    stems = PitchedStems(drums=drums)
    out = shift_playback(stems, 0, _LengthBreakingShifter(), PlaybackStrategy.SPLIT)
    # r == 0: no stem is shifted, so the drums pass through sample-for-sample.
    assert np.allclose(out.samples, drums.samples)


def test_strategy_type_mismatches_raise() -> None:
    fake = _LengthBreakingShifter()
    with pytest.raises(TypeError):
        shift_playback({StemKind.BASS: _noise(10)}, -2, fake, PlaybackStrategy.WHOLE)
    with pytest.raises(TypeError):
        shift_playback(_noise(10), -2, fake, PlaybackStrategy.SPLIT)


# --- softening ------------------------------------------------------------- #

def test_softening_zero_is_a_noop() -> None:
    curve = F0Curve(hz=_hz_from_midi(np.array([60.0, 72.0])), hop_seconds=0.01)
    assert soften_f0(curve, 0.0) is curve


def test_softening_shrinks_the_span_and_keeps_unvoiced_and_length() -> None:
    low = _hz_from_midi(np.full(50, 55.0))
    high = _hz_from_midi(np.full(50, 79.0))
    hz = np.concatenate([low, [0.0] * 10, high])
    curve = F0Curve(hz=hz, hop_seconds=0.01)
    soft = soften_f0(curve, 0.25)
    assert soft.frames == curve.frames
    assert np.all(soft.hz[50:60] == 0.0)  # unvoiced frames untouched
    voiced_before = hz[hz > 0]
    voiced_after = soft.hz[soft.hz > 0]
    assert np.ptp(voiced_after) < np.ptp(voiced_before)


def test_softening_rejects_out_of_range_amount() -> None:
    curve = F0Curve(hz=_hz_from_midi(np.full(10, 60.0)), hop_seconds=0.01)
    with pytest.raises(ValueError):
        soften_f0(curve, 1.5)


# --- quality probe --------------------------------------------------------- #

def test_measure_artifacts_scores_clean_low_and_broken_high() -> None:
    clean = AudioBuffer(
        samples=(0.1 * np.sin(np.linspace(0, 100, SR))).astype(np.float32)[None, :],
        sample_rate=SR,
    )
    assert measure_artifacts(clean).penalty < 0.05

    broken = clean.samples.copy()
    broken[0, ::20] = np.nan  # 5% non-finite -- a clearly broken conversion
    stats = measure_artifacts(AudioBuffer(samples=broken, sample_rate=SR))
    assert stats.nonfinite_fraction == pytest.approx(0.05, abs=0.01)
    assert stats.penalty > measure_artifacts(clean).penalty
    assert stats.penalty > 0.1


class _FakeConversion:
    """A conversion backend that degrades output in proportion to |semitones|."""

    def __init__(self) -> None:
        self.loaded = False
        self.unloaded = False

    def info(self) -> BackendInfo:
        return BackendInfo("fakervc", "בדיקה", available=True)

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:
        self.loaded = True

    def convert(self, audio: AudioBuffer, f0: F0Curve, params: ConversionParams) -> AudioBuffer:
        bump = abs(params.semitones) / 40.0
        out = audio.samples + bump * np.sign(np.sin(np.arange(audio.samples.shape[-1])))
        return AudioBuffer(samples=out.astype(np.float32), sample_rate=audio.sample_rate)

    def unload(self) -> None:
        self.unloaded = True


def test_probe_builds_a_curve_and_always_unloads() -> None:
    ref = AudioBuffer(samples=(0.1 * np.ones((1, SR))).astype(np.float32), sample_rate=SR)
    f0 = F0Curve(hz=np.full(100, 220.0), hop_seconds=0.01)
    backend = _FakeConversion()
    curve = probe_quality_curve(ref, f0, backend, VoiceHandle("v", tmp_root()))

    assert backend.loaded and backend.unloaded
    shifts = [int(s) for s, _ in curve.to_profile_list()]
    assert shifts == sorted(DEFAULT_PROBE_SHIFTS)
    assert curve.similarity_available is False
    # Every penalty is a real number in [0, 1].
    assert all(0.0 <= p <= 1.0 for _, p in curve.to_profile_list())


def test_probe_uses_injected_similarity_when_present() -> None:
    ref = AudioBuffer(samples=(0.1 * np.ones((1, SR))).astype(np.float32), sample_rate=SR)
    f0 = F0Curve(hz=np.full(100, 220.0), hop_seconds=0.01)
    curve = probe_quality_curve(
        ref, f0, _FakeConversion(), VoiceHandle("v", tmp_root()),
        similarity_fn=lambda a, b: 1.0,
    )
    assert curve.similarity_available is True


def tmp_root():  # type: ignore[no-untyped-def]
    from pathlib import Path
    return Path(".")


# --- profile round-trip carries the measured curve ------------------------- #

def test_profile_quality_curve_roundtrips_through_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    base = _voice("v", cl=45, ch=57, al=40, ah=60, med=50)
    assert base.quality_vs_shift is None
    curve = [[-12.0, 0.2], [0.0, 0.0], [12.0, 0.3]]
    withc = base.with_quality_curve(curve)

    path = withc.save(tmp_path / "v.json")
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["quality_vs_shift"] == curve

    loaded = VoiceProfile.load(path)
    assert loaded.quality_vs_shift == curve
    assert VoiceProfile.load(base.save(tmp_path / "b.json")).quality_vs_shift is None
