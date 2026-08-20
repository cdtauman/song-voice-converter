"""Analysis engine: the decision math, covered without loading an F0 model.

Everything here runs against numpy arrays or a fake extractor, exactly like the
separation tests run against a fake backend. The real FCPE/RMVPE paths are
exercised by tools/bench_analysis.py, which needs torch and weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from svc_engine.analysis import (
    analyze_vocal,
    compute_range,
    estimate_key,
    find_voiced_segments,
    hz_to_midi,
    midi_to_hz,
    midi_to_note_name,
    pick_preview,
)
from svc_engine.analysis.f0 import resample_f0
from svc_engine.analysis.key import _MAJOR_PROFILE, _MINOR_PROFILE, chroma_from_audio
from svc_engine.analysis.range import weighted_percentile
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve

# --------------------------------------------------------------------------- #
# midi
# --------------------------------------------------------------------------- #

def test_midi_anchors() -> None:
    assert hz_to_midi(440.0) == pytest.approx(69.0)
    assert midi_to_hz(69.0) == pytest.approx(440.0)
    assert midi_to_note_name(69.0) == "A4"
    assert midi_to_note_name(60.0) == "C4"


def test_unvoiced_hz_becomes_nan() -> None:
    out = hz_to_midi(np.array([440.0, 0.0, -1.0, 220.0]))
    assert np.isnan(out[1]) and np.isnan(out[2])
    assert out[0] == pytest.approx(69.0)
    assert out[3] == pytest.approx(57.0)


def test_octave_is_twelve_semitones() -> None:
    assert hz_to_midi(880.0) - hz_to_midi(440.0) == pytest.approx(12.0)


# --------------------------------------------------------------------------- #
# range
# --------------------------------------------------------------------------- #

def test_weighted_percentile_matches_numpy_for_equal_weights() -> None:
    values = np.linspace(0.0, 100.0, 101)
    weights = np.ones_like(values)
    got = weighted_percentile(values, weights, np.array([5.0, 50.0, 95.0]))
    exp = np.percentile(values, [5, 50, 95])
    assert np.allclose(got, exp, atol=1.0)


def test_range_excludes_unvoiced_frames() -> None:
    hz = np.array([440.0, 0.0, 0.0, 440.0, 440.0])
    stats = compute_range(hz)
    assert stats.voiced_frames == 3
    assert stats.voiced_fraction == pytest.approx(3 / 5)
    assert stats.p50 == pytest.approx(69.0)


def test_energy_weight_pulls_median_toward_the_loud_note() -> None:
    # Two pitches; the higher one is far louder, so the weighted median sits on it.
    hz = np.array([220.0, 220.0, 440.0, 440.0])
    quiet, loud = 0.01, 1.0
    weights = np.array([quiet, quiet, loud, loud])
    stats = compute_range(hz, weights)
    assert stats.p50 == pytest.approx(69.0, abs=0.5)   # A4, the loud note
    unweighted = compute_range(hz)
    assert unweighted.p50 == pytest.approx(63.0, abs=0.5)  # midpoint of the two


def test_range_of_pure_silence_is_nan() -> None:
    stats = compute_range(np.zeros(100))
    assert stats.voiced_frames == 0
    assert np.isnan(stats.p50)
    assert stats.comfortable_span == 0.0


def test_range_weights_length_must_match() -> None:
    with pytest.raises(ValueError):
        compute_range(np.array([440.0, 440.0]), np.array([1.0]))


# --------------------------------------------------------------------------- #
# key
# --------------------------------------------------------------------------- #

def test_key_recovers_the_profile_it_was_built_from() -> None:
    # A chroma shaped exactly like the C-major profile must read as C major.
    est = estimate_key(_MAJOR_PROFILE.copy())
    assert est.tonic == 0
    assert est.mode == "major"
    assert est.correlation > 0.99


def test_key_rotation_moves_the_tonic() -> None:
    # Rotate the major profile so degree 0 sits on G (7 semitones up).
    g_major = np.roll(_MAJOR_PROFILE, 7)
    est = estimate_key(g_major)
    assert est.name == "G major"


def test_key_tells_major_from_minor() -> None:
    est = estimate_key(np.roll(_MINOR_PROFILE, 9))  # A minor
    assert est.mode == "minor"
    assert est.tonic == 9


def test_chroma_from_a_pure_tone_peaks_on_its_pitch_class() -> None:
    sr = 22050
    t = np.arange(sr * 2) / sr
    tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)  # A4 -> pitch class 9
    chroma = chroma_from_audio(tone, sr)
    assert int(np.argmax(chroma)) == 9


def test_empty_chroma_is_handled() -> None:
    est = estimate_key(np.zeros(12))
    assert est.correlation == 0.0


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #

def test_segments_bridge_short_gaps_and_drop_blips() -> None:
    hop = 0.01
    hz = np.zeros(1000)
    hz[100:400] = 200.0          # 3s phrase
    hz[250:255] = 0.0            # 50ms gap inside it -> bridged
    hz[600:602] = 200.0          # 20ms blip -> dropped
    hz[700:950] = 200.0          # 2.5s phrase
    segs = find_voiced_segments(hz, hop, min_voiced_seconds=0.2, bridge_gap_seconds=0.1)
    assert len(segs) == 2
    assert segs[0].start_seconds == pytest.approx(1.0, abs=0.02)
    assert segs[0].end_seconds == pytest.approx(4.0, abs=0.02)


def test_segments_never_bridge_the_head() -> None:
    hop = 0.01
    hz = np.zeros(500)
    hz[400:] = 200.0
    segs = find_voiced_segments(hz, hop, min_voiced_seconds=0.2, bridge_gap_seconds=10.0)
    assert len(segs) == 1
    assert segs[0].start_seconds == pytest.approx(4.0, abs=0.02)


def test_all_silence_has_no_segments() -> None:
    assert find_voiced_segments(np.zeros(500), 0.01) == []


# --------------------------------------------------------------------------- #
# preview picker
# --------------------------------------------------------------------------- #

def test_preview_prefers_the_sung_window_over_silence() -> None:
    hop = 0.01
    n = 6000  # 60s
    hz = np.zeros(n)
    # A steady, repeating sung region in the middle; silence on both ends.
    hz[1000:4000] = 220.0
    choice = pick_preview(hz, hop, window_seconds=10.0, stride_seconds=5.0)
    assert choice.best is not None
    mid = (choice.best.start_seconds + choice.best.end_seconds) / 2
    assert 10.0 <= mid <= 40.0
    assert choice.best.density > 0.5


def test_preview_returns_all_candidate_windows() -> None:
    hz = np.full(3000, 220.0)
    choice = pick_preview(hz, 0.01, window_seconds=10.0, stride_seconds=5.0)
    assert len(choice.windows) >= 1
    assert choice.best in choice.windows


def test_preview_of_short_clip_is_the_whole_clip() -> None:
    hz = np.full(500, 220.0)  # 5s, shorter than the 25s window
    choice = pick_preview(hz, 0.01, window_seconds=25.0)
    assert choice.best is not None
    assert choice.best.start_seconds == pytest.approx(0.0)
    assert choice.best.end_seconds == pytest.approx(5.0, abs=0.05)


# --------------------------------------------------------------------------- #
# resample_f0 (pure numpy half of the F0 adapter)
# --------------------------------------------------------------------------- #

def test_resample_f0_nearest_preserves_unvoiced() -> None:
    src = np.array([0.0, 440.0, 0.0])  # unvoiced, voiced, unvoiced over 0.3s
    out = resample_f0(src, duration=0.3, hop_seconds=0.05)
    assert out.size == 6
    # The middle frame's pitch must never bleed a fake value into its neighbours.
    assert set(np.unique(out)).issubset({0.0, 440.0})


def test_resample_f0_empty() -> None:
    assert resample_f0(np.zeros(0), 1.0, 0.01).size == 0


# --------------------------------------------------------------------------- #
# report orchestration, via a fake extractor (no torch)
# --------------------------------------------------------------------------- #

class _FakeExtractor:
    """Returns a constant pitch for the whole clip -- enough to drive report."""

    def __init__(self, hz: float) -> None:
        self.hz = hz

    def info(self) -> BackendInfo:
        return BackendInfo(backend_id="fake", display_name_he="בדיקה", available=True)

    def extract(
        self, audio: AudioBuffer, device: DeviceHint, hop_seconds: float = 0.01
    ) -> F0Curve:
        n = int(round(audio.seconds / hop_seconds))
        return F0Curve(hz=np.full(n, self.hz), hop_seconds=hop_seconds)

    def unload(self) -> None:
        pass


def _tone(hz: float, seconds: float, sr: int = 44100) -> AudioBuffer:
    t = np.arange(int(sr * seconds)) / sr
    wave = (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return AudioBuffer(samples=wave[None, :], sample_rate=sr)


def test_analyze_vocal_produces_a_serialisable_report() -> None:
    import json

    audio = _tone(220.0, 30.0)
    report = analyze_vocal(audio, _FakeExtractor(220.0), source="tone.wav")

    assert report.f0_method == "fake"
    assert report.range.p50 == pytest.approx(57.0, abs=0.3)  # A3
    assert report.voiced_fraction > 0.9
    assert report.preview.best is not None

    payload = json.dumps(report.to_dict(), ensure_ascii=False)
    back = json.loads(payload)
    assert back["range"]["median_note"] == "A3"
    assert back["f0_track"]  # decimated curve is present for the UI
