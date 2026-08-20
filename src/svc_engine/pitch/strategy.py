"""The shift decision -- the core of the whole project.

Research section 4 (``docs/research.md``): a song sung in one voice has to be
moved to sit naturally in another. Every shift ``s`` (in semitones) splits into

    s = 12*k + r        k = whole octaves, r in [-6, +5]

where the ``k`` octaves are *free of the playback*: a note and the note an octave
below it are the same note harmonically, so shifting the singing by a whole
octave keeps everything in key without touching the backing track. Only the
``r`` part changes the key of the song, and only ``r`` forces the playback to
move with it. That asymmetry is the reason a man can sing a woman's karaoke song:
he drops it an octave and the band never moves.

The engine turns that into a cost and minimises it. The cost weighs four things
that are computed here from measured data, plus one that is *measured per voice*
and stored in the profile:

    cost(s) = w1 * mass of the song's notes outside the target's comfort band
            + w2 * mass outside the target's absolute range      (heavy)
            + w3 * |shifted median - target median|              (in octaves)
            + w4 * playback penalty(|r|)                         (0 when r == 0)
            + w5 * measured quality-vs-shift penalty(s)          (0 when unmeasured)

The weights ``w1..w5`` are **calibration parameters, not constants** -- a
documented starting point here, tuned in the Phase 4 benchmark over three voice
profiles (``docs/testing.md`` section 5a, ``docs/roadmap.md`` Phase 4). The
``w5`` term is *absent, not faked* until the quality curve is measured against a
running conversion model: an unmeasured voice contributes zero there rather than
a guess (see ``pitch/quality_probe.py`` and ``profiles/profile.py``).

Pure numpy. No torch, no audio. The arithmetic is unit-tested against fixed
profiles exactly as ``docs/roadmap.md`` Phase 4 requires ("unit tests on the
maths: for predefined profiles, is the recommendation the expected one?").
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from svc_engine.analysis.midi import hz_to_midi, midi_to_note_name
from svc_engine.analysis.range import weighted_percentile
from svc_engine.profiles.profile import VoiceProfile

__all__ = [
    "CostWeights",
    "PitchDistribution",
    "ShiftCandidate",
    "ShiftDecision",
    "decide_shift",
    "decompose",
    "playback_penalty",
]


def decompose(s: int) -> tuple[int, int]:
    """Split a shift into whole octaves and an in-octave remainder ``r in [-6, +5]``.

    ``s = 12*k + r``. The remainder is the tritone-centred residue: it is what,
    and only what, forces the playback to move. ``r == 0`` means a pure octave
    shift, which the backing track never has to follow.
    """
    r = ((s + 6) % 12) - 6
    k = (s - r) // 12
    return int(k), int(r)


def playback_penalty(r: int) -> float:
    """How much moving the playback by ``r`` semitones is expected to hurt it.

    Normalised to [0, 1] over the possible ``|r| in {0..6}`` so it is comparable
    to the mass terms during weight calibration. Zero at ``r == 0`` -- an octave
    shift asks nothing of the playback. This is the *shape* of the penalty; the
    playback benchmark (``docs/testing.md`` 5a) decides how strongly it counts.
    """
    return abs(r) / 6.0


@dataclass(frozen=True)
class CostWeights:
    """The five weights of the cost function.

    Defaults are a documented **starting point**, not a calibrated result. Phase
    4's benchmark tunes them over three voice profiles; until then the shift
    engine runs on these and every report says so. ``w_quality`` multiplies the
    measured quality-vs-shift curve, which is 0 for any voice whose curve has not
    yet been measured -- so on an unmeasured voice this term simply drops out.
    """

    comfort: float = 1.0        # w1: notes outside the comfortable band
    absolute: float = 4.0       # w2: notes outside the absolute range (heavy)
    median: float = 1.0         # w3: distance of the shifted median from target
    playback: float = 0.5       # w4: cost of moving the playback by r
    quality: float = 1.0        # w5: measured quality-vs-shift penalty

    def to_dict(self) -> dict[str, float]:
        return {
            "comfort": self.comfort,
            "absolute": self.absolute,
            "median": self.median,
            "playback": self.playback,
            "quality": self.quality,
        }


@dataclass(frozen=True)
class PitchDistribution:
    """The song's sung pitches as energy-weighted mass over MIDI semitones.

    This is the honest input to the mass terms: "how much of the singing lands
    outside the target's range if we shift by ``s``" is a fraction of held,
    audible pitch, not a count of raw frames. Built from an F0 curve and a
    matching per-frame energy, exactly like ``analysis/range.py``; weights sum to
    one so the mass terms are fractions in [0, 1].
    """

    midi: np.ndarray            # voiced-frame pitches, semitones
    weight: np.ndarray          # normalised energy weights, sum == 1
    median: float               # energy-weighted p50, semitones

    @classmethod
    def from_f0(
        cls, f0_hz: np.ndarray, energy: np.ndarray | None = None
    ) -> PitchDistribution:
        hz = np.asarray(f0_hz, dtype=np.float64).ravel()
        midi = hz_to_midi(hz)
        voiced = ~np.isnan(midi)
        if energy is None:
            w = np.ones(hz.size, dtype=np.float64)
        else:
            w = np.clip(np.asarray(energy, dtype=np.float64).ravel(), 0.0, None)
            if w.size != hz.size:
                raise ValueError(
                    f"energy length {w.size} does not match f0 length {hz.size}"
                )
        keep = voiced & (w > 0.0)
        if not keep.any():
            raise ValueError("no voiced, audible frames; cannot build a pitch distribution")
        mv = midi[keep]
        wv = w[keep]
        wv = wv / wv.sum()
        median = float(weighted_percentile(mv, wv, np.array([50.0]))[0])
        return cls(midi=mv, weight=wv, median=median)

    def mass_outside(self, low: float, high: float, shift: float = 0.0) -> float:
        """Fraction of energy-weighted pitch mass falling outside ``[low, high]``
        once the whole distribution is moved by ``shift`` semitones."""
        moved = self.midi + shift
        outside = (moved < low) | (moved > high)
        return float(self.weight[outside].sum())


@dataclass(frozen=True)
class ShiftCandidate:
    """One evaluated shift, with the cost broken out so nothing is a black box."""

    semitones: int
    octaves: int                # k
    remainder: int              # r in [-6, +5]
    cost: float
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def needs_playback_shift(self) -> bool:
        return self.remainder != 0

    def to_dict(self) -> dict[str, object]:
        return {
            "semitones": self.semitones,
            "octaves": self.octaves,
            "remainder": self.remainder,
            "playback_semitones": self.remainder,
            "needs_playback_shift": self.needs_playback_shift,
            "cost": round(self.cost, 5),
            "breakdown": {k: round(v, 5) for k, v in self.breakdown.items()},
        }


@dataclass(frozen=True)
class ShiftDecision:
    """The recommended shift and the runners-up, ready to explain or serialise."""

    best: ShiftCandidate
    alternatives: tuple[ShiftCandidate, ...]
    weights: CostWeights
    song_median: float
    target: VoiceProfile
    quality_measured: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "best": self.best.to_dict(),
            "alternatives": [c.to_dict() for c in self.alternatives],
            "weights": self.weights.to_dict(),
            "song_median_midi": round(self.song_median, 3),
            "song_median_note": midi_to_note_name(self.song_median),
            "target_voice": self.target.name,
            "target_median_note": midi_to_note_name(self.target.median),
            # False means the w5 term was zero everywhere: the voice has no
            # measured quality-vs-shift curve yet (Phase 4/5 measurement).
            "quality_curve_measured": self.quality_measured,
        }


def evaluate_shift(
    dist: PitchDistribution,
    target: VoiceProfile,
    weights: CostWeights,
    s: int,
) -> ShiftCandidate:
    """Cost of a single shift, with every term recorded."""
    k, r = decompose(s)

    comfort_mass = dist.mass_outside(target.comfort_low, target.comfort_high, s)
    absolute_mass = dist.mass_outside(target.abs_low, target.abs_high, s)
    median_gap = abs((dist.median + s) - target.median) / 12.0
    playback = playback_penalty(r)
    quality = _quality_penalty(target, s)

    breakdown = {
        "comfort": weights.comfort * comfort_mass,
        "absolute": weights.absolute * absolute_mass,
        "median": weights.median * median_gap,
        "playback": weights.playback * playback,
        "quality": weights.quality * quality,
    }
    return ShiftCandidate(
        semitones=s,
        octaves=k,
        remainder=r,
        cost=float(sum(breakdown.values())),
        breakdown=breakdown,
    )


def decide_shift(
    dist: PitchDistribution,
    target: VoiceProfile,
    weights: CostWeights | None = None,
    search: range | None = None,
    n_alternatives: int = 2,
) -> ShiftDecision:
    """Rank every candidate shift by cost and return the best plus runners-up.

    ``search`` defaults to every integer semitone in ``[-24, +24]`` -- two
    octaves each way covers female<->male and then some, and evaluating fifty
    candidates over a distribution of a few thousand frames is free. Ties break
    toward the smaller ``|s|`` and then the smaller ``|r|``, so an octave-clean
    answer wins a dead heat.
    """
    weights = weights or CostWeights()
    search = search or range(-24, 25)

    candidates = [evaluate_shift(dist, target, weights, s) for s in search]
    candidates.sort(key=lambda c: (round(c.cost, 9), abs(c.semitones), abs(c.remainder)))

    best = candidates[0]
    alternatives = tuple(_distinct_alternatives(candidates[1:], best, n_alternatives))
    return ShiftDecision(
        best=best,
        alternatives=alternatives,
        weights=weights,
        song_median=dist.median,
        target=target,
        quality_measured=_has_quality_curve(target),
    )


def _distinct_alternatives(
    ranked: list[ShiftCandidate], best: ShiftCandidate, n: int
) -> list[ShiftCandidate]:
    """Runners-up that differ meaningfully from the winner and each other.

    Two shifts an octave apart with the same ``r`` say the same musical thing;
    listing both wastes the user's attention. Alternatives must bring a new
    remainder ``r`` (a genuinely different playback trade-off)."""
    seen_r = {best.remainder}
    out: list[ShiftCandidate] = []
    for c in ranked:
        if c.remainder in seen_r:
            continue
        seen_r.add(c.remainder)
        out.append(c)
        if len(out) >= n:
            break
    return out


def _has_quality_curve(target: VoiceProfile) -> bool:
    curve = getattr(target, "quality_vs_shift", None)
    return bool(curve)


def _quality_penalty(target: VoiceProfile, s: int) -> float:
    """Measured quality-vs-shift penalty at ``s``, or 0 if the voice has no curve.

    The curve is a set of (shift, penalty) samples measured once per voice
    against a running conversion model (``pitch/quality_probe.py``). We linearly
    interpolate between measured shifts and hold the endpoints flat. A voice
    without a curve returns 0 -- the term is absent, never invented.
    """
    curve = getattr(target, "quality_vs_shift", None)
    if not curve:
        return 0.0
    shifts = np.array([float(p[0]) for p in curve], dtype=np.float64)
    penalties = np.array([float(p[1]) for p in curve], dtype=np.float64)
    order = np.argsort(shifts)
    return float(np.interp(float(s), shifts[order], penalties[order]))
