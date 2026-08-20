"""Measure how a voice's quality holds up across shifts -- the ``w5`` curve.

Research section 4.3 step 3: the formant penalty in the cost function used to be
a number I made up. The fix is to *measure* it, once per voice: run a short test
conversion at a handful of shifts, score each, and store the resulting
(shift -> penalty) curve in the voice profile. Then the cost function reads a
curve measured on *this* voice instead of a guess (``pitch/strategy.py``).

**What is Phase 4 here and what is not.** This module is the harness and the
scoring: it drives *an injected* ``ConversionBackend`` and turns its output into
a ``QualityCurve``. It deliberately does not implement a conversion backend --
RVC inference is Phase 5 (``docs/roadmap.md``), and Phase 4 must not start it. So
the curve is measured for real only once a backend exists (at voice import or
training, Phase 9); until then a voice's ``quality_vs_shift`` stays absent and
the ``w5`` term is zero, never invented.

The artifact scoring is pure numpy and real. Speaker similarity needs a speaker
embedding model, which is Phase 5+ material, so it is an *injected* function that
defaults to unavailable rather than being faked here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from svc_engine.backends.base import AudioBuffer, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionBackend, ConversionParams, VoiceHandle

__all__ = [
    "ArtifactStats",
    "QualitySample",
    "QualityCurve",
    "measure_artifacts",
    "probe_quality_curve",
    "DEFAULT_PROBE_SHIFTS",
]

#: The shifts research 4.3 step 3 measures: down a fifth, an octave, an octave
#: and a fourth, and the same upward -- enough to see where the voice degrades.
DEFAULT_PROBE_SHIFTS: tuple[int, ...] = (0, -5, -12, -17, 5, 12)

#: A speaker-similarity comparator: (reference, converted) -> similarity in
#: [0, 1]. None means "no similarity model available" -- Phase 5+ supplies one.
SimilarityFn = Callable[[AudioBuffer, AudioBuffer], float]


@dataclass(frozen=True)
class ArtifactStats:
    """Objective, model-free signs that a conversion went wrong."""

    clip_fraction: float       # samples at full scale
    click_score: float         # sudden sample-to-sample jumps
    nonfinite_fraction: float  # NaN/inf leaking through

    @property
    def penalty(self) -> float:
        """A single [0, 1] artifact penalty. Non-finite output is the worst
        possible sign and is weighted to dominate."""
        return float(
            min(
                1.0,
                4.0 * self.nonfinite_fraction
                + 2.0 * self.clip_fraction
                + self.click_score,
            )
        )


def measure_artifacts(audio: AudioBuffer, click_threshold: float = 0.5) -> ArtifactStats:
    """Cheap, honest artifact detection over a converted buffer.

    Clipping and non-finite samples are counted directly. "Clicks" are the
    fraction of neighbouring-sample jumps larger than ``click_threshold`` in
    normalised amplitude -- a discontinuity that size in a vocal is not singing,
    it is a glitch.
    """
    x = np.asarray(audio.samples, dtype=np.float64)
    total = x.size
    if total == 0:
        return ArtifactStats(0.0, 0.0, 0.0)

    finite = np.isfinite(x)
    nonfinite_fraction = float((~finite).sum()) / total

    safe = np.where(finite, x, 0.0)
    clip_fraction = float((np.abs(safe) >= 0.999).sum()) / total

    if safe.shape[-1] > 1:
        jumps = np.abs(np.diff(safe, axis=-1))
        click_score = float((jumps > click_threshold).mean())
    else:
        click_score = 0.0

    return ArtifactStats(
        clip_fraction=clip_fraction,
        click_score=click_score,
        nonfinite_fraction=nonfinite_fraction,
    )


@dataclass(frozen=True)
class QualitySample:
    """One probed shift: what it cost the voice."""

    shift: int
    artifacts: ArtifactStats
    similarity: float | None  # None when no speaker model was supplied

    @property
    def penalty(self) -> float:
        """Combined [0, 1] penalty. Similarity, when measured, adds ``1 - sim``;
        when absent the penalty rests on the artifact score alone."""
        base = self.artifacts.penalty
        if self.similarity is None:
            return base
        return float(min(1.0, base + (1.0 - self.similarity)))


@dataclass(frozen=True)
class QualityCurve:
    """The per-voice quality-vs-shift curve, ready for the profile and the cost."""

    samples: tuple[QualitySample, ...]
    similarity_available: bool

    def to_profile_list(self) -> list[list[float]]:
        """``[[shift, penalty], ...]`` -- exactly what ``VoiceProfile`` stores and
        ``strategy._quality_penalty`` interpolates."""
        return [[float(s.shift), round(s.penalty, 5)] for s in sorted(
            self.samples, key=lambda s: s.shift
        )]

    def to_dict(self) -> dict[str, object]:
        return {
            "similarity_available": self.similarity_available,
            "samples": [
                {
                    "shift": s.shift,
                    "penalty": round(s.penalty, 5),
                    "similarity": None if s.similarity is None else round(s.similarity, 5),
                    "clip_fraction": round(s.artifacts.clip_fraction, 6),
                    "click_score": round(s.artifacts.click_score, 6),
                    "nonfinite_fraction": round(s.artifacts.nonfinite_fraction, 6),
                }
                for s in sorted(self.samples, key=lambda s: s.shift)
            ],
        }


def probe_quality_curve(
    reference_vocal: AudioBuffer,
    f0: F0Curve,
    backend: ConversionBackend,
    voice: VoiceHandle,
    device: DeviceHint | None = None,
    shifts: Sequence[int] = DEFAULT_PROBE_SHIFTS,
    similarity_fn: SimilarityFn | None = None,
    base_params: ConversionParams | None = None,
) -> QualityCurve:
    """Run the test conversion at each shift and score the output.

    ``backend`` is injected: this harness never constructs a conversion engine of
    its own. With a real backend and a real voice this produces the curve that
    Phase 4/5 stores in the profile; in a test it runs against a fake backend so
    the scoring and aggregation are covered without the heavy stack.
    """
    device = device or DeviceHint()
    base = base_params or ConversionParams()
    backend.load(voice, device)

    samples: list[QualitySample] = []
    try:
        for s in shifts:
            params = ConversionParams(
                semitones=int(s),
                index_rate=base.index_rate,
                protect=base.protect,
                rms_mix_rate=base.rms_mix_rate,
                filter_radius=base.filter_radius,
                formant_shift=base.formant_shift,
            )
            converted = backend.convert(reference_vocal, f0, params)
            similarity = (
                None if similarity_fn is None else float(similarity_fn(reference_vocal, converted))
            )
            samples.append(
                QualitySample(
                    shift=int(s),
                    artifacts=measure_artifacts(converted),
                    similarity=similarity,
                )
            )
    finally:
        backend.unload()

    return QualityCurve(samples=tuple(samples), similarity_available=similarity_fn is not None)
