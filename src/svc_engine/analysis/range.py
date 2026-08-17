"""Vocal range as an energy-weighted MIDI distribution.

Research section 4.3, step 1: the singer's profile in this song is not "lowest
note to highest note" -- one grunt on a low note must not widen the range the
pitch engine plans against. It is a set of percentiles over the pitches the
singer actually *held*, weighted by how loud and how long each was. A note sung
for a whole phrase counts; a passing sixteenth barely does.

Pure numpy: the arithmetic here is exactly what the Phase 4 cost function reads,
and it is covered by unit tests that never load an F0 model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from svc_engine.analysis.midi import hz_to_midi, midi_to_note_name

__all__ = ["RangeStats", "compute_range", "weighted_percentile"]


@dataclass(frozen=True)
class RangeStats:
    """Where a voice lives, in MIDI semitones. `nan` means no voiced frame."""

    p05: float
    p25: float
    p50: float
    p75: float
    p95: float
    min_midi: float
    max_midi: float
    voiced_fraction: float
    voiced_frames: int

    @property
    def comfortable_span(self) -> float:
        """p05..p95 in semitones -- the working range, robust to outliers."""
        if np.isnan(self.p05) or np.isnan(self.p95):
            return 0.0
        return self.p95 - self.p05

    @property
    def full_span(self) -> float:
        if np.isnan(self.min_midi) or np.isnan(self.max_midi):
            return 0.0
        return self.max_midi - self.min_midi

    def to_dict(self) -> dict[str, object]:
        def named(value: float) -> str | None:
            return None if np.isnan(value) else midi_to_note_name(value)

        return {
            "unit": "midi",
            "p05": _num(self.p05),
            "p25": _num(self.p25),
            "p50": _num(self.p50),
            "p75": _num(self.p75),
            "p95": _num(self.p95),
            "min": _num(self.min_midi),
            "max": _num(self.max_midi),
            "median_note": named(self.p50),
            "low_note": named(self.p05),
            "high_note": named(self.p95),
            "comfortable_span_semitones": _num(self.comfortable_span),
            "full_span_semitones": _num(self.full_span),
            "voiced_fraction": _num(self.voiced_fraction),
            "voiced_frames": self.voiced_frames,
        }


def _num(value: float) -> float | None:
    return None if np.isnan(value) else round(float(value), 3)


def weighted_percentile(
    values: np.ndarray, weights: np.ndarray, percentiles: np.ndarray
) -> np.ndarray:
    """Percentiles of `values` weighted by `weights`.

    Linear interpolation on the cumulative weight midpoints -- the standard
    "type 7"-style estimator generalised to weights, so equal weights reproduce
    ordinary percentiles closely. Assumes at least one positive weight.
    """
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cum = np.cumsum(w)
    total = cum[-1]
    # Midpoint of each sample's weight interval, normalised to [0, 1].
    midpoints = (cum - 0.5 * w) / total
    return np.interp(np.asarray(percentiles, dtype=np.float64) / 100.0, midpoints, v)


def compute_range(
    f0_hz: np.ndarray,
    weights: np.ndarray | None = None,
) -> RangeStats:
    """Energy-weighted MIDI percentiles over the voiced frames of an F0 curve.

    `weights` is a per-frame non-negative energy (e.g. RMS). If omitted every
    voiced frame counts equally -- still correct, just not loudness-aware.
    """
    hz = np.asarray(f0_hz, dtype=np.float64).ravel()
    midi = hz_to_midi(hz)
    voiced = ~np.isnan(midi)
    total_frames = int(hz.size)

    if weights is None:
        w = np.ones(total_frames, dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size != total_frames:
            raise ValueError(
                f"weights length {w.size} does not match f0 length {total_frames}"
            )
        w = np.clip(w, 0.0, None)

    keep = voiced & (w > 0.0)
    voiced_frames = int(keep.sum())
    voiced_fraction = voiced_frames / total_frames if total_frames else 0.0

    if voiced_frames == 0:
        nan = float("nan")
        return RangeStats(nan, nan, nan, nan, nan, nan, nan, voiced_fraction, 0)

    mv = midi[keep]
    wv = w[keep]
    p05, p25, p50, p75, p95 = weighted_percentile(
        mv, wv, np.array([5.0, 25.0, 50.0, 75.0, 95.0])
    )
    return RangeStats(
        p05=float(p05),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p95=float(p95),
        min_midi=float(mv.min()),
        max_midi=float(mv.max()),
        voiced_fraction=voiced_fraction,
        voiced_frames=voiced_frames,
    )
