"""Where the singing is, and where it isn't.

The preview picker needs to avoid silent intros and long instrumental breaks,
and Phase 5 will convert only the sung regions. A frame is a candidate for
"voiced" when the F0 model committed to a pitch there; short gaps inside a
phrase are bridged and short blips of noise are dropped, so what comes out is
phrase-shaped, not frame-flickery.

Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["Segment", "find_voiced_segments", "voiced_mask", "voiced_fraction"]


@dataclass(frozen=True)
class Segment:
    start_seconds: float
    end_seconds: float

    @property
    def duration(self) -> float:
        return self.end_seconds - self.start_seconds

    def to_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start_seconds, 3),
            "end": round(self.end_seconds, 3),
            "duration": round(self.duration, 3),
        }


def voiced_mask(
    f0_hz: np.ndarray,
    energy: np.ndarray | None = None,
    energy_floor_ratio: float = 0.04,
) -> np.ndarray:
    """Per-frame boolean: a committed pitch, and (if given) enough energy.

    The energy gate is relative to the loudest frame, so it tracks the mix
    rather than an absolute level -- a quiet recording is not all "silence".
    """
    hz = np.asarray(f0_hz, dtype=np.float64).ravel()
    mask = hz > 0.0
    if energy is not None:
        e = np.asarray(energy, dtype=np.float64).ravel()
        if e.size != mask.size:
            raise ValueError(f"energy length {e.size} != f0 length {mask.size}")
        peak = float(e.max()) if e.size else 0.0
        if peak > 0.0:
            mask &= e >= energy_floor_ratio * peak
    return mask


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open [start, end) index runs of True in a boolean array."""
    if not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.where(edges == 1)[0] + 1)
    ends = list(np.where(edges == -1)[0] + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends, strict=True))


def find_voiced_segments(
    f0_hz: np.ndarray,
    hop_seconds: float,
    energy: np.ndarray | None = None,
    min_voiced_seconds: float = 0.2,
    bridge_gap_seconds: float = 0.25,
    energy_floor_ratio: float = 0.04,
) -> list[Segment]:
    """Phrase-shaped voiced regions in seconds.

    Gaps shorter than `bridge_gap_seconds` inside singing are bridged (a
    breath, a consonant, a short rest); runs shorter than `min_voiced_seconds`
    are dropped (a stray voiced frame in noise).
    """
    mask = voiced_mask(f0_hz, energy, energy_floor_ratio)
    if not mask.any():
        return []

    bridge = int(round(bridge_gap_seconds / hop_seconds))
    filled = mask.copy()
    for start, end in _runs(~mask):
        interior = start > 0 and end < mask.size  # never bridge head/tail
        if interior and (end - start) <= bridge:
            filled[start:end] = True

    min_frames = max(1, int(round(min_voiced_seconds / hop_seconds)))
    segments: list[Segment] = []
    for start, end in _runs(filled):
        if (end - start) < min_frames:
            continue
        segments.append(Segment(start * hop_seconds, end * hop_seconds))
    return segments


def voiced_fraction(segments: list[Segment], total_seconds: float) -> float:
    if total_seconds <= 0.0:
        return 0.0
    sung = sum(s.duration for s in segments)
    return min(1.0, sung / total_seconds)
