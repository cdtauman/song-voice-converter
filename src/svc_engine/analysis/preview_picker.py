"""Pick the ~25 seconds that best represents a song.

Research section 7: separation is expensive and conversion is cheap, so the app
separates once and then previews a short slice many times. That slice has to be
*worth* previewing -- a silent intro or an instrumental bridge tells the user
nothing about how the voice will sound.

Four measurable things make a window representative, scored per candidate and
combined by weights. The weights are a documented starting point, not a
calibrated law -- like every threshold in this project (see docs/testing.md),
they belong to a benchmark, and the picker returns the full breakdown so that
benchmark has something to calibrate against.

Pure numpy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from svc_engine.analysis.midi import hz_to_midi

__all__ = ["WindowScore", "PreviewChoice", "PreviewWeights", "pick_preview"]


@dataclass(frozen=True)
class PreviewWeights:
    density: float = 0.40      # how much of the window is actually sung
    stability: float = 0.20    # steady pitch, not a scribble of octave jumps
    width: float = 0.15        # covers a fair spread of the singer's range
    recurrence: float = 0.25   # content that repeats -> probably the chorus

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.density, self.stability, self.width, self.recurrence)


@dataclass(frozen=True)
class WindowScore:
    start_seconds: float
    end_seconds: float
    score: float
    density: float
    stability: float
    width: float
    recurrence: float

    def to_dict(self) -> dict[str, float]:
        return {
            "start": round(self.start_seconds, 3),
            "end": round(self.end_seconds, 3),
            "score": round(self.score, 4),
            "density": round(self.density, 4),
            "stability": round(self.stability, 4),
            "width": round(self.width, 4),
            "recurrence": round(self.recurrence, 4),
        }


@dataclass(frozen=True)
class PreviewChoice:
    best: WindowScore | None
    windows: tuple[WindowScore, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        return {
            "chosen": self.best.to_dict() if self.best else None,
            "candidates": [w.to_dict() for w in self.windows],
        }


#: A frozen singleton so it can be a default argument without B008.
_DEFAULT_WEIGHTS = PreviewWeights()


def _minmax(raw: np.ndarray) -> np.ndarray:
    """Scale to [0, 1]. A flat metric contributes nothing rather than NaN."""
    lo, hi = float(raw.min()), float(raw.max())
    if hi - lo < 1e-12:
        return np.zeros_like(raw)
    return (raw - lo) / (hi - lo)


def _signature(midi_window: np.ndarray, weight_window: np.ndarray) -> np.ndarray:
    """Energy-weighted pitch-class histogram of a window, for recurrence."""
    sig = np.zeros(12, dtype=np.float64)
    voiced = ~np.isnan(midi_window)
    if not voiced.any():
        return sig
    pc = np.mod(np.rint(midi_window[voiced]).astype(int), 12)
    np.add.at(sig, pc, weight_window[voiced])
    norm = np.linalg.norm(sig)
    return sig / norm if norm > 0 else sig


def pick_preview(
    f0_hz: np.ndarray,
    hop_seconds: float,
    energy: np.ndarray | None = None,
    window_seconds: float = 25.0,
    stride_seconds: float = 5.0,
    weights: PreviewWeights = _DEFAULT_WEIGHTS,
) -> PreviewChoice:
    """Score every window and return the best plus the full field."""
    hz = np.asarray(f0_hz, dtype=np.float64).ravel()
    n = hz.size
    if n == 0:
        return PreviewChoice(best=None)

    midi = hz_to_midi(hz)
    if energy is None:
        energy = (~np.isnan(midi)).astype(np.float64)
    else:
        energy = np.asarray(energy, dtype=np.float64).ravel()
        if energy.size != n:
            raise ValueError(f"energy length {energy.size} != f0 length {n}")

    win = max(1, int(round(window_seconds / hop_seconds)))
    stride = max(1, int(round(stride_seconds / hop_seconds)))

    if n <= win:
        starts = [0]
        win = n
    else:
        starts = list(range(0, n - win + 1, stride))
        if starts[-1] != n - win:
            starts.append(n - win)

    raw = np.zeros((len(starts), 4), dtype=np.float64)
    signatures: list[np.ndarray] = []
    for i, s in enumerate(starts):
        mw = midi[s : s + win]
        ew = energy[s : s + win]
        voiced = ~np.isnan(mw)
        density = float(voiced.mean())

        if voiced.sum() >= 2:
            deltas = np.abs(np.diff(mw[voiced]))
            stability = 1.0 / (1.0 + float(np.mean(deltas)))
            p05, p95 = np.percentile(mw[voiced], [5, 95])
            width = float(p95 - p05)
        else:
            stability = 0.0
            width = 0.0

        raw[i] = (density, stability, width, 0.0)
        signatures.append(_signature(mw, ew))

    # Recurrence: how strongly a window's pitch content echoes elsewhere in the
    # song. Compared only against windows that do not overlap it, so a window is
    # never rewarded for resembling its own near-duplicate neighbour.
    for i, s in enumerate(starts):
        best_sim = 0.0
        for j, t in enumerate(starts):
            if i == j or abs(s - t) < win:
                continue
            best_sim = max(best_sim, float(np.dot(signatures[i], signatures[j])))
        raw[i, 3] = best_sim

    norm = np.column_stack([_minmax(raw[:, k]) for k in range(4)])
    w = np.array(weights.as_tuple(), dtype=np.float64)
    combined = norm @ w

    scored = [
        WindowScore(
            start_seconds=starts[i] * hop_seconds,
            end_seconds=(starts[i] + win) * hop_seconds,
            score=float(combined[i]),
            density=float(raw[i, 0]),
            stability=float(raw[i, 1]),
            width=float(raw[i, 2]),
            recurrence=float(raw[i, 3]),
        )
        for i in range(len(starts))
    ]
    best = max(scored, key=lambda ws: ws.score)
    return PreviewChoice(best=best, windows=tuple(scored))
