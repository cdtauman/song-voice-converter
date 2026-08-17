"""Frame-level features that the analysis stages share.

The one that matters is per-frame energy on the *same* time grid as the F0
curve: the range statistic weights pitches by it, the segmenter gates on it, and
the preview picker scores with it. Computing it once, here, keeps those three
honest about lining up frame-for-frame with the pitches.

Pure numpy.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_rms", "align_to"]


def frame_rms(
    mono: np.ndarray,
    sample_rate: int,
    hop_seconds: float,
    win_seconds: float = 0.04,
) -> np.ndarray:
    """RMS energy per frame, one frame every `hop_seconds`, centred windows."""
    x = np.asarray(mono, dtype=np.float64).ravel()
    hop = max(1, int(round(hop_seconds * sample_rate)))
    win = max(hop, int(round(win_seconds * sample_rate)))
    half = win // 2
    n_frames = int(np.ceil(x.size / hop)) if x.size else 0
    padded = np.pad(x, (half, half), mode="constant")
    out = np.empty(n_frames, dtype=np.float64)
    for i in range(n_frames):
        start = i * hop
        frame = padded[start : start + win]
        out[i] = np.sqrt(np.mean(frame**2)) if frame.size else 0.0
    return out


def align_to(values: np.ndarray, n_frames: int) -> np.ndarray:
    """Trim or edge-pad a per-frame array to exactly `n_frames`.

    F0 models and the RMS helper round frame counts slightly differently; a
    one- or two-frame mismatch is normal and must not desync the weights from
    the pitches. Padding uses the edge value, not zero, so a trailing pad does
    not read as sudden silence.
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == n_frames:
        return values
    if values.size > n_frames:
        return values[:n_frames]
    if values.size == 0:
        return np.zeros(n_frames, dtype=np.float64)
    pad = np.full(n_frames - values.size, values[-1], dtype=np.float64)
    return np.concatenate([values, pad])
