"""Match the converted vocal's loudness envelope to the original performance.

RVC's `rms_mix_rate` blends the converted signal's own RMS envelope with the
source's, so the cover breathes like the original take instead of RVC's flatter
dynamics. This is the numpy equivalent of RVC-Project's `change_rms`
(infer/vc/pipeline.py, MIT); ours avoids torch so it can be tested, and matches
the same formula: `out *= rms_src^(1-rate) * rms_conv^(rate-1)`.

`rate` is the weight on the *converted* envelope: 1.0 leaves the conversion
untouched, 0.0 forces the source's envelope onto it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["frame_rms", "blend_rms"]


def frame_rms(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """RMS at RVC's cadence: one point per half second, over half-second frames."""
    x = np.asarray(signal, dtype=np.float64)
    hop = max(1, sample_rate // 2)
    frame = hop * 2
    if x.size == 0:
        return np.zeros(1, dtype=np.float64)
    pad = frame // 2
    padded = np.pad(x, (pad, pad), mode="constant")
    n = 1 + (len(padded) - frame) // hop if len(padded) >= frame else 1
    out = np.empty(max(1, n), dtype=np.float64)
    for i in range(len(out)):
        seg = padded[i * hop : i * hop + frame]
        out[i] = np.sqrt(np.mean(seg * seg)) if seg.size else 0.0
    return out


def _interp_to(values: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a short envelope to `length` samples."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == 1:
        return np.full(length, values[0], dtype=np.float64)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, values)


def blend_rms(
    source: np.ndarray,
    source_sr: int,
    converted: np.ndarray,
    converted_sr: int,
    rate: float,
) -> np.ndarray:
    """Scale `converted` so its loudness follows the source, weighted by `rate`."""
    converted = np.asarray(converted, dtype=np.float64)
    if converted.size == 0 or rate >= 1.0:
        return converted.astype(np.float32)

    rms_src = _interp_to(frame_rms(source, source_sr), converted.shape[0])
    rms_conv = _interp_to(frame_rms(converted, converted_sr), converted.shape[0])
    rms_conv = np.maximum(rms_conv, 1e-6)

    scaled = converted * (
        np.power(rms_src, 1.0 - rate) * np.power(rms_conv, rate - 1.0)
    )
    return scaled.astype(np.float32)
