"""Conservative removal of conversion clicks and invalid samples.

The detector repairs only very short outlier runs.  A real drum transient or
plosive lasts longer and is deliberately left alone; a one-sample discontinuity
from a chunk/model failure is interpolated from its neighbours.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt

from svc_engine.backends.base import AudioBuffer

__all__ = ["RepairReport", "repair_artifacts"]


@dataclass(frozen=True)
class RepairReport:
    repaired_samples: int
    nonfinite_samples: int
    peak_before: float
    peak_after: float


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(mask.astype(np.int8), (1, 1))
    changes = np.flatnonzero(np.diff(padded))
    return [(int(a), int(b)) for a, b in changes.reshape(-1, 2)]


def repair_artifacts(
    audio: AudioBuffer,
    *,
    sensitivity: float = 10.0,
    max_click_samples: int = 3,
) -> tuple[AudioBuffer, RepairReport]:
    """Replace non-finite values and isolated click outliers, preserving length."""
    samples = np.asarray(audio.samples, dtype=np.float32).copy()
    before = float(np.nanmax(np.abs(samples))) if samples.size else 0.0
    finite = np.isfinite(samples)
    nonfinite = int(np.size(finite) - np.count_nonzero(finite))
    samples[~finite] = 0.0
    repaired = 0

    for channel in range(samples.shape[0]):
        x = samples[channel]
        if x.size < 7:
            continue
        baseline = medfilt(x, kernel_size=5).astype(np.float32)
        residual = np.abs(x - baseline)
        centre = float(np.median(residual))
        mad = 1.4826 * float(np.median(np.abs(residual - centre)))
        # The absolute floor prevents quantisation noise in near-silence from
        # being "repaired". The local edge test rejects sustained transients.
        threshold = max(0.03, centre + sensitivity * max(mad, 1e-6))
        candidates = residual > threshold
        candidates[:2] = False
        candidates[-2:] = False
        for start, end in _runs(candidates):
            if end - start > max_click_samples:
                continue
            left, right = float(x[start - 1]), float(x[end])
            x[start:end] = np.linspace(left, right, end - start + 2, dtype=np.float32)[1:-1]
            repaired += end - start

    after = float(np.max(np.abs(samples))) if samples.size else 0.0
    return (
        AudioBuffer(samples=samples, sample_rate=audio.sample_rate),
        RepairReport(repaired, nonfinite, before, after),
    )
