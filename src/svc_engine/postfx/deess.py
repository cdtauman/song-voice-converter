"""A gentle, signal-dependent de-esser built from standard SciPy filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import butter, sosfiltfilt

from svc_engine.backends.base import AudioBuffer

__all__ = ["DeEssReport", "deess"]


@dataclass(frozen=True)
class DeEssReport:
    active_fraction: float
    maximum_reduction_db: float
    crossover_hz: float


def deess(
    audio: AudioBuffer,
    *,
    crossover_hz: float = 5500.0,
    max_reduction_db: float = 5.0,
) -> tuple[AudioBuffer, DeEssReport]:
    """Reduce only high-band energy that dominates the local full-band RMS."""
    nyquist = audio.sample_rate / 2.0
    if audio.frames < 32 or crossover_hz >= nyquist * 0.95:
        return audio, DeEssReport(0.0, 0.0, crossover_hz)
    sos = butter(4, crossover_hz / nyquist, btype="highpass", output="sos")
    high = sosfiltfilt(sos, audio.samples, axis=1).astype(np.float32)
    full_power = gaussian_filter1d(
        np.mean(audio.samples.astype(np.float64) ** 2, axis=0),
        sigma=max(1.0, audio.sample_rate * 0.004),
    )
    high_power = gaussian_filter1d(
        np.mean(high.astype(np.float64) ** 2, axis=0),
        sigma=max(1.0, audio.sample_rate * 0.004),
    )
    ratio = high_power / np.maximum(full_power, 1e-9)
    # Starts acting when the sibilant band carries >35% of local energy and
    # reaches the configured maximum at 75%.
    activity = np.clip((ratio - 0.35) / 0.40, 0.0, 1.0)
    activity = gaussian_filter1d(activity, sigma=max(1.0, audio.sample_rate * 0.002))
    reduction_db = max_reduction_db * activity
    high_gain = np.power(10.0, -reduction_db / 20.0)
    samples = audio.samples - high + high * high_gain[None, :]
    return (
        AudioBuffer(samples=samples.astype(np.float32), sample_rate=audio.sample_rate),
        DeEssReport(
            active_fraction=float(np.mean(activity > 0.05)),
            maximum_reduction_db=float(np.max(reduction_db)),
            crossover_hz=crossover_hz,
        ),
    )
