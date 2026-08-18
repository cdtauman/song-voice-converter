"""The three Phase-6 acoustic-space strategies (A/B/C)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from scipy.signal import fftconvolve, lfilter

from svc_engine.audio.buffers import add, fit_length, rms
from svc_engine.audio.io import match_channels
from svc_engine.backends.base import AudioBuffer

__all__ = [
    "AmbienceMeasurement",
    "AmbienceResult",
    "AmbienceStrategy",
    "apply_ambience",
    "measure_ambience",
]


class AmbienceStrategy(StrEnum):
    ORIGINAL = "A"
    PARAMETRIC = "B"
    HYBRID = "C"


@dataclass(frozen=True)
class AmbienceMeasurement:
    available: bool
    rt60_seconds: float
    wet_dry_ratio: float
    note: str = ""


@dataclass(frozen=True)
class AmbienceResult:
    vocal: AudioBuffer
    strategy: AmbienceStrategy
    measurement: AmbienceMeasurement
    original_layer_gain: float


def measure_ambience(
    dry_vocal: AudioBuffer,
    ambience: AudioBuffer | None,
) -> AmbienceMeasurement:
    """Estimate wet/dry level and RT60 from the removed room layer."""
    if ambience is None or not ambience.samples.size or rms(ambience) < 1e-7:
        return AmbienceMeasurement(False, 0.8, 0.0, "no separated ambience layer")
    wet_ratio = float(np.clip(rms(ambience) / max(rms(dry_vocal), 1e-7), 0.01, 0.45))
    mono = np.mean(np.asarray(ambience.samples, dtype=np.float64), axis=0)
    hop = max(1, int(ambience.sample_rate * 0.02))
    usable = (mono.size // hop) * hop
    if usable < hop * 8:
        return AmbienceMeasurement(True, 0.8, wet_ratio, "tail too short for RT60 fit")
    energy = np.mean(mono[:usable].reshape(-1, hop) ** 2, axis=1)
    # Schroeder decay curve: reverse-integrated energy. Fit -5..-35 dB.
    decay = np.cumsum(energy[::-1])[::-1]
    decay_db = 10.0 * np.log10(np.maximum(decay / max(decay[0], 1e-15), 1e-12))
    t = np.arange(decay_db.size) * hop / ambience.sample_rate
    region = (decay_db <= -5.0) & (decay_db >= -35.0)
    if np.count_nonzero(region) < 5:
        rt60 = 0.8
        note = "insufficient decay span; used conservative RT60"
    else:
        slope = float(np.polyfit(t[region], decay_db[region], 1)[0])
        rt60 = float(np.clip(-60.0 / slope, 0.2, 4.0)) if slope < -1.0 else 0.8
        note = ""
    return AmbienceMeasurement(True, rt60, wet_ratio, note)


def _impulse_response(sample_rate: int, rt60: float) -> np.ndarray:
    length = max(32, int(sample_rate * min(4.0, max(0.2, rt60))))
    t = np.arange(length, dtype=np.float64) / sample_rate
    # Deterministic filtered noise makes a dense tail; sparse early reflections
    # provide room cues without pulling in another model or plugin.
    rng = np.random.default_rng(6006)
    noise = lfilter([1.0], [1.0, -0.72], rng.standard_normal(length))
    decay = np.power(10.0, -3.0 * t / max(rt60, 0.05))
    ir = noise * decay
    ir[: max(1, int(0.01 * sample_rate))] *= np.linspace(
        0.0, 1.0, max(1, int(0.01 * sample_rate))
    )
    for delay, gain in ((0.023, 0.7), (0.041, 0.5), (0.067, 0.35)):
        index = int(delay * sample_rate)
        if index < length:
            ir[index] += gain
    norm = np.sqrt(np.sum(ir * ir))
    return (ir / max(norm, 1e-9)).astype(np.float32)


def _synthetic_reverb(vocal: AudioBuffer, measurement: AmbienceMeasurement) -> AudioBuffer:
    ir = _impulse_response(vocal.sample_rate, measurement.rt60_seconds)
    wet = np.vstack(
        [fftconvolve(channel, ir, mode="full")[: vocal.frames] for channel in vocal.samples]
    ).astype(np.float32)
    wet_buffer = AudioBuffer(samples=wet, sample_rate=vocal.sample_rate)
    gain = measurement.wet_dry_ratio * rms(vocal) / max(rms(wet_buffer), 1e-9)
    return AudioBuffer(samples=wet * gain, sample_rate=vocal.sample_rate)


def apply_ambience(
    converted_vocal: AudioBuffer,
    reference_dry: AudioBuffer,
    original_ambience: AudioBuffer | None,
    strategy: AmbienceStrategy = AmbienceStrategy.PARAMETRIC,
    *,
    hybrid_original_gain: float = 0.15,
) -> AmbienceResult:
    """Apply A (original), B (new measured room), or C (new + quiet original)."""
    strategy = AmbienceStrategy(strategy)
    measurement = measure_ambience(reference_dry, original_ambience)
    if not measurement.available or original_ambience is None:
        return AmbienceResult(converted_vocal, strategy, measurement, 0.0)

    original = match_channels(fit_length(original_ambience, converted_vocal.frames),
                              converted_vocal.channels)
    if strategy is AmbienceStrategy.ORIGINAL:
        return AmbienceResult(add(converted_vocal, original), strategy, measurement, 1.0)

    synthetic = _synthetic_reverb(converted_vocal, measurement)
    vocal = add(converted_vocal, synthetic)
    if strategy is AmbienceStrategy.HYBRID:
        original_quiet = AudioBuffer(
            samples=(original.samples * hybrid_original_gain).astype(np.float32),
            sample_rate=original.sample_rate,
        )
        vocal = add(vocal, original_quiet)
        return AmbienceResult(vocal, strategy, measurement, hybrid_original_gain)
    return AmbienceResult(vocal, strategy, measurement, 0.0)
