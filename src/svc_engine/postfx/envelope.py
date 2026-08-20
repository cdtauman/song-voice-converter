"""Match the converted vocal's loudness envelope to the original performance."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter1d

from svc_engine.backends.base import AudioBuffer

__all__ = ["EnvelopeReport", "envelope_correlation", "match_envelope"]


@dataclass(frozen=True)
class EnvelopeReport:
    correlation_before: float | None
    correlation_after: float | None
    rms_mix_rate: float
    gain_min_db: float
    gain_max_db: float


def _envelope(audio: AudioBuffer, hop_seconds: float = 0.02) -> np.ndarray:
    mono = np.mean(np.asarray(audio.samples, dtype=np.float64), axis=0)
    hop = max(1, int(audio.sample_rate * hop_seconds))
    usable = (mono.size // hop) * hop
    if usable == 0:
        return np.array([0.0])
    frames = mono[:usable].reshape(-1, hop)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    return gaussian_filter1d(rms, sigma=2.0, mode="nearest")


def _align(values: np.ndarray, size: int) -> np.ndarray:
    if values.size == size:
        return values
    return np.interp(np.linspace(0, 1, size), np.linspace(0, 1, values.size), values)


def _correlation(a: np.ndarray, b: np.ndarray) -> float | None:
    if a.size < 3 or b.size < 3 or np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def envelope_correlation(reference: AudioBuffer, candidate: AudioBuffer) -> float | None:
    a = _envelope(reference)
    b = _align(_envelope(candidate), a.size)
    return _correlation(a, b)


def match_envelope(
    reference: AudioBuffer,
    converted: AudioBuffer,
    rms_mix_rate: float = 0.25,
    *,
    max_gain_db: float = 9.0,
) -> tuple[AudioBuffer, EnvelopeReport]:
    """Blend source and converted RMS envelopes using RVC's rate convention.

    0.0 follows the source completely, 1.0 keeps the conversion envelope.
    Gain is smoothed and bounded so breaths/silence cannot explode.
    """
    rate = float(np.clip(rms_mix_rate, 0.0, 1.0))
    ref_env = _envelope(reference)
    out_env = _align(_envelope(converted), ref_env.size)
    before = _correlation(ref_env, out_env)
    eps = 1e-5
    target = np.power(np.maximum(ref_env, eps), 1.0 - rate) * np.power(
        np.maximum(out_env, eps), rate
    )
    frame_gain = target / np.maximum(out_env, eps)
    limit = 10.0 ** (max_gain_db / 20.0)
    frame_gain = np.clip(frame_gain, 1.0 / limit, limit)
    frame_gain = gaussian_filter1d(frame_gain, sigma=3.0, mode="nearest")
    gain = np.interp(
        np.linspace(0.0, 1.0, converted.frames),
        np.linspace(0.0, 1.0, frame_gain.size),
        frame_gain,
    )
    samples = (converted.samples.astype(np.float64) * gain[None, :]).astype(np.float32)
    result = AudioBuffer(samples=samples, sample_rate=converted.sample_rate)
    after = envelope_correlation(reference, result)
    gain_db = 20.0 * np.log10(np.maximum(frame_gain, 1e-9))
    return result, EnvelopeReport(
        before, after, rate, float(np.min(gain_db)), float(np.max(gain_db))
    )
