"""Sample-accurate buffer arithmetic.

Every stage of the pipeline must hand back audio the exact length it received.
A single sample of drift per stage becomes an audible flam by the time the
vocal is mixed back against the instrumental, so length handling lives here and
is used everywhere rather than re-derived per module.
"""

from __future__ import annotations

import numpy as np

from svc_engine.backends.base import AudioBuffer

__all__ = [
    "silence_like",
    "fit_length",
    "align_all",
    "add",
    "subtract",
    "peak",
    "rms",
    "is_silent",
    "crossfade_join",
]


def silence_like(reference: AudioBuffer) -> AudioBuffer:
    return AudioBuffer(
        samples=np.zeros_like(reference.samples), sample_rate=reference.sample_rate
    )


def fit_length(buffer: AudioBuffer, frames: int) -> AudioBuffer:
    """Trim or zero-pad to exactly `frames`. Never resamples."""
    have = buffer.frames
    if have == frames:
        return buffer
    if have > frames:
        return AudioBuffer(
            samples=buffer.samples[:, :frames].copy(), sample_rate=buffer.sample_rate
        )
    pad = np.zeros((buffer.channels, frames - have), dtype=np.float32)
    return AudioBuffer(
        samples=np.concatenate([buffer.samples, pad], axis=1), sample_rate=buffer.sample_rate
    )


def align_all(buffers: list[AudioBuffer], frames: int | None = None) -> list[AudioBuffer]:
    """Bring a set of stems to a common length -- the shortest, unless told."""
    if not buffers:
        return []
    target = frames if frames is not None else min(b.frames for b in buffers)
    return [fit_length(b, target) for b in buffers]


def _binary(a: AudioBuffer, b: AudioBuffer, sign: float) -> AudioBuffer:
    if a.sample_rate != b.sample_rate:
        raise ValueError(f"sample rate mismatch: {a.sample_rate} vs {b.sample_rate}")
    channels = max(a.channels, b.channels)
    frames = min(a.frames, b.frames)

    def widen(buf: AudioBuffer) -> np.ndarray:
        data = buf.samples[:, :frames]
        if buf.channels == channels:
            return data
        return np.repeat(data, channels // buf.channels, axis=0)[:channels]

    return AudioBuffer(
        samples=(widen(a) + sign * widen(b)).astype(np.float32), sample_rate=a.sample_rate
    )


def add(a: AudioBuffer, b: AudioBuffer) -> AudioBuffer:
    return _binary(a, b, 1.0)


def subtract(a: AudioBuffer, b: AudioBuffer) -> AudioBuffer:
    """`a - b`. Used to derive a stem a model did not return directly."""
    return _binary(a, b, -1.0)


def peak(buffer: AudioBuffer) -> float:
    return float(np.max(np.abs(buffer.samples))) if buffer.samples.size else 0.0


def rms(buffer: AudioBuffer) -> float:
    if not buffer.samples.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(buffer.samples, dtype=np.float64))))


def is_silent(buffer: AudioBuffer, threshold_db: float = -60.0) -> bool:
    level = rms(buffer)
    if level <= 0.0:
        return True
    return 20.0 * float(np.log10(level)) < threshold_db


def crossfade_join(
    left: AudioBuffer, right: AudioBuffer, overlap_frames: int
) -> AudioBuffer:
    """Join two chunks with a linear crossfade over their overlap.

    Linear, deliberately -- the gains sum to exactly 1 everywhere. These two
    chunks are consecutive windows over the *same* source, so in the overlap
    region they carry nearly identical, in-phase audio. An equal-power
    (cos/sin) fade is the right choice for crossfading unrelated material, but
    on correlated content its gains sum to as much as √2, producing a +3dB bump
    at every seam instead of removing one.
    """
    if left.sample_rate != right.sample_rate:
        raise ValueError("sample rate mismatch")
    overlap = max(0, min(overlap_frames, left.frames, right.frames))
    if overlap == 0:
        return AudioBuffer(
            samples=np.concatenate([left.samples, right.samples], axis=1),
            sample_rate=left.sample_rate,
        )

    fade_in = np.linspace(0.0, 1.0, overlap, endpoint=False, dtype=np.float32)
    fade_out = (1.0 - fade_in).astype(np.float32)

    head = left.samples[:, : left.frames - overlap]
    blend = left.samples[:, left.frames - overlap :] * fade_out + (
        right.samples[:, :overlap] * fade_in
    )
    tail = right.samples[:, overlap:]
    return AudioBuffer(
        samples=np.concatenate([head, blend, tail], axis=1).astype(np.float32),
        sample_rate=left.sample_rate,
    )
