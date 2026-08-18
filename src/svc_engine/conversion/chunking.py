"""Convert a long vocal in overlapping chunks, with no audible seam.

Phase 5 DoD: "no audible seams between chunks". A conversion backend works on a
bounded window at a time -- to cap memory, and because some engines degrade on
very long inputs -- so a five-minute vocal is cut into overlapping windows,
each converted, then reassembled with a crossfade over the overlap.

The arithmetic is chosen so reassembly is *sample-exact*: with hop = chunk -
overlap, windows starting at `i*hop`, and each converted chunk kept at its input
length, the crossfade sum telescopes back to the original length. That length
guarantee matters as much here as in `audio.buffers`: a vocal one sample longer
than the instrumental flams when they are mixed. This module never resamples and
never changes total length.

It is deliberately backend-agnostic: it drives an injected `ConversionBackend`
(RVC in the MVP, anything else later) and is pure numpy, so it is fully tested
without the heavy stack -- the same shape as `pitch.quality_probe`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from svc_engine.audio.buffers import crossfade_join, fit_length
from svc_engine.backends.base import AudioBuffer, F0Curve
from svc_engine.backends.conversion import ConversionBackend, ConversionParams

__all__ = ["ChunkWindow", "ChunkPlan", "plan_chunks", "convert_in_chunks"]

#: Defaults. A ~30s window keeps peak memory bounded on a long song; 0.5s of
#: overlap is long enough for an inaudible linear crossfade between two windows
#: over the same source (see audio.buffers.crossfade_join) without wasting work.
DEFAULT_CHUNK_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 0.5

ProgressHook = Callable[[int, int], None]


@dataclass(frozen=True)
class ChunkWindow:
    """One window over the source, in samples. `[start, end)`."""

    index: int
    start: int
    end: int

    @property
    def frames(self) -> int:
        return self.end - self.start


@dataclass(frozen=True)
class ChunkPlan:
    """How a signal of `total_frames` is cut into overlapping windows."""

    total_frames: int
    chunk_frames: int
    overlap_frames: int
    windows: tuple[ChunkWindow, ...]

    @property
    def hop_frames(self) -> int:
        return self.chunk_frames - self.overlap_frames

    @property
    def count(self) -> int:
        return len(self.windows)


def plan_chunks(
    total_frames: int,
    sample_rate: int,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> ChunkPlan:
    """Lay out the windows for a signal of `total_frames`.

    A signal that fits in one chunk yields a single full-length window and no
    crossfade. Overlap is clamped below the chunk so the hop is always positive.
    """
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    chunk_frames = max(1, int(round(chunk_seconds * sample_rate)))
    overlap_frames = max(0, int(round(overlap_seconds * sample_rate)))
    # Overlap must leave a positive hop, or windows never advance.
    overlap_frames = min(overlap_frames, chunk_frames - 1)

    if total_frames <= chunk_frames:
        window = ChunkWindow(index=0, start=0, end=total_frames)
        return ChunkPlan(total_frames, chunk_frames, 0, (window,))

    hop = chunk_frames - overlap_frames
    windows: list[ChunkWindow] = []
    start = 0
    index = 0
    while start < total_frames:
        end = min(start + chunk_frames, total_frames)
        windows.append(ChunkWindow(index=index, start=start, end=end))
        if end >= total_frames:
            break
        start += hop
        index += 1
    return ChunkPlan(total_frames, chunk_frames, overlap_frames, tuple(windows))


def _slice_f0(f0: F0Curve, start: int, end: int, sample_rate: int) -> F0Curve:
    """The stretch of the F0 curve that lines up with samples `[start, end)`."""
    frames = f0.frames
    if frames == 0:
        return f0
    hop_samples = max(1.0, f0.hop_seconds * sample_rate)
    f_start = int(np.floor(start / hop_samples))
    f_end = int(np.ceil(end / hop_samples))
    f_start = max(0, min(f_start, frames))
    f_end = max(f_start, min(f_end, frames))
    return F0Curve(hz=np.asarray(f0.hz)[f_start:f_end], hop_seconds=f0.hop_seconds)


def convert_in_chunks(
    audio: AudioBuffer,
    f0: F0Curve,
    backend: ConversionBackend,
    params: ConversionParams,
    chunk_seconds: float = DEFAULT_CHUNK_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
    on_progress: ProgressHook | None = None,
) -> AudioBuffer:
    """Convert `audio` window by window and reassemble seamlessly.

    `backend` is already loaded. Each converted window is forced back to its
    input length (defending the seam-free reconstruction against a backend that
    rounds), then crossfaded into the running result. The output length equals
    `audio.frames` exactly.
    """
    plan = plan_chunks(audio.frames, audio.sample_rate, chunk_seconds, overlap_seconds)
    pieces: list[AudioBuffer] = []
    for window in plan.windows:
        chunk = AudioBuffer(
            samples=audio.samples[:, window.start : window.end],
            sample_rate=audio.sample_rate,
        )
        f0_slice = _slice_f0(f0, window.start, window.end, audio.sample_rate)
        converted = backend.convert(chunk, f0_slice, params)
        # Enforce the per-chunk length contract so total length stays exact.
        converted = fit_length(converted, window.frames)
        pieces.append(converted)
        if on_progress is not None:
            on_progress(window.index + 1, plan.count)

    result = _join(pieces, plan.overlap_frames)
    return fit_length(result, audio.frames)


def _join(pieces: Sequence[AudioBuffer], overlap_frames: int) -> AudioBuffer:
    """Crossfade-join consecutive converted windows into one buffer."""
    if not pieces:
        raise ValueError("no chunks to join")
    result = pieces[0]
    for piece in pieces[1:]:
        result = crossfade_join(result, piece, overlap_frames)
    return result
