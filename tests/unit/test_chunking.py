"""Chunked conversion: exact-length reassembly and seam-free crossfades."""

from __future__ import annotations

import numpy as np
import pytest

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionParams, VoiceHandle
from svc_engine.conversion.chunking import convert_in_chunks, plan_chunks

SR = 44100


class _Identity:
    """Returns its input unchanged -- the reference for seam-free reassembly."""

    def info(self) -> BackendInfo:
        return BackendInfo("id", "id", available=True)

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:  # pragma: no cover
        pass

    def convert(self, audio: AudioBuffer, f0: F0Curve, params: ConversionParams) -> AudioBuffer:
        return AudioBuffer(samples=audio.samples.copy(), sample_rate=audio.sample_rate)

    def unload(self) -> None:  # pragma: no cover
        pass


class _PerChunkConstant:
    """Adds a per-call constant, so each window differs -- stresses the seams."""

    def __init__(self) -> None:
        self.calls = 0

    def info(self) -> BackendInfo:
        return BackendInfo("c", "c", available=True)

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:  # pragma: no cover
        pass

    def convert(self, audio: AudioBuffer, f0: F0Curve, params: ConversionParams) -> AudioBuffer:
        self.calls += 1
        return AudioBuffer(
            samples=audio.samples + float(self.calls), sample_rate=audio.sample_rate
        )

    def unload(self) -> None:  # pragma: no cover
        pass


class _WrongLength:
    """A misbehaving backend that changes length -- must not break the total."""

    def info(self) -> BackendInfo:
        return BackendInfo("w", "w", available=True)

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:  # pragma: no cover
        pass

    def convert(self, audio: AudioBuffer, f0: F0Curve, params: ConversionParams) -> AudioBuffer:
        return AudioBuffer(samples=audio.samples[:, :-5], sample_rate=audio.sample_rate)

    def unload(self) -> None:  # pragma: no cover
        pass


def _tone(frames: int, channels: int = 1) -> AudioBuffer:
    t = np.arange(frames, dtype=np.float32)
    wave = 0.3 * np.sin(2 * np.pi * 220.0 * t / SR)
    samples = np.tile(wave, (channels, 1)).astype(np.float32)
    return AudioBuffer(samples=samples, sample_rate=SR)


def _f0(frames_of_audio: int) -> F0Curve:
    n = max(1, frames_of_audio // 441)  # 10ms hop
    return F0Curve(hz=np.full(n, 220.0, dtype=np.float64), hop_seconds=0.01)


# --- planning -------------------------------------------------------------- #

def test_plan_single_window_when_short() -> None:
    plan = plan_chunks(SR, SR, chunk_seconds=30.0, overlap_seconds=0.5)
    assert plan.count == 1
    assert plan.windows[0].start == 0 and plan.windows[0].end == SR
    assert plan.overlap_frames == 0


def test_plan_tiles_with_hop_equal_chunk_minus_overlap() -> None:
    total = 10 * SR
    plan = plan_chunks(total, SR, chunk_seconds=3.0, overlap_seconds=0.5)
    assert plan.count > 1
    assert plan.hop_frames == plan.chunk_frames - plan.overlap_frames
    # windows advance by hop and cover the whole signal
    for i, w in enumerate(plan.windows[:-1]):
        assert w.start == i * plan.hop_frames
        assert w.frames == plan.chunk_frames
    assert plan.windows[-1].end == total


def test_plan_rejects_empty() -> None:
    with pytest.raises(ValueError):
        plan_chunks(0, SR)


# --- reassembly ------------------------------------------------------------ #

@pytest.mark.parametrize("frames", [SR, 3 * SR + 137, 7 * SR, 10 * SR + 1])
@pytest.mark.parametrize("channels", [1, 2])
def test_identity_reassembles_bit_exact(frames: int, channels: int) -> None:
    audio = _tone(frames, channels)
    out = convert_in_chunks(
        audio, _f0(frames), _Identity(), ConversionParams(),
        chunk_seconds=2.0, overlap_seconds=0.25,
    )
    assert out.frames == frames
    assert out.channels == channels
    # Linear crossfade of two identical overlapping windows sums to the original.
    assert np.allclose(out.samples, audio.samples, atol=1e-5)


def test_total_length_exact_even_with_misbehaving_backend() -> None:
    frames = 5 * SR + 999
    audio = _tone(frames)
    out = convert_in_chunks(
        audio, _f0(frames), _WrongLength(), ConversionParams(),
        chunk_seconds=1.5, overlap_seconds=0.2,
    )
    assert out.frames == frames


def test_no_nans_and_bounded_seam_with_varying_chunks() -> None:
    frames = 6 * SR
    audio = _tone(frames)
    out = convert_in_chunks(
        audio, _f0(frames), _PerChunkConstant(), ConversionParams(),
        chunk_seconds=2.0, overlap_seconds=0.5,
    )
    assert out.frames == frames
    assert np.all(np.isfinite(out.samples))
    # No single-sample discontinuity larger than any per-chunk step: the
    # crossfade spreads each +1 constant across the overlap, so neighbouring
    # differences stay far below the raw between-chunk jump of ~1.0.
    jumps = np.abs(np.diff(out.samples, axis=-1))
    assert float(jumps.max()) < 0.2


def test_progress_reports_each_window() -> None:
    frames = 5 * SR
    seen: list[tuple[int, int]] = []
    convert_in_chunks(
        _tone(frames), _f0(frames), _Identity(), ConversionParams(),
        chunk_seconds=1.0, overlap_seconds=0.1,
        on_progress=lambda done, total: seen.append((done, total)),
    )
    assert seen[-1][0] == seen[-1][1]  # ended at 100%
    assert [d for d, _ in seen] == list(range(1, len(seen) + 1))
