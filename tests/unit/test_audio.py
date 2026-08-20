"""Audio ingest, buffer arithmetic and export.

The ffmpeg-backed tests are skipped when ffmpeg is missing so that a checkout
without it still runs the pure-numpy half.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from svc_engine.audio import buffers, io
from svc_engine.backends.base import AudioBuffer
from svc_engine.errors import EngineError, ErrorCode

has_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg is not installed",
)


def tone(seconds: float = 1.0, hz: float = 440.0, channels: int = 2, rate: int = 44100):
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    mono = (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)[None, :]
    return AudioBuffer(samples=np.repeat(mono, channels, axis=0), sample_rate=rate)


# --------------------------------------------------------------------------- #
# buffer arithmetic
# --------------------------------------------------------------------------- #

def test_fit_length_trims_and_pads_exactly() -> None:
    buf = tone(0.5)
    assert buffers.fit_length(buf, 1000).frames == 1000
    assert buffers.fit_length(buf, buf.frames + 500).frames == buf.frames + 500
    # Padding must be silence, not a repeat of the tail.
    padded = buffers.fit_length(buf, buf.frames + 500)
    assert np.all(padded.samples[:, buf.frames :] == 0.0)


def test_fit_length_is_identity_at_the_right_length() -> None:
    buf = tone(0.25)
    assert buffers.fit_length(buf, buf.frames) is buf


def test_subtract_then_add_reconstructs_the_original() -> None:
    """The property every derived stem relies on."""
    mix = tone(0.5, hz=220.0)
    part = tone(0.5, hz=660.0)
    rest = buffers.subtract(mix, part)
    back = buffers.add(rest, part)
    assert np.allclose(back.samples, mix.samples, atol=1e-6)


def test_align_all_uses_the_shortest() -> None:
    aligned = buffers.align_all([tone(0.5), tone(0.3), tone(0.7)])
    assert len({b.frames for b in aligned}) == 1
    assert aligned[0].frames == tone(0.3).frames


def test_is_silent_distinguishes_silence_from_a_quiet_signal() -> None:
    loud = tone(0.2)
    quiet = AudioBuffer(samples=loud.samples * 1e-5, sample_rate=loud.sample_rate)
    assert not buffers.is_silent(loud)
    assert buffers.is_silent(quiet)
    assert buffers.is_silent(buffers.silence_like(loud))


def test_crossfade_join_has_no_level_dip_at_the_seam() -> None:
    """Equal power, not linear -- a linear fade dips ~3dB and is audible."""
    rate = 44100
    steady = AudioBuffer(
        samples=np.ones((1, rate), dtype=np.float32) * 0.5, sample_rate=rate
    )
    overlap = 4410
    joined = buffers.crossfade_join(steady, steady, overlap)

    assert joined.frames == 2 * rate - overlap
    seam = joined.samples[0, rate - overlap : rate]
    assert np.allclose(seam, 0.5, atol=1e-3), "equal-power crossfade should hold level"


def test_crossfade_join_without_overlap_concatenates() -> None:
    a, b = tone(0.1), tone(0.1)
    joined = buffers.crossfade_join(a, b, 0)
    assert joined.frames == a.frames + b.frames


# --------------------------------------------------------------------------- #
# ffmpeg round trip
# --------------------------------------------------------------------------- #

@has_ffmpeg
def test_wav_round_trip_preserves_length_and_signal(tmp_path: Path) -> None:
    original = tone(0.75, hz=330.0)
    path = io.save_wav(original, tmp_path / "t.wav", bit_depth=24)
    back = io.load_audio(path)

    assert back.sample_rate == original.sample_rate
    assert back.channels == original.channels
    assert back.frames == original.frames, "length must survive a round trip exactly"
    assert np.max(np.abs(back.samples - original.samples)) < 1e-4


@has_ffmpeg
def test_load_audio_resamples_to_the_project_rate(tmp_path: Path) -> None:
    original = tone(1.0, rate=48000)
    path = io.save_wav(original, tmp_path / "t48.wav")
    back = io.load_audio(path)

    assert back.sample_rate == io.DEFAULT_SAMPLE_RATE
    assert abs(back.seconds - original.seconds) < 0.01


@has_ffmpeg
def test_load_audio_slice_decodes_only_the_requested_window(tmp_path: Path) -> None:
    path = io.save_wav(tone(4.0), tmp_path / "long.wav")
    part = io.load_audio(path, start_seconds=1.0, duration_seconds=0.5)
    assert abs(part.seconds - 0.5) < 0.02


@has_ffmpeg
def test_load_audio_mono_collapses_channels(tmp_path: Path) -> None:
    path = io.save_wav(tone(0.5), tmp_path / "stereo.wav")
    assert io.load_audio(path, mono=True).channels == 1


@has_ffmpeg
def test_probe_reports_what_the_file_is(tmp_path: Path) -> None:
    path = io.save_wav(tone(2.0), tmp_path / "p.wav")
    info = io.probe(path)
    assert info.channels == 2
    assert info.sample_rate == 44100
    assert abs(info.seconds - 2.0) < 0.05


@has_ffmpeg
def test_mp3_export_is_readable_and_roughly_the_right_length(tmp_path: Path) -> None:
    original = tone(1.0)
    path = io.save_mp3(original, tmp_path / "t.mp3")
    back = io.load_audio(path)
    # MP3 pads; exact length is not a contract for a lossy export.
    assert abs(back.seconds - original.seconds) < 0.1


def test_probe_of_a_missing_file_is_a_hebrew_error() -> None:
    with pytest.raises(EngineError) as caught:
        io.probe(Path("no_such_song.mp3"))
    assert caught.value.code is ErrorCode.AUDIO_UNSUPPORTED
    assert caught.value.user_message.action


@has_ffmpeg
def test_a_corrupt_file_fails_with_a_message_not_a_crash(tmp_path: Path) -> None:
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"this is definitely not audio" * 100)
    with pytest.raises(EngineError) as caught:
        io.load_audio(broken)
    assert caught.value.code is ErrorCode.AUDIO_UNSUPPORTED


def test_match_channels_both_directions() -> None:
    stereo = tone(0.1, channels=2)
    mono = io.match_channels(stereo, 1)
    assert mono.channels == 1
    assert io.match_channels(mono, 2).channels == 2
