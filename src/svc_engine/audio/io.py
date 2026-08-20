"""Audio ingest and export, all of it through ffmpeg.

ffmpeg is the single decoder on purpose: it opens everything the user is likely
to drop on the window (mp3, m4a, flac, ogg, wma, video containers), and it is
already a shipped dependency for the mixing stage. Nothing else in the engine
needs to know what a container is.

The in-memory contract is `AudioBuffer`: shape (channels, samples), float32,
44.1kHz unless a caller asks otherwise. Every stage downstream assumes it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svc_engine.backends.base import AudioBuffer
from svc_engine.errors import EngineError, ErrorCode

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "AudioInfo",
    "ffmpeg_exe",
    "ffprobe_exe",
    "probe",
    "load_audio",
    "save_wav",
    "save_mp3",
    "save_audio",
    "to_mono",
    "match_channels",
]

#: Everything in the pipeline runs at 44.1kHz. RVC resamples internally to its
#: own rate; keeping one project-wide rate avoids a chain of drifting lengths.
DEFAULT_SAMPLE_RATE = 44100

#: soxr at very high quality. ffmpeg's default resampler is audibly worse and
#: this costs almost nothing at song lengths.
_RESAMPLE_FILTER = "aresample=resampler=soxr:precision=28"

_DECODE_TIMEOUT = 3600


@dataclass(frozen=True)
class AudioInfo:
    """What a file claims to be, before we decode it."""

    path: Path
    seconds: float
    sample_rate: int
    channels: int
    codec: str
    format_name: str


def _which(name: str, code: ErrorCode = ErrorCode.FFMPEG_MISSING) -> str:
    exe = shutil.which(name)
    if not exe:
        raise EngineError(code, f"{name} not found on PATH")
    return exe


def ffmpeg_exe() -> str:
    return _which("ffmpeg")


def ffprobe_exe() -> str:
    return _which("ffprobe")


def probe(path: Path | str) -> AudioInfo:
    """Read stream metadata without decoding. Raises `E_AUDIO_UNSUPPORTED`."""
    path = Path(path)
    if not path.exists():
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, f"file not found: {path}")

    cmd = [
        ffprobe_exe(), "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=codec_name,sample_rate,channels:format=duration,format_name",
        "-of", "json", str(path),
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, str(exc)) from exc
    if proc.returncode != 0:
        raise EngineError(
            ErrorCode.AUDIO_UNSUPPORTED,
            proc.stderr.decode("utf-8", "replace").strip()[:400],
        )

    try:
        data = json.loads(proc.stdout.decode("utf-8", "replace"))
        stream = (data.get("streams") or [{}])[0]
        fmt = data.get("format") or {}
    except (json.JSONDecodeError, IndexError) as exc:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, "unreadable ffprobe output") from exc

    if not stream:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, "no audio stream in file")

    return AudioInfo(
        path=path,
        seconds=float(fmt.get("duration") or 0.0),
        sample_rate=int(stream.get("sample_rate") or 0),
        channels=int(stream.get("channels") or 0),
        codec=str(stream.get("codec_name") or ""),
        format_name=str(fmt.get("format_name") or ""),
    )


def load_audio(
    path: Path | str,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    mono: bool = False,
    start_seconds: float | None = None,
    duration_seconds: float | None = None,
) -> AudioBuffer:
    """Decode any file ffmpeg can open into float32 (channels, samples).

    `start_seconds` / `duration_seconds` decode a slice only -- the preview
    stage in Phase 3 needs 12 seconds out of a 5 minute song and should not pay
    for the rest.
    """
    path = Path(path)
    info = probe(path)
    channels = 1 if mono else max(1, min(info.channels or 2, 2))

    cmd = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-v", "error"]
    if start_seconds is not None:
        cmd += ["-ss", f"{max(0.0, start_seconds):.6f}"]
    cmd += ["-i", str(path)]
    if duration_seconds is not None:
        cmd += ["-t", f"{max(0.0, duration_seconds):.6f}"]
    cmd += [
        "-map", "a:0",
        "-af", _RESAMPLE_FILTER,
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-f", "f32le", "-acodec", "pcm_f32le", "-",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_DECODE_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, str(exc)) from exc
    if proc.returncode != 0:
        raise EngineError(
            ErrorCode.AUDIO_UNSUPPORTED,
            proc.stderr.decode("utf-8", "replace").strip()[:400],
        )

    flat = np.frombuffer(proc.stdout, dtype="<f4")
    usable = (flat.size // channels) * channels
    samples = flat[:usable].reshape(-1, channels).T.astype(np.float32, copy=True)
    if samples.size == 0:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, "decoded to zero samples")
    return AudioBuffer(samples=samples, sample_rate=sample_rate)


def to_mono(buffer: AudioBuffer) -> AudioBuffer:
    if buffer.channels == 1:
        return buffer
    mono = buffer.samples.mean(axis=0, keepdims=True).astype(np.float32)
    return AudioBuffer(samples=mono, sample_rate=buffer.sample_rate)


def match_channels(buffer: AudioBuffer, channels: int) -> AudioBuffer:
    """Force a buffer to `channels`, duplicating mono or averaging down."""
    if buffer.channels == channels:
        return buffer
    if channels == 1:
        return to_mono(buffer)
    if buffer.channels == 1:
        wide = np.repeat(buffer.samples, channels, axis=0).astype(np.float32)
        return AudioBuffer(samples=wide, sample_rate=buffer.sample_rate)
    return AudioBuffer(samples=buffer.samples[:channels].copy(), sample_rate=buffer.sample_rate)


def _pipe_to_ffmpeg(buffer: AudioBuffer, out_path: Path, encode_args: list[str]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = np.ascontiguousarray(buffer.samples.T.astype("<f4")).tobytes()
    cmd = [
        ffmpeg_exe(), "-hide_banner", "-nostdin", "-v", "error", "-y",
        "-f", "f32le", "-ar", str(buffer.sample_rate), "-ac", str(buffer.channels),
        "-i", "-", *encode_args, str(out_path),
    ]
    try:
        proc = subprocess.run(cmd, input=raw, capture_output=True, timeout=_DECODE_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(ErrorCode.INTERNAL, f"ffmpeg encode failed: {exc}") from exc
    if proc.returncode != 0:
        raise EngineError(
            ErrorCode.INTERNAL,
            "ffmpeg encode failed: " + proc.stderr.decode("utf-8", "replace").strip()[:400],
        )


def save_wav(buffer: AudioBuffer, path: Path | str, bit_depth: int = 24) -> Path:
    """24-bit by default: the intermediate stems are re-processed several times."""
    codec = {16: "pcm_s16le", 24: "pcm_s24le", 32: "pcm_f32le"}.get(bit_depth)
    if codec is None:
        raise ValueError(f"unsupported bit depth: {bit_depth}")
    path = Path(path)
    _pipe_to_ffmpeg(buffer, path, ["-acodec", codec])
    return path


def save_mp3(buffer: AudioBuffer, path: Path | str, bitrate: str = "320k") -> Path:
    path = Path(path)
    _pipe_to_ffmpeg(buffer, path, ["-acodec", "libmp3lame", "-b:a", bitrate])
    return path


def save_audio(buffer: AudioBuffer, path: Path | str) -> Path:
    """Dispatch on file extension. Unknown extensions become WAV."""
    path = Path(path)
    if path.suffix.lower() == ".mp3":
        return save_mp3(buffer, path)
    if path.suffix.lower() == ".flac":
        _pipe_to_ffmpeg(buffer, path, ["-acodec", "flac", "-sample_fmt", "s32"])
        return path
    return save_wav(buffer, path)
