"""ffmpeg-based summing, compression, EBU-R128 normalization and limiting."""

from __future__ import annotations

import json
import math
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svc_engine.audio.buffers import fit_length, peak, rms
from svc_engine.audio.io import ffmpeg_exe, match_channels
from svc_engine.backends.base import AudioBuffer
from svc_engine.errors import EngineError, ErrorCode

__all__ = ["MixConfig", "MixReport", "integrated_lufs", "mix_and_master"]

_TIMEOUT = 3600


@dataclass(frozen=True)
class MixConfig:
    target_lufs: float = -14.0
    true_peak_db: float = -1.0
    loudness_range: float = 11.0
    vocal_gain_db: float = 0.0
    playback_gain_db: float = 0.0
    compress: bool = True


@dataclass(frozen=True)
class MixReport:
    target_lufs: float
    integrated_lufs: float
    peak_dbfs: float
    clipped: bool
    frames: int
    used_filters: tuple[str, ...]


def integrated_lufs(audio: AudioBuffer) -> float:
    """Measure integrated loudness with pyloudnorm (MIT), as research.md specifies."""
    if not audio.samples.size or rms(audio) < 1e-10:
        return float("-inf")
    import pyloudnorm as pyln

    samples = audio.samples.T.astype(np.float64)
    minimum = max(1, int(audio.sample_rate * 0.5))
    if samples.shape[0] < minimum:
        samples = np.pad(samples, ((0, minimum - samples.shape[0]), (0, 0)))
    meter = pyln.Meter(audio.sample_rate)
    return float(meter.integrated_loudness(samples))


def _raw(path: Path, audio: AudioBuffer, channels: int, frames: int) -> None:
    fitted = match_channels(fit_length(audio, frames), channels)
    path.write_bytes(np.ascontiguousarray(fitted.samples.T.astype("<f4")).tobytes())


def _inputs(vocal_path: Path, playback_path: Path, sample_rate: int, channels: int) -> list[str]:
    one = ["-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels)]
    return [*one, "-i", str(vocal_path), *one, "-i", str(playback_path)]


def _premix_filter(cfg: MixConfig) -> str:
    chain = (
        f"[0:a]volume={cfg.vocal_gain_db:.4f}dB[v];"
        f"[1:a]volume={cfg.playback_gain_db:.4f}dB[p];"
        "[v][p]amix=inputs=2:duration=longest:dropout_transition=0:normalize=0"
    )
    if cfg.compress:
        chain += ",acompressor=threshold=0.125:ratio=2:attack=20:release=250:makeup=1"
    return chain


def _run(cmd: list[str]) -> subprocess.CompletedProcess[bytes]:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=_TIMEOUT)
    except (OSError, subprocess.SubprocessError) as exc:
        raise EngineError(ErrorCode.INTERNAL, f"ffmpeg mix failed: {exc}") from exc
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip()[-1200:]
        raise EngineError(ErrorCode.INTERNAL, "ffmpeg mix failed: " + detail)
    return proc


def _analysis(stderr: bytes) -> dict[str, str]:
    text = stderr.decode("utf-8", "replace")
    matches = re.findall(r"\{\s*\"input_i\".*?\}", text, flags=re.DOTALL)
    if not matches:
        raise EngineError(ErrorCode.INTERNAL, "ffmpeg loudnorm returned no measurement")
    data = json.loads(matches[-1])
    return {str(k): str(v) for k, v in data.items()}


def mix_and_master(
    vocal: AudioBuffer,
    playback: AudioBuffer,
    config: MixConfig | None = None,
    *,
    work_dir: Path | None = None,
) -> tuple[AudioBuffer, MixReport]:
    """Sum and master entirely through ffmpeg's LGPL filters.

    loudnorm runs in two passes. The first measures this exact sum; the second
    applies those measurements and a final look-ahead limiter. No home-grown
    compressor or limiter sits in the distributable chain.
    """
    cfg = config or MixConfig()
    if vocal.sample_rate != playback.sample_rate:
        raise ValueError("sample rate mismatch")
    frames = playback.frames
    channels = max(vocal.channels, playback.channels)
    if rms(vocal) < 1e-10 and rms(playback) < 1e-10:
        silent = AudioBuffer(
            samples=np.zeros((channels, frames), dtype=np.float32),
            sample_rate=playback.sample_rate,
        )
        return silent, MixReport(cfg.target_lufs, float("-inf"), float("-inf"), False,
                                 frames, ("amix", "loudnorm", "alimiter"))

    parent = str(work_dir) if work_dir is not None else None
    if work_dir is not None:
        work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="songvoice-mix-", dir=parent) as temp:
        temp_dir = Path(temp)
        vocal_path, playback_path = temp_dir / "vocal.f32", temp_dir / "playback.f32"
        _raw(vocal_path, vocal, channels, frames)
        _raw(playback_path, playback, channels, frames)
        base = [ffmpeg_exe(), "-hide_banner", "-nostdin", "-v", "info", "-y"]
        inputs = _inputs(vocal_path, playback_path, playback.sample_rate, channels)
        pre = _premix_filter(cfg)
        first_filter = (
            f"{pre},loudnorm=I={cfg.target_lufs}:TP={cfg.true_peak_db}:"
            f"LRA={cfg.loudness_range}:print_format=json[out]"
        )
        first = _run([*base, *inputs, "-filter_complex", first_filter, "-map", "[out]",
                      "-f", "null", "-"])
        measured = _analysis(first.stderr)

        limit = 10.0 ** (cfg.true_peak_db / 20.0)
        loudnorm = (
            f"loudnorm=I={cfg.target_lufs}:TP={cfg.true_peak_db}:LRA={cfg.loudness_range}:"
            f"measured_I={measured['input_i']}:measured_TP={measured['input_tp']}:"
            f"measured_LRA={measured['input_lra']}:"
            f"measured_thresh={measured['input_thresh']}:offset={measured['target_offset']}:"
            "linear=true:print_format=summary"
        )
        second_filter = f"{pre},{loudnorm},alimiter=limit={limit:.8f}:level=false[out]"
        second = _run([
            *base, *inputs, "-filter_complex", second_filter, "-map", "[out]",
            "-frames:a", str(frames), "-ar", str(playback.sample_rate),
            "-ac", str(channels), "-f", "f32le", "-acodec", "pcm_f32le", "-",
        ])

    flat = np.frombuffer(second.stdout, dtype="<f4")
    usable = (flat.size // channels) * channels
    samples = flat[:usable].reshape(-1, channels).T.astype(np.float32, copy=True)
    mastered = fit_length(AudioBuffer(samples=samples, sample_rate=playback.sample_rate), frames)
    lufs = integrated_lufs(mastered)
    sample_peak = peak(mastered)
    peak_db = 20.0 * math.log10(sample_peak) if sample_peak > 0.0 else float("-inf")
    filters = ("amix", "acompressor", "loudnorm", "alimiter") if cfg.compress else (
        "amix", "loudnorm", "alimiter"
    )
    return mastered, MixReport(
        cfg.target_lufs, lufs, peak_db, bool(sample_peak > 1.0), mastered.frames, filters
    )
