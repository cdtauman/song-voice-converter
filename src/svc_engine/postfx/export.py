"""User-safe output naming and the Phase-6 export contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from svc_engine.audio import io as audio_io
from svc_engine.backends.base import AudioBuffer

__all__ = ["ExportResult", "clean_output_path", "export_audio"]

_BAD = re.compile(r"[<>:\"/\\|?*\x00-\x1f]")
_RESERVED = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)),
             *(f"LPT{i}" for i in range(1, 10))}


@dataclass(frozen=True)
class ExportResult:
    path: Path
    format: str
    quality: str


def clean_output_path(path: Path | str) -> Path:
    path = Path(path)
    stem = _BAD.sub("-", path.stem).strip(" .") or "songvoice-cover"
    if stem.upper() in _RESERVED:
        stem = f"{stem}-cover"
    suffix = path.suffix.lower()
    if suffix not in {".wav", ".mp3", ".flac"}:
        suffix = ".wav"
    return path.with_name(stem + suffix)


def export_audio(audio: AudioBuffer, path: Path | str) -> ExportResult:
    clean = clean_output_path(path)
    if clean.suffix == ".mp3":
        audio_io.save_mp3(audio, clean, bitrate="320k")
        return ExportResult(clean, "mp3", "320 kbps")
    if clean.suffix == ".flac":
        audio_io.save_audio(audio, clean)
        return ExportResult(clean, "flac", "lossless")
    audio_io.save_wav(audio, clean, bit_depth=24)
    return ExportResult(clean, "wav", "24-bit PCM")
