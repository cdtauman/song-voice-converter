"""Audio ingest, buffer arithmetic and export."""

from svc_engine.audio.buffers import (
    add,
    align_all,
    crossfade_join,
    fit_length,
    is_silent,
    peak,
    rms,
    silence_like,
    subtract,
)
from svc_engine.audio.io import (
    DEFAULT_SAMPLE_RATE,
    AudioInfo,
    ffmpeg_exe,
    ffprobe_exe,
    load_audio,
    match_channels,
    probe,
    save_audio,
    save_mp3,
    save_wav,
    to_mono,
)

__all__ = [
    "DEFAULT_SAMPLE_RATE",
    "AudioInfo",
    "add",
    "align_all",
    "crossfade_join",
    "ffmpeg_exe",
    "ffprobe_exe",
    "fit_length",
    "is_silent",
    "load_audio",
    "match_channels",
    "peak",
    "probe",
    "rms",
    "save_audio",
    "save_mp3",
    "save_wav",
    "silence_like",
    "subtract",
    "to_mono",
]
