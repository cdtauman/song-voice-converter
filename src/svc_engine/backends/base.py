"""Shared value types used across all engine backends.

Phase 1 defines the vocabulary only. No backend is implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

__all__ = [
    "AudioBuffer",
    "F0Curve",
    "BackendInfo",
    "DeviceHint",
]


@dataclass(frozen=True)
class AudioBuffer:
    """Interleaved-free audio: shape (channels, samples), float32, range ~[-1, 1]."""

    samples: np.ndarray
    sample_rate: int

    @property
    def channels(self) -> int:
        return int(self.samples.shape[0])

    @property
    def frames(self) -> int:
        return int(self.samples.shape[1])

    @property
    def seconds(self) -> float:
        return self.frames / float(self.sample_rate)


@dataclass(frozen=True)
class F0Curve:
    """Fundamental frequency over time.

    `hz` holds 0.0 for unvoiced frames. `hop_seconds` is the spacing between frames.
    """

    hz: np.ndarray
    hop_seconds: float

    @property
    def frames(self) -> int:
        return int(self.hz.shape[0])


@dataclass(frozen=True)
class DeviceHint:
    """What the caller would like the backend to run on."""

    prefer_gpu: bool = True
    device_index: int = 0
    max_vram_mb: int | None = None


@dataclass(frozen=True)
class BackendInfo:
    """Identity and capabilities of a concrete backend implementation."""

    backend_id: str
    display_name_he: str
    available: bool
    unavailable_reason: str | None = None
    supports_gpu: bool = False
    capabilities: frozenset[str] = field(default_factory=frozenset)
    model_root: Path | None = None
