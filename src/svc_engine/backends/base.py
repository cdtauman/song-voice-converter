"""Shared value types used across all engine backends.

Phase 1 defines the vocabulary only. No backend is implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from svc_engine.compute.devices import ComputeBackend, DeviceInfo

__all__ = [
    "AudioBuffer",
    "BackendInfo",
    "ComputeBackend",
    "DeviceHint",
    "DeviceInfo",
    "F0Curve",
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
    """Which compute device the caller wants this backend to use.

    Produced by `SupportMatrix.device_for(component, manager)`, so it already
    reflects what that component has actually been proven to run on.
    """

    backend: ComputeBackend = ComputeBackend.CPU
    device_index: int = 0
    max_vram_mb: int | None = None

    @property
    def torch_device(self) -> str:
        if self.backend is ComputeBackend.CPU:
            return "cpu"
        return f"{self.backend.value}:{self.device_index}"

    @classmethod
    def from_device(cls, device: DeviceInfo, max_vram_mb: int | None = None) -> DeviceHint:
        return cls(backend=device.backend, device_index=device.index, max_vram_mb=max_vram_mb)


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
