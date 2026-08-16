"""SeparationBackend -- splitting a mixed song into stems."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, runtime_checkable

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint

__all__ = ["SeparationBackend", "SeparationRequest", "Stems", "StemKind"]


class StemKind(StrEnum):
    VOCALS = "vocals"
    INSTRUMENTAL = "instrumental"
    LEAD = "lead"
    BACKING = "backing"
    AMBIENCE = "ambience"
    BASS = "bass"
    DRUMS = "drums"
    OTHER = "other"


@dataclass(frozen=True)
class Stems:
    """Result of one separation pass. All stems share length and sample rate."""

    parts: dict[StemKind, AudioBuffer]
    model_id: str

    def get(self, kind: StemKind) -> AudioBuffer | None:
        return self.parts.get(kind)


@dataclass(frozen=True)
class SeparationRequest:
    model_id: str
    wanted: frozenset[StemKind] = field(
        default_factory=lambda: frozenset({StemKind.VOCALS, StemKind.INSTRUMENTAL})
    )
    overlap: int = 4
    chunk_seconds: float | None = None
    batch_size: int = 1


@runtime_checkable
class SeparationBackend(Protocol):
    def info(self) -> BackendInfo:
        ...

    def list_models(self) -> list[str]:
        """Model ids this backend can run."""
        ...

    def separate(
        self,
        audio: AudioBuffer,
        request: SeparationRequest,
        device: DeviceHint,
    ) -> Stems:
        ...

    def unload(self) -> None:
        ...
