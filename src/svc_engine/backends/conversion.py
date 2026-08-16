"""ConversionBackend -- the voice conversion interface.

RVC v2 is the MVP implementation, but nothing above this module may assume it.
See docs/architecture.md section 3a.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve

__all__ = ["ConversionBackend", "ConversionParams", "VoiceHandle"]


@dataclass(frozen=True)
class VoiceHandle:
    """Points at a voice in the user's library. The backend interprets the files."""

    voice_id: str
    root: Path


@dataclass(frozen=True)
class ConversionParams:
    """Knobs shared by conversion backends.

    Values a given backend does not understand are ignored, not an error.
    Defaults here are community starting points, not calibrated recommendations
    -- calibration happens in Phase 5/6.
    """

    semitones: int = 0
    index_rate: float = 0.70
    protect: float = 0.33
    rms_mix_rate: float = 0.25
    filter_radius: int = 3
    formant_shift: float = 0.0


@runtime_checkable
class ConversionBackend(Protocol):
    """Converts a vocal performance into a target voice."""

    def info(self) -> BackendInfo:
        """Identity and whether this backend can run right now."""
        ...

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:
        """Bring a voice model into memory. Idempotent for the same voice."""
        ...

    def convert(
        self,
        audio: AudioBuffer,
        f0: F0Curve,
        params: ConversionParams,
    ) -> AudioBuffer:
        """Run conversion. Output length must match the input length exactly."""
        ...

    def unload(self) -> None:
        """Release the model and any GPU memory it holds."""
        ...
