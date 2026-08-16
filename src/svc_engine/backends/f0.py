"""F0Extractor -- fundamental frequency (pitch) detection.

RMVPE is the accuracy default, FCPE the fast path for previews.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve

__all__ = ["F0Extractor"]


@runtime_checkable
class F0Extractor(Protocol):
    def info(self) -> BackendInfo:
        ...

    def extract(
        self,
        audio: AudioBuffer,
        device: DeviceHint,
        hop_seconds: float = 0.01,
    ) -> F0Curve:
        ...

    def unload(self) -> None:
        ...
