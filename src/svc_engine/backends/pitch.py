"""PitchShifter -- length-preserving pitch shifting.

Default implementation target is `python-stretch` (MIT). Rubber Band (GPL) may
only ever appear as a development-time comparison and must never be packaged.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from svc_engine.backends.base import AudioBuffer, BackendInfo

__all__ = ["PitchShifter"]


@runtime_checkable
class PitchShifter(Protocol):
    """Shifts pitch without changing duration.

    Hard contract: the returned buffer has exactly the same number of frames and
    the same sample rate as the input. Timing is never allowed to drift.
    """

    def info(self) -> BackendInfo:
        ...

    def shift(self, audio: AudioBuffer, semitones: float) -> AudioBuffer:
        ...
