"""Engine backend interfaces.

Every external AI model or DSP library the engine uses sits behind one of the
four protocols defined here. Swapping an implementation must never require
changes outside this package.

Phase 1 defines interfaces only -- there are no concrete implementations yet.
"""

from svc_engine.backends.base import (
    AudioBuffer,
    BackendInfo,
    DeviceHint,
    F0Curve,
)
from svc_engine.backends.conversion import ConversionBackend, ConversionParams
from svc_engine.backends.f0 import F0Extractor
from svc_engine.backends.pitch import PitchShifter
from svc_engine.backends.separation import SeparationBackend, SeparationRequest, Stems

__all__ = [
    "AudioBuffer",
    "BackendInfo",
    "ConversionBackend",
    "ConversionParams",
    "DeviceHint",
    "F0Curve",
    "F0Extractor",
    "PitchShifter",
    "SeparationBackend",
    "SeparationRequest",
    "Stems",
]

#: The four interfaces required by the approved architecture. Used by tests.
REQUIRED_PROTOCOLS: tuple[str, ...] = (
    "ConversionBackend",
    "SeparationBackend",
    "PitchShifter",
    "F0Extractor",
)
