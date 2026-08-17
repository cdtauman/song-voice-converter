"""Splitting a song into stems."""

from svc_engine.separation.backends import AudioSeparatorBackend, PymssBackend
from svc_engine.separation.cleanup import CleanupResult, VocalCleanup
from svc_engine.separation.ensemble import EnsembleMode, combine_buffers, combine_stems
from svc_engine.separation.pipeline import (
    Progress,
    ProgressHook,
    SeparationOutcome,
    SeparationPipeline,
)
from svc_engine.separation.quality import (
    PROFILES,
    CleanupStep,
    QualityLevel,
    SeparationProfile,
    profile_for,
)

__all__ = [
    "PROFILES",
    "AudioSeparatorBackend",
    "CleanupResult",
    "CleanupStep",
    "EnsembleMode",
    "Progress",
    "ProgressHook",
    "PymssBackend",
    "QualityLevel",
    "SeparationOutcome",
    "SeparationPipeline",
    "SeparationProfile",
    "VocalCleanup",
    "combine_buffers",
    "combine_stems",
    "profile_for",
]
