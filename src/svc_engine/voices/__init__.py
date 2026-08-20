"""The voice library: the user's catalogue of target voices.

A "voice" is a folder under `paths.voices/<voice_id>/` holding an RVC model, an
optional retrieval index, an optional avatar and sample, a range profile, and a
manifest (`voice.json`). This package owns the *catalogue* -- discovery,
validation, health, consent and import -- and never touches torch. The actual
model weights are interpreted by `conversion.rvc` behind the `ConversionBackend`
interface. See docs/architecture.md section 5.
"""

from __future__ import annotations

from svc_engine.voices.importer import ImportResult, import_voice_from_zip
from svc_engine.voices.library import VoiceLibrary
from svc_engine.voices.manifest import (
    VOICE_FILE,
    HealthState,
    HealthStatus,
    RecommendedParams,
    VoiceManifest,
    VoiceSource,
)

__all__ = [
    "VOICE_FILE",
    "HealthState",
    "HealthStatus",
    "ImportResult",
    "RecommendedParams",
    "VoiceLibrary",
    "VoiceManifest",
    "VoiceSource",
    "import_voice_from_zip",
]
