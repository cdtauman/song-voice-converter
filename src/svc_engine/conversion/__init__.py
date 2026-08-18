"""The voice-conversion stage.

Phase 5: turn a separated vocal into the target voice, then wire the whole
song-to-cover pipeline together (separation -> analysis -> pitch -> conversion
-> remix). The heavy RVC engine lives in `conversion.rvc` behind the
`ConversionBackend` interface and is imported lazily; everything in this package
above that interface is torch-free.
"""

from __future__ import annotations

from svc_engine.conversion.chunking import ChunkPlan, convert_in_chunks, plan_chunks
from svc_engine.conversion.pipeline import (
    ConversionOutcome,
    ConversionPipeline,
    mix_cover,
    params_for_voice,
)

__all__ = [
    "ChunkPlan",
    "ConversionOutcome",
    "ConversionPipeline",
    "convert_in_chunks",
    "mix_cover",
    "params_for_voice",
    "plan_chunks",
]
