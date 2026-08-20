"""Phase 6 post-processing: repair, melody, dynamics, space and mastering."""

from svc_engine.postfx.ambience import (
    AmbienceMeasurement,
    AmbienceResult,
    AmbienceStrategy,
    apply_ambience,
    measure_ambience,
)
from svc_engine.postfx.deess import DeEssReport, deess
from svc_engine.postfx.envelope import EnvelopeReport, envelope_correlation, match_envelope
from svc_engine.postfx.export import ExportResult, clean_output_path, export_audio
from svc_engine.postfx.melody_check import (
    MelodyReport,
    check_melody,
    correct_global_pitch,
)
from svc_engine.postfx.mix import MixConfig, MixReport, integrated_lufs, mix_and_master
from svc_engine.postfx.pipeline import PostFxConfig, PostFxOutcome, PostFxPipeline, PostFxReport
from svc_engine.postfx.repair import RepairReport, repair_artifacts

__all__ = [
    "AmbienceMeasurement",
    "AmbienceResult",
    "AmbienceStrategy",
    "DeEssReport",
    "EnvelopeReport",
    "ExportResult",
    "MelodyReport",
    "MixConfig",
    "MixReport",
    "PostFxConfig",
    "PostFxOutcome",
    "PostFxPipeline",
    "PostFxReport",
    "RepairReport",
    "apply_ambience",
    "check_melody",
    "clean_output_path",
    "correct_global_pitch",
    "deess",
    "envelope_correlation",
    "export_audio",
    "integrated_lufs",
    "match_envelope",
    "measure_ambience",
    "mix_and_master",
    "repair_artifacts",
]
