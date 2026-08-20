"""Advanced conversion controls and deterministic preview auto-tuning."""

from svc_engine.tuning.config import PARAMETER_HELP_HE, AdvancedConfig
from svc_engine.tuning.optimizer import (
    CandidateResult,
    TuningResult,
    auto_tune,
    candidate_grid,
    score_audio,
)

__all__ = [
    "AdvancedConfig",
    "CandidateResult",
    "PARAMETER_HELP_HE",
    "TuningResult",
    "auto_tune",
    "candidate_grid",
    "score_audio",
]
