"""The pitch engine: decide how far to shift a song, and carry the shift out.

This is the heart of the project (``docs/research.md`` section 4, ``docs/roadmap.md``
Phase 4). Given the song's measured pitch distribution and a target voice's
profile, ``strategy`` finds the shift ``s = 12k + r`` that seats the singing most
naturally in the new voice, ``explain`` says it in plain Hebrew, ``shifter`` and
``instrumental`` carry it out length-preservingly, ``quality_probe`` measures the
per-voice quality curve the cost function reads, and ``softening`` is the optional
melody-compressing switch.

The decision maths (``strategy``, ``explain``, ``softening``) is pure numpy and
always importable. ``shifter`` imports python-stretch lazily and ``quality_probe``
takes an injected conversion backend, so importing this package never drags the
heavy stack along -- the same rule the analysis package follows.
"""

from __future__ import annotations

from svc_engine.pitch.explain import explain_candidate, explain_decision
from svc_engine.pitch.instrumental import PitchedStems, PlaybackStrategy, shift_playback
from svc_engine.pitch.quality_probe import (
    DEFAULT_PROBE_SHIFTS,
    ArtifactStats,
    QualityCurve,
    QualitySample,
    measure_artifacts,
    probe_quality_curve,
)
from svc_engine.pitch.shifter import PythonStretchShifter
from svc_engine.pitch.softening import soften_f0
from svc_engine.pitch.strategy import (
    CostWeights,
    PitchDistribution,
    ShiftCandidate,
    ShiftDecision,
    decide_shift,
    decompose,
    playback_penalty,
)

__all__ = [
    "DEFAULT_PROBE_SHIFTS",
    "ArtifactStats",
    "CostWeights",
    "PitchDistribution",
    "PitchedStems",
    "PlaybackStrategy",
    "PythonStretchShifter",
    "QualityCurve",
    "QualitySample",
    "ShiftCandidate",
    "ShiftDecision",
    "decide_shift",
    "decompose",
    "explain_candidate",
    "explain_decision",
    "measure_artifacts",
    "playback_penalty",
    "probe_quality_curve",
    "shift_playback",
    "soften_f0",
]
