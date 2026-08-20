"""The analysis engine: understand a song before converting it.

Given a separated vocal, this package measures the singer's range, the song's
key, where the singing is, and which slice best represents it -- everything the
Phase 4 pitch engine and the Phase 8 preview need. See docs/roadmap.md Phase 3
and docs/research.md sections 4, 5 and 7.

The math (`range`, `key`, `segments`, `preview_picker`, `midi`, `features`) is
pure numpy and always importable. The F0 models (`f0`, `rmvpe_model`) import
torch lazily, so bringing in this package never drags the heavy stack along.
"""

from __future__ import annotations

from svc_engine.analysis.key import KeyEstimate, estimate_key, estimate_key_from_audio
from svc_engine.analysis.midi import hz_to_midi, midi_to_hz, midi_to_note_name
from svc_engine.analysis.preview_picker import (
    PreviewChoice,
    PreviewWeights,
    WindowScore,
    pick_preview,
)
from svc_engine.analysis.range import RangeStats, compute_range
from svc_engine.analysis.report import AnalysisReport, analyze_vocal, plot_report
from svc_engine.analysis.segments import Segment, find_voiced_segments, voiced_fraction

__all__ = [
    "AnalysisReport",
    "KeyEstimate",
    "PreviewChoice",
    "PreviewWeights",
    "RangeStats",
    "Segment",
    "WindowScore",
    "analyze_vocal",
    "compute_range",
    "estimate_key",
    "estimate_key_from_audio",
    "find_voiced_segments",
    "hz_to_midi",
    "midi_to_hz",
    "midi_to_note_name",
    "pick_preview",
    "plot_report",
    "voiced_fraction",
]
