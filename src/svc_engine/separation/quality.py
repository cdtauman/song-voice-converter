"""The three quality levels, spelled out.

"Fast / balanced / maximum" is one button in the UI and one flag on the CLI. It
has to mean something precise underneath, or the levels drift into folklore.
Each profile below fixes every knob that changes the result: which checkpoints
run, how big a window, how much overlap, whether an ensemble happens, and which
cleanup passes are in the chain.

Nothing here is measured yet. These are starting positions, and the benchmark in
Phase 10 is what turns them into defaults -- exactly as docs/testing.md requires
of every threshold in this project.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from svc_engine.backends.separation import StemKind
from svc_engine.separation.ensemble import EnsembleMode

__all__ = [
    "QualityLevel",
    "CleanupStep",
    "SeparationProfile",
    "PROFILES",
    "profile_for",
]


class QualityLevel(StrEnum):
    FAST = "fast"
    BALANCED = "balanced"
    MAX = "max"


class CleanupStep(StrEnum):
    """Optional passes over the separated vocal, in the order they may run."""

    DENOISE = "denoise"
    DEREVERB = "dereverb"
    DEECHO = "deecho"
    #: Splits the vocal into lead and backing vocals.
    KARAOKE = "karaoke"


@dataclass(frozen=True)
class SeparationProfile:
    level: QualityLevel
    display_name_he: str
    description_he: str
    #: Catalogue ids. More than one means the results get ensembled.
    models: tuple[str, ...]
    ensemble_mode: EnsembleMode
    segment_size: int
    #: How many times each sample is processed and averaged. 1 is contiguous
    #: chunks; higher costs proportionally more time and hides chunk seams.
    overlap: int
    batch_size: int
    #: Passes that run without being asked for. Cleanup can always be requested
    #: explicitly at any level -- this is only what the level turns on by itself.
    cleanup: tuple[CleanupStep, ...] = ()
    wanted: frozenset[StemKind] = field(
        default_factory=lambda: frozenset({StemKind.VOCALS, StemKind.INSTRUMENTAL})
    )

    @property
    def is_ensemble(self) -> bool:
        return len(self.models) > 1


#: `sep_melband_kim` is the primary model at every level. It has both the
#: highest measured vocal SDR of the candidates (12.60) and the only verified
#: permissive licence among them -- so quality and distributability point the
#: same way, and there is no reason to trade one for the other.
PROFILES: dict[QualityLevel, SeparationProfile] = {
    QualityLevel.FAST: SeparationProfile(
        level=QualityLevel.FAST,
        display_name_he="מהיר",
        description_he="מעבר אחד על השיר, בלי חפיפה. הכי מהיר, ומספיק כדי לשמוע לאן זה הולך.",
        models=("sep_melband_kim",),
        ensemble_mode=EnsembleMode.NONE,
        segment_size=256,
        overlap=1,
        batch_size=1,
    ),
    QualityLevel.BALANCED: SeparationProfile(
        level=QualityLevel.BALANCED,
        display_name_he="מאוזן",
        description_he="ברירת המחדל. אותו מודל, שני מעברים חופפים — פחות תפרים בין מקטעים.",
        models=("sep_melband_kim",),
        ensemble_mode=EnsembleMode.NONE,
        segment_size=256,
        overlap=2,
        batch_size=1,
    ),
    QualityLevel.MAX: SeparationProfile(
        level=QualityLevel.MAX,
        display_name_he="איכות מרבית",
        description_he="שני מודלים משולבים וארבעה מעברים חופפים. איטי משמעותית.",
        models=("sep_melband_kim", "sep_melband_kim_ft2"),
        ensemble_mode=EnsembleMode.MEDIAN,
        segment_size=256,
        overlap=4,
        batch_size=1,
    ),
}


def profile_for(level: QualityLevel | str) -> SeparationProfile:
    return PROFILES[QualityLevel(level)]
