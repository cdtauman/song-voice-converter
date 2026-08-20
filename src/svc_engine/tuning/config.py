"""Validated Phase-10 controls shared by RPC, jobs and the desktop UI."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from svc_engine.backends.conversion import ConversionParams
from svc_engine.pitch import PlaybackStrategy
from svc_engine.postfx import AmbienceStrategy, PostFxConfig

__all__ = ["AdvancedConfig", "PARAMETER_HELP_HE"]


PARAMETER_HELP_HE = {
    "index_rate": "עוצמת השימוש ב-index: יותר דמיון לקול, אך ערך גבוה מדי עלול להוסיף חספוס.",
    "protect": "מגן על עיצורים ונשימות מפני החלפה אגרסיבית.",
    "rms_mix_rate": "כמה מהדינמיקה של השירה המקורית נשמרת בתוצאה.",
    "filter_radius": "החלקת עקומת הגובה; שימושי בעיקר בתיקוני גובה רועשים.",
    "formant_shift": "שינוי צבע הקול בלי לשנות את התווים.",
    "target_lufs": "עוצמת היעד של המאסטר. ‎-14 LUFS היא ברירת מחדל בטוחה.",
    "ambience_strategy": "A מחזירה חדר מקורי, B בונה חדר נקי, C משלבת ביניהם.",
    "playback_strategy": "שלם מזיז את כל הליווי; מפוצל שומר תופים ומזיז שכבות טונאליות.",
    "f0_method": "מחלץ גובה: RMVPE לאיכות מלאה, FCPE לתצוגה מהירה.",
    "deess_enabled": "מפחית שורקנות בעיצורים כמו ס׳ ו־ש׳.",
    "melody_correction": "מתקן סטייה גלובלית שהתגלתה אחרי ההמרה.",
}


@dataclass(frozen=True)
class AdvancedConfig:
    index_rate: float = 0.70
    protect: float = 0.33
    rms_mix_rate: float = 0.25
    filter_radius: int = 3
    formant_shift: float = 0.0
    target_lufs: float = -14.0
    ambience_strategy: str = "B"
    playback_strategy: str = "A"
    f0_method: str = "auto"
    deess_enabled: bool = True
    melody_correction: bool = True
    auto_tune: bool = False

    def __post_init__(self) -> None:
        ranges = {
            "index_rate": (self.index_rate, 0.0, 1.0),
            "protect": (self.protect, 0.0, 0.5),
            "rms_mix_rate": (self.rms_mix_rate, 0.0, 1.0),
            "formant_shift": (self.formant_shift, -12.0, 12.0),
            "target_lufs": (self.target_lufs, -70.0, -5.0),
        }
        for name, (value, low, high) in ranges.items():
            if not low <= float(value) <= high:
                raise ValueError(f"{name} must be between {low} and {high}")
        if not 0 <= self.filter_radius <= 7:
            raise ValueError("filter_radius must be between 0 and 7")
        if self.ambience_strategy not in {"A", "B", "C"}:
            raise ValueError("ambience_strategy must be A, B or C")
        if self.playback_strategy not in {item.value for item in PlaybackStrategy}:
            raise ValueError("invalid playback_strategy")
        if self.f0_method not in {"auto", "rmvpe", "fcpe"}:
            raise ValueError("invalid f0_method")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> AdvancedConfig:
        if not raw:
            return cls()
        known = {key: value for key, value in raw.items() if key in cls.__dataclass_fields__}
        return cls(**known)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def conversion_params(self, semitones: int = 0) -> ConversionParams:
        return ConversionParams(
            semitones=semitones,
            index_rate=self.index_rate,
            protect=self.protect,
            rms_mix_rate=self.rms_mix_rate,
            filter_radius=self.filter_radius,
            formant_shift=self.formant_shift,
        )

    def postfx_config(self) -> PostFxConfig:
        return PostFxConfig(
            ambience_strategy=AmbienceStrategy(self.ambience_strategy),
            rms_mix_rate=self.rms_mix_rate,
            target_lufs=self.target_lufs,
            melody_correction=self.melody_correction,
            deess_enabled=self.deess_enabled,
        )

    @property
    def playback(self) -> PlaybackStrategy:
        return PlaybackStrategy(self.playback_strategy)
