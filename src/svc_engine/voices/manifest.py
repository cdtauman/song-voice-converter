"""The voice manifest -- `voice.json` -- and the on-disk layout of one voice.

This is deliberately pure: dataclasses plus JSON, no torch, no audio. It is the
one place that knows what files make up a voice and what a voice claims about
itself, so the library, the importer and the conversion backend all agree.

Consent is a first-class field, not a formality: SongVoice will only ever use a
voice the user made or was explicitly allowed to use (README, docs/architecture
section 5), so a manifest without a confirmed consent flag is not a usable
voice.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "VOICE_FILE",
    "MODEL_FILE",
    "INDEX_FILE",
    "PROFILE_FILE",
    "SAMPLE_FILE",
    "AVATAR_FILE",
    "MANIFEST_VERSION",
    "HealthStatus",
    "HealthState",
    "RecommendedParams",
    "VoiceSource",
    "VoiceManifest",
    "slugify",
    "is_canonical_voice_id",
]

#: File names inside a voice folder. Fixed so every part of the app agrees.
VOICE_FILE = "voice.json"
MODEL_FILE = "model.pth"
INDEX_FILE = "model.index"
PROFILE_FILE = "profile.json"
SAMPLE_FILE = "sample.wav"
AVATAR_FILE = "avatar.png"

MANIFEST_VERSION = 1


class VoiceSource(StrEnum):
    """Where a voice came from -- shown to the user, and a consent audit trail."""

    TRAINED_LOCALLY = "trained_locally"
    IMPORTED = "imported"


class HealthStatus(StrEnum):
    OK = "ok"
    MISSING_MODEL = "missing_model"
    CORRUPT_MODEL = "corrupt_model"
    NO_CONSENT = "no_consent"
    UNKNOWN = "unknown"


#: Hebrew, user-facing, one line each. The UI shows these next to the voice card.
_HEALTH_HE: dict[HealthStatus, str] = {
    HealthStatus.OK: "תקין",
    HealthStatus.MISSING_MODEL: "חסר קובץ המודל של הקול",
    HealthStatus.CORRUPT_MODEL: "קובץ המודל של הקול פגום או לא בפורמט מוכר",
    HealthStatus.NO_CONSENT: "הקול לא אושר לשימוש",
    HealthStatus.UNKNOWN: "מצב הקול לא נבדק",
}


@dataclass(frozen=True)
class HealthState:
    """The result of the last health check on a voice."""

    status: HealthStatus = HealthStatus.UNKNOWN
    checked_at: str = ""

    @property
    def ok(self) -> bool:
        return self.status is HealthStatus.OK

    @property
    def note_he(self) -> str:
        return _HEALTH_HE.get(self.status, _HEALTH_HE[HealthStatus.UNKNOWN])

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status.value, "checked_at": self.checked_at}

    @classmethod
    def from_dict(cls, data: dict | None) -> HealthState:
        if not data:
            return cls()
        try:
            status = HealthStatus(str(data.get("status") or "unknown"))
        except ValueError:
            status = HealthStatus.UNKNOWN
        return cls(status=status, checked_at=str(data.get("checked_at") or ""))


@dataclass(frozen=True)
class RecommendedParams:
    """Community starting points, per docs/research.md 3.5. Not calibrated yet."""

    index_rate: float = 0.70
    protect: float = 0.33
    rms_mix_rate: float = 0.25

    def to_dict(self) -> dict[str, float]:
        return {
            "index_rate": round(self.index_rate, 4),
            "protect": round(self.protect, 4),
            "rms_mix_rate": round(self.rms_mix_rate, 4),
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> RecommendedParams:
        if not data:
            return cls()
        d = cls()
        return cls(
            index_rate=float(data.get("index_rate", d.index_rate)),
            protect=float(data.get("protect", d.protect)),
            rms_mix_rate=float(data.get("rms_mix_rate", d.rms_mix_rate)),
        )


@dataclass(frozen=True)
class VoiceManifest:
    """Everything `voice.json` records about one voice."""

    voice_id: str
    display_name: str
    source: VoiceSource
    consent_confirmed: bool
    consent_note: str = ""
    created_at: str = ""
    sample_rate: int = 0
    rvc_version: str = "v2"
    has_index: bool = False
    has_sample: bool = False
    has_avatar: bool = False
    recommended: RecommendedParams = field(default_factory=RecommendedParams)
    health: HealthState = field(default_factory=HealthState)
    notes_he: str = ""

    @property
    def usable(self) -> bool:
        """A voice may be used only with confirmed consent and a healthy model."""
        return self.consent_confirmed and self.health.ok

    def to_dict(self) -> dict[str, object]:
        return {
            "version": MANIFEST_VERSION,
            "id": self.voice_id,
            "display_name": self.display_name,
            "source": self.source.value,
            "consent_confirmed": self.consent_confirmed,
            "consent_note": self.consent_note,
            "created_at": self.created_at,
            "sample_rate": self.sample_rate,
            "rvc_version": self.rvc_version,
            "has_index": self.has_index,
            "has_sample": self.has_sample,
            "has_avatar": self.has_avatar,
            "recommended": self.recommended.to_dict(),
            "health": self.health.to_dict(),
            "notes_he": self.notes_he,
        }

    @classmethod
    def from_dict(cls, data: dict) -> VoiceManifest:
        try:
            source = VoiceSource(str(data.get("source") or "imported"))
        except ValueError:
            source = VoiceSource.IMPORTED
        return cls(
            voice_id=str(data.get("id") or ""),
            display_name=str(data.get("display_name") or data.get("id") or ""),
            source=source,
            consent_confirmed=bool(data.get("consent_confirmed", False)),
            consent_note=str(data.get("consent_note") or ""),
            created_at=str(data.get("created_at") or ""),
            sample_rate=int(data.get("sample_rate") or 0),
            rvc_version=str(data.get("rvc_version") or "v2"),
            has_index=bool(data.get("has_index", False)),
            has_sample=bool(data.get("has_sample", False)),
            has_avatar=bool(data.get("has_avatar", False)),
            recommended=RecommendedParams.from_dict(data.get("recommended")),
            health=HealthState.from_dict(data.get("health")),
            notes_he=str(data.get("notes_he") or ""),
        )

    def with_health(self, health: HealthState) -> VoiceManifest:
        from dataclasses import replace

        return replace(self, health=health)

    def save(self, voice_dir: Path | str) -> Path:
        path = Path(voice_dir) / VOICE_FILE
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, voice_dir: Path | str) -> VoiceManifest:
        path = Path(voice_dir) / VOICE_FILE
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, fallback: str = "voice") -> str:
    """A filesystem-safe voice id. Hebrew display names keep their own field;
    the id stays ASCII because it becomes a directory the C++/AI tooling opens,
    and some of that tooling still breaks on non-ASCII paths (config.Paths.work
    makes the same choice)."""
    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return slug or fallback


def is_canonical_voice_id(value: object) -> bool:
    """True only for the ASCII ids this package is allowed to put on disk."""
    return isinstance(value, str) and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value) is not None
