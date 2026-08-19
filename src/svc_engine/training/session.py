"""Crash-safe state for one five-step voice-training wizard session."""

from __future__ import annotations

import datetime as dt
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from svc_engine.jobs._io import atomic_write_json, read_json_object, validate_identifier
from svc_engine.voices.manifest import slugify

__all__ = ["SessionStage", "TrainingSession", "TrainingSessionStore"]


class SessionStage(StrEnum):
    RECORDINGS = "recordings"
    QUALITY = "quality"
    CLEANING = "cleaning"
    TRAINING = "training"
    PAUSED = "paused"
    FINALIZING = "finalizing"
    READY = "ready"
    FAILED = "failed"


def _now() -> str:
    return dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat()


@dataclass
class TrainingSession:
    session_id: str
    voice_id: str
    display_name: str
    recordings: tuple[str, ...]
    consent_confirmed: bool
    consent_note: str
    root: Path
    stage: SessionStage = SessionStage.RECORDINGS
    progress: float = 0.0
    message_he: str = "ההקלטות נוספו."
    quality: dict[str, Any] | None = None
    dataset_seconds: float = 0.0
    current_epoch: int = 0
    total_epochs: int = 200
    estimated_remaining_seconds: float | None = None
    worker_pid: int | None = None
    error_he: str = ""
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)

    @property
    def dataset_dir(self) -> Path:
        return self.root / "dataset"

    @property
    def applio_experiment(self) -> str:
        return f"svc-{self.session_id}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "session_id": self.session_id,
            "voice_id": self.voice_id,
            "display_name": self.display_name,
            "recordings": list(self.recordings),
            "consent_confirmed": self.consent_confirmed,
            "consent_note": self.consent_note,
            "root": str(self.root),
            "stage": self.stage.value,
            "progress": self.progress,
            "message_he": self.message_he,
            "quality": self.quality,
            "dataset_seconds": self.dataset_seconds,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "estimated_remaining_seconds": self.estimated_remaining_seconds,
            "worker_pid": self.worker_pid,
            "error_he": self.error_he,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingSession:
        return cls(
            session_id=validate_identifier(str(raw["session_id"]), label="training session id"),
            voice_id=str(raw["voice_id"]),
            display_name=str(raw["display_name"]),
            recordings=tuple(str(path) for path in raw.get("recordings", [])),
            consent_confirmed=bool(raw.get("consent_confirmed")),
            consent_note=str(raw.get("consent_note") or ""),
            root=Path(str(raw["root"])),
            stage=SessionStage(str(raw.get("stage") or SessionStage.RECORDINGS.value)),
            progress=float(raw.get("progress") or 0.0),
            message_he=str(raw.get("message_he") or ""),
            quality=dict(raw["quality"]) if isinstance(raw.get("quality"), dict) else None,
            dataset_seconds=float(raw.get("dataset_seconds") or 0.0),
            current_epoch=int(raw.get("current_epoch") or 0),
            total_epochs=int(raw.get("total_epochs") or 200),
            estimated_remaining_seconds=(
                float(raw["estimated_remaining_seconds"])
                if raw.get("estimated_remaining_seconds") is not None
                else None
            ),
            worker_pid=int(raw["worker_pid"]) if raw.get("worker_pid") else None,
            error_he=str(raw.get("error_he") or ""),
            created_at=str(raw.get("created_at") or _now()),
            updated_at=str(raw.get("updated_at") or _now()),
        )


class TrainingSessionStore:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self,
        display_name: str,
        recordings: Sequence[Path | str],
        *,
        consent_confirmed: bool,
        consent_note: str,
        total_epochs: int = 200,
    ) -> TrainingSession:
        if not consent_confirmed:
            raise ValueError("explicit consent is required before training")
        if not recordings:
            raise ValueError("at least one recording is required")
        session_id = uuid.uuid4().hex[:16]
        base = slugify(display_name)
        voice_id = f"{base}-{session_id[:6]}"
        session_root = self.root / session_id
        session_root.mkdir(parents=True)
        session = TrainingSession(
            session_id=session_id,
            voice_id=voice_id,
            display_name=display_name.strip() or "קול חדש",
            recordings=tuple(str(Path(path).resolve()) for path in recordings),
            consent_confirmed=True,
            consent_note=consent_note.strip(),
            root=session_root,
            total_epochs=max(20, min(1000, int(total_epochs))),
        )
        self.save(session)
        return session

    def save(self, session: TrainingSession) -> None:
        session.updated_at = _now()
        atomic_write_json(self._path(session.session_id), session.to_dict())

    def load(self, session_id: str) -> TrainingSession:
        return TrainingSession.from_dict(read_json_object(self._path(session_id)))

    def list(self) -> list[TrainingSession]:
        sessions: list[TrainingSession] = []
        for path in self.root.glob("*/session.json"):
            try:
                sessions.append(TrainingSession.from_dict(read_json_object(path)))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(sessions, key=lambda item: item.updated_at, reverse=True)

    def _path(self, session_id: str) -> Path:
        safe = validate_identifier(session_id, label="training session id")
        return self.root / safe / "session.json"
