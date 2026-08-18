"""Crash-safe snapshots used to discover and resume unfinished jobs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from svc_engine.jobs._io import atomic_write_json, read_json_object, validate_identifier

_SCHEMA = 1

__all__ = ["JobState", "RecoveryStore", "StepState"]


class StepState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CACHED = "cached"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobState(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class StepSnapshot:
    status: StepState = StepState.PENDING
    cache_key: str | None = None
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "cache_key": self.cache_key,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> StepSnapshot:
        return cls(
            status=StepState(str(raw.get("status", StepState.PENDING.value))),
            cache_key=str(raw["cache_key"]) if raw.get("cache_key") else None,
            started_at=float(raw["started_at"]) if raw.get("started_at") is not None else None,
            finished_at=float(raw["finished_at"]) if raw.get("finished_at") is not None else None,
            error=str(raw["error"]) if raw.get("error") else None,
        )


@dataclass
class RecoverySnapshot:
    job_id: str
    name: str
    plan_signature: str
    status: JobState
    steps: dict[str, StepSnapshot]
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    error: str | None = None

    @property
    def recoverable(self) -> bool:
        return self.status in {JobState.PENDING, JobState.RUNNING, JobState.FAILED}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "job_id": self.job_id,
            "name": self.name,
            "plan_signature": self.plan_signature,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "error": self.error,
            "steps": {key: value.to_dict() for key, value in sorted(self.steps.items())},
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> RecoverySnapshot:
        if int(raw.get("schema", -1)) != _SCHEMA:
            raise ValueError("unsupported recovery snapshot schema")
        job_id = validate_identifier(str(raw["job_id"]), label="job id")
        steps_raw = raw.get("steps")
        if not isinstance(steps_raw, dict):
            raise ValueError("recovery steps must be an object")
        return cls(
            job_id=job_id,
            name=str(raw["name"]),
            plan_signature=str(raw["plan_signature"]),
            status=JobState(str(raw["status"])),
            created_at=float(raw["created_at"]),
            updated_at=float(raw["updated_at"]),
            error=str(raw["error"]) if raw.get("error") else None,
            steps={
                str(key): StepSnapshot.from_dict(dict(value))
                for key, value in steps_raw.items()
            },
        )


class RecoveryStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def create(
        self, *, job_id: str, name: str, plan_signature: str, step_ids: list[str]
    ) -> RecoverySnapshot:
        validate_identifier(job_id, label="job id")
        now = time.time()
        snapshot = RecoverySnapshot(
            job_id=job_id,
            name=name,
            plan_signature=plan_signature,
            status=JobState.PENDING,
            steps={step_id: StepSnapshot() for step_id in step_ids},
            created_at=now,
            updated_at=now,
        )
        self.save(snapshot)
        return snapshot

    def save(self, snapshot: RecoverySnapshot) -> None:
        snapshot.updated_at = time.time()
        atomic_write_json(self._path(snapshot.job_id), snapshot.to_dict())

    def load(self, job_id: str) -> RecoverySnapshot | None:
        path = self._path(job_id)
        if not path.is_file():
            return None
        return RecoverySnapshot.from_dict(read_json_object(path))

    def discover(self) -> list[RecoverySnapshot]:
        found: list[RecoverySnapshot] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                snapshot = RecoverySnapshot.from_dict(read_json_object(path))
            except (OSError, ValueError, KeyError, TypeError):
                # User data is never destroyed merely because it is unreadable.
                continue
            if snapshot.recoverable:
                found.append(snapshot)
        return sorted(found, key=lambda item: item.updated_at, reverse=True)

    def remove(self, job_id: str) -> None:
        self._path(job_id).unlink(missing_ok=True)

    def _path(self, job_id: str) -> Path:
        return self.root / f"{validate_identifier(job_id, label='job id')}.json"
