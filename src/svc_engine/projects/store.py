"""Save and load user projects without exposing partially written JSON."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from svc_engine.jobs._io import atomic_write_json, read_json_object, validate_identifier

_SCHEMA = 1

__all__ = ["Project", "ProjectStore"]


@dataclass(frozen=True)
class Project:
    project_id: str
    name: str
    data: dict[str, Any]
    created_at: float
    updated_at: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "project_id": self.project_id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Project:
        if int(raw.get("schema", -1)) != _SCHEMA:
            raise ValueError("unsupported project schema")
        data = raw.get("data")
        if not isinstance(data, dict):
            raise ValueError("project data must be an object")
        return cls(
            project_id=validate_identifier(str(raw["project_id"]), label="project id"),
            name=str(raw["name"]),
            data=data,
            created_at=float(raw["created_at"]),
            updated_at=float(raw["updated_at"]),
        )


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, project_id: str, *, name: str, data: dict[str, Any]) -> Project:
        validate_identifier(project_id, label="project id")
        if not isinstance(data, dict):
            raise TypeError("project data must be a dictionary")
        path = self._path(project_id)
        previous = self.load(project_id) if path.is_file() else None
        now = time.time()
        project = Project(
            project_id=project_id,
            name=name,
            data=data,
            created_at=previous.created_at if previous else now,
            updated_at=now,
        )
        if path.is_file():
            # A last-known-good copy makes accidental corruption recoverable.
            if previous is None:  # pragma: no cover - guarded by the load above
                raise RuntimeError("existing project could not be read")
            atomic_write_json(path.with_suffix(".json.bak"), previous.to_dict())
        atomic_write_json(path, project.to_dict())
        return project

    def load(self, project_id: str) -> Project:
        path = self._path(project_id)
        try:
            return Project.from_dict(read_json_object(path))
        except (OSError, ValueError, KeyError, TypeError) as primary:
            backup = path.with_suffix(".json.bak")
            if backup.is_file():
                return Project.from_dict(read_json_object(backup))
            raise primary

    def list(self) -> list[Project]:
        projects: list[Project] = []
        for path in sorted(self.root.glob("*.json")):
            try:
                projects.append(Project.from_dict(read_json_object(path)))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return sorted(projects, key=lambda item: item.updated_at, reverse=True)

    def _path(self, project_id: str) -> Path:
        return self.root / f"{validate_identifier(project_id, label='project id')}.json"
