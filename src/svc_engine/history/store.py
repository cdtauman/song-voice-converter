"""Durable job metadata in SQLite; audio and intermediate data stay in files."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["HistoryEntry", "HistoryStore"]


@dataclass(frozen=True)
class HistoryEntry:
    job_id: str
    name: str
    status: str
    created_at: float
    updated_at: float
    progress: float
    current_step: str | None
    error: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "progress": self.progress,
            "current_step": self.current_step,
            "error": self.error,
        }


class HistoryStore:
    def __init__(self, database: Path) -> None:
        self.database = Path(database)
        self.database.parent.mkdir(parents=True, exist_ok=True)
        self._initialise()

    def record(
        self,
        *,
        job_id: str,
        name: str,
        status: str,
        progress: float,
        current_step: str | None = None,
        error: str | None = None,
        created_at: float | None = None,
    ) -> None:
        now = time.time()
        created = created_at if created_at is not None else now
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO job_history
                    (job_id, name, status, created_at, updated_at, progress, current_step, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET
                    name=excluded.name,
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    progress=excluded.progress,
                    current_step=excluded.current_step,
                    error=excluded.error
                """,
                (job_id, name, status, created, now, progress, current_step, error),
            )

    def get(self, job_id: str) -> HistoryEntry | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT job_id, name, status, created_at, updated_at, progress, "
                "current_step, error "
                "FROM job_history WHERE job_id = ?",
                (job_id,),
            ).fetchone()
        return self._from_row(row) if row else None

    def list(self, *, limit: int = 100) -> list[HistoryEntry]:
        if not 1 <= limit <= 1000:
            raise ValueError("history limit must be between 1 and 1000")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT job_id, name, status, created_at, updated_at, progress, "
                "current_step, error "
                "FROM job_history ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._from_row(row) for row in rows]

    def _initialise(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS job_history (
                    job_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    progress REAL NOT NULL CHECK(progress >= 0 AND progress <= 1),
                    current_step TEXT,
                    error TEXT
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_job_history_updated "
                "ON job_history(updated_at DESC)"
            )

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database, timeout=10.0)

    @staticmethod
    def _from_row(row: tuple[Any, ...]) -> HistoryEntry:
        return HistoryEntry(
            job_id=str(row[0]),
            name=str(row[1]),
            status=str(row[2]),
            created_at=float(row[3]),
            updated_at=float(row[4]),
            progress=float(row[5]),
            current_step=str(row[6]) if row[6] is not None else None,
            error=str(row[7]) if row[7] is not None else None,
        )
