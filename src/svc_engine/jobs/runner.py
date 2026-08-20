"""Dependency-aware execution with progress, cache reuse and crash recovery."""

from __future__ import annotations

import contextlib
import errno
import hashlib
import json
import logging
import shutil
import sqlite3
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from svc_engine.config import Paths, Settings
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.history import HistoryStore
from svc_engine.jobs._io import validate_identifier
from svc_engine.jobs.cache import StepCache, cache_key
from svc_engine.jobs.cancel import CancellationToken
from svc_engine.jobs.recovery import (
    JobState,
    RecoverySnapshot,
    RecoveryStore,
    StepState,
)

__all__ = [
    "Job",
    "JobProgress",
    "JobResult",
    "JobRunner",
    "Step",
    "StepContext",
    "StepResult",
]


StepAction = Callable[["StepContext"], Mapping[str, Path]]
ProgressCallback = Callable[["JobProgress"], None]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Step:
    """One deterministic node in a job graph.

    ``version`` must change whenever the implementation changes in a way that
    can affect outputs. Parameters and input file *contents* are hashed by the
    runner; dependencies contribute their own keys.
    """

    step_id: str
    action: StepAction
    parameters: Mapping[str, Any] = field(default_factory=dict)
    input_files: tuple[Path, ...] = ()
    needs: tuple[str, ...] = ()
    version: str = "1"
    weight: float = 1.0
    vram_hint_mb: int = 0
    cancellable: bool = True

    def __post_init__(self) -> None:
        validate_identifier(self.step_id, label="step id")
        if self.weight <= 0:
            raise ValueError("step weight must be positive")
        if self.vram_hint_mb < 0:
            raise ValueError("vram hint must be non-negative")


@dataclass(frozen=True)
class Job:
    name: str
    steps: tuple[Step, ...]
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        validate_identifier(self.job_id, label="job id")
        if not self.name.strip():
            raise ValueError("job name must not be empty")
        if not self.steps:
            raise ValueError("a job needs at least one step")


@dataclass(frozen=True)
class JobProgress:
    job_id: str
    step_id: str
    step_fraction: float
    overall_fraction: float
    message_he: str


@dataclass(frozen=True)
class StepResult:
    step_id: str
    cache_key: str
    outputs: dict[str, Path]
    cache_hit: bool
    seconds: float


@dataclass(frozen=True)
class JobResult:
    job_id: str
    status: JobState
    steps: dict[str, StepResult]
    resumed: bool
    seconds: float


class StepContext:
    """The only mutable surface given to a step implementation."""

    def __init__(
        self,
        *,
        job_id: str,
        step: Step,
        output_dir: Path,
        dependencies: dict[str, dict[str, Path]],
        token: CancellationToken,
        report: Callable[[float, str], None],
    ) -> None:
        self.job_id = job_id
        self.step_id = step.step_id
        self.parameters = dict(step.parameters)
        self.input_files = tuple(Path(path) for path in step.input_files)
        self.output_dir = output_dir
        self.dependencies = dependencies
        self.vram_hint_mb = step.vram_hint_mb
        self.cancellable = step.cancellable
        self._token = token
        self._report = report

    def check_cancelled(self) -> None:
        self._token.raise_if_cancelled()

    def progress(self, fraction: float, message_he: str = "מעבדים…") -> None:
        if not 0.0 <= fraction <= 1.0:
            raise ValueError("step progress must be between 0 and 1")
        self.check_cancelled()
        self._report(fraction, message_he)


class JobRunner:
    def __init__(
        self,
        paths: Paths,
        *,
        settings: Settings | None = None,
        cache: StepCache | None = None,
        recovery: RecoveryStore | None = None,
        history: HistoryStore | None = None,
    ) -> None:
        self.paths = paths
        self.paths.ensure()
        self.settings = settings or Settings()
        self.cache = cache or StepCache(paths.cache / "steps")
        self.recovery = recovery or RecoveryStore(paths.root / "jobs")
        self.history = history or HistoryStore(paths.db)
        self.work_root = paths.work / "jobs"
        self.work_root.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        job: Job,
        *,
        cancellation: CancellationToken | None = None,
        on_progress: ProgressCallback | None = None,
    ) -> JobResult:
        ordered = self._topological(job)
        signature = self._plan_signature(ordered)
        existing = self.recovery.load(job.job_id)
        resumed = existing is not None and existing.recoverable
        if existing is not None and existing.plan_signature != signature:
            raise ValueError("cannot resume a job with a different step graph")
        snapshot = existing or self.recovery.create(
            job_id=job.job_id,
            name=job.name,
            plan_signature=signature,
            step_ids=[step.step_id for step in ordered],
        )
        if set(snapshot.steps) != {step.step_id for step in ordered}:
            raise ValueError("recovery snapshot does not match the job steps")

        token = cancellation or CancellationToken()
        workspace = self._prepare_workspace(job.job_id)
        active_ids = {item.job_id for item in self.recovery.discover()}
        self._cleanup_orphan_workspaces(except_ids=active_ids | {job.job_id})
        self.cache.remove_abandoned_temporaries()
        started = time.monotonic()
        snapshot.status = JobState.RUNNING
        snapshot.error = None
        self.recovery.save(snapshot)
        self.history.record(
            job_id=job.job_id,
            name=job.name,
            status=JobState.RUNNING.value,
            progress=0.0,
            created_at=snapshot.created_at,
        )

        total_weight = sum(step.weight for step in ordered)
        completed_weight = 0.0
        results: dict[str, StepResult] = {}
        cache_keys: dict[str, str] = {}
        try:
            for step in ordered:
                token.raise_if_cancelled()
                dependency_keys = [cache_keys[dependency] for dependency in step.needs]
                key = cache_key(
                    step_id=step.step_id,
                    version=step.version,
                    parameters=step.parameters,
                    input_files=step.input_files,
                    dependency_keys=dependency_keys,
                )
                cache_keys[step.step_id] = key
                state = snapshot.steps[step.step_id]
                step_started = time.monotonic()
                hit = self.cache.lookup(key)
                if hit is not None:
                    state.status = StepState.CACHED
                    state.cache_key = key
                    state.started_at = state.started_at or time.time()
                    state.finished_at = time.time()
                    state.error = None
                    self.recovery.save(snapshot)
                    result = StepResult(
                        step_id=step.step_id,
                        cache_key=key,
                        outputs=hit.outputs,
                        cache_hit=True,
                        seconds=time.monotonic() - step_started,
                    )
                else:
                    state.status = StepState.RUNNING
                    state.cache_key = key
                    state.started_at = time.time()
                    state.finished_at = None
                    state.error = None
                    self.recovery.save(snapshot)
                    self._record_progress(
                        job,
                        step,
                        snapshot,
                        completed_weight / total_weight,
                        0.0,
                        "מתחילים את השלב…",
                        on_progress,
                    )
                    partial = workspace / f"{step.step_id}.partial-{uuid.uuid4().hex}"
                    partial.mkdir(parents=True)

                    base_weight = completed_weight
                    current_step = step

                    def report(
                        fraction: float,
                        message: str,
                        *,
                        _base_weight: float = base_weight,
                        _current_step: Step = current_step,
                    ) -> None:
                        overall = (
                            _base_weight + _current_step.weight * fraction
                        ) / total_weight
                        self._record_progress(
                            job,
                            _current_step,
                            snapshot,
                            overall,
                            fraction,
                            message,
                            on_progress,
                        )

                    context = StepContext(
                        job_id=job.job_id,
                        step=step,
                        output_dir=partial,
                        dependencies={
                            dependency: results[dependency].outputs
                            for dependency in step.needs
                        },
                        token=token,
                        report=report,
                    )
                    try:
                        produced = dict(step.action(context))
                        token.raise_if_cancelled()
                        entry = self.cache.publish(key, produced)
                    finally:
                        shutil.rmtree(partial, ignore_errors=True)
                    state.status = StepState.COMPLETED
                    state.finished_at = time.time()
                    self.recovery.save(snapshot)
                    result = StepResult(
                        step_id=step.step_id,
                        cache_key=key,
                        outputs=entry.outputs,
                        cache_hit=False,
                        seconds=time.monotonic() - step_started,
                    )

                results[step.step_id] = result
                completed_weight += step.weight
                self._record_progress(
                    job,
                    step,
                    snapshot,
                    completed_weight / total_weight,
                    1.0,
                    "השלב הושלם.",
                    on_progress,
                )

            snapshot.status = JobState.COMPLETED
            snapshot.error = None
            self.recovery.save(snapshot)
            self.history.record(
                job_id=job.job_id,
                name=job.name,
                status=JobState.COMPLETED.value,
                progress=1.0,
                created_at=snapshot.created_at,
            )
            max_bytes = max(0, int(self.settings.keep_cache_gb * 1024**3))
            try:
                self.cache.enforce_limit(
                    max_bytes, protected_keys=set(cache_keys.values())
                )
            except OSError:
                # Quota maintenance must not turn a completed cover into a
                # failed job merely because an antivirus temporarily locked an
                # old cache file on Windows.
                log.warning("cache quota cleanup failed", exc_info=True)
            return JobResult(
                job_id=job.job_id,
                status=JobState.COMPLETED,
                steps=results,
                resumed=resumed,
                seconds=time.monotonic() - started,
            )
        except EngineError as exc:
            if exc.code is ErrorCode.CANCELLED:
                self._mark_terminal(snapshot, JobState.CANCELLED, str(exc))
                self.history.record(
                    job_id=job.job_id,
                    name=job.name,
                    status=JobState.CANCELLED.value,
                    progress=completed_weight / total_weight,
                    error=exc.detail,
                    created_at=snapshot.created_at,
                )
            else:
                self._mark_terminal(snapshot, JobState.FAILED, str(exc))
                self.history.record(
                    job_id=job.job_id,
                    name=job.name,
                    status=JobState.FAILED.value,
                    progress=completed_weight / total_weight,
                    error=str(exc),
                    created_at=snapshot.created_at,
                )
            raise
        except OSError as exc:
            if exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112:
                translated = EngineError(ErrorCode.DISK_FULL, str(exc))
                # The failed cache publication and step scratch have already
                # been removed. If the volume is still completely full, the
                # prior RUNNING snapshot remains a valid recovery point.
                with contextlib.suppress(OSError, sqlite3.Error):
                    self._mark_terminal(snapshot, JobState.FAILED, str(translated))
                    self.history.record(
                        job_id=job.job_id,
                        name=job.name,
                        status=JobState.FAILED.value,
                        progress=completed_weight / total_weight,
                        error=translated.detail,
                        created_at=snapshot.created_at,
                    )
                raise translated from exc
            self._mark_terminal(snapshot, JobState.FAILED, f"OSError: {exc}")
            self.history.record(
                job_id=job.job_id,
                name=job.name,
                status=JobState.FAILED.value,
                progress=completed_weight / total_weight,
                error=f"OSError: {exc}",
                created_at=snapshot.created_at,
            )
            raise
        except BaseException as exc:
            self._mark_terminal(snapshot, JobState.FAILED, f"{type(exc).__name__}: {exc}")
            self.history.record(
                job_id=job.job_id,
                name=job.name,
                status=JobState.FAILED.value,
                progress=completed_weight / total_weight,
                error=f"{type(exc).__name__}: {exc}",
                created_at=snapshot.created_at,
            )
            raise
        finally:
            # Cache entries are immutable and snapshots carry every durable
            # result, so no successful data lives in scratch space.
            shutil.rmtree(workspace, ignore_errors=True)

    def recoverable_jobs(self) -> list[RecoverySnapshot]:
        return self.recovery.discover()

    def cleanup(self) -> dict[str, int]:
        active = {snapshot.job_id for snapshot in self.recovery.discover()}
        workspaces = self._cleanup_orphan_workspaces(except_ids=active)
        temporaries = self.cache.remove_abandoned_temporaries()
        max_bytes = max(0, int(self.settings.keep_cache_gb * 1024**3))
        stats = self.cache.enforce_limit(max_bytes)
        return {
            "workspaces_removed": workspaces,
            "cache_temporaries_removed": temporaries,
            "cache_entries": stats.entries,
            "cache_size_bytes": stats.size_bytes,
        }

    def _record_progress(
        self,
        job: Job,
        step: Step,
        snapshot: RecoverySnapshot,
        overall: float,
        step_fraction: float,
        message: str,
        callback: ProgressCallback | None,
    ) -> None:
        bounded = min(1.0, max(0.0, overall))
        self.history.record(
            job_id=job.job_id,
            name=job.name,
            status=JobState.RUNNING.value,
            progress=bounded,
            current_step=step.step_id,
            created_at=snapshot.created_at,
        )
        if callback is not None:
            callback(
                JobProgress(
                    job_id=job.job_id,
                    step_id=step.step_id,
                    step_fraction=step_fraction,
                    overall_fraction=bounded,
                    message_he=message,
                )
            )

    def _prepare_workspace(self, job_id: str) -> Path:
        workspace = self.work_root / validate_identifier(job_id, label="job id")
        resolved = workspace.resolve()
        if resolved.parent != self.work_root.resolve():
            raise ValueError("job workspace escaped the work root")
        shutil.rmtree(resolved, ignore_errors=True)
        resolved.mkdir(parents=True)
        return resolved

    def _cleanup_orphan_workspaces(self, *, except_ids: set[str]) -> int:
        removed = 0
        root = self.work_root.resolve()
        for child in self.work_root.iterdir():
            if not child.is_dir() or child.name in except_ids:
                continue
            resolved = child.resolve()
            if resolved.parent != root:
                continue
            shutil.rmtree(resolved)
            removed += 1
        return removed

    def _mark_terminal(
        self, snapshot: RecoverySnapshot, status: JobState, error: str
    ) -> None:
        snapshot.status = status
        snapshot.error = error
        for step in snapshot.steps.values():
            if step.status is StepState.RUNNING:
                step.status = (
                    StepState.CANCELLED if status is JobState.CANCELLED else StepState.FAILED
                )
                step.finished_at = time.time()
                step.error = error
        self.recovery.save(snapshot)

    @staticmethod
    def _plan_signature(steps: list[Step]) -> str:
        descriptors = [
            {
                "id": step.step_id,
                "needs": list(step.needs),
                "version": step.version,
                "weight": step.weight,
                "cancellable": step.cancellable,
            }
            for step in steps
        ]
        payload = json.dumps(descriptors, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _topological(job: Job) -> list[Step]:
        by_id: dict[str, Step] = {}
        for step in job.steps:
            if step.step_id in by_id:
                raise ValueError(f"duplicate step id: {step.step_id}")
            by_id[step.step_id] = step
        for step in job.steps:
            missing = set(step.needs) - set(by_id)
            if missing:
                raise ValueError(f"step {step.step_id} has missing dependencies: {sorted(missing)}")

        ordered: list[Step] = []
        temporary: set[str] = set()
        permanent: set[str] = set()

        def visit(step_id: str) -> None:
            if step_id in permanent:
                return
            if step_id in temporary:
                raise ValueError("job graph contains a dependency cycle")
            temporary.add(step_id)
            for dependency in by_id[step_id].needs:
                visit(dependency)
            temporary.remove(step_id)
            permanent.add(step_id)
            ordered.append(by_id[step_id])

        for step in job.steps:
            visit(step.step_id)
        return ordered
