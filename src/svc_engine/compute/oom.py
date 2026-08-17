"""Recovering from "out of memory" without asking the user anything.

The ladder is fixed by docs/architecture.md section 4:

    chunk / 2  ->  batch = 1  ->  minimal overlap  ->  CPU

Each rung is a strictly smaller memory footprint than the one before it, and the
last rung always succeeds because CPU memory is the machine's RAM. The user sees
one sentence -- "מתאימים את ההגדרות לכרטיס המסך…" -- and never a stack trace.

This lives in `compute` rather than in `separation` because conversion in
Phase 5 hits exactly the same wall with exactly the same remedies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import TypeVar

from svc_engine.compute.devices import ComputeBackend
from svc_engine.errors import EngineError, ErrorCode

__all__ = [
    "ResourcePlan",
    "OomStep",
    "is_oom_error",
    "oom_ladder",
    "run_with_oom_ladder",
]

log = logging.getLogger(__name__)

T = TypeVar("T")

#: Never subdivide below this -- past it the chunk is shorter than the model's
#: own receptive field and the output degrades instead of merely being slower.
MIN_SEGMENT_SIZE = 32

#: Substrings that identify an out-of-memory failure across CUDA, XPU and host
#: allocators. Matching on text is unpleasant, but the exception *types* differ
#: per backend and torch does not expose a common base class for them.
_OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    # Level Zero / SYCL, which is what an Intel XPU allocation failure surfaces as.
    "out_of_device_memory",
    "out_of_host_memory",
    "cublas_status_alloc_failed",
    "failed to allocate",
    "insufficient memory",
    "allocation failed",
    "bad_alloc",
)


@dataclass(frozen=True)
class ResourcePlan:
    """The memory-relevant knobs, shared by every heavy backend.

    `overlap` is a coverage factor: 1 means contiguous chunks, 4 means every
    sample is processed four times and the passes are averaged.
    """

    segment_size: int = 256
    batch_size: int = 1
    overlap: int = 2
    backend: ComputeBackend = ComputeBackend.CPU


@dataclass(frozen=True)
class OomStep:
    """One rung. `index` is 1-based so it can go straight into a message."""

    plan: ResourcePlan
    index: int
    total: int
    reason_he: str

    @property
    def message_he(self) -> str:
        if self.index == 1:
            return ""
        return f"מתאימים את ההגדרות לכרטיס המסך… (ניסיון {self.index} מתוך {self.total})"


def is_oom_error(exc: BaseException) -> bool:
    """True when `exc` is a memory exhaustion failure from any backend."""
    if isinstance(exc, MemoryError):
        return True
    if type(exc).__name__ in {"OutOfMemoryError", "OutOfMemoryException"}:
        return True
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _OOM_MARKERS)


def oom_ladder(
    plan: ResourcePlan, cpu_fallback: bool = True
) -> Iterator[ResourcePlan]:
    """Yield `plan`, then progressively cheaper variants of it.

    On a plan that already runs on CPU the accelerator rungs are pointless, so
    only the segment reduction is attempted.
    """
    yield plan

    if plan.segment_size > MIN_SEGMENT_SIZE:
        plan = replace(plan, segment_size=max(MIN_SEGMENT_SIZE, plan.segment_size // 2))
        yield plan

    if plan.batch_size > 1:
        plan = replace(plan, batch_size=1)
        yield plan

    if plan.overlap > 1:
        plan = replace(plan, overlap=1)
        yield plan

    if cpu_fallback and plan.backend is not ComputeBackend.CPU:
        yield replace(plan, backend=ComputeBackend.CPU)


_REASONS = (
    "הגדרות רגילות",
    "מקטעים קטנים יותר",
    "עיבוד אחד בכל פעם",
    "חפיפה מינימלית",
    "מעבר לעיבוד על המעבד",
)


def run_with_oom_ladder(
    plan: ResourcePlan,
    run: Callable[[ResourcePlan], T],
    on_step: Callable[[OomStep], None] | None = None,
    cpu_fallback: bool = True,
) -> T:
    """Call `run` down the ladder until it stops running out of memory.

    Only out-of-memory failures are retried. Anything else is a real bug or a
    real bad input and is raised immediately -- retrying it four times with less
    memory would just make the same error take four times as long.
    """
    rungs = list(oom_ladder(plan, cpu_fallback=cpu_fallback))
    last: BaseException | None = None

    for index, rung in enumerate(rungs, start=1):
        step = OomStep(
            plan=rung,
            index=index,
            total=len(rungs),
            reason_he=_REASONS[min(index - 1, len(_REASONS) - 1)],
        )
        if on_step is not None:
            on_step(step)
        try:
            return run(rung)
        except BaseException as exc:  # noqa: BLE001  re-raised below unless OOM
            if not is_oom_error(exc):
                raise
            last = exc
            log.warning(
                "out of memory at rung %d/%d (%s); backing off",
                index, len(rungs), step.reason_he,
            )

    raise EngineError(
        ErrorCode.GPU_OOM,
        f"exhausted the fallback ladder after {len(rungs)} attempts: {last}",
    )
