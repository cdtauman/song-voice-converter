"""Cooperative job cancellation and the process-level three-second fallback."""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable
from typing import Protocol

from svc_engine.errors import EngineError, ErrorCode

__all__ = ["CancellationToken", "cancel_process"]


class CancellationToken:
    """Thread-safe flag checked between steps and inside long-running loops."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise EngineError(ErrorCode.CANCELLED, "job cancellation requested")


class _Process(Protocol):
    def poll(self) -> int | None: ...
    def wait(self, timeout: float | None = None) -> int: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...


def cancel_process(
    process: _Process,
    *,
    request_cancel: Callable[[], None] | None = None,
    timeout: float = 3.0,
    poll_interval: float = 0.02,
) -> bool:
    """Request a clean stop, then terminate/kill an unresponsive worker.

    Returns ``True`` when the cooperative request was enough and ``False`` when
    the process boundary had to enforce cancellation. The total cooperative
    wait is capped at the architecture's three-second deadline.
    """
    if timeout < 0:
        raise ValueError("timeout must be non-negative")
    if process.poll() is not None:
        return True
    if request_cancel is not None:
        request_cancel()
    started = time.monotonic()
    deadline = started + timeout
    # Reserve time inside the same deadline for terminate and kill. This keeps
    # the *total* cancellation latency below three seconds, not just the clean
    # request window.
    cooperative_deadline = started + max(0.0, timeout - min(0.5, timeout / 2.0))
    while process.poll() is None and time.monotonic() < cooperative_deadline:
        time.sleep(
            min(poll_interval, max(0.0, cooperative_deadline - time.monotonic()))
        )
    if process.poll() is not None:
        return True

    process.terminate()
    try:
        remaining = max(0.0, deadline - time.monotonic())
        process.wait(timeout=min(0.25, remaining))
    except (subprocess.TimeoutExpired, TimeoutError):
        process.kill()
        process.wait(timeout=max(0.0, deadline - time.monotonic()))
    return False
