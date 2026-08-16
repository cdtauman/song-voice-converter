"""UI-side client that drives the engine as a separate process.

Hard rule enforced by tests: nothing under svc_app may import torch or any AI
library. All heavy work happens in the engine process.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
from pathlib import Path
from typing import Any

from svc_engine.rpc.protocol import Request, decode_response, encode

__all__ = ["EngineClient", "EngineUnavailable"]


class EngineUnavailable(RuntimeError):
    """The engine process could not be started or died unexpectedly."""


class EngineClient:
    """Starts `svc serve` and exchanges line-delimited JSON with it.

    Killing the process is the cancellation mechanism -- see docs/architecture.md.
    """

    def __init__(self, python_executable: str | None = None, cwd: Path | None = None) -> None:
        self._exe = python_executable or sys.executable
        self._cwd = cwd
        self._proc: subprocess.Popen[str] | None = None
        self._ids = itertools.count(1)

    # -- lifecycle --------------------------------------------------------- #

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        try:
            self._proc = subprocess.Popen(
                [self._exe, "-m", "svc_engine.cli", "serve"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=self._cwd,
            )
        except OSError as exc:
            raise EngineUnavailable(str(exc)) from exc

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        if proc.poll() is not None:
            return
        try:
            if proc.stdin:
                proc.stdin.close()
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- calls ------------------------------------------------------------- #

    def call(self, method: str, **params: Any) -> Any:
        if not self.is_running:
            self.start()
        proc = self._proc
        if proc is None or proc.stdin is None or proc.stdout is None:
            raise EngineUnavailable("engine process has no pipes")

        req = Request(id=str(next(self._ids)), method=method, params=params)
        try:
            proc.stdin.write(encode(req))
            proc.stdin.flush()
            line = proc.stdout.readline()
        except (OSError, ValueError) as exc:
            raise EngineUnavailable(str(exc)) from exc

        if not line:
            raise EngineUnavailable("engine closed the connection")

        resp = decode_response(line)
        if not resp.ok:
            raise EngineUnavailable(resp.error_message_he or resp.error_code or "unknown error")
        return resp.result

    def ping(self, echo: str = "hello") -> dict[str, Any]:
        return dict(self.call("ping", echo=echo))

    def doctor(self) -> dict[str, Any]:
        return dict(self.call("doctor"))

    # -- context manager --------------------------------------------------- #

    def __enter__(self) -> EngineClient:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
