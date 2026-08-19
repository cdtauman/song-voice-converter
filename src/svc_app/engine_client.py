"""UI-side client that drives the engine as a separate process.

Hard rule enforced by tests: nothing under svc_app may import torch or any AI
library. All heavy work happens in the engine process.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from svc_engine.rpc.protocol import Request, decode_event, decode_response, encode

__all__ = ["EngineCallError", "EngineClient", "EngineUnavailable"]

EventCallback = Callable[[str, dict[str, Any]], None]


class EngineUnavailable(RuntimeError):
    """The engine process could not be started or died unexpectedly."""


class EngineCallError(RuntimeError):
    """A structured, user-safe failure returned by the engine."""

    def __init__(self, code: str, message_he: str) -> None:
        self.code = code
        self.message_he = message_he
        super().__init__(message_he)


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
            # Reserve the final half-second for forceful termination so the
            # complete cancellation path, not merely the polite wait, fits the
            # Phase-7 three-second deadline.
            proc.wait(timeout=2.5)
        except (subprocess.TimeoutExpired, OSError):
            with contextlib.suppress(OSError):
                proc.terminate()
            try:
                proc.wait(timeout=0.25)
            except (subprocess.TimeoutExpired, OSError):
                with contextlib.suppress(OSError):
                    proc.kill()
                with contextlib.suppress(subprocess.TimeoutExpired, OSError):
                    proc.wait(timeout=0.25)

    def cancel_current(self) -> None:
        """Cancel current work at the process boundary within three seconds.

        Closing stdin is the cooperative request for an idle server. A busy
        engine cannot read another RPC message, so ``stop`` enforces the
        architecture's kill fallback. The next call starts a fresh engine;
        Phase-7 recovery resumes from the last atomically completed step.
        """
        self.stop()

    @property
    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # -- calls ------------------------------------------------------------- #

    def call(
        self,
        method: str,
        *,
        on_event: EventCallback | None = None,
        **params: Any,
    ) -> Any:
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

        while line:
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EngineUnavailable("engine returned an invalid message") from exc
            if "event" not in raw:
                break
            event = decode_event(line)
            if event.id == req.id and on_event is not None:
                on_event(event.event, event.data)
            try:
                line = proc.stdout.readline()
            except (OSError, ValueError) as exc:
                raise EngineUnavailable(str(exc)) from exc

        if not line:
            raise EngineUnavailable("engine closed the connection")
        resp = decode_response(line)
        if resp.id != req.id:
            raise EngineUnavailable("engine response id does not match the request")
        if not resp.ok:
            raise EngineCallError(
                resp.error_code or "E_INTERNAL",
                resp.error_message_he or "משהו השתבש. נסה שוב.",
            )
        return resp.result

    def ping(self, echo: str = "hello") -> dict[str, Any]:
        return dict(self.call("ping", echo=echo))

    def doctor(self) -> dict[str, Any]:
        return dict(self.call("doctor"))

    def recoverable_jobs(self) -> list[dict[str, Any]]:
        return list(self.call("jobs.recoverable"))

    def job_history(self, limit: int = 100) -> list[dict[str, Any]]:
        return list(self.call("jobs.history", limit=limit))

    def cleanup_jobs(self) -> dict[str, int]:
        return dict(self.call("jobs.cleanup"))

    def cache_stats(self) -> dict[str, int]:
        return dict(self.call("cache.stats"))

    def list_projects(self) -> list[dict[str, Any]]:
        return list(self.call("projects.list"))

    def load_project(self, project_id: str) -> dict[str, Any]:
        return dict(self.call("projects.load", project_id=project_id))

    def save_project(self, project_id: str, name: str, data: dict[str, Any]) -> dict[str, Any]:
        return dict(self.call("projects.save", project_id=project_id, name=name, data=data))

    def settings(self) -> dict[str, Any]:
        return dict(self.call("settings.get"))

    def save_settings(self, **values: Any) -> dict[str, Any]:
        return dict(self.call("settings.save", **values))

    def voices(self) -> list[dict[str, Any]]:
        return list(self.call("voices.list"))

    def import_voice(
        self,
        archive: str,
        display_name: str,
        *,
        consent_confirmed: bool,
        consent_note: str = "",
    ) -> dict[str, Any]:
        return dict(
            self.call(
                "voices.import",
                archive=archive,
                display_name=display_name,
                consent_confirmed=consent_confirmed,
                consent_note=consent_note,
            )
        )

    def remove_voice(self, voice_id: str) -> None:
        self.call("voices.remove", voice_id=voice_id)

    def update_voice(self, voice_id: str, **values: Any) -> dict[str, Any]:
        return dict(self.call("voices.update", voice_id=voice_id, **values))

    def check_voice(self, voice_id: str) -> dict[str, Any]:
        return dict(self.call("voices.health", voice_id=voice_id))

    def create_training(
        self,
        display_name: str,
        recordings: list[str],
        *,
        consent_confirmed: bool,
        consent_note: str,
    ) -> dict[str, Any]:
        return dict(
            self.call(
                "training.create",
                display_name=display_name,
                recordings=recordings,
                consent_confirmed=consent_confirmed,
                consent_note=consent_note,
            )
        )

    def training_sessions(self) -> list[dict[str, Any]]:
        return list(self.call("training.list"))

    def inspect_training(self, session_id: str) -> dict[str, Any]:
        return dict(self.call("training.inspect", session_id=session_id))

    def prepare_training(self, session_id: str, separate_mix: bool = True) -> dict[str, Any]:
        return dict(self.call("training.prepare", session_id=session_id, separate_mix=separate_mix))

    def start_training(self, session_id: str) -> dict[str, Any]:
        return dict(self.call("training.start", session_id=session_id))

    def training_status(self, session_id: str) -> dict[str, Any]:
        return dict(self.call("training.status", session_id=session_id))

    def pause_training(self, session_id: str) -> dict[str, Any]:
        return dict(self.call("training.pause", session_id=session_id))

    def preview_cover(
        self,
        *,
        song: str,
        voice_id: str,
        quality: str,
        preview_seconds: float = 30.0,
        on_event: EventCallback | None = None,
        advanced: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return dict(
            self.call(
                "covers.preview",
                song=song,
                voice_id=voice_id,
                quality=quality,
                preview_seconds=preview_seconds,
                advanced=advanced or {},
                on_event=on_event,
            )
        )

    def render_cover(
        self,
        *,
        song: str,
        voice_id: str,
        quality: str,
        output: str,
        on_event: EventCallback | None = None,
        advanced: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return dict(
            self.call(
                "covers.run",
                song=song,
                voice_id=voice_id,
                quality=quality,
                output=output,
                advanced=advanced or {},
                on_event=on_event,
            )
        )

    def resume_cover(
        self,
        job_id: str,
        *,
        on_event: EventCallback | None = None,
    ) -> dict[str, Any]:
        return dict(self.call("covers.resume", job_id=job_id, on_event=on_event))

    # -- context manager --------------------------------------------------- #

    def __enter__(self) -> EngineClient:
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()
