"""Engine-side RPC loop.

The transport stays free of AI imports. Phase 7 adds durable project/job
inspection methods; concrete processing jobs remain engine-owned Python graphs,
never executable callables supplied over JSON.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

from svc_engine.config import Paths, load_settings, paths
from svc_engine.diag import run_all_checks
from svc_engine.diag.report import overall_status
from svc_engine.errors import EngineError, ErrorCode, message_for
from svc_engine.logging_setup import get_logger
from svc_engine.rpc.protocol import PROTOCOL_VERSION, Request, Response, encode

log = get_logger(__name__)

Handler = Callable[[dict[str, Any]], Any]

__all__ = ["Server", "serve_stdio"]


class Server:
    def __init__(self, app_paths: Paths | None = None) -> None:
        self._paths = app_paths or paths()
        self._paths.ensure()
        self._handlers: dict[str, Handler] = {
            "ping": self._ping,
            "doctor": self._doctor,
            "jobs.recoverable": self._jobs_recoverable,
            "jobs.history": self._jobs_history,
            "jobs.cleanup": self._jobs_cleanup,
            "cache.stats": self._cache_stats,
            "projects.list": self._projects_list,
            "projects.load": self._projects_load,
            "projects.save": self._projects_save,
        }

    # -- methods ----------------------------------------------------------- #

    def _ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "protocol": PROTOCOL_VERSION, "echo": params.get("echo")}

    def _doctor(self, params: dict[str, Any]) -> dict[str, Any]:
        results = run_all_checks(self._paths.work)
        return {
            "overall": overall_status(results).value,
            "checks": [
                {
                    "key": r.key,
                    "label_he": r.label_he,
                    "status": r.status.value,
                    "message_he": r.message_he,
                    "detail": r.detail,
                }
                for r in results
            ],
        }

    def _jobs_recoverable(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.jobs import RecoveryStore

        snapshots = RecoveryStore(self._paths.root / "jobs").discover()
        return [
            {
                "job_id": item.job_id,
                "name": item.name,
                "status": item.status.value,
                "updated_at": item.updated_at,
                "completed_steps": sum(
                    step.status.value in {"completed", "cached"} for step in item.steps.values()
                ),
                "total_steps": len(item.steps),
            }
            for item in snapshots
        ]

    def _jobs_history(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.history import HistoryStore

        limit = int(params.get("limit", 100))
        return [item.to_dict() for item in HistoryStore(self._paths.db).list(limit=limit)]

    def _jobs_cleanup(self, params: dict[str, Any]) -> dict[str, int]:
        from svc_engine.jobs import JobRunner

        return JobRunner(self._paths, settings=load_settings(self._paths)).cleanup()

    def _cache_stats(self, params: dict[str, Any]) -> dict[str, int]:
        from svc_engine.jobs import StepCache

        stats = StepCache(self._paths.cache / "steps").stats()
        return {"entries": stats.entries, "size_bytes": stats.size_bytes}

    def _projects_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.projects import ProjectStore

        return [item.to_dict() for item in ProjectStore(self._paths.projects).list()]

    def _projects_load(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.projects import ProjectStore

        return ProjectStore(self._paths.projects).load(str(params["project_id"])).to_dict()

    def _projects_save(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.projects import ProjectStore

        data = params.get("data")
        if not isinstance(data, dict):
            raise ValueError("project data must be an object")
        return ProjectStore(self._paths.projects).save(
            str(params["project_id"]), name=str(params["name"]), data=data
        ).to_dict()

    # -- dispatch ---------------------------------------------------------- #

    def handle(self, req: Request) -> Response:
        handler = self._handlers.get(req.method)
        if handler is None:
            msg = message_for(ErrorCode.INTERNAL)
            return Response(
                id=req.id, ok=False,
                error_code=ErrorCode.INTERNAL.value, error_message_he=msg.render(),
            )
        try:
            return Response(id=req.id, ok=True, result=handler(req.params))
        except EngineError as exc:
            log.warning("method %s failed: %s", req.method, exc)
            return Response(
                id=req.id, ok=False,
                error_code=exc.code.value, error_message_he=exc.user_message.render(),
            )
        except Exception:  # noqa: BLE001  the engine must never die on one bad call
            log.exception("unhandled error in method %s", req.method)
            msg = message_for(ErrorCode.INTERNAL)
            return Response(
                id=req.id, ok=False,
                error_code=ErrorCode.INTERNAL.value, error_message_he=msg.render(),
            )


def serve_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Read requests from stdin, write responses to stdout, until EOF."""
    from svc_engine.rpc.protocol import decode_request

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = Server()
    log.info("engine rpc ready (protocol v%d)", PROTOCOL_VERSION)

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = decode_request(line)
        except Exception:  # noqa: BLE001
            log.warning("dropped malformed request line")
            continue
        stdout.write(encode(server.handle(req)))
        stdout.flush()
