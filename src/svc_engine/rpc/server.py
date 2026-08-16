"""Engine-side RPC loop.

Phase 1 exposes two methods: `ping` and `doctor`. Job methods arrive in Phase 7.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from typing import Any, TextIO

from svc_engine.config import paths
from svc_engine.diag import run_all_checks
from svc_engine.diag.report import overall_status
from svc_engine.errors import EngineError, ErrorCode, message_for
from svc_engine.logging_setup import get_logger
from svc_engine.rpc.protocol import PROTOCOL_VERSION, Request, Response, encode

log = get_logger(__name__)

Handler = Callable[[dict[str, Any]], Any]

__all__ = ["Server", "serve_stdio"]


class Server:
    def __init__(self) -> None:
        self._handlers: dict[str, Handler] = {
            "ping": self._ping,
            "doctor": self._doctor,
        }

    # -- methods ----------------------------------------------------------- #

    def _ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "protocol": PROTOCOL_VERSION, "echo": params.get("echo")}

    def _doctor(self, params: dict[str, Any]) -> dict[str, Any]:
        p = paths()
        results = run_all_checks(p.work)
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
