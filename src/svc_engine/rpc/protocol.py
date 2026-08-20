"""Line-delimited JSON-RPC between the UI process and the engine process.

One JSON object per line, UTF-8. Deliberately boring: the UI must never need to
import anything from the engine's dependency stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Event",
    "Request",
    "Response",
    "decode_event",
    "decode_request",
    "decode_response",
    "encode",
]

PROTOCOL_VERSION = 1


@dataclass(frozen=True)
class Request:
    id: str
    method: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"v": PROTOCOL_VERSION, "id": self.id, "method": self.method, "params": self.params}


@dataclass(frozen=True)
class Response:
    id: str
    ok: bool
    result: Any = None
    error_code: str | None = None
    error_message_he: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"v": PROTOCOL_VERSION, "id": self.id, "ok": self.ok}
        if self.ok:
            d["result"] = self.result
        else:
            d["error"] = {"code": self.error_code, "message_he": self.error_message_he}
        return d


@dataclass(frozen=True)
class Event:
    """An in-flight notification emitted before a request's final response."""

    id: str
    event: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "v": PROTOCOL_VERSION,
            "id": self.id,
            "event": self.event,
            "data": self.data,
        }


def encode(obj: Request | Response | Event) -> str:
    return json.dumps(obj.to_dict(), ensure_ascii=False) + "\n"


def decode_request(line: str) -> Request:
    d = json.loads(line)
    return Request(id=str(d["id"]), method=str(d["method"]), params=dict(d.get("params") or {}))


def decode_response(line: str) -> Response:
    d = json.loads(line)
    if d.get("ok"):
        return Response(id=str(d["id"]), ok=True, result=d.get("result"))
    err = d.get("error") or {}
    return Response(
        id=str(d["id"]),
        ok=False,
        error_code=err.get("code"),
        error_message_he=err.get("message_he"),
    )


def decode_event(line: str) -> Event:
    d = json.loads(line)
    if "event" not in d:
        raise ValueError("message is not an event")
    return Event(
        id=str(d["id"]),
        event=str(d["event"]),
        data=dict(d.get("data") or {}),
    )
