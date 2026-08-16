"""RPC protocol and the ping/doctor round trip."""

from __future__ import annotations

import io
import json

from svc_engine.rpc import Request, Response, Server, decode_request, decode_response, encode
from svc_engine.rpc.server import serve_stdio


def test_request_roundtrip() -> None:
    req = Request(id="7", method="ping", params={"echo": "שלום"})
    back = decode_request(encode(req))
    assert back == req


def test_response_roundtrip_ok() -> None:
    resp = Response(id="7", ok=True, result={"pong": True})
    back = decode_response(encode(resp))
    assert back.ok and back.result == {"pong": True}


def test_response_roundtrip_error() -> None:
    resp = Response(id="7", ok=False, error_code="E_DISK_FULL", error_message_he="אין מקום.")
    back = decode_response(encode(resp))
    assert not back.ok
    assert back.error_code == "E_DISK_FULL"
    assert back.error_message_he == "אין מקום."


def test_encoding_is_one_line_utf8() -> None:
    line = encode(Request(id="1", method="ping", params={"echo": "עברית"}))
    assert line.endswith("\n")
    assert line.count("\n") == 1
    assert "עברית" in line  # ensure_ascii=False keeps Hebrew readable in logs


def test_server_ping() -> None:
    resp = Server().handle(Request(id="1", method="ping", params={"echo": "hi"}))
    assert resp.ok
    assert resp.result["pong"] is True
    assert resp.result["echo"] == "hi"


def test_server_doctor_returns_checks() -> None:
    resp = Server().handle(Request(id="2", method="doctor"))
    assert resp.ok
    assert resp.result["overall"] in {"ok", "warn", "fail"}
    assert len(resp.result["checks"]) >= 8


def test_unknown_method_returns_hebrew_error_not_exception() -> None:
    resp = Server().handle(Request(id="3", method="no_such_method"))
    assert not resp.ok
    assert resp.error_code == "E_INTERNAL"
    assert resp.error_message_he


def test_serve_stdio_handles_a_full_session() -> None:
    stdin = io.StringIO(
        encode(Request(id="1", method="ping", params={"echo": "a"}))
        + "\n"  # blank lines are ignored
        + "{{ not json\n"  # malformed lines are dropped, not fatal
        + encode(Request(id="2", method="ping", params={"echo": "b"}))
    )
    stdout = io.StringIO()
    serve_stdio(stdin, stdout)

    lines = [line for line in stdout.getvalue().splitlines() if line.strip()]
    assert len(lines) == 2
    assert json.loads(lines[0])["result"]["echo"] == "a"
    assert json.loads(lines[1])["result"]["echo"] == "b"
