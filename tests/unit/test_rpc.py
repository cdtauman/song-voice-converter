"""RPC protocol and the ping/doctor round trip."""

from __future__ import annotations

import io
import json
from pathlib import Path

from svc_engine.config import paths
from svc_engine.rpc import (
    Event,
    Request,
    Response,
    Server,
    decode_event,
    decode_request,
    decode_response,
    encode,
)
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


def test_progress_event_roundtrip() -> None:
    event = Event(id="8", event="progress", data={"fraction": 0.25, "message_he": "מפרידים…"})
    assert decode_event(encode(event)) == event


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


def test_phase7_project_cache_and_history_methods(tmp_path: Path) -> None:
    server = Server(paths(tmp_path))
    saved = server.handle(
        Request(
            id="save",
            method="projects.save",
            params={"project_id": "demo", "name": "דמו", "data": {"quality": "max"}},
        )
    )
    listed = server.handle(Request(id="list", method="projects.list"))
    loaded = server.handle(
        Request(id="load", method="projects.load", params={"project_id": "demo"})
    )
    cache = server.handle(Request(id="cache", method="cache.stats"))
    history = server.handle(Request(id="history", method="jobs.history"))
    recoverable = server.handle(Request(id="recovery", method="jobs.recoverable"))

    assert saved.ok and saved.result["name"] == "דמו"
    assert listed.ok and len(listed.result) == 1
    assert loaded.ok and loaded.result["data"]["quality"] == "max"
    assert cache.ok and cache.result == {"entries": 0, "size_bytes": 0}
    assert history.ok and history.result == []
    assert recoverable.ok and recoverable.result == []


def test_phase8_cover_rpc_streams_progress(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    def fake_run_cover(_paths, **kwargs):  # type: ignore[no-untyped-def]
        kwargs["on_progress"](0.5, "ממירים את הקול…")
        return {"output": str(tmp_path / "preview.wav"), "preview": True}

    monkeypatch.setattr("svc_engine.workflows.run_cover", fake_run_cover)
    events: list[Event] = []
    response = Server(paths(tmp_path)).handle(
        Request(
            id="preview",
            method="covers.preview",
            params={"song": "song.wav", "voice_id": "voice", "quality": "balanced"},
        ),
        on_event=events.append,
    )

    assert response.ok
    assert response.result["preview"] is True
    assert events == [
        Event(
            id="preview",
            event="progress",
            data={"fraction": 0.5, "message_he": "ממירים את הקול…"},
        )
    ]


def test_phase8_settings_rpc_roundtrip(tmp_path: Path) -> None:
    server = Server(paths(tmp_path))
    saved = server.handle(
        Request(
            id="save-settings",
            method="settings.save",
            params={"quality": "max", "theme": "dark", "target_lufs": -15.0},
        )
    )
    loaded = server.handle(Request(id="get-settings", method="settings.get"))

    assert saved.ok and saved.result["quality"] == "max"
    assert loaded.ok and loaded.result["theme"] == "dark"
    assert loaded.result["target_lufs"] == -15.0
