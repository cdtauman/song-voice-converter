"""UI engine-client streaming and structured-error coverage."""

from __future__ import annotations

import io
from typing import Any

from svc_app.engine_client import EngineCallError, EngineClient
from svc_engine.rpc import Event, Response, encode


class FakeProcess:
    def __init__(self, output: str) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(output)
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


def test_client_delivers_progress_events_before_result(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    output = encode(
        Event(id="1", event="progress", data={"fraction": 0.5, "message_he": "מעבדים…"})
    ) + encode(Response(id="1", ok=True, result={"output": "cover.wav"}))
    process = FakeProcess(output)
    monkeypatch.setattr("svc_app.engine_client.subprocess.Popen", lambda *_a, **_k: process)
    events: list[tuple[str, dict[str, Any]]] = []

    result = EngineClient().call(
        "covers.run", on_event=lambda name, data: events.append((name, data))
    )

    assert result == {"output": "cover.wav"}
    assert events == [("progress", {"fraction": 0.5, "message_he": "מעבדים…"})]


def test_client_preserves_engine_error_code(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    process = FakeProcess(
        encode(
            Response(
                id="1",
                ok=False,
                error_code="E_NO_VOCALS",
                error_message_he="לא זוהתה שירה. נסה קובץ אחר.",
            )
        )
    )
    monkeypatch.setattr("svc_app.engine_client.subprocess.Popen", lambda *_a, **_k: process)

    try:
        EngineClient().call("covers.preview")
    except EngineCallError as exc:
        assert exc.code == "E_NO_VOCALS"
        assert "נסה" in exc.message_he
    else:
        raise AssertionError("structured engine error was not raised")
