"""Every error the user can see must be in Hebrew and must say what to do."""

from __future__ import annotations

import re

import pytest

from svc_engine.errors import EngineError, ErrorCode, message_for

HEBREW = re.compile(r"[֐-׿]")


@pytest.mark.parametrize("code", list(ErrorCode), ids=lambda c: c.value)
def test_every_code_has_a_hebrew_message(code: ErrorCode) -> None:
    msg = message_for(code)
    assert HEBREW.search(msg.what), f"{code.value}: 'what' is not Hebrew"
    assert HEBREW.search(msg.action), f"{code.value}: 'action' is not Hebrew"


@pytest.mark.parametrize("code", list(ErrorCode), ids=lambda c: c.value)
def test_messages_are_not_placeholders(code: ErrorCode) -> None:
    msg = message_for(code)
    assert len(msg.what.strip()) >= 8
    assert len(msg.action.strip()) >= 8
    assert msg.render() == f"{msg.what} {msg.action}"


def test_unknown_code_falls_back_instead_of_raising() -> None:
    assert message_for("not-a-code") is message_for(ErrorCode.INTERNAL)  # type: ignore[arg-type]


def test_engine_error_carries_code_and_message() -> None:
    exc = EngineError(ErrorCode.DISK_FULL, "need 4GB")
    assert exc.code is ErrorCode.DISK_FULL
    assert "need 4GB" in str(exc)
    assert HEBREW.search(exc.user_message.render())


def test_no_stack_trace_leaks_into_user_message() -> None:
    exc = EngineError(ErrorCode.INTERNAL, "Traceback (most recent call last): ...")
    assert "Traceback" not in exc.user_message.render()
