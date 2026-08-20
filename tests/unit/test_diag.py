"""`svc doctor` must always produce a readable Hebrew report and never raise."""

from __future__ import annotations

import json
import re
from pathlib import Path

from svc_engine.diag import Status, render_json, render_text, run_all_checks
from svc_engine.diag.checks import (
    REQUIRED_FFMPEG_FILTERS,
    check_acceleration_summary,
    check_backend_interfaces,
    check_component_backends,
    check_cuda,
    check_disk,
    check_ffmpeg,
    check_graphics_hardware,
    check_python,
    check_xpu,
    ffmpeg_build_is_gpl,
)
from svc_engine.diag.report import overall_status

HEBREW = re.compile(r"[֐-׿]")


def test_run_all_checks_returns_results(tmp_path: Path) -> None:
    results = run_all_checks(tmp_path)
    assert len(results) >= 10
    keys = {r.key for r in results}
    for expected in ("os", "python", "ram", "disk", "cuda", "xpu", "accel",
                     "components", "ffmpeg", "backends"):
        assert expected in keys


def test_every_check_has_a_hebrew_label(tmp_path: Path) -> None:
    for r in run_all_checks(tmp_path):
        assert HEBREW.search(r.label_he), f"{r.key} label is not Hebrew"
        assert r.message_he.strip(), f"{r.key} has an empty message"


def test_individual_checks_never_raise() -> None:
    for fn in (check_python, check_cuda, check_xpu, check_acceleration_summary,
               check_component_backends, check_ffmpeg, check_backend_interfaces):
        result = fn()
        assert result.status in tuple(Status)


def test_graphics_hardware_check_reports_something() -> None:
    results = check_graphics_hardware()
    assert results
    for r in results:
        assert r.status in tuple(Status)
        assert HEBREW.search(r.label_he)


def test_missing_nvidia_is_information_not_failure() -> None:
    """Intel is a first-class target, so 'no CUDA here' must not read as broken."""
    result = check_cuda()
    assert result.status is not Status.FAIL


def test_missing_intel_is_information_not_failure() -> None:
    result = check_xpu()
    assert result.status is not Status.FAIL


def test_info_status_does_not_poison_the_overall_verdict() -> None:
    from svc_engine.diag.checks import CheckResult

    results = [
        CheckResult("a", "תווית", Status.OK, "הודעה"),
        CheckResult("b", "תווית", Status.INFO, "הודעה"),
    ]
    assert overall_status(results) is Status.OK


def test_disk_check_handles_missing_directory(tmp_path: Path) -> None:
    """A path that does not exist yet must resolve upward, not explode."""
    result = check_disk(tmp_path / "a" / "b" / "c")
    assert result.status in tuple(Status)


def test_backend_interfaces_check_passes() -> None:
    assert check_backend_interfaces().status is Status.OK


def test_required_ffmpeg_filters_cover_the_mixing_stage() -> None:
    assert {"loudnorm", "alimiter"} <= set(REQUIRED_FFMPEG_FILTERS)


def test_ffmpeg_gpl_build_flag_is_not_silently_missed() -> None:
    assert ffmpeg_build_is_gpl("configuration: --enable-gpl --enable-libx264")
    assert not ffmpeg_build_is_gpl("configuration: --disable-gpl --enable-shared")


def test_windows_11_is_not_reported_as_windows_10() -> None:
    """platform.release() lies on Windows 11; the report must not repeat the lie."""
    import platform
    import sys

    if sys.platform != "win32":
        return
    from svc_engine.diag.checks import _windows_name

    name = _windows_name()
    try:
        build = int(platform.version().split(".")[-1])
    except (ValueError, IndexError):
        return
    if build >= 22000:
        assert "Windows 11" in name, name


def test_render_text_is_readable(tmp_path: Path) -> None:
    text = render_text(run_all_checks(tmp_path))
    assert "SongVoice" in text
    assert HEBREW.search(text)


def test_render_json_is_valid(tmp_path: Path) -> None:
    payload = json.loads(render_json(run_all_checks(tmp_path)))
    assert payload["overall"] in {"ok", "warn", "fail"}
    assert isinstance(payload["checks"], list)
    assert all("status" in c for c in payload["checks"])


def test_overall_status_precedence() -> None:
    from svc_engine.diag.checks import CheckResult

    def r(s: Status) -> CheckResult:
        return CheckResult("k", "תווית", s, "הודעה")

    assert overall_status([r(Status.OK), r(Status.OK)]) is Status.OK
    assert overall_status([r(Status.OK), r(Status.WARN)]) is Status.WARN
    assert overall_status([r(Status.WARN), r(Status.FAIL)]) is Status.FAIL
