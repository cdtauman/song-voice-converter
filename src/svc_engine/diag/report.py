"""Renders check results for humans (console) and for machines (JSON)."""

from __future__ import annotations

import json
from dataclasses import asdict

from svc_engine.diag.checks import CheckResult, Status

__all__ = ["render_text", "render_json", "overall_status"]

_ICON = {Status.OK: "✅", Status.INFO: "ℹ️ ", Status.WARN: "⚠️ ", Status.FAIL: "❌"}


def overall_status(results: list[CheckResult]) -> Status:
    """INFO is neutral by design -- "no NVIDIA here" must not read as a problem."""
    if any(r.status is Status.FAIL for r in results):
        return Status.FAIL
    if any(r.status is Status.WARN for r in results):
        return Status.WARN
    return Status.OK


def render_text(results: list[CheckResult], verbose: bool = False) -> str:
    width = max((len(r.label_he) for r in results), default=0)
    lines: list[str] = ["בדיקת מערכת — SongVoice", ""]
    for r in results:
        lines.append(f"{_ICON[r.status]} {r.label_he.ljust(width)}  {r.message_he}")
        if verbose and r.detail:
            lines.append(f"      {r.detail}")

    lines.append("")
    status = overall_status(results)
    if status is Status.OK:
        lines.append("המערכת מוכנה.")
    elif status is Status.WARN:
        lines.append("המערכת עובדת, אבל יש דברים ששווה לשפר (מסומנים ב-⚠️).")
    else:
        failed = [r.label_he for r in results if r.status is Status.FAIL]
        lines.append("חסרים דברים כדי שהמערכת תעבוד: " + ", ".join(failed))
    return "\n".join(lines)


def render_json(results: list[CheckResult]) -> str:
    payload = {
        "overall": overall_status(results).value,
        "checks": [asdict(r) | {"status": r.status.value} for r in results],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
