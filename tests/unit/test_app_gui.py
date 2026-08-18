"""Phase-8 GUI contract tests run with Qt's offscreen platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from svc_app.i18n import error_text  # noqa: E402
from svc_app.main import MainWindow  # noqa: E402
from svc_app.screens import CoverWizard  # noqa: E402
from svc_engine.errors import ErrorCode  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.cancelled = False
        self.saved: list[tuple[str, str, dict[str, Any]]] = []

    def voices(self) -> list[dict[str, Any]]:
        return []

    def list_projects(self) -> list[dict[str, Any]]:
        return []

    def settings(self) -> dict[str, Any]:
        return {
            "quality": "balanced",
            "target_lufs": -14.0,
            "keep_cache_gb": 20.0,
            "allow_model_downloads": True,
            "theme": "system",
        }

    def save_project(self, project_id: str, name: str, data: dict[str, Any]) -> dict[str, Any]:
        self.saved.append((project_id, name, data))
        return {"project_id": project_id, "name": name, "data": data}

    def cancel_current(self) -> None:
        self.cancelled = True


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_all_main_screens_inherit_rtl() -> None:
    _app()
    window = MainWindow(FakeClient())  # type: ignore[arg-type]
    try:
        assert window.layoutDirection() is Qt.LayoutDirection.RightToLeft
        assert window.stack.count() == 4
        assert window.wizard.pages.count() == 7
        for index in range(window.stack.count()):
            assert window.stack.widget(index).layoutDirection() is Qt.LayoutDirection.RightToLeft
        for index in range(window.wizard.pages.count()):
            assert (
                window.wizard.pages.widget(index).layoutDirection()
                is Qt.LayoutDirection.RightToLeft
            )
    finally:
        window.close()


def test_wizard_covers_song_voice_quality_preview_and_result(tmp_path: Path) -> None:
    _app()
    wizard = CoverWizard()
    song = tmp_path / "שיר בדיקה.wav"
    song.write_bytes(b"RIFF")
    wizard.set_voices(
        [
            {
                "id": "demo-voice",
                "display_name": "קול לדוגמה",
                "usable": True,
                "health_note_he": "תקין",
            }
        ]
    )
    wizard.drop_zone.set_file(str(song))
    wizard._voice_selected("demo-voice")

    assert wizard.request_data() == {
        "song": str(song),
        "voice_id": "demo-voice",
        "quality": "balanced",
    }
    wizard.show_processing("preview")
    wizard.update_progress(0.42, "ממירים את הקול…")
    assert wizard.progress.value() == 420
    wizard.show_recommendation(
        {
            "output": str(tmp_path / "preview.wav"),
            "recommendation": {"semitones": -12, "playback_semitones": 0},
        }
    )
    assert "-12" in wizard.recommend_title.text()
    wizard.show_result({"output": str(tmp_path / "cover.wav"), "summary_he": "הקאבר נשמר בהצלחה."})
    assert wizard.pages.currentIndex() == 6
    assert wizard.result_name.text() == "cover.wav"


def test_every_engine_error_has_hebrew_title_and_suggested_action() -> None:
    for code in ErrorCode:
        title, action = error_text(code.value)
        assert title.strip()
        assert action.strip()
        assert any("\u0590" <= character <= "\u05ff" for character in title + action)


def test_opening_gui_offers_to_resume_recoverable_cover(monkeypatch, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    class RecoverableClient(FakeClient):
        def recoverable_jobs(self) -> list[dict[str, Any]]:
            return [
                {
                    "job_id": "cover-recovery",
                    "kind": "cover",
                    "source": str(tmp_path / "שיר.wav"),
                    "voice_id": "demo-voice",
                    "preview": True,
                }
            ]

    _app()
    client = RecoverableClient()
    window = MainWindow(client)  # type: ignore[arg-type]
    started: list[Any] = []
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(
        window,
        "_start_worker",
        lambda operation, on_success: started.append((operation, on_success)),
    )
    try:
        window._offer_recovery()
        assert window.wizard.pages.currentIndex() == 3
        assert window.wizard.progress_message.text() == "מכינים תצוגה מקדימה…"
        assert len(started) == 1
    finally:
        window.close()
