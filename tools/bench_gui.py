"""Phase-8 visual/RTL gate for every simple-mode screen.

Run on Windows (the offscreen Qt plugin has no system fonts):

    python tools/bench_gui.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QImage, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication, QWidget  # noqa: E402

from svc_app.design import Theme, apply_theme  # noqa: E402
from svc_app.main import MainWindow  # noqa: E402

RESULTS = REPO / "benchmark" / "results" / "gui"


class DemoClient:
    def voices(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "demo-voice",
                "display_name": "קול לדוגמה",
                "usable": True,
                "health_note_he": "תקין",
                "has_profile": True,
            },
            {
                "id": "needs-profile",
                "display_name": "קול שדורש תיקון",
                "usable": False,
                "health_note_he": "חסר פרופיל מנעד",
                "has_profile": False,
            },
        ]

    def list_projects(self) -> list[dict[str, Any]]:
        return [
            {
                "project_id": "demo",
                "name": "השיר לדוגמה",
                "updated_at": "19.8.2026 · 14:30",
                "data": {},
            }
        ]

    def settings(self) -> dict[str, Any]:
        return {
            "quality": "balanced",
            "theme": "system",
            "target_lufs": -14.0,
            "keep_cache_gb": 20.0,
            "allow_model_downloads": True,
        }

    def cancel_current(self) -> None:
        return None


def _rtl_tree(widget: QWidget) -> bool:
    candidates = [widget, *widget.findChildren(QWidget)]
    return all(item.layoutDirection() is Qt.LayoutDirection.RightToLeft for item in candidates)


def main() -> int:
    app = QApplication.instance() or QApplication([])
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    apply_theme(app, Theme.LIGHT)
    window = MainWindow(DemoClient())  # type: ignore[arg-type]
    window.show()
    app.processEvents()
    window.wizard.set_voices(DemoClient().voices())
    window.wizard.song = "C:/מוזיקה/שיר לדוגמה.wav"
    window.wizard.voice_id = "demo-voice"
    window.wizard.show_recommendation(
        {
            "output": "C:/preview.wav",
            "recommendation": {"semitones": -12, "playback_semitones": 0},
        }
    )
    window.wizard.show_result(
        {"output": "C:/מוזיקה/שיר לדוגמה-SongVoice.wav", "summary_he": "הקאבר מוכן."}
    )

    RESULTS.mkdir(parents=True, exist_ok=True)
    captures: list[tuple[str, QImage]] = []
    report: list[dict[str, Any]] = []
    for index, name in enumerate(
        ["song", "voice", "quality", "processing", "recommendation", "preview", "result"]
    ):
        window._navigate(0)
        window.wizard._show(index)
        if index == 3:
            window.wizard.update_progress(0.58, "ממירים את הקול…")
        app.processEvents()
        image = window.grab().toImage()
        captures.append((name, image))
        report.append(
            {
                "screen": f"wizard-{name}",
                "rtl": _rtl_tree(window.wizard.pages.widget(index)),
                "width": image.width(),
                "height": image.height(),
            }
        )

    for index, name in [(1, "voices"), (2, "projects"), (3, "settings")]:
        window._navigate(index)
        app.processEvents()
        image = window.grab().toImage()
        captures.append((name, image))
        report.append(
            {
                "screen": f"screen-{name}",
                "rtl": _rtl_tree(window.stack.widget(index)),
                "width": image.width(),
                "height": image.height(),
            }
        )

    thumb_width = 472
    thumb_height = 304
    sheet = QImage(thumb_width * 2, thumb_height * 5, QImage.Format.Format_RGB32)
    sheet.fill(QColor("white"))
    painter = QPainter(sheet)
    for position, (_name, image) in enumerate(captures):
        thumbnail = image.scaled(
            thumb_width,
            thumb_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawImage((position % 2) * thumb_width, (position // 2) * thumb_height, thumbnail)
    painter.end()
    sheet.save(str(RESULTS / "phase8-contact-sheet.png"))

    payload = {
        "phase": 8,
        "screens": report,
        "all_rtl": all(item["rtl"] for item in report),
        "result": "pass" if all(item["rtl"] for item in report) else "fail",
    }
    (RESULTS / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    window.close()
    return 0 if payload["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
