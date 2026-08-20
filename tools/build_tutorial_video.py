"""Build the short, silent Hebrew quick-start video from the real Qt UI."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

if os.name == "nt":
    # The offscreen Qt plugin does not load Windows system fonts and renders
    # Hebrew as tofu. The native renderer is required for a readable artifact.
    os.environ["QT_QPA_PLATFORM"] = "windows"
else:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtGui import QColor, QFont, QPainter  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from svc_app.design import Theme, apply_theme  # noqa: E402
from svc_app.main import MainWindow  # noqa: E402


class DemoClient:
    def voices(self) -> list[dict[str, Any]]:
        return [
            {
                "id": "demo",
                "display_name": "הקול שלי",
                "usable": True,
                "health_note_he": "תקין ומוכן לקאבר",
                "has_profile": True,
            }
        ]

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

    def cancel_current(self) -> None:
        return None


def _caption(image, text: str):  # type: ignore[no-untyped-def]
    painter = QPainter(image)
    painter.fillRect(0, image.height() - 72, image.width(), 72, QColor(20, 24, 40, 225))
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Segoe UI", 20, QFont.Weight.Bold))
    painter.drawText(
        24,
        image.height() - 72,
        image.width() - 48,
        72,
        Qt.AlignmentFlag.AlignCenter,
        text,
    )
    painter.end()
    return image


def main() -> int:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SystemExit("ffmpeg is required to build the tutorial video")
    output = REPO / "docs" / "media" / "songvoice-quickstart-he.mp4"
    output.parent.mkdir(parents=True, exist_ok=True)
    build_root = REPO / "build"
    build_root.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication([])
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    apply_theme(app, Theme.LIGHT)
    window = MainWindow(DemoClient())  # type: ignore[arg-type]
    window.show()
    app.processEvents()
    window.wizard.set_voices(DemoClient().voices())

    with tempfile.TemporaryDirectory(prefix="songvoice-tutorial-", dir=build_root) as raw_temp:
        temp = Path(raw_temp)
        slides: list[tuple[str, int]] = []

        def capture(name: str, caption: str, seconds: int = 8) -> None:
            app.processEvents()
            image = _caption(window.grab().toImage(), caption)
            path = temp / f"{len(slides):02d}-{name}.png"
            if not image.save(str(path)):
                raise RuntimeError(f"could not save {path}")
            slides.append((path.as_posix(), seconds))

        window._navigate(0)
        window.wizard._show(0)
        capture("song", "1 · בוחרים שיר")
        window._navigate(1)
        capture("voice", "2 · מוסיפים קול מורשה")
        window._navigate(0)
        window.wizard.song = "C:/מוזיקה/שיר לדוגמה.wav"
        window.wizard.voice_id = "demo"
        window.wizard._show(2)
        capture("quality", "3 · בוחרים איכות ויוצרים Preview")
        window.wizard.show_recommendation(
            {
                "output": "C:/מוזיקה/preview.wav",
                "recommendation": {"semitones": -12, "playback_semitones": 0},
            }
        )
        capture("preview", "4 · מאזינים להשוואה ויוצרים קאבר מלא")
        window._navigate(5)
        capture("help", "עזרה ופתרון תקלות זמינים גם ללא אינטרנט")

        concat = temp / "slides.txt"
        lines: list[str] = []
        for path, seconds in slides:
            lines.extend((f"file '{path}'", f"duration {seconds}"))
        lines.append(f"file '{slides[-1][0]}'")
        concat.write_text("\n".join(lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(concat),
                "-vf",
                "fps=24,format=yuv420p",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "24",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    window.close()
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
