"""SongVoice Hebrew RTL desktop application."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from svc_app.design import Theme, apply_theme
from svc_app.engine_client import EngineCallError, EngineClient
from svc_app.i18n import error_text
from svc_app.runtime import configure_bundled_runtime
from svc_app.screens import (
    BenchmarkScreen,
    CoverWizard,
    HelpScreen,
    ProjectsScreen,
    SettingsScreen,
    VoiceLibraryScreen,
)
from svc_app.screens.first_run import FirstRunDialog


class EngineWorker(QObject):
    progress = Signal(float, str)
    succeeded = Signal(dict)
    failed = Signal(str, str)
    finished = Signal()

    def __init__(self, operation: Callable[[Callable[[str, dict[str, Any]], None]], dict]) -> None:
        super().__init__()
        self.operation = operation

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation(self._event)
        except EngineCallError as exc:
            self.failed.emit(exc.code, exc.message_he)
        except Exception:
            self.failed.emit("E_INTERNAL", "")
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()

    def _event(self, event: str, data: dict[str, Any]) -> None:
        if event == "progress":
            self.progress.emit(
                float(data.get("fraction") or 0.0), str(data.get("message_he") or "")
            )


class MainWindow(QMainWindow):
    def __init__(self, client: EngineClient | None = None) -> None:
        super().__init__()
        self.client = client or EngineClient()
        self._thread: QThread | None = None
        self._worker: EngineWorker | None = None
        self._cancelled = False
        self.setWindowTitle("SongVoice · קאבר בקול שלך")
        self.resize(1180, 760)
        self.setMinimumSize(900, 620)
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)

        root = QWidget()
        root.setObjectName("AppRoot")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        self.setCentralWidget(root)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(218)
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(16, 22, 16, 18)
        brand = QLabel("SongVoice")
        brand.setStyleSheet("font-size: 23px; font-weight: 800;")
        tagline = QLabel("קאבר בקול שלך")
        tagline.setProperty("muted", True)
        side.addWidget(brand)
        side.addWidget(tagline)
        side.addSpacing(28)

        self.stack = QStackedWidget()
        self.wizard = CoverWizard()
        self.library = VoiceLibraryScreen(self.client)
        self.projects = ProjectsScreen(self.client)
        self.benchmark = BenchmarkScreen()
        self.settings = SettingsScreen(self.client)
        self.help = HelpScreen()
        for page in (
            self.wizard,
            self.library,
            self.projects,
            self.benchmark,
            self.settings,
            self.help,
        ):
            self.stack.addWidget(page)

        self.nav_buttons: list[QPushButton] = []
        for label, index in [
            ("♫  קאבר חדש", 0),
            ("●  ספריית קולות", 1),
            ("▣  פרויקטים", 2),
            ("◫  מעבדת השוואה", 3),
            ("⚙  הגדרות", 4),
            ("?  עזרה", 5),
        ]:
            button = QPushButton(label)
            button.setCheckable(True)
            button.setProperty("nav", True)
            button.clicked.connect(lambda _checked=False, i=index: self._navigate(i))
            side.addWidget(button)
            self.nav_buttons.append(button)
        side.addStretch()
        privacy = QLabel("🔒 העיבוד מקומי\nהשמע לא עוזב את המחשב")
        privacy.setProperty("muted", True)
        privacy.setWordWrap(True)
        side.addWidget(privacy)
        shell.addWidget(sidebar)
        shell.addWidget(self.stack, 1)

        self.wizard.preview_requested.connect(self._start_preview)
        self.wizard.full_requested.connect(self._start_full)
        self.wizard.cancel_requested.connect(self._cancel)
        self.library.changed.connect(self.wizard.set_voices)
        self.projects.open_requested.connect(self._open_project)
        self.settings.theme_changed.connect(self._theme_changed)
        self.settings.advanced_changed.connect(self.wizard.advanced_toggle.setChecked)
        self.settings.update_requested.connect(self._check_for_update)

        self._navigate(0)
        self.library.refresh()
        QTimer.singleShot(0, self._offer_recovery)

    def offer_first_run(self) -> None:
        try:
            status = self.client.provisioning_status()
        except Exception:
            return
        if not status.get("complete"):
            FirstRunDialog(self.client, status, self).exec()

    def _check_for_update(self) -> None:
        try:
            result = self.client.check_for_update()
        except EngineCallError as exc:
            QMessageBox.warning(self, "בדיקת העדכון נכשלה", exc.message_he)
            return
        except Exception:
            QMessageBox.warning(self, "בדיקת העדכון נכשלה", "בדוק את החיבור ונסה שוב.")
            return
        release = result.get("release")
        if not result.get("available") or not isinstance(release, dict):
            QMessageBox.information(self, "עדכונים", "הגרסה המותקנת היא העדכנית ביותר.")
            return
        answer = QMessageBox.question(
            self,
            "עדכון זמין",
            f"גרסה {release.get('version')} זמינה. להוריד כעת?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        progress = QProgressDialog("מורידים עדכון מאומת…", "", 0, 1000, self)
        progress.setCancelButton(None)
        progress.setWindowTitle("עדכון SongVoice")
        progress.setAutoClose(False)
        progress.show()

        def operation(event):  # type: ignore[no-untyped-def]
            return self.client.stage_update(release, on_event=event)

        def on_success(_result: dict[str, Any]) -> None:
            progress.close()
            QMessageBox.information(
                self, "העדכון מוכן", "העדכון המאומת יותקן אוטומטית בהפעלה הבאה."
            )

        def on_progress(fraction: float, message: str) -> None:
            progress.setValue(round(fraction * 1000))
            progress.setLabelText(message or "מורידים עדכון מאומת…")

        def on_failure(code: str, message_he: str) -> None:
            progress.close()
            self._show_error(code, message_he)

        self._start_worker(
            operation,
            on_success,
            on_progress=on_progress,
            on_failure=on_failure,
        )

    def _navigate(self, index: int) -> None:
        if self._thread is not None and self._thread.isRunning() and index != 0:
            return
        self.stack.setCurrentIndex(index)
        for position, button in enumerate(self.nav_buttons):
            button.setChecked(position == index)
        if index == 1:
            self.library.refresh()
        elif index == 2:
            self.projects.refresh()
        elif index == 4:
            self.settings.refresh()

    def _start_preview(self, request: dict[str, Any]) -> None:
        if not request.get("song") or not request.get("voice_id"):
            return
        self._save_project(request)
        self.wizard.show_processing("preview")
        self._start_worker(
            lambda event: self.client.preview_cover(
                song=request["song"],
                voice_id=request["voice_id"],
                quality=request["quality"],
                advanced=request.get("advanced"),
                on_event=event,
            ),
            self.wizard.show_recommendation,
        )

    def _start_full(self, request: dict[str, Any]) -> None:
        output = _available_output(Path(request["song"]))
        self.wizard.show_processing("full")
        self._start_worker(
            lambda event: self.client.render_cover(
                song=request["song"],
                voice_id=request["voice_id"],
                quality=request["quality"],
                output=str(output),
                advanced=request.get("advanced"),
                on_event=event,
            ),
            self.wizard.show_result,
        )

    def _start_worker(
        self,
        operation: Callable[[Callable[[str, dict[str, Any]], None]], dict],
        on_success: Callable[[dict], None],
        on_progress: Callable[[float, str], None] | None = None,
        on_failure: Callable[[str, str], None] | None = None,
    ) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        self._cancelled = False
        thread = QThread(self)
        worker = EngineWorker(operation)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(on_progress or self.wizard.update_progress)
        worker.succeeded.connect(on_success)
        worker.failed.connect(on_failure or self._show_error)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._worker_finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    def _cancel(self) -> None:
        if self._thread is None or not self._thread.isRunning():
            return
        self._cancelled = True
        self.client.cancel_current()
        self.wizard.cancelled()

    def _offer_recovery(self) -> None:
        recoverable = getattr(self.client, "recoverable_jobs", None)
        if not callable(recoverable):
            return
        try:
            jobs = [item for item in recoverable() if item.get("kind") == "cover"]
        except Exception:
            return
        if not jobs:
            return
        job = jobs[0]
        answer = QMessageBox.question(
            self,
            "נמצאה עבודה שלא הסתיימה",
            "מצאנו קאבר שנעצר. להמשיך מהשלב האחרון שהושלם?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        source = str(job.get("source") or "")
        voice_id = str(job.get("voice_id") or "")
        if source:
            self.wizard.drop_zone.set_file(source)
        if voice_id:
            self.wizard._voice_selected(voice_id)
        preview = bool(job.get("preview"))
        self.wizard.show_processing("preview" if preview else "full")
        self._start_worker(
            lambda event: self.client.resume_cover(str(job["job_id"]), on_event=event),
            self.wizard.show_recommendation if preview else self.wizard.show_result,
        )

    @Slot(str, str)
    def _show_error(self, code: str, fallback: str) -> None:
        if self._cancelled:
            return
        title, action = error_text(code, fallback)
        QMessageBox.warning(self, title, action)
        self.wizard.cancelled()

    @Slot()
    def _worker_finished(self) -> None:
        self._thread = None
        self._worker = None

    def _save_project(self, request: dict[str, Any]) -> None:
        source = Path(request["song"])
        digest = hashlib.sha256(str(source.resolve()).encode("utf-8")).hexdigest()[:16]
        # Persistence must not block a cover; the engine log keeps details.
        with contextlib.suppress(Exception):
            self.client.save_project(
                f"song-{digest}",
                source.stem,
                {
                    "song": str(source),
                    "voice_id": request["voice_id"],
                    "quality": request["quality"],
                    "advanced": request.get("advanced") or {},
                },
            )

    def _open_project(self, data: dict[str, object]) -> None:
        self.wizard.load_project(data)
        self._navigate(0)

    def _theme_changed(self, value: str) -> None:
        app = QApplication.instance()
        if isinstance(app, QApplication):
            apply_theme(app, Theme(value))

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        self.client.cancel_current()
        event.accept()


def _available_output(source: Path) -> Path:
    candidate = source.with_name(f"{source.stem}-SongVoice.wav")
    counter = 2
    while candidate.exists():
        candidate = source.with_name(f"{source.stem}-SongVoice-{counter}.wav")
        counter += 1
    return candidate


def main(argv: list[str] | None = None) -> int:
    configure_bundled_runtime()
    raw_args = list(sys.argv[1:] if argv is None else argv)
    if "--training-worker" in raw_args:
        worker_parser = argparse.ArgumentParser(add_help=False)
        worker_parser.add_argument("--training-worker", action="store_true")
        worker_parser.add_argument("--home", type=Path, required=True)
        worker_parser.add_argument("session_id")
        worker_args = worker_parser.parse_args(raw_args)
        from svc_engine.training.worker import run

        return run(worker_args.home, worker_args.session_id)
    parser = argparse.ArgumentParser(prog="songvoice", add_help=True)
    parser.add_argument("--engine", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--screenshot", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.engine:
        from svc_engine.rpc.server import serve_stdio

        serve_stdio()
        return 0
    instance = QApplication.instance()
    app = instance if isinstance(instance, QApplication) else QApplication(sys.argv[:1])
    app.setApplicationName("SongVoice")
    app.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
    apply_theme(app, Theme.SYSTEM)
    window = MainWindow()
    window.show()
    if not args.smoke_test:
        QTimer.singleShot(0, window.offer_first_run)
    if args.screenshot:

        def save_screenshot() -> None:
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            window.grab().save(str(args.screenshot))
            if args.smoke_test:
                window.close()

        QTimer.singleShot(500, save_screenshot)
    elif args.smoke_test:
        QTimer.singleShot(500, window.close)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
