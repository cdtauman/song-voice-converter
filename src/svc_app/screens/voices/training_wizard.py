"""Five-step Hebrew wizard for creating a local RVC voice."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QThread, QTimer, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from svc_app.engine_client import EngineCallError, EngineClient

__all__ = ["TrainingWizardDialog"]


class _Task(QObject):
    succeeded = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, operation: Callable[[], dict[str, Any]]) -> None:
        super().__init__()
        self.operation = operation

    @Slot()
    def run(self) -> None:
        try:
            result = self.operation()
        except EngineCallError as exc:
            self.failed.emit(exc.message_he)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self.failed.emit(str(exc))
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()


def _page(title: str, subtitle: str) -> tuple[QWidget, QVBoxLayout]:
    widget = QWidget()
    layout = QVBoxLayout(widget)
    heading = QLabel(title)
    heading.setObjectName("Title")
    explanation = QLabel(subtitle)
    explanation.setObjectName("Subtitle")
    explanation.setWordWrap(True)
    layout.addWidget(heading)
    layout.addWidget(explanation)
    layout.addSpacing(12)
    return widget, layout


class TrainingWizardDialog(QDialog):
    voice_ready = Signal()

    def __init__(
        self,
        client: EngineClient,
        parent: QWidget | None = None,
        session: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(parent)
        self.client = client
        self.session: dict[str, Any] = session or {}
        self.recordings: list[str] = []
        self._thread: QThread | None = None
        self._task: _Task | None = None
        self.setWindowTitle("יצירת קול חדש")
        self.resize(760, 600)
        root = QVBoxLayout(self)
        self.steps = QLabel("1. הקלטות  ←  2. איכות  ←  3. ניקוי  ←  4. אימון  ←  5. מוכן")
        self.steps.setProperty("muted", True)
        root.addWidget(self.steps)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)

        first, layout = _page(
            "הוסף הקלטות", "בחר הקלטות של אותו אדם. 15 דקות או יותר יתנו תוצאה יציבה יותר."
        )
        self.name = QLineEdit()
        self.name.setPlaceholderText("שם הקול שיוצג בספרייה")
        self.files = QListWidget()
        add = QPushButton("הוסף הקלטות")
        add.clicked.connect(self._add_files)
        self.mixed = QCheckBox("ההקלטות כוללות מוזיקה או ליווי — הפרד אותם אוטומטית")
        self.mixed.setChecked(True)
        self.consent = QCheckBox("אני מאשר/ת שזה הקול שלי או שקיבלתי רשות מפורשת ליצור ממנו מודל")
        inspect = QPushButton("בדוק את איכות החומר")
        inspect.setProperty("primary", True)
        inspect.clicked.connect(self._inspect)
        layout.addWidget(self.name)
        layout.addWidget(self.files, 1)
        layout.addWidget(add)
        layout.addWidget(self.mixed)
        layout.addWidget(self.consent)
        layout.addWidget(inspect)
        self.pages.addWidget(first)

        quality, layout = _page("בדיקת איכות", "הבדיקה מסתיימת לפני האימון ומסבירה מה כדאי לתקן.")
        self.quality_summary = QLabel()
        self.quality_summary.setWordWrap(True)
        self.quality_issues = QLabel()
        self.quality_issues.setWordWrap(True)
        self.clean_button = QPushButton("נקה והכן את ההקלטות")
        self.clean_button.setProperty("primary", True)
        self.clean_button.clicked.connect(self._prepare)
        layout.addWidget(self.quality_summary)
        layout.addWidget(self.quality_issues, 1)
        layout.addWidget(self.clean_button)
        self.pages.addWidget(quality)

        cleaning, layout = _page(
            "ניקוי", "מפרידים ליווי, מסירים רעש והדהוד, חותכים שקטים ומכינים מקטעים."
        )
        self.clean_progress = QProgressBar()
        self.clean_progress.setRange(0, 0)
        self.clean_status = QLabel("ממתין להתחלה…")
        self.begin_training = QPushButton("התחל אימון")
        self.begin_training.setProperty("primary", True)
        self.begin_training.setEnabled(False)
        self.begin_training.clicked.connect(self._start)
        layout.addWidget(self.clean_progress)
        layout.addWidget(self.clean_status)
        layout.addStretch()
        layout.addWidget(self.begin_training)
        self.pages.addWidget(cleaning)

        training, layout = _page(
            "אימון", "SongVoice שומר נקודת ביקורת. אפשר לעצור ולחזור גם לאחר פתיחת האפליקציה מחדש."
        )
        self.training_progress = QProgressBar()
        self.training_progress.setRange(0, 1000)
        self.training_status_label = QLabel()
        self.training_status_label.setWordWrap(True)
        self.eta = QLabel()
        self.eta.setProperty("muted", True)
        self.pause_button = QPushButton("עצור אימון")
        self.pause_button.clicked.connect(self._toggle_pause)
        layout.addWidget(self.training_progress)
        layout.addWidget(self.training_status_label)
        layout.addWidget(self.eta)
        layout.addStretch()
        layout.addWidget(self.pause_button)
        self.pages.addWidget(training)

        ready, layout = _page("הקול מוכן", "המודל, דוגמת השמע ופרופיל המנעד נוספו לספרייה.")
        ready_text = QLabel("אפשר לבחור את הקול במסך ״קאבר חדש״ ולהשתמש בו מיד.")
        ready_text.setWordWrap(True)
        close = QPushButton("סיום")
        close.setProperty("primary", True)
        close.clicked.connect(self.accept)
        layout.addWidget(ready_text)
        layout.addStretch()
        layout.addWidget(close)
        self.pages.addWidget(ready)

        self.poller = QTimer(self)
        self.poller.setInterval(1200)
        self.poller.timeout.connect(self._poll)
        if session:
            self._show_session(session)

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "בחירת הקלטות",
            "",
            "Audio (*.wav *.flac *.mp3 *.m4a *.ogg)",
        )
        for path in files:
            if path not in self.recordings:
                self.recordings.append(path)
                self.files.addItem(Path(path).name)

    def _run(
        self, operation: Callable[[], dict[str, Any]], success: Callable[[dict[str, Any]], None]
    ) -> None:
        if self._thread is not None and self._thread.isRunning():
            return
        thread = QThread(self)
        task = _Task(operation)
        task.moveToThread(thread)
        thread.started.connect(task.run)
        task.succeeded.connect(success)
        task.failed.connect(self._failed)
        task.finished.connect(thread.quit)
        task.finished.connect(task.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_task)
        self._thread, self._task = thread, task
        thread.start()

    def _clear_task(self) -> None:
        self._thread = None
        self._task = None

    def _inspect(self) -> None:
        if not self.recordings or not self.name.text().strip() or not self.consent.isChecked():
            QMessageBox.warning(self, "חסר מידע", "יש לבחור הקלטות, לתת שם ולאשר הרשאה.")
            return
        self._run(
            lambda: self.client.inspect_training(
                str(
                    self.client.create_training(
                        self.name.text().strip(),
                        self.recordings,
                        consent_confirmed=True,
                        consent_note="אושר באשף האימון של SongVoice",
                    )["session_id"]
                )
            ),
            self._quality_ready,
        )

    def _quality_ready(self, session: dict[str, Any]) -> None:
        self.session = session
        quality = dict(session.get("quality") or {})
        self.quality_summary.setText(str(quality.get("summary_he") or ""))
        issues = [dict(issue) for issue in quality.get("issues", [])]
        self.quality_issues.setText(
            "\n\n".join(
                f"• {issue.get('message_he')}\n  {issue.get('action_he')}" for issue in issues
            )
            or "לא נמצאו בעיות בולטות."
        )
        self.clean_button.setEnabled(bool(quality.get("can_train")))
        self.pages.setCurrentIndex(1)

    def _prepare(self) -> None:
        self.pages.setCurrentIndex(2)
        self.clean_status.setText("מפרידים ומנקים את ההקלטות…")
        self._run(
            lambda: self.client.prepare_training(
                str(self.session["session_id"]), separate_mix=self.mixed.isChecked()
            ),
            self._prepared,
        )

    def _prepared(self, session: dict[str, Any]) -> None:
        self.session = session
        self.clean_progress.setRange(0, 100)
        self.clean_progress.setValue(100)
        self.clean_status.setText("החומר נקי ומוכן לאימון.")
        self.begin_training.setEnabled(True)

    def _start(self) -> None:
        self.pages.setCurrentIndex(3)
        try:
            self.session = self.client.start_training(str(self.session["session_id"]))
        except EngineCallError as exc:
            self._failed(exc.message_he)
            return
        self.poller.start()
        self._render_training(self.session)

    def _poll(self) -> None:
        try:
            session = self.client.training_status(str(self.session["session_id"]))
        except Exception:
            return
        self.session = session
        self._show_session(session)

    def _toggle_pause(self) -> None:
        stage = str(self.session.get("stage") or "")
        try:
            if stage in {"paused", "failed"}:
                self.session = self.client.start_training(str(self.session["session_id"]))
                self.poller.start()
            else:
                self.session = self.client.pause_training(str(self.session["session_id"]))
        except EngineCallError as exc:
            self._failed(exc.message_he)
            return
        self._render_training(self.session)

    def _show_session(self, session: dict[str, Any]) -> None:
        self.session = session
        stage = str(session.get("stage") or "")
        if stage == "quality":
            self._quality_ready(session)
        elif stage == "cleaning":
            self.pages.setCurrentIndex(2)
        elif stage in {"training", "paused", "failed", "finalizing"}:
            self.pages.setCurrentIndex(3)
            self._render_training(session)
            if stage in {"training", "finalizing"}:
                self.poller.start()
        elif stage == "ready":
            self.poller.stop()
            self.pages.setCurrentIndex(4)
            self.voice_ready.emit()

    def _render_training(self, session: dict[str, Any]) -> None:
        self.training_progress.setValue(int(float(session.get("progress") or 0.0) * 1000))
        self.training_status_label.setText(str(session.get("message_he") or ""))
        remaining = session.get("estimated_remaining_seconds")
        if remaining is None:
            self.eta.setText("הערכת הזמן תתעדכן לאחר epoch ראשון.")
        else:
            minutes = max(0, int(float(remaining) // 60))
            self.eta.setText(f"זמן משוער שנותר: כ־{minutes // 60} שעות ו־{minutes % 60} דקות")
        paused = str(session.get("stage")) in {"paused", "failed"}
        self.pause_button.setText("המשך אימון" if paused else "עצור אימון")

    def _failed(self, message: str) -> None:
        QMessageBox.warning(self, "לא הצלחנו להשלים את הפעולה", message or "נסה שוב.")
