"""Hebrew first-run wizard; all heavy work remains in the engine process."""

from __future__ import annotations

import contextlib
from typing import Any

from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QDialog,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from svc_app.engine_client import EngineCallError, EngineClient


class _ProvisionWorker(QObject):
    progress = Signal(float, str)
    succeeded = Signal(dict)
    failed = Signal(str)
    finished = Signal()

    def __init__(self, client: EngineClient) -> None:
        super().__init__()
        self.client = client

    @Slot()
    def run(self) -> None:
        try:
            result = self.client.provision(self._event)
        except EngineCallError as exc:
            self.failed.emit(exc.message_he)
        except Exception:
            self.failed.emit("לא הצלחנו להשלים את ההכנה. אפשר לנסות שוב.")
        else:
            self.succeeded.emit(result)
        finally:
            self.finished.emit()

    def _event(self, event: str, data: dict[str, Any]) -> None:
        if event == "progress":
            self.progress.emit(
                float(data.get("fraction") or 0.0), str(data.get("message_he") or "")
            )


class FirstRunDialog(QDialog):
    def __init__(self, client: EngineClient, status: dict[str, Any], parent=None) -> None:  # type: ignore[no-untyped-def]
        super().__init__(parent)
        self.client = client
        self._thread: QThread | None = None
        self._worker: _ProvisionWorker | None = None
        self.setWindowTitle("הכנה ראשונית · SongVoice")
        self.setMinimumWidth(560)
        layout = QVBoxLayout(self)
        title = QLabel("מכינים את SongVoice לקאבר הראשון")
        title.setStyleSheet("font-size: 24px; font-weight: 800;")
        device = str(status.get("device_name") or "מעבד")
        backend = str(status.get("backend") or "cpu").upper()
        self.detail = QLabel(
            f"זוהתה חומרה: {device} ({backend}). נבדוק את סביבת PyTorch המשובצת "
            "ונוריד קבצי מודל מאומתים. השמע שלך נשאר במחשב."
        )
        self.detail.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.start_button = QPushButton("התחל הכנה")
        self.start_button.setProperty("primary", True)
        self.start_button.clicked.connect(self.start)
        layout.addWidget(title)
        layout.addWidget(self.detail)
        layout.addWidget(self.progress)
        layout.addWidget(self.start_button)

    def start(self) -> None:
        if self._thread is not None:
            return
        self.start_button.setEnabled(False)
        thread = QThread(self)
        worker = _ProvisionWorker(self.client)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self._progress)
        worker.succeeded.connect(self._succeeded)
        worker.failed.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finished)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(float, str)
    def _progress(self, fraction: float, message: str) -> None:
        self.progress.setValue(round(max(0.0, min(1.0, fraction)) * 1000))
        if message:
            self.detail.setText(message)

    @Slot(dict)
    def _succeeded(self, _result: dict[str, Any]) -> None:
        self.progress.setValue(1000)
        self.detail.setText("הכול מוכן. אפשר ליצור קאבר ראשון.")
        self.start_button.setText("כניסה ל־SongVoice")
        self.start_button.setEnabled(True)
        with_context = self.start_button.clicked
        with contextlib.suppress(RuntimeError):
            with_context.disconnect()
        self.start_button.clicked.connect(self.accept)

    @Slot(str)
    def _failed(self, message: str) -> None:
        self.detail.setText(message)
        self.start_button.setText("נסה שוב")
        self.start_button.setEnabled(True)

    @Slot()
    def _finished(self) -> None:
        self._thread = None
        self._worker = None
