"""Keyboard-accessible audio drag and drop target."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QKeyEvent, QMouseEvent
from PySide6.QtWidgets import QFileDialog, QFrame, QLabel, QVBoxLayout

_AUDIO = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


class DropZone(QFrame):
    file_selected = Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.setProperty("card", True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(190)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.icon = QLabel("♫")
        self.icon.setStyleSheet("font-size: 42px; font-weight: 700;")
        self.title = QLabel("גרור לכאן שיר")
        self.title.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.hint = QLabel("או לחץ לבחירת קובץ · WAV, MP3, M4A, FLAC")
        self.hint.setProperty("muted", True)
        for widget in (self.icon, self.title, self.hint):
            widget.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(widget)

    def choose(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "בחירת שיר", "", "קובצי שמע (*.wav *.mp3 *.m4a *.flac *.ogg *.aac)"
        )
        if path:
            self.set_file(path)

    def set_file(self, path: str) -> None:
        if Path(path).suffix.lower() not in _AUDIO:
            return
        self.title.setText(Path(path).name)
        self.hint.setText("השיר נבחר · לחץ כדי להחליף")
        self.file_selected.emit(path)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self.choose()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self.choose()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if len(urls) == 1 and Path(urls[0].toLocalFile()).suffix.lower() in _AUDIO:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        urls = event.mimeData().urls()
        if urls:
            self.set_file(urls[0].toLocalFile())
            event.acceptProposedAction()
