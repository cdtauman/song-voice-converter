"""Selectable target-voice card."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout


class VoiceCard(QFrame):
    selected = Signal(str)

    def __init__(self, voice: dict[str, object]) -> None:
        super().__init__()
        self.voice_id = str(voice.get("id") or "")
        self.setProperty("card", True)
        self.setProperty("selected", False)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        layout = QHBoxLayout(self)
        avatar = str(voice.get("avatar_path") or "")
        badge = QLabel("●")
        badge.setFixedSize(56, 56)
        badge.setStyleSheet("font-size: 24px;")
        if avatar:
            pixmap = QPixmap(avatar)
            if not pixmap.isNull():
                badge.setPixmap(
                    pixmap.scaled(
                        badge.size(),
                        Qt.AspectRatioMode.KeepAspectRatioByExpanding,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                )
        text = QVBoxLayout()
        name = QLabel(str(voice.get("display_name") or self.voice_id))
        name.setStyleSheet("font-size: 17px; font-weight: 700;")
        status = (
            "מוכן לקאבר"
            if voice.get("usable")
            else str(voice.get("health_note_he") or "הקול אינו מוכן")
        )
        state = QLabel(status)
        state.setProperty("muted", True)
        text.addWidget(name)
        text.addWidget(state)
        layout.addWidget(badge)
        layout.addLayout(text, 1)
        self.setEnabled(bool(voice.get("usable")))

    def set_selected(self, selected: bool) -> None:
        self.setProperty("selected", selected)
        self.style().unpolish(self)
        self.style().polish(self)

    def _choose(self) -> None:
        if self.isEnabled():
            self.selected.emit(self.voice_id)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton:
            self._choose()
        super().mousePressEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            self._choose()
            return
        super().keyPressEvent(event)
