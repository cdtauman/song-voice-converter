"""Voice library with explicit consent-gated import."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from svc_app.engine_client import EngineCallError, EngineClient
from svc_app.widgets import VoiceCard


class VoiceLibraryScreen(QWidget):
    changed = Signal(list)

    def __init__(self, client: EngineClient) -> None:
        super().__init__()
        self.client = client
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("ספריית קולות")
        title.setObjectName("Title")
        subtitle = QLabel("קולות שייבאת ושיש לך רשות מפורשת להשתמש בהם")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        add = QPushButton("הוסף קול מקובץ ZIP")
        add.setProperty("primary", True)
        add.clicked.connect(self._import)
        header.addLayout(title_box, 1)
        header.addWidget(add)
        root.addLayout(header)
        self.container = QWidget()
        self.cards = QVBoxLayout(self.container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.container)
        root.addWidget(scroll, 1)

    def refresh(self) -> list[dict[str, object]]:
        try:
            voices = self.client.voices()
        except Exception:
            voices = []
        while self.cards.count():
            item = self.cards.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        if not voices:
            empty = QLabel("הספרייה עדיין ריקה. הוסף קול כדי ליצור קאבר ראשון.")
            empty.setProperty("muted", True)
            self.cards.addWidget(empty)
        for voice in voices:
            row = QWidget()
            row_layout = QHBoxLayout(row)
            card = VoiceCard(voice)
            remove = QPushButton("הסר")
            remove.setProperty("danger", True)
            voice_id = str(voice.get("id") or "")
            remove.clicked.connect(
                lambda _checked=False, selected_id=voice_id: self._remove(selected_id)
            )
            row_layout.addWidget(card, 1)
            row_layout.addWidget(remove)
            self.cards.addWidget(row)
        self.cards.addStretch()
        self.changed.emit(voices)
        return voices

    def _import(self) -> None:
        archive, _ = QFileDialog.getOpenFileName(self, "בחירת חבילת קול", "", "ZIP (*.zip)")
        if not archive:
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle("אישור הרשאה לקול")
        dialog.setText(
            "SongVoice מאפשרת שימוש רק בקול שיצרת בעצמך או שקיבלת רשות מפורשת להשתמש בו."
        )
        consent = QCheckBox("אני מאשר/ת שיש לי רשות להשתמש בקול הזה")
        name = QLineEdit(Path(archive).stem)
        form_widget = QWidget()
        form = QFormLayout(form_widget)
        form.addRow("שם שיוצג:", name)
        form.addRow(consent)
        dialog_layout = dialog.layout()
        if isinstance(dialog_layout, QGridLayout):
            dialog_layout.addWidget(form_widget, 1, 0, 1, dialog_layout.columnCount())
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel)
        ok = dialog.button(QMessageBox.StandardButton.Ok)
        ok.setText("הוסף קול")
        ok.setEnabled(False)
        consent.toggled.connect(ok.setEnabled)
        if dialog.exec() != QMessageBox.StandardButton.Ok:
            return
        try:
            result = self.client.import_voice(
                archive,
                name.text().strip() or Path(archive).stem,
                consent_confirmed=True,
                consent_note="אושר בממשק SongVoice",
            )
        except EngineCallError as exc:
            QMessageBox.warning(self, "לא הצלחנו להוסיף את הקול", exc.message_he)
            return
        QMessageBox.information(self, "הקול נוסף", str(result.get("summary_he") or ""))
        self.refresh()

    def _remove(self, voice_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "הסרת קול",
            "להסיר את הקול מהספרייה? לא ניתן לבטל פעולה זו.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.client.remove_voice(voice_id)
        self.refresh()
