"""Voice library with explicit consent-gated import."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from svc_app.engine_client import EngineCallError, EngineClient
from svc_app.screens.voices import TrainingWizardDialog
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
        actions = QVBoxLayout()
        train = QPushButton("צור קול מהקלטות")
        train.setProperty("primary", True)
        train.clicked.connect(self._train)
        add = QPushButton("ייבא קול מקובץ ZIP")
        add.clicked.connect(self._import)
        self.resume = QPushButton("המשך אימון שנעצר")
        self.resume.clicked.connect(self._resume_training)
        self.resume.hide()
        actions.addWidget(train)
        actions.addWidget(add)
        actions.addWidget(self.resume)
        header.addLayout(title_box, 1)
        header.addLayout(actions)
        root.addLayout(header)
        self.container = QWidget()
        self.cards = QGridLayout(self.container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.container)
        root.addWidget(scroll, 1)

    def refresh(self) -> list[dict[str, object]]:
        try:
            voices = self.client.voices()
        except Exception:
            voices = []
        try:
            sessions = self.client.training_sessions()
        except Exception:
            sessions = []
        self._resumable = [
            session
            for session in sessions
            if str(session.get("stage")) not in {"ready", "recordings"}
        ]
        self.resume.setVisible(bool(self._resumable))
        while self.cards.count():
            item = self.cards.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        if not voices:
            empty = QLabel("הספרייה עדיין ריקה. הוסף קול כדי ליצור קאבר ראשון.")
            empty.setProperty("muted", True)
            self.cards.addWidget(empty, 0, 0, 1, 2)
        for position, voice in enumerate(voices):
            voice_id = str(voice.get("id") or "")
            row = QWidget()
            row.setProperty("card", True)
            row_layout = QVBoxLayout(row)
            card = VoiceCard(voice)
            controls = QHBoxLayout()
            rename = QPushButton("שנה שם")
            rename.clicked.connect(lambda _checked=False, selected=voice: self._rename(selected))
            image = QPushButton("תמונה")
            image.clicked.connect(
                lambda _checked=False, selected_id=voice_id: self._image(selected_id)
            )
            sample = QPushButton("דוגמה")
            sample.clicked.connect(lambda _checked=False, selected=voice: self._sample(selected))
            health = QPushButton("בדוק")
            health.clicked.connect(
                lambda _checked=False, selected_id=voice_id: self._health(selected_id)
            )
            remove = QPushButton("הסר")
            remove.setProperty("danger", True)
            remove.clicked.connect(
                lambda _checked=False, selected_id=voice_id: self._remove(selected_id)
            )
            for button in (rename, image, sample, health, remove):
                controls.addWidget(button)
            row_layout.addWidget(card, 1)
            row_layout.addLayout(controls)
            self.cards.addWidget(row, position // 2, position % 2)
        self.cards.setRowStretch((len(voices) + 1) // 2, 1)
        self.changed.emit(voices)
        return voices

    def _train(self) -> None:
        dialog = TrainingWizardDialog(self.client, self)
        dialog.voice_ready.connect(self.refresh)
        dialog.exec()
        self.refresh()

    def _resume_training(self) -> None:
        if not getattr(self, "_resumable", None):
            return
        dialog = TrainingWizardDialog(self.client, self, self._resumable[0])
        dialog.voice_ready.connect(self.refresh)
        dialog.exec()
        self.refresh()

    def _rename(self, voice: dict[str, object]) -> None:
        value, ok = QInputDialog.getText(
            self,
            "שינוי שם הקול",
            "שם חדש:",
            text=str(voice.get("display_name") or ""),
        )
        if ok and value.strip():
            self.client.update_voice(str(voice.get("id") or ""), display_name=value.strip())
            self.refresh()

    def _image(self, voice_id: str) -> None:
        image, _ = QFileDialog.getOpenFileName(self, "בחירת תמונה", "", "PNG (*.png)")
        if image:
            self.client.update_voice(voice_id, avatar=image)
            self.refresh()

    def _sample(self, voice: dict[str, object]) -> None:
        existing = str(voice.get("sample_path") or "")
        if existing and Path(existing).is_file():
            QDesktopServices.openUrl(QUrl.fromLocalFile(existing))
            return
        sample, _ = QFileDialog.getOpenFileName(
            self, "בחירת דוגמת שמע", "", "Audio (*.wav *.flac *.mp3 *.m4a *.ogg)"
        )
        if sample:
            self.client.update_voice(str(voice.get("id") or ""), sample=sample)
            self.refresh()

    def _health(self, voice_id: str) -> None:
        result = self.client.check_voice(voice_id)
        QMessageBox.information(self, "בדיקת תקינות", str(result.get("health_note_he") or ""))
        self.refresh()

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
        self._finish_import(archive, name.text().strip() or Path(archive).stem)

    def _finish_import(self, archive: str, display_name: str) -> bool:
        """Import, refresh both library views, and report a clear result."""
        try:
            result = self.client.import_voice(
                archive,
                display_name,
                consent_confirmed=True,
                consent_note="אושר בממשק SongVoice",
            )
        except EngineCallError as exc:
            QMessageBox.warning(self, "לא הצלחנו להוסיף את הקול", exc.message_he)
            return False
        self.refresh()
        QMessageBox.information(self, "הקול נוסף", str(result.get("summary_he") or ""))
        return True

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
