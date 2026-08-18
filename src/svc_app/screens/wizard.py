"""The simple-mode cover wizard: song to result in seven explicit steps."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from svc_app.widgets import ABPlayer, DropZone, VoiceCard


class CoverWizard(QWidget):
    preview_requested = Signal(dict)
    full_requested = Signal(dict)
    cancel_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.song = ""
        self.voice_id = ""
        self.quality = "balanced"
        self.preview_result: dict[str, object] = {}
        self._voice_cards: list[VoiceCard] = []
        self._processing_mode = "preview"

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        root.setSpacing(18)
        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("קאבר חדש")
        title.setObjectName("Title")
        subtitle = QLabel("שיר אחד, קול אחד, ותוצאה שאפשר לשמוע לפני השמירה")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        self.step_pill = QLabel("שלב 1 מתוך 7 · בחירת שיר")
        self.step_pill.setObjectName("StepPill")
        header.addLayout(title_box, 1)
        header.addWidget(self.step_pill, 0, Qt.AlignmentFlag.AlignTop)
        root.addLayout(header)
        self.pages = QStackedWidget()
        root.addWidget(self.pages, 1)

        self.pages.addWidget(self._song_page())
        self.pages.addWidget(self._voice_page())
        self.pages.addWidget(self._quality_page())
        self.pages.addWidget(self._processing_page())
        self.pages.addWidget(self._recommendation_page())
        self.pages.addWidget(self._preview_page())
        self.pages.addWidget(self._result_page())

    def _page_shell(self, heading: str, text: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(16)
        heading_label = QLabel(heading)
        heading_label.setStyleSheet("font-size: 22px; font-weight: 700;")
        description = QLabel(text)
        description.setWordWrap(True)
        description.setProperty("muted", True)
        layout.addWidget(heading_label)
        layout.addWidget(description)
        return page, layout

    def _song_page(self) -> QWidget:
        page, layout = self._page_shell(
            "איזה שיר נהפוך לקאבר?",
            "השיר נשאר במחשב שלך. שום קובץ שמע לא נשלח לרשת.",
        )
        self.drop_zone = DropZone()
        self.drop_zone.file_selected.connect(self._song_selected)
        layout.addWidget(self.drop_zone)
        layout.addStretch()
        self.song_next = _primary_button("המשך לבחירת קול")
        self.song_next.setEnabled(False)
        self.song_next.clicked.connect(lambda: self._show(1))
        layout.addWidget(self.song_next, 0, Qt.AlignmentFlag.AlignLeft)
        return page

    def _voice_page(self) -> QWidget:
        page, layout = self._page_shell(
            "באיזה קול להשתמש?",
            "מוצגים רק קולות עם מודל תקין, פרופיל מנעד ואישור שימוש.",
        )
        self.voice_container = QWidget()
        self.voice_layout = QVBoxLayout(self.voice_container)
        self.voice_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.voice_container)
        layout.addWidget(scroll, 1)
        buttons = QHBoxLayout()
        back = QPushButton("חזרה")
        back.clicked.connect(lambda: self._show(0))
        self.voice_next = _primary_button("המשך לאיכות")
        self.voice_next.setEnabled(False)
        self.voice_next.clicked.connect(lambda: self._show(2))
        buttons.addWidget(back)
        buttons.addStretch()
        buttons.addWidget(self.voice_next)
        layout.addLayout(buttons)
        return page

    def _quality_page(self) -> QWidget:
        page, layout = self._page_shell(
            "בחר רמת איכות",
            "אפשר להתחיל ב‘מאוזנת’. לפני הקאבר המלא נכין תצוגה מקדימה קצרה.",
        )
        group = QButtonGroup(self)
        options = [
            ("fast", "מהירה", "לבדיקה זריזה ולמחשבים חלשים"),
            ("balanced", "מאוזנת", "הבחירה המומלצת לרוב השירים"),
            ("max", "מרבית", "איכות גבוהה יותר וזמן עיבוד ארוך יותר"),
        ]
        for value, title, hint in options:
            card = QFrame()
            card.setProperty("card", True)
            card_layout = QVBoxLayout(card)
            radio = QRadioButton(title)
            radio.setProperty("quality", value)
            radio.setStyleSheet("font-size: 17px; font-weight: 700;")
            radio.toggled.connect(
                lambda checked, v=value: self._set_quality(v) if checked else None
            )
            group.addButton(radio)
            card_layout.addWidget(radio)
            hint_label = QLabel(hint)
            hint_label.setProperty("muted", True)
            card_layout.addWidget(hint_label)
            layout.addWidget(card)
            if value == "balanced":
                radio.setChecked(True)
        layout.addStretch()
        buttons = QHBoxLayout()
        back = QPushButton("חזרה")
        back.clicked.connect(lambda: self._show(1))
        start = _primary_button("הכן תצוגה מקדימה")
        start.clicked.connect(self._request_preview)
        buttons.addWidget(back)
        buttons.addStretch()
        buttons.addWidget(start)
        layout.addLayout(buttons)
        return page

    def _processing_page(self) -> QWidget:
        page, layout = self._page_shell(
            "מעבדים את השיר",
            "אפשר לבטל בכל רגע. פעולות שכבר הושלמו נשמרות בבטחה.",
        )
        card = QFrame()
        card.setProperty("card", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        self.progress_message = QLabel("מתחילים…")
        self.progress_message.setStyleSheet("font-size: 18px; font-weight: 700;")
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.progress_detail = QLabel("0%")
        self.progress_detail.setProperty("muted", True)
        card_layout.addWidget(self.progress_message)
        card_layout.addWidget(self.progress)
        card_layout.addWidget(self.progress_detail)
        layout.addWidget(card)
        layout.addStretch()
        cancel = QPushButton("בטל עיבוד")
        cancel.setProperty("danger", True)
        cancel.clicked.connect(self.cancel_requested)
        layout.addWidget(cancel, 0, Qt.AlignmentFlag.AlignLeft)
        return page

    def _recommendation_page(self) -> QWidget:
        page, layout = self._page_shell(
            "מצאנו התאמה לקול",
            "המלצת הגובה מחושבת לפי המנעד של השיר ושל הקול שבחרת.",
        )
        card = QFrame()
        card.setProperty("card", True)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(24, 24, 24, 24)
        self.recommend_title = QLabel("המלצה")
        self.recommend_title.setStyleSheet("font-size: 24px; font-weight: 700;")
        self.recommend_detail = QLabel()
        self.recommend_detail.setWordWrap(True)
        self.recommend_detail.setProperty("muted", True)
        card_layout.addWidget(self.recommend_title)
        card_layout.addWidget(self.recommend_detail)
        layout.addWidget(card)
        layout.addStretch()
        next_button = _primary_button("שמע תצוגה מקדימה")
        next_button.clicked.connect(self._open_preview)
        layout.addWidget(next_button, 0, Qt.AlignmentFlag.AlignLeft)
        return page

    def _preview_page(self) -> QWidget:
        page, layout = self._page_shell(
            "לפני שממשיכים — איך זה נשמע?",
            "החלף בין המקור לקאבר באותה נקודת זמן. זהו קטע קצר בלבד.",
        )
        card = QFrame()
        card.setProperty("card", True)
        card_layout = QVBoxLayout(card)
        self.player = ABPlayer()
        card_layout.addWidget(self.player)
        layout.addWidget(card)
        layout.addStretch()
        buttons = QHBoxLayout()
        back = QPushButton("שנה בחירה")
        back.clicked.connect(lambda: self._show(2))
        full = _primary_button("צור את הקאבר המלא")
        full.clicked.connect(self._request_full)
        buttons.addWidget(back)
        buttons.addStretch()
        buttons.addWidget(full)
        layout.addLayout(buttons)
        return page

    def _result_page(self) -> QWidget:
        page, layout = self._page_shell(
            "הקאבר מוכן",
            "אפשר להאזין, לפתוח את התיקייה או להתחיל קאבר נוסף.",
        )
        card = QFrame()
        card.setProperty("card", True)
        card_layout = QVBoxLayout(card)
        self.result_name = QLabel()
        self.result_name.setStyleSheet("font-size: 20px; font-weight: 700;")
        self.result_summary = QLabel()
        self.result_summary.setWordWrap(True)
        self.result_summary.setProperty("muted", True)
        card_layout.addWidget(self.result_name)
        card_layout.addWidget(self.result_summary)
        layout.addWidget(card)
        layout.addStretch()
        again = _primary_button("קאבר חדש")
        again.clicked.connect(self.reset)
        layout.addWidget(again, 0, Qt.AlignmentFlag.AlignLeft)
        return page

    def set_voices(self, voices: list[dict[str, object]]) -> None:
        while self.voice_layout.count():
            item = self.voice_layout.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._voice_cards = []
        if not voices:
            empty = QLabel("אין עדיין קולות מוכנים. הוסף קול דרך ‘ספריית קולות’.")
            empty.setWordWrap(True)
            empty.setProperty("muted", True)
            self.voice_layout.addWidget(empty)
        for voice in voices:
            card = VoiceCard(voice)
            card.selected.connect(self._voice_selected)
            self.voice_layout.addWidget(card)
            self._voice_cards.append(card)
        self.voice_layout.addStretch()

    def load_project(self, data: dict[str, object]) -> None:
        song = str(data.get("song") or "")
        if song:
            self.drop_zone.set_file(song)
        self.voice_id = str(data.get("voice_id") or "")
        self.quality = str(data.get("quality") or "balanced")
        self._voice_selected(self.voice_id)
        self._show(2 if self.song and self.voice_id else 0)

    def show_processing(self, mode: str) -> None:
        self._processing_mode = mode
        self.progress.setValue(0)
        self.progress_detail.setText("0%")
        self.progress_message.setText(
            "מכינים תצוגה מקדימה…" if mode == "preview" else "יוצרים את הקאבר המלא…"
        )
        self._show(3)

    def update_progress(self, fraction: float, message: str) -> None:
        value = round(max(0.0, min(1.0, fraction)) * 1000)
        self.progress.setValue(value)
        self.progress_detail.setText(f"{value / 10:.0f}%")
        if message:
            self.progress_message.setText(message)

    def show_recommendation(self, result: dict[str, object]) -> None:
        self.preview_result = result
        raw_recommendation = result.get("recommendation")
        recommendation = raw_recommendation if isinstance(raw_recommendation, dict) else {}
        shift = int(recommendation.get("semitones") or 0)
        playback = int(recommendation.get("playback_semitones") or 0)
        self.recommend_title.setText(f"להזיז את הקול ב־{shift:+d} חצאי טונים")
        if playback:
            detail = f"כדי לשמור על ההרמוניה, גם הליווי יוזז ב־{playback:+d} חצאי טונים."
        else:
            detail = "הליווי נשאר בגובה המקורי — אין צורך להזיז אותו."
        self.recommend_detail.setText(detail)
        self._show(4)

    def show_result(self, result: dict[str, object]) -> None:
        output = str(result.get("output") or "")
        self.result_name.setText(Path(output).name)
        self.result_summary.setText(str(result.get("summary_he") or "הקאבר נשמר בהצלחה."))
        self._show(6)

    def cancelled(self) -> None:
        self.progress_message.setText("העיבוד בוטל")
        self._show(2)

    def reset(self) -> None:
        self.player.stop()
        self.song = ""
        self.voice_id = ""
        self.preview_result = {}
        self.song_next.setEnabled(False)
        self.voice_next.setEnabled(False)
        self.drop_zone.title.setText("גרור לכאן שיר")
        self.drop_zone.hint.setText("או לחץ לבחירת קובץ · WAV, MP3, M4A, FLAC")
        self._show(0)

    def _song_selected(self, path: str) -> None:
        self.song = path
        self.song_next.setEnabled(bool(path))

    def _voice_selected(self, voice_id: str) -> None:
        self.voice_id = voice_id
        for card in self._voice_cards:
            card.set_selected(card.voice_id == voice_id)
        self.voice_next.setEnabled(
            any(card.voice_id == voice_id and card.isEnabled() for card in self._voice_cards)
        )

    def _set_quality(self, value: str) -> None:
        self.quality = value

    def _request_preview(self) -> None:
        self.preview_requested.emit(self.request_data())

    def _request_full(self) -> None:
        self.player.stop()
        self.full_requested.emit(self.request_data())

    def _open_preview(self) -> None:
        self.player.set_sources(self.song, str(self.preview_result.get("output") or ""))
        self._show(5)

    def request_data(self) -> dict[str, str]:
        return {"song": self.song, "voice_id": self.voice_id, "quality": self.quality}

    def _show(self, index: int) -> None:
        labels = ["בחירת שיר", "בחירת קול", "איכות", "עיבוד", "המלצה", "Preview", "תוצאה"]
        self.pages.setCurrentIndex(index)
        self.step_pill.setText(f"שלב {index + 1} מתוך 7 · {labels[index]}")


def _primary_button(text: str) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("primary", True)
    return button
