"""Synchronous multi-variant player with an optional blind identity layer."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class ABPlayer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._players: list[QMediaPlayer] = []
        self._outputs: list[QAudioOutput] = []
        self._buttons: list[QPushButton] = []
        self._labels: list[str] = []
        self._blind = False
        self._active = 0
        layout = QVBoxLayout(self)
        self.label = QLabel("בחר מקור להשמעה")
        self.label.setProperty("muted", True)
        layout.addWidget(self.label)
        self.buttons = QHBoxLayout()
        layout.addLayout(self.buttons)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0, 0)
        self.position.sliderMoved.connect(self._seek)
        layout.addWidget(self.position)
        self.set_variants([])

    def set_sources(self, original: str, preview: str) -> None:
        self.set_variants([("המקור", original), ("התוצאה", preview)])
        self.label.setText("A · המקור  /  B · הקאבר — המעבר שומר על אותה נקודת זמן")

    def set_variants(self, variants: list[tuple[str, str]], *, blind: bool = False) -> None:
        self.stop()
        while self.buttons.count():
            item = self.buttons.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        self._players = []
        self._outputs = []
        self._buttons = []
        self._labels = [label for label, _path in variants]
        self._blind = blind
        self._active = 0
        for index, (_label, path) in enumerate(variants):
            player = QMediaPlayer(self)
            output = QAudioOutput(self)
            player.setAudioOutput(output)
            output.setVolume(0.8)
            player.setSource(QUrl.fromLocalFile(path))
            player.positionChanged.connect(self._position_changed)
            player.durationChanged.connect(self._duration_changed)
            button = QPushButton(self._button_text(index))
            button.clicked.connect(lambda _checked=False, i=index: self._switch(i))
            self.buttons.addWidget(button)
            self._players.append(player)
            self._outputs.append(output)
            self._buttons.append(button)
        self.label.setText(
            "מצב עיוור — זהות הגרסאות מוסתרת" if blind else "בחר גרסה להשמעה"
        )

    def set_blind(self, blind: bool) -> None:
        self._blind = blind
        for index, button in enumerate(self._buttons):
            button.setText(self._button_text(index))
        self.label.setText(
            "מצב עיוור — זהות הגרסאות מוסתרת" if blind else "זהות הגרסאות גלויה"
        )

    def _button_text(self, index: int) -> str:
        alias = chr(ord("A") + index)
        return alias if self._blind else f"{alias} · {self._labels[index]}"

    def stop(self) -> None:
        for player in self._players:
            player.stop()

    def _switch(self, index: int) -> None:
        if not self._players or not 0 <= index < len(self._players):
            return
        position = self._players[self._active].position()
        for player in self._players:
            player.pause()
        self._active = index
        self._players[index].setPosition(position)
        self._players[index].play()

    def _seek(self, position: int) -> None:
        for player in self._players:
            player.setPosition(position)

    def _position_changed(self, position: int) -> None:
        if self.sender() is self._players[self._active] and not self.position.isSliderDown():
            self.position.setValue(position)

    def _duration_changed(self, duration: int) -> None:
        if self.sender() is self._players[self._active]:
            self.position.setRange(0, duration)
