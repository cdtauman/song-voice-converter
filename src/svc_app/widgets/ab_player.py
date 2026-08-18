"""Synchronous A/B player for the original and generated preview."""

from __future__ import annotations

from PySide6.QtCore import Qt, QUrl
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSlider, QVBoxLayout, QWidget


class ABPlayer(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._players = [QMediaPlayer(self), QMediaPlayer(self)]
        self._outputs = [QAudioOutput(self), QAudioOutput(self)]
        for player, output in zip(self._players, self._outputs, strict=True):
            player.setAudioOutput(output)
            output.setVolume(0.8)
        self._active = 0
        layout = QVBoxLayout(self)
        self.label = QLabel("בחר מקור להשמעה")
        self.label.setProperty("muted", True)
        layout.addWidget(self.label)
        buttons = QHBoxLayout()
        self.original_button = QPushButton("A · המקור")
        self.preview_button = QPushButton("B · התוצאה")
        self.original_button.clicked.connect(lambda: self._switch(0))
        self.preview_button.clicked.connect(lambda: self._switch(1))
        buttons.addWidget(self.original_button)
        buttons.addWidget(self.preview_button)
        layout.addLayout(buttons)
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0, 0)
        self.position.sliderMoved.connect(self._seek)
        layout.addWidget(self.position)
        for player in self._players:
            player.positionChanged.connect(self._position_changed)
            player.durationChanged.connect(self._duration_changed)

    def set_sources(self, original: str, preview: str) -> None:
        self._players[0].setSource(QUrl.fromLocalFile(original))
        self._players[1].setSource(QUrl.fromLocalFile(preview))
        self.label.setText("A · המקור  /  B · הקאבר — המעבר שומר על אותה נקודת זמן")

    def stop(self) -> None:
        for player in self._players:
            player.stop()

    def _switch(self, index: int) -> None:
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
