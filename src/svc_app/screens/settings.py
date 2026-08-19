"""Application defaults, including the Phase-10 advanced-mode preference."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from svc_app.engine_client import EngineClient


class SettingsScreen(QWidget):
    theme_changed = Signal(str)
    advanced_changed = Signal(bool)
    update_requested = Signal()

    def __init__(self, client: EngineClient) -> None:
        super().__init__()
        self.client = client
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        title = QLabel("הגדרות")
        title.setObjectName("Title")
        subtitle = QLabel("ברירות מחדל בטוחות למצב הפשוט")
        subtitle.setObjectName("Subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        form = QFormLayout()
        form.setVerticalSpacing(16)
        self.quality = QComboBox()
        self.quality.addItem("מהירה", "fast")
        self.quality.addItem("מאוזנת", "balanced")
        self.quality.addItem("מרבית", "max")
        self.theme = QComboBox()
        self.theme.addItem("לפי Windows", "system")
        self.theme.addItem("בהיר", "light")
        self.theme.addItem("כהה", "dark")
        self.target_lufs = QDoubleSpinBox()
        self.target_lufs.setRange(-70.0, -5.0)
        self.target_lufs.setSuffix(" LUFS")
        self.target_lufs.setDecimals(1)
        self.cache_gb = QDoubleSpinBox()
        self.cache_gb.setRange(0.0, 500.0)
        self.cache_gb.setSuffix(" GB")
        self.downloads = QCheckBox("אפשר הורדה אוטומטית של מודלים חסרים")
        self.updates = QCheckBox("בדוק עדכונים אוטומטית")
        self.advanced = QCheckBox("הצג כברירת מחדל את בקרות המצב המתקדם")
        form.addRow("איכות ברירת מחדל:", self.quality)
        form.addRow("ערכת צבעים:", self.theme)
        form.addRow("עוצמת יעד:", self.target_lufs)
        form.addRow("מכסת מטמון:", self.cache_gb)
        form.addRow("הורדות:", self.downloads)
        form.addRow("עדכונים:", self.updates)
        form.addRow("מצב מתקדם:", self.advanced)
        root.addLayout(form)
        root.addStretch()
        save = QPushButton("שמור הגדרות")
        save.setProperty("primary", True)
        save.clicked.connect(self.save)
        root.addWidget(save)
        check_update = QPushButton("בדוק עדכון עכשיו")
        check_update.clicked.connect(self.update_requested)
        root.addWidget(check_update)

    def refresh(self) -> None:
        settings = self.client.settings()
        self.quality.setCurrentIndex(
            max(0, self.quality.findData(settings.get("quality", "balanced")))
        )
        self.target_lufs.setValue(float(settings.get("target_lufs", -14.0)))
        self.cache_gb.setValue(float(settings.get("keep_cache_gb", 20.0)))
        self.downloads.setChecked(bool(settings.get("allow_model_downloads", True)))
        self.updates.setChecked(bool(settings.get("check_updates", True)))
        self.theme.setCurrentIndex(max(0, self.theme.findData(settings.get("theme", "system"))))
        self.advanced.setChecked(bool(settings.get("advanced_mode", False)))

    def save(self) -> None:
        self.client.save_settings(
            quality=str(self.quality.currentData()),
            target_lufs=self.target_lufs.value(),
            keep_cache_gb=self.cache_gb.value(),
            allow_model_downloads=self.downloads.isChecked(),
            check_updates=self.updates.isChecked(),
            theme=str(self.theme.currentData()),
            advanced_mode=self.advanced.isChecked(),
        )
        self.theme_changed.emit(str(self.theme.currentData()))
        self.advanced_changed.emit(self.advanced.isChecked())
