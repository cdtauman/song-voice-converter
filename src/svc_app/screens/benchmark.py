"""Phase-10 result viewer: metrics table and sample-synchronous blind A/B."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from svc_app.widgets import ABPlayer


class BenchmarkScreen(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._folder: Path | None = None
        self._manifest: dict[str, Any] = {}
        self._variants: list[tuple[str, str]] = []
        self._row_variant_ids: list[str] = []
        self._labels: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        header = QHBoxLayout()
        titles = QVBoxLayout()
        title = QLabel("מעבדת השוואה")
        title.setObjectName("Title")
        subtitle = QLabel("טבלת מדדים והאזנת A/B עיוורת באותה נקודת זמן")
        subtitle.setObjectName("Subtitle")
        titles.addWidget(title)
        titles.addWidget(subtitle)
        open_button = QPushButton("פתח תיקיית תוצאות")
        open_button.setProperty("primary", True)
        open_button.clicked.connect(self._choose_folder)
        header.addLayout(titles, 1)
        header.addWidget(open_button)
        root.addLayout(header)

        card = QFrame()
        card.setProperty("card", True)
        card_layout = QVBoxLayout(card)
        controls = QHBoxLayout()
        self.blind = QCheckBox("מצב עיוור")
        self.blind.setChecked(True)
        self.blind.toggled.connect(self._blind_changed)
        reveal = QPushButton("חשוף זהויות")
        reveal.clicked.connect(lambda: self.blind.setChecked(False))
        controls.addWidget(self.blind)
        controls.addWidget(reveal)
        controls.addStretch()
        card_layout.addLayout(controls)
        self.player = ABPlayer()
        card_layout.addWidget(self.player)
        root.addWidget(card)

        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["גרסה", "חזרה", "מצב", "שניות", "RAM MB", "VRAM MB", "הגדרות"]
        )
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table, 1)
        self.status = QLabel("בחר תיקייה שנוצרה על־ידי svc-bench.")
        self.status.setProperty("muted", True)
        root.addWidget(self.status)

    def _choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "בחר תיקיית תוצאות benchmark")
        if selected:
            try:
                self.load_results(Path(selected))
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                QMessageBox.warning(self, "אי אפשר לפתוח את התוצאות", str(exc))

    def load_results(self, folder: Path | str) -> None:
        root = Path(folder)
        manifest_path = root / "manifest.json"
        csv_path = root / "results.csv"
        if not manifest_path.is_file() or not csv_path.is_file():
            raise ValueError("התיקייה חייבת להכיל manifest.json ו־results.csv")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("schema", -1)) != 1:
            raise ValueError("גרסת manifest אינה נתמכת")
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.table.setRowCount(len(rows))
        labels = {
            str(item["id"]): str(item.get("label") or item["id"])
            for item in manifest.get("variants", [])
            if isinstance(item, dict) and item.get("id")
        }
        self._row_variant_ids = [row.get("variant_id", "") for row in rows]
        self._labels = labels
        for row_index, row in enumerate(rows):
            values = [
                labels.get(row.get("variant_id", ""), row.get("variant_id", "")),
                row.get("repetition", ""),
                row.get("status", ""),
                row.get("seconds", ""),
                row.get("peak_ram_mb", ""),
                row.get("peak_vram_mb", ""),
                row.get("settings", ""),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        first_audio: dict[str, str] = {}
        for row in rows:
            audio = row.get("audio") or ""
            variant_id = row.get("variant_id") or ""
            if audio and variant_id not in first_audio:
                first_audio[variant_id] = str((root / audio).resolve())
        blind_map = manifest.get("blind_map") or {}
        self._aliases = (
            {str(variant_id): str(alias) for alias, variant_id in blind_map.items()}
            if isinstance(blind_map, dict)
            else {}
        )
        order = [str(value) for value in blind_map.values()] if isinstance(blind_map, dict) else []
        order.extend(item for item in first_audio if item not in order)
        self._variants = [
            (labels.get(item, item), first_audio[item]) for item in order if item in first_audio
        ]
        self._folder = root
        self._manifest = manifest
        self.player.set_variants(self._variants, blind=self.blind.isChecked())
        self._blind_changed(self.blind.isChecked())
        self.status.setText(
            f"נטענו {len(rows)} ריצות ו־{len(self._variants)} גרסאות · "
            f"{manifest.get('name', '')}"
        )

    def _blind_changed(self, checked: bool) -> None:
        self.player.set_blind(checked)
        for row, variant_id in enumerate(self._row_variant_ids):
            label = (
                self._aliases.get(variant_id, "גרסה")
                if checked
                else self._labels.get(variant_id, variant_id)
            )
            self.table.setItem(row, 0, QTableWidgetItem(label))
