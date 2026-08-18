"""Saved-project browser backed by the Phase-7 atomic project store."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from svc_app.engine_client import EngineClient


class ProjectsScreen(QWidget):
    open_requested = Signal(dict)

    def __init__(self, client: EngineClient) -> None:
        super().__init__()
        self.client = client
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 24, 32, 24)
        title = QLabel("פרויקטים")
        title.setObjectName("Title")
        subtitle = QLabel("בחירות שנשמרו אוטומטית — אפשר לחזור ולהפיק שוב")
        subtitle.setObjectName("Subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)
        self.container = QWidget()
        self.rows = QVBoxLayout(self.container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.container)
        root.addWidget(scroll, 1)

    def refresh(self) -> None:
        try:
            projects = self.client.list_projects()
        except Exception:
            projects = []
        while self.rows.count():
            item = self.rows.takeAt(0)
            widget = item.widget() if item is not None else None
            if widget is not None:
                widget.deleteLater()
        if not projects:
            empty = QLabel("אין עדיין פרויקטים. הבחירה הראשונה תישמר כאן אוטומטית.")
            empty.setProperty("muted", True)
            self.rows.addWidget(empty)
        for project in projects:
            card = QFrame()
            card.setProperty("card", True)
            layout = QHBoxLayout(card)
            text = QVBoxLayout()
            name = QLabel(str(project.get("name") or "פרויקט ללא שם"))
            name.setStyleSheet("font-size: 17px; font-weight: 700;")
            updated = QLabel(str(project.get("updated_at") or ""))
            updated.setProperty("muted", True)
            text.addWidget(name)
            text.addWidget(updated)
            open_button = QPushButton("פתח")
            project_id = str(project.get("project_id") or "")
            open_button.clicked.connect(
                lambda _checked=False, selected_id=project_id: self._open(selected_id)
            )
            layout.addLayout(text, 1)
            layout.addWidget(open_button)
            self.rows.addWidget(card)
        self.rows.addStretch()

    def _open(self, project_id: str) -> None:
        project = self.client.load_project(project_id)
        self.open_requested.emit(dict(project.get("data") or {}))
