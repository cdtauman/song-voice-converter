"""Colour, typography and QSS for the Hebrew desktop interface."""

from __future__ import annotations

from enum import StrEnum

from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication


class Theme(StrEnum):
    SYSTEM = "system"
    LIGHT = "light"
    DARK = "dark"


_COLORS = {
    Theme.LIGHT: {
        "bg": "#F6F7FB",
        "surface": "#FFFFFF",
        "surface_alt": "#EEF1F8",
        "text": "#172033",
        "muted": "#667085",
        "primary": "#6750E8",
        "primary_hover": "#563ED7",
        "border": "#D8DDEA",
        "success": "#147D64",
        "danger": "#B42318",
    },
    Theme.DARK: {
        "bg": "#111421",
        "surface": "#1B2031",
        "surface_alt": "#252B3E",
        "text": "#F4F6FC",
        "muted": "#ABB3C7",
        "primary": "#9B8AFB",
        "primary_hover": "#B2A5FF",
        "border": "#343B52",
        "success": "#5DD6B1",
        "danger": "#FF8A80",
    },
}


def resolved_theme(app: QApplication, theme: Theme) -> Theme:
    if theme is not Theme.SYSTEM:
        return theme
    return (
        Theme.DARK
        if app.palette().color(QPalette.ColorRole.Window).lightness() < 128
        else Theme.LIGHT
    )


def apply_theme(app: QApplication, theme: Theme = Theme.SYSTEM) -> Theme:
    selected = resolved_theme(app, theme)
    c = _COLORS[selected]
    app.setStyle("Fusion")
    app.setStyleSheet(
        f"""
        * {{ font-family: "Segoe UI", "Arial"; font-size: 14px; color: {c["text"]}; }}
        QMainWindow, QWidget#AppRoot {{ background: {c["bg"]}; }}
        QFrame#Sidebar, QFrame[card="true"], QDialog {{
            background: {c["surface"]}; border: 1px solid {c["border"]}; border-radius: 14px;
        }}
        QLabel#Title {{ font-size: 28px; font-weight: 700; }}
        QLabel#Subtitle, QLabel[muted="true"] {{ color: {c["muted"]}; }}
        QLabel#StepPill {{
            background: {c["surface_alt"]}; color: {c["primary"]}; padding: 7px 12px;
            border-radius: 12px; font-weight: 600;
        }}
        QPushButton {{
            min-height: 38px; padding: 0 18px; background: {c["surface_alt"]};
            border: 1px solid {c["border"]}; border-radius: 10px;
        }}
        QPushButton:hover {{ border-color: {c["primary"]}; }}
        QPushButton:disabled {{ color: {c["muted"]}; background: {c["bg"]}; }}
        QPushButton[primary="true"] {{
            color: white; background: {c["primary"]}; border-color: {c["primary"]};
            font-weight: 700;
        }}
        QPushButton[primary="true"]:hover {{ background: {c["primary_hover"]}; }}
        QPushButton[danger="true"] {{ color: {c["danger"]}; }}
        QPushButton[nav="true"] {{ text-align: right; border: 0; background: transparent; }}
        QPushButton[nav="true"]:checked {{ background: {c["surface_alt"]}; color: {c["primary"]}; }}
        QLineEdit, QComboBox, QDoubleSpinBox {{
            min-height: 38px; padding: 0 10px; background: {c["surface"]};
            border: 1px solid {c["border"]}; border-radius: 9px;
        }}
        QProgressBar {{
            min-height: 14px; background: {c["surface_alt"]}; border: 0; border-radius: 7px;
            text-align: center;
        }}
        QProgressBar::chunk {{ background: {c["primary"]}; border-radius: 7px; }}
        QSlider::groove:horizontal {{ height: 5px; background: {c["border"]}; border-radius: 2px; }}
        QSlider::handle:horizontal {{
            width: 16px; margin: -6px 0; background: {c["primary"]}; border-radius: 8px;
        }}
        QScrollArea {{ border: 0; background: transparent; }}
        QFrame[selected="true"] {{
            border: 2px solid {c["primary"]}; background: {c["surface_alt"]};
        }}
        """
    )
    return selected
