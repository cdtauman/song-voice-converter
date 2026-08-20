"""Built-in Hebrew help for the release build."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QScrollArea, QVBoxLayout, QWidget

from svc_app import __version__


def _section(title: str, body: str) -> QFrame:
    card = QFrame()
    card.setProperty("card", True)
    layout = QVBoxLayout(card)
    layout.setContentsMargins(20, 16, 20, 16)
    heading = QLabel(title)
    heading.setStyleSheet("font-size: 18px; font-weight: 700;")
    text = QLabel(body)
    text.setWordWrap(True)
    text.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    layout.addWidget(heading)
    layout.addWidget(text)
    return card


class HelpScreen(QWidget):
    """Short, actionable help that remains available without internet access."""

    def __init__(self) -> None:
        super().__init__()
        self.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        root = QVBoxLayout(self)
        root.setContentsMargins(32, 28, 32, 28)

        title = QLabel("עזרה")
        title.setObjectName("Title")
        subtitle = QLabel(f"SongVoice {__version__} · מדריך מהיר שעובד גם ללא אינטרנט")
        subtitle.setObjectName("Subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        sections = QVBoxLayout(content)
        sections.setContentsMargins(0, 18, 10, 10)
        sections.setSpacing(12)
        sections.addWidget(
            _section(
                "קאבר ראשון — ארבעה צעדים",
                "1. הוסף קול לספרייה ואשר שיש לך הרשאה להשתמש בו.\n"
                "2. בחר שיר במסך 'קאבר חדש'.\n"
                "3. בחר קול ואיכות, וצור תצוגה מקדימה.\n"
                "4. האזן להשוואה ורק אז צור את השיר המלא.",
            )
        )
        sections.addWidget(
            _section(
                "אם פעולה נעצרה",
                "אפשר לפתוח שוב את SongVoice ולהמשיך מהשלב האחרון שהושלם. "
                "אם הורדת מודל נקטעה, נסה שוב לאחר שהחיבור חזר — ההורדה ממשיכה "
                "מהקובץ החלקי ולא מתחילה מחדש.",
            )
        )
        sections.addWidget(
            _section(
                "פתרון תקלות נפוצות",
                "קובץ שמע לא נפתח: נסה MP3, WAV או M4A תקין.\n"
                "אין מקום בדיסק: פנה מקום והפעל שוב; פלט חלקי אינו מחליף תוצאה תקינה.\n"
                "לא זוהתה שירה: נסה קובץ אחר או מיקס שבו השירה ברורה יותר.\n"
                "חסר מודל: ודא שהורדות אוטומטיות פעילות בהגדרות ובדוק את החיבור.",
            )
        )
        sections.addWidget(
            _section(
                "פרטיות והרשאה",
                "עיבוד השמע מתבצע מקומית. השתמש רק בקול שיצרת בעצמך או שקיבלת "
                "רשות מפורשת להשתמש בו. קובצי השמע והקולות אינם נשלחים לענן.",
            )
        )
        sections.addWidget(
            _section(
                "מידע לאבחון",
                "הפעל 'svc doctor --json' ממסוף רק אם התמיכה ביקשה זאת. "
                "הפקודה בודקת רכיבי מערכת ואינה מעלה קבצים.",
            )
        )
        sections.addStretch()
        scroll.setWidget(content)
        root.addWidget(scroll, 1)


__all__ = ["HelpScreen"]
