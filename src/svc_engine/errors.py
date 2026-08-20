"""Engine errors and their Hebrew user-facing messages.

Rule from docs/architecture.md section 8: every error the user can see must say
(1) what happened and (2) what to do about it. Never a stack trace.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["ErrorCode", "UserMessage", "EngineError", "message_for"]


class ErrorCode(StrEnum):
    GPU_OOM = "E_GPU_OOM"
    GPU_MISSING = "E_GPU_MISSING"
    MODEL_MISSING = "E_MODEL_MISSING"
    MODEL_CORRUPT = "E_MODEL_CORRUPT"
    AUDIO_UNSUPPORTED = "E_AUDIO_UNSUPPORTED"
    VOICE_CORRUPT = "E_VOICE_CORRUPT"
    CONSENT_REQUIRED = "E_CONSENT_REQUIRED"
    DISK_FULL = "E_DISK_FULL"
    NO_VOCALS = "E_NO_VOCALS"
    FFMPEG_MISSING = "E_FFMPEG_MISSING"
    DOWNLOAD_FAILED = "E_DOWNLOAD_FAILED"
    CANCELLED = "E_CANCELLED"
    BACKEND_UNAVAILABLE = "E_BACKEND_UNAVAILABLE"
    INTERNAL = "E_INTERNAL"


@dataclass(frozen=True)
class UserMessage:
    """What the user reads. `what` explains, `action` tells them what to do."""

    what: str
    action: str

    def render(self) -> str:
        return f"{self.what} {self.action}".strip()


_MESSAGES: dict[ErrorCode, UserMessage] = {
    ErrorCode.GPU_OOM: UserMessage(
        "לכרטיס המסך נגמר הזיכרון.",
        "אנחנו מתאימים את ההגדרות אוטומטית וממשיכים.",
    ),
    ErrorCode.GPU_MISSING: UserMessage(
        "לא נמצא כרטיס מסך של NVIDIA.",
        "אפשר להמשיך על המעבד, אבל זה יהיה איטי משמעותית.",
    ),
    ErrorCode.MODEL_MISSING: UserMessage(
        "חסר קובץ שנחוץ לפעולה הזו.",
        "נוריד אותו עכשיו — זה ייקח כדקה.",
    ),
    ErrorCode.MODEL_CORRUPT: UserMessage(
        "אחד הקבצים שהורדנו פגום.",
        "נמחק אותו ונוריד מחדש.",
    ),
    ErrorCode.AUDIO_UNSUPPORTED: UserMessage(
        "לא הצלחנו לפתוח את קובץ השמע.",
        "נסה קובץ MP3, WAV או M4A.",
    ),
    ErrorCode.VOICE_CORRUPT: UserMessage(
        "קובץ הקול פגום או לא שלם.",
        "אפשר להסיר את הקול ולהוסיף אותו מחדש.",
    ),
    ErrorCode.CONSENT_REQUIRED: UserMessage(
        "אפשר להוסיף אך ורק קול שיצרת בעצמך או שיש לך רשות מפורשת להשתמש בו.",
        "אשר שיש לך הרשאה לקול הזה ונסה שוב.",
    ),
    ErrorCode.DISK_FULL: UserMessage(
        "אין מספיק מקום פנוי בדיסק.",
        "פנה מקום ונסה שוב.",
    ),
    ErrorCode.NO_VOCALS: UserMessage(
        "לא זיהינו שירה בקובץ.",
        "אולי זה שיר אינסטרומנטלי? נסה קובץ אחר.",
    ),
    ErrorCode.FFMPEG_MISSING: UserMessage(
        "רכיב עיבוד השמע חסר.",
        "נסה להתקין את התוכנה מחדש.",
    ),
    ErrorCode.DOWNLOAD_FAILED: UserMessage(
        "ההורדה נכשלה.",
        "בדוק את חיבור האינטרנט ולחץ על 'נסה שוב'.",
    ),
    ErrorCode.CANCELLED: UserMessage(
        "הפעולה בוטלה.",
        "אפשר להתחיל מחדש בכל רגע.",
    ),
    ErrorCode.BACKEND_UNAVAILABLE: UserMessage(
        "אחד ממנועי העיבוד אינו זמין.",
        "הרץ את בדיקת המערכת כדי לראות מה חסר.",
    ),
    ErrorCode.INTERNAL: UserMessage(
        "משהו השתבש אצלנו.",
        "הפרטים נשמרו ביומן. נסה שוב.",
    ),
}


def message_for(code: ErrorCode) -> UserMessage:
    """Never raises. An unmapped code falls back to the generic internal message."""
    return _MESSAGES.get(code, _MESSAGES[ErrorCode.INTERNAL])


class EngineError(Exception):
    """Any failure the engine reports upward. Carries a Hebrew message."""

    def __init__(self, code: ErrorCode, detail: str = "") -> None:
        self.code = code
        self.detail = detail
        self.user_message = message_for(code)
        super().__init__(f"{code.value}: {detail}" if detail else code.value)
