"""No raw technical error reaches the interface."""

from __future__ import annotations

_ERRORS = {
    "E_GPU_OOM": ("אין מספיק זיכרון גרפי", "בחר איכות מהירה יותר ונסה שוב."),
    "E_GPU_MISSING": ("לא נמצא מאיץ מתאים", "אפשר לנסות שוב על המעבד דרך ההגדרות."),
    "E_MODEL_MISSING": ("חסר קובץ עיבוד", "אפשר הורדת מודלים בהגדרות ונסה שוב."),
    "E_MODEL_CORRUPT": ("קובץ עיבוד נפגם", "הפעל בדיקת מערכת והורד אותו מחדש."),
    "E_AUDIO_UNSUPPORTED": ("לא הצלחנו לפתוח את השיר", "בחר קובץ WAV, MP3, M4A או FLAC."),
    "E_VOICE_CORRUPT": ("הקול שבחרת אינו מוכן", "הסר אותו מספריית הקולות והוסף מחדש."),
    "E_CONSENT_REQUIRED": ("נדרשת הרשאה לקול", "הוסף את הקול מחדש ואשר שיש לך רשות להשתמש בו."),
    "E_DISK_FULL": ("אין מספיק מקום בדיסק", "פנה מקום ובחר שוב יצירת קאבר."),
    "E_NO_VOCALS": ("לא זיהינו שירה", "נסה שיר אחר או הקלטה שבה השירה ברורה יותר."),
    "E_FFMPEG_MISSING": ("רכיב השמע חסר", "הפעל בדיקת מערכת או התקן את התוכנה מחדש."),
    "E_DOWNLOAD_FAILED": ("ההורדה נעצרה", "בדוק את החיבור ולחץ על נסה שוב."),
    "E_CANCELLED": ("העיבוד בוטל", "אפשר להתחיל מחדש בכל רגע."),
    "E_BACKEND_UNAVAILABLE": ("מנוע העיבוד אינו זמין", "פתח הגדרות והפעל בדיקת מערכת."),
    "E_INTERNAL": ("משהו השתבש", "הפרטים נשמרו ביומן. נסה שוב."),
}


def error_text(code: str, fallback: str = "") -> tuple[str, str]:
    title, action = _ERRORS.get(code, _ERRORS["E_INTERNAL"])
    if fallback and code not in _ERRORS:
        return title, action
    return title, action
