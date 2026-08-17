# פנקס קוד צד-שלישי

מרשם של **כל** קוד חיצוני שנכנס ל-SongVoice — בין אם כתלות, כקוד מותאם או כקוד
שהוכנס לריפו.

**חובה לפי [reuse-policy.md](reuse-policy.md):** לכל רשומה — הריפו, הקבצים המדויקים,
ה-commit, והרישיון. בלי זה אי אפשר לתחזק ואי אפשר לוודא ציות לרישיון.

---

## 1. מקורות ייחוס מאושרים

רישיונות אומתו מול GitHub API ב-**17.8.2026**. ה-commit הוא ה-HEAD באותו תאריך,
לצורך שחזוריות.

| מקור | רישיון | HEAD בתאריך הבדיקה |
|------|--------|---------------------|
| [mason369/AI-RVC](https://github.com/mason369/AI-RVC) | MIT | `ecbe4da80d5a13f40ed85257b293a81e1e8b3313` |
| [IAHispano/Applio](https://github.com/IAHispano/Applio) | MIT | `085197e738ce9dd4c0bae1e0a74df5de25b89444` |
| [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) | MIT | `81eed5e8f68b6bed1789f682fe78cdd324495afc` |

---

## 2. קוד שהוכנס לריפו (vendored)

**נכון ל-Phase 1: אין.**

כשיוכנס קוד — כל רשומה תכלול:

| שדה | דוגמה |
|------|--------|
| נתיב אצלנו | `src/svc_engine/conversion/rvc/vendor/` |
| מקור | `RVC-Project/Retrieval-based-Voice-Conversion-WebUI` |
| קבצים מקוריים | `infer/lib/infer_pack/models.py` |
| commit | `81eed5e8…` |
| רישיון | MIT |
| הודעת זכויות יוצרים | נשמרה בראש כל קובץ + `LICENSE-THIRD-PARTY` |
| מה שונה | תיעוד השינויים |
| למה הוכנס ולא נעטף | הנימוק |

**כללים לקוד מוכנס:**

1. הודעת הרישיון והזכויות של המקור **נשמרת בראש כל קובץ**.
2. הקוד נכנס לתיקיית `vendor/` נפרדת — לא מעורבב בקוד שלנו.
3. `ruff` ו-`mypy` **לא** רצים עליו (הוא לא שלנו לתקן).
4. כל שינוי מתועד כאן.
5. **לא נכנס שום קוד GPL/AGPL.**

---

## 3. תלויות ריצה

הרשימה המלאה נמצאת ב-`constraints.txt` (נעול ע"י ה-Compatibility Spike)
ומבוקרת אוטומטית:

```bash
python tools/audit_constraints_licenses.py
```

**תוצאת הבדיקה האחרונה (17.8.2026, 86 חבילות):**

| | |
|---|---|
| הפרות GPL / AGPL | **0** |
| LGPL (מותר בקישור דינמי) | 1 — `soxr` |
| ללא הצהרת רישיון | 1 — `llvmlite` |

---

## 4. מודלים

⚠️ **רישיון MIT על ריפו לא אומר שהמודלים שבתוכו הם MIT.**
מודלים מבוקרים בנפרד — טבלת ה-audit המלאה נמצאת ב-[models.md](models.md).
