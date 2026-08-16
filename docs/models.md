# רשימת רכיבים, מודלים וספריות — כולל audit רישוי

**עודכן:** 16 באוגוסט 2026 — סבב ביקורת ראשון.
**מדיניות:** הליבה נבנית כאילו התוכנה תופץ. רכיבי GPL מותרים **רק** בכלי פיתוח ובנצ'מרק שלא נארזים.

---

## 1. 🆕 טבלת Audit רישוי — מודלים

**שיטת האימות:** שאילתה ישירה ל-`huggingface.co/api/models/<id>` ולשדה `license` ב-GitHub API,
בתאריך **16.8.2026**. עמודת "מקור האימות" מציינת מאיפה הגיע הנתון.

| Checkpoint / משקולות | תפקיד | URL | רישיון מאומת | מקור האימות | קוד ה-inference | רישיון הקוד | סטטוס |
|---|---|---|---|---|---|---|---|
| `bgkb/bs_polarformer` | הפרדה ראשית — **ברירת מחדל** | [HF](https://huggingface.co/bgkb/bs_polarformer) | **MIT** | HF API `cardData.license` | MSST / pymss | MIT | ✅ **בליבה** |
| `pcunwa/BS-Roformer-Leap` | הפרדה ראשית — חלופה | [HF](https://huggingface.co/pcunwa/BS-Roformer-Leap) | **אין הצהרה** | HF API — שדה ריק | MSST / pymss | MIT | ⚠️ **פרטי בלבד** |
| `pcunwa/BS-Roformer-Revive` | הפרדה — מועמד | [HF](https://huggingface.co/pcunwa/BS-Roformer-Revive) | **אין הצהרה** | HF API — שדה ריק | MSST | MIT | ⚠️ פרטי בלבד |
| Mel-Band Roformer Kim (vocals) | הפרדה — מהיר ויציב | מופץ דרך UVR / audio-separator | **טעון אימות** | לא אותר repo מקור חד-משמעי | audio-separator | MIT | 🔍 **לבדוק ב-Phase 2** |
| Mel-Band Roformer Karaoke | ליד ↔ ווקאלים מלווים | דרך audio-separator | **טעון אימות** | — | audio-separator | MIT | 🔍 לבדוק |
| Mel-Band Roformer DeReverb/DeEcho | הסרת הדהוד | דרך audio-separator | **טעון אימות** | — | audio-separator | MIT | 🔍 לבדוק |
| Mel-Band Roformer Denoise | ניקוי רעש | דרך audio-separator | **טעון אימות** | — | audio-separator | MIT | 🔍 לבדוק |
| HTDemucs4 FT | פיצול בס/תופים/שאר | [GitHub](https://github.com/facebookresearch/demucs) | **MIT** | GitHub API | demucs | MIT | ✅ בליבה |
| RMVPE (משקולות) | זיהוי F0 — עיקרי | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | HF API | [Dream-High/RMVPE](https://github.com/Dream-High/RMVPE) | Apache-2.0 | ✅ בליבה |
| ContentVec / HuBERT (`hubert_base.pt`) | הפרדת תוכן מגוון | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | HF API | [contentvec](https://github.com/auspicious3000/contentvec) | MIT | ✅ בליבה |
| FCPE | זיהוי F0 — מהיר ל-Preview | `torchfcpe` | MIT | PyPI / repo | torchfcpe | MIT | ✅ בליבה |
| `blaise-tk/TITAN` | מודל בסיס לאימון RVC | [HF](https://huggingface.co/blaise-tk/TITAN) | **Apache-2.0** | HF API `cardData.license` | RVC trainer | MIT | ✅ בליבה |
| RVC v2 pretrained (רשמי) | מודל בסיס חלופי | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | HF API | RVC-Project | MIT | ✅ בליבה |
| Whisper small | מדידת WER בבנצ'מרק | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | **Apache-2.0** | HF API | [openai/whisper](https://github.com/openai/whisper) | MIT | 🔬 פיתוח בלבד |
| מודל embedding לדובר | מדידת דמיון בבנצ'מרק | ייבחר ב-Phase 3 | טעון אימות | — | — | — | 🔬 פיתוח בלבד |
| Seed-VC checkpoints | benchmark השוואתי | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | **GPL-3.0** | GitHub API | seed-vc | GPL-3.0 | 🔬 **סביבה מבודדת בלבד** |

### מסקנת ה-Audit

> **גרסה הניתנת להפצה אינה דורשת ויתור על איכות.**
> מודל ההפרדה MIT (`bs_polarformer`, SDR 11.76) מדורג **מעל** המודל חסר הרישיון
> (`BS-Roformer-Leap`, SDR 11.74). מודל הבסיס לאימון (TITAN) הוא Apache-2.0.
> המשקולות של RMVPE ו-HuBERT מגיעות מריפו MIT.
>
> **הפעולה היחידה שנותרה:** לאמת את ארבעת מודלי ה-Mel-Band Roformer שמגיעים דרך
> UVR/audio-separator. זו משימה ב-Phase 2, לא חסם.

### חבילות התקנה

- **התקנה ראשונה (הכרחית):** מודל הפרדה אחד + RMVPE + FCPE + ContentVec ≈ **600MB**
- **התקנה מלאה (כל המודלים):** ≈ **3.5GB**

---

## 2. ⚠️ גרסאות תלויות — לא ננעלות עדיין

**כל טבלאות הספריות למטה מציינות גרסאות שנצפו ב-16.8.2026 — הן נקודת פתיחה לניסוי, לא נעילה.**
המטריצה הסופית נקבעת ב-**Compatibility Spike של Phase 1** על החומרה שלך, ונשמרת
בקובץ `constraints.txt` נעול בריפו.

### 2.1 ליבה

| ספריה | נצפה | תפקיד | רישיון | בליבה? |
|-------|------|-------|--------|--------|
| `torch` | ייקבע ב-Spike | מנוע ה-AI | BSD | ✅ |
| `numpy`, `scipy` | — | חישוב, פילטרים | BSD | ✅ |
| `soundfile` | — | קריאה/כתיבה WAV | BSD | ✅ |
| `soxr` | 1.1.0 | דגימה מחדש VHQ | LGPL | ✅ |
| `librosa` | 1.0.0 | ניתוח, chroma, זיהוי סולם | ISC | ✅ |
| `pyloudnorm` | 0.2.0 | מדידת LUFS | MIT | ✅ |
| ffmpeg (build LGPL) | — | קליטה, ייצוא, **לימיטר/קומפרסור/loudnorm** | LGPL | ✅ |

### 2.2 הפרדה

| ספריה | נצפה | רישיון | בליבה? |
|-------|------|--------|--------|
| `audio-separator` | 0.44.5 | MIT | ✅ |
| `pymss` | 2.1.3 | MIT | ✅ |

### 2.3 המרת קול

| רכיב | מקור | רישיון | בליבה? |
|------|------|--------|--------|
| קוד inference של RVC v2 | חילוץ מ-RVC-Project / Applio | MIT | ✅ |
| `torchfcpe` | 0.0.4 | MIT | ✅ |

### 2.4 🆕 הזזת גובה — החלטה מעודכנת

| ספריה | נצפה | איכות | רישיון | wheels ל-Windows? | סטטוס |
|-------|------|-------|--------|--------------------|-------|
| **`python-stretch`** (עוטף Signalsmith Stretch) | **0.3.1** | מצוינת | **MIT** | ✅ **כן** — `cp311-win_amd64` ואחרים | ✅ **מועמד ברירת מחדל, נבדק ב-Spike** |
| `pyrubberband` + Rubber Band CLI | 0.4.0 | מצוינת | **GPL-2.0** | דורש התקנה נפרדת | ⛔ **הוצא מהליבה** — גיבוי פיתוח בלבד |
| SoundTouch | — | בינונית | LGPL | — | ➖ לא נדרש |

> **תיקון מהסבב הקודם:** קבעתי בטעות ש-Signalsmith Stretch אינו זמין כחבילת Python
> ודורש בנייה מ-C++. **אומת ב-PyPI:** `python-stretch` 0.3.1 מספקת wheels מוכנים
> ל-Windows x64 עבור CPython 3.8–3.12 (כולל 3.11), ברישיון MIT.
> הריפו: [gregogiudici/python-stretch](https://github.com/gregogiudici/python-stretch).

### 2.5 ממשק ואריזה

| ספריה | נצפה | תפקיד | רישיון | בליבה? |
|-------|------|-------|--------|--------|
| `PySide6` | 6.11.1 | ממשק, RTL נטיבי | LGPL-3.0 | ✅ (קישור דינמי) |
| `uv` | — | ניהול תלויות | MIT/Apache | 🔧 פיתוח |
| `PyInstaller` | — | אריזה ל-exe | GPL + חריגה לשימוש מסחרי | 🔧 כלי בנייה |
| `Inno Setup` | — | installer | חינמי, מותר מסחרית | 🔧 כלי בנייה |

### 2.6 פונטים

| פונט | תפקיד | רישיון |
|------|-------|--------|
| **Rubik** / **Assistant** | טקסט ראשי בעברית | OFL — מותר לארוז |
| **Heebo** | כותרות (אופציונלי) | OFL |

---

## 3. מה **לא** ניכנס אליו, ולמה

| רכיב | רישיון | למה לא | מה במקום |
|------|--------|--------|----------|
| `pedalboard` (Spotify) | GPL-3.0 | מדביק | **פילטרים של ffmpeg** (LGPL) |
| `matchering` | GPL-3.0 | מדביק, לא הכרחי ל-MVP | — |
| `praat-parselmouth` | GPL | מדביק | `formant_shift` המובנה בפורקים של RVC |
| Rubber Band | GPL-2.0 | מדביק | **`python-stretch`** (MIT) |
| קוד של `MSST-WebUI` | AGPL-3.0 | הכי מדביק | MSST (MIT) כמקור לקונפיגים |
| `Seed-VC` בליבה | GPL-3.0 | מדביק + בארכיון | RVC. Seed-VC רק כ-benchmark מבודד |
| `HQ-SVC` | — | דורש Linux/WSL | — |
| Gradio / WebUI | — | לא רוצים דפדפן או שרת מקומי | PySide6 |

> **🆕 שינוי מהסבב הקודם:** קודם כתבתי "נממש EQ/קומפרסור/לימיטר בעצמנו ב-numpy".
> **זו הייתה החלטה גרועה** — לימיטר שקוף באמת (true-peak, oversampling, lookahead)
> קשה בהרבה ממה שזה נשמע. **אנחנו כבר אורזים ffmpeg**, ויש בו `alimiter`, `acompressor`
> ו-`loudnorm` ברישיון LGPL. EQ בלבד יבוצע עם פילטרים סטנדרטיים מ-`scipy.signal`.

---

## 4. שיקוף המודלים (Mirroring) — משימה חובה

**הבעיה:** סקירה מיולי 2026 מצאה שרק 37 מתוך 89 המודלים בקטלוג של `audio-separator`
עדיין ניתנים להורדה. מודלי קהילה נעלמים.

**הפתרון:** ריפו HuggingFace **פרטי** משלנו עם עותק של כל מודל שאנחנו תלויים בו,
עם `sha256` רשום ב-`models.json`. ההורדה מנסה קודם את המראה שלנו, ואז את המקור.

**⚠️ הערת רישוי:** שיקוף של checkpoint **ללא רישיון מוצהר** לריפו פרטי הוא לשימוש
עצמי. הכללה בגרסה מופצת תדרוש אישור מפורש מהיוצר. זו הסיבה ש-`bs_polarformer` (MIT)
הוא ברירת המחדל ולא `BS-Roformer-Leap`.

מבנה ה-mirroring נבנה ב-Phase 2; המילוי בפועל ב-Phase 11.
