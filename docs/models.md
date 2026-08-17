# רשימת רכיבים, מודלים וספריות — כולל audit רישוי

**עודכן:** 17 באוגוסט 2026 — Phase 2.
**מדיניות:** הליבה נבנית כאילו התוכנה תופץ. רכיבי GPL מותרים **רק** בכלי פיתוח ובנצ'מרק שלא נארזים.

> ## ⚠️ עדכון Phase 2 (17.8.2026) — קרא את זה לפני הטבלה שמתחת
>
> ה-audit שנדרש כאן בוצע במלואו, **והתוצאה שינתה שתי החלטות.** הפירוט המלא
> ב-[phase-reports/phase-2.md](phase-reports/phase-2.md) §2–§3; התמצית:
>
> **1. `bgkb/bs_polarformer` אינו זמין כמודל.** הריפו מכיל `.yaml` בלבד —
> **אין בו משקולות.** ההחלטה לבחור בו כברירת מחדל (החלטה #3) התבססה על טבלת SDR
> ולא על בדיקה שאפשר להוריד אותו.
>
> **2. ברירת המחדל החדשה טובה יותר:** `KimberleyJSN/melbandroformer` הוא
> **MIT מאומת** וגם בעל ה-SDR הגבוה ביותר במבחן (12.60). אין כאן פשרה בין
> רישוי לאיכות.
>
> **3. חמישה מתוך שבעה checkpoints כבר מוחזרים 404** מריפו ההפצה של UVR.
> לכולם אותרו מראות חיות ב-HuggingFace, וכל כתובת בקטלוג נבדקה בפועל.
> `tools/check_model_catalogue.py --check-urls` מריץ את הבדיקה מחדש.
>
> **4. מודלי הניקוי אינם מתירניים:** DeReverb הוא **GPL-3.0**, DeEcho הוא
> **CC-BY-NC-SA-4.0**. הם פעילים לשימוש פרטי ומכובים אוטומטית בבנייה להפצה.
>
> **מקור האמת התפעולי הוא כעת `src/svc_engine/data/models.json`** — הקטלוג
> שהקוד קורא, עם רישיון, מקור אימות, גודל ותאריך לכל רשומה. הטבלה שלמטה היא
> התיעוד ההיסטורי של Phase 0.

---

## 1. טבלת Audit רישוי — מודלים

**שיטת האימות:** שאילתה ישירה ל-`huggingface.co/api/models/<id>` ולשדה `license` ב-GitHub API,
בתאריך **16.8.2026**. עמודת "מקור האימות" מציינת מאיפה הגיע הנתון.

| Checkpoint / משקולות | תפקיד | URL | רישיון מאומת | מקור האימות | קוד ה-inference | רישיון הקוד | סטטוס |
|---|---|---|---|---|---|---|---|
| `bgkb/bs_polarformer` | ~~הפרדה ראשית — ברירת מחדל~~ | [HF](https://huggingface.co/bgkb/bs_polarformer) | MIT | HF API `cardData.license` | MSST / pymss | MIT | ⛔ **בוטל ב-Phase 2 — אין בריפו משקולות, רק `.yaml`** |
| `pcunwa/BS-Roformer-Leap` | הפרדה ראשית — חלופה | [HF](https://huggingface.co/pcunwa/BS-Roformer-Leap) | **אין הצהרה** | HF API — שדה ריק | MSST / pymss | MIT | ⚠️ **פרטי בלבד** |
| `pcunwa/BS-Roformer-Revive` | הפרדה — מועמד | [HF](https://huggingface.co/pcunwa/BS-Roformer-Revive) | **אין הצהרה** | HF API — שדה ריק | MSST | MIT | ⚠️ פרטי בלבד |
| Mel-Band Roformer Kim (vocals) | הפרדה ראשית — **ברירת מחדל חדשה** | [HF](https://huggingface.co/KimberleyJSN/melbandroformer) | **MIT** | HF API `cardData.license`, 17.8.2026 | audio-separator | MIT | ✅ **בליבה** — SDR 12.60, הגבוה במבחן |
| Mel-Band Roformer Kim FT2 | הפרדה — שותף ל-ensemble | [HF](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) | **אין הצהרה** | HF API — שדה ריק, 17.8.2026 | audio-separator | MIT | ⚠️ **פרטי בלבד** |
| BS-Roformer viperx 1297 | הפרדה — קטן ומהיר | UVR model repo | **אין הצהרה** | לא אותר repo מקור, 17.8.2026 | audio-separator | MIT | ⚠️ פרטי בלבד |
| Mel-Band Roformer Karaoke (aufr33/viperx) | ליד ↔ ווקאלים מלווים | UVR model repo | **אין הצהרה** | לא אותר repo מקור, 17.8.2026 | audio-separator | MIT | ⚠️ פרטי בלבד |
| Mel-Band Roformer DeReverb (anvuew) | הסרת הדהוד + שכבת ההדהוד | [HF](https://huggingface.co/anvuew/dereverb_mel_band_roformer) | **GPL-3.0** | HF API, 17.8.2026 | audio-separator | MIT | ⛔ **אסור בהפצה** |
| Mel-Band Roformer DeEcho (Sucial) | הסרת הד | [HF](https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer) | **CC-BY-NC-SA-4.0** | HF API, 17.8.2026 | audio-separator | MIT | ⛔ **אסור בהפצה** (לא-מסחרי) |
| Mel-Band Roformer Denoise (aufr33) | ניקוי רעש | [HF](https://huggingface.co/poiqazwsx/melband-roformer-denoise) | **אין הצהרה** | HF API — שדה ריק, 17.8.2026 | audio-separator | MIT | ⚠️ פרטי בלבד |
| HTDemucs4 FT | פיצול בס/תופים/שאר | [GitHub](https://github.com/facebookresearch/demucs) | **MIT** | GitHub API | demucs | MIT | ✅ בליבה |
| RMVPE (משקולות) | זיהוי F0 — עיקרי | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | HF API `cardData.license`, 18.8.2026 | RVC `infer/rmvpe.py` (מוטמע, ראה third-party §1.3) | MIT | ✅ בליבה — Phase 3 |
| ContentVec / HuBERT (`hubert_base.pt`) | הפרדת תוכן מגוון | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | HF API | [contentvec](https://github.com/auspicious3000/contentvec) | MIT | ✅ בליבה |
| FCPE | זיהוי F0 — מהיר ל-Preview | `torchfcpe` | MIT | PyPI / repo | torchfcpe | MIT | ✅ בליבה |
| `blaise-tk/TITAN` | מודל בסיס לאימון RVC | [HF](https://huggingface.co/blaise-tk/TITAN) | **Apache-2.0** | HF API `cardData.license` | RVC trainer | MIT | ✅ בליבה |
| RVC v2 pretrained (רשמי) | מודל בסיס חלופי | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | HF API | RVC-Project | MIT | ✅ בליבה |
| Whisper small | מדידת WER בבנצ'מרק | [openai/whisper-small](https://huggingface.co/openai/whisper-small) | **Apache-2.0** | HF API | [openai/whisper](https://github.com/openai/whisper) | MIT | 🔬 פיתוח בלבד |
| מודל embedding לדובר | מדידת דמיון בבנצ'מרק | ייבחר ב-Phase 3 | טעון אימות | — | — | — | 🔬 פיתוח בלבד |
| Seed-VC checkpoints | benchmark השוואתי | [Plachtaa/seed-vc](https://github.com/Plachtaa/seed-vc) | **GPL-3.0** | GitHub API | seed-vc | GPL-3.0 | 🔬 **סביבה מבודדת בלבד** |

### מסקנת ה-Audit — כפי שעודכנה ב-Phase 2

> **המסקנה המקורית מתקיימת, דרך מודל אחר.**
> "גרסה הניתנת להפצה אינה דורשת ויתור על איכות" נשאר נכון: מודל ההפרדה
> **MIT (`KimberleyJSN/melbandroformer`, SDR 12.60)** מדורג מעל **כל** המועמדים
> חסרי הרישיון. מודל הבסיס לאימון (TITAN) הוא Apache-2.0, והמשקולות של RMVPE
> ו-HuBERT מגיעות מריפו MIT.
>
> **הסייג שנוסף:** המסקנה חלה על ה**הפרדה**. שלבי ה**ניקוי** (DeReverb, DeEcho,
> Denoise, Karaoke) נשענים כרגע על מודלים שאינם מתירניים, ובגרסה מופצת הם
> יכובו. זו יכולת חסרה בהפצה, לא בעיית איכות בהפרדה עצמה.
>
> **המשימה הפתוחה החדשה:** למצוא מודל DeReverb מתירני, או להסתמך על אסטרטגיה
> B/C ב-Phase 6 (ייצור הד חדש) שאינה דורשת מודל DeReverb בזמן ריצה אצל המשתמש.

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
