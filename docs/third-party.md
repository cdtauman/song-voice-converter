# מרשם קוד ומודלים חיצוניים

**נוצר:** 17 באוגוסט 2026 — Phase 2, השלב הראשון שבו נעשה שימוש חוזר בפועל.

המסמך הזה ממלא את חובה 3 ב-[reuse-policy.md](reuse-policy.md#רישוי--חובות-לפני-כל-שימוש-חוזר):
**לתעד את הריפו, הקבצים המדויקים והרישיון של כל דבר חיצוני שנשען עליו.**

**כלל:** אין שורה כאן = אין שימוש. אין רישיון מאומת = אין שורה.

---

## 1. קוד חיצוני

### 1.1 ספריות שנצרכות כתלות (`pip`) — לא מועתקות

זו הצורה המועדפת: **שימוש ישיר**, הדרגה הראשונה בסולם ההחלטה.

| חבילה | גרסה נעולה | רישיון | מה נלקח | דרגה בסולם |
|-------|-------------|--------|----------|-------------|
| [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) | 0.44.5 | MIT | inference של BS-Roformer, Mel-Band Roformer, MDX23C, MDX, Demucs | **3 — עטיפה במתאם שלנו** (`AudioSeparatorBackend`) |
| [`pymss`](https://github.com/pymss-project/pymss) | 2.1.3 | MIT | inference + מרשם מודלים ומראה עצמאיים | **3 — עטיפה במתאם שלנו** (`PymssBackend`) |
| [`python-stretch`](https://github.com/gregogiudici/python-stretch) | 0.3.1 | MIT | הזזת גובה (Signalsmith Stretch) | 1 — שימוש ישיר (Phase 4) |
| [`torchfcpe`](https://pypi.org/project/torchfcpe/) | 0.0.4 | MIT | זיהוי F0 מהיר | 1 — שימוש ישיר (Phase 3) |
| `ffmpeg` | build חיצוני | LGPL | קליטה, ייצוא, `loudnorm`/`alimiter`/`acompressor` | 1 — קישור דינמי, נקרא כתהליך |

### 1.3 קוד שהועתק לריפו (דרגה 4 — הכנסת תת-קבוצה)

| מה | ריפו + commit | קובץ מקור | היעד אצלנו | רישיון |
|----|----------------|-----------|-------------|--------|
| ארכיטקטורת RMVPE + inference | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) @ `81eed5e` | `infer/rmvpe.py` | `src/svc_engine/analysis/rmvpe_model.py` | **MIT** |

**למה הועתק ולא נצרך כתלות:** ל-RMVPE אין חבילת `pip` עצמאית ויציבה; מחלקות הרשת
(`E2E`, `DeepUnet`, `MelSpectrogram` וכו') חייבות להתאים בדיוק לשמות ולצורות
שבהן אומן `rmvpe.pt`, אחרת `load_state_dict` נכשל. **מה שהוסר מהמקור:** מסלול
ה-fp16, ענף ה-DirectML/ONNX, לכידת CUDA-graph, והתלות ב-`configs.config` —
כולם קשרו את הקוד לאפליקציה המארחת. הארכיטקטורה עצמה הועתקה מילה-במילה.

טענת "אף שורת קוד לא הועתקה" מ-Phase 2 כבר אינה נכונה החל מ-Phase 3: זו ההעתקה
הראשונה, והיא היחידה עד כה. דרגה 4 נוספת צפויה ב-Phase 5 (RVC inference).

### 1.2 התאמות שנעשו מעל הספריות, ולמה

הן חיות במתאמים שלנו, לא בקוד החיצוני:

| מה | למה | איפה |
|----|-----|------|
| הורדות מודלים משלנו | ל-audio-separator אין checksum, אין resume ואין retry. על קובץ של 900MB זה ההבדל בין "עובד" לבין checkpoint חצי שנכתב בשקט | `resources/download.py` |
| מסלול Intel XPU | `setup_torch_device` מכיר CUDA / MPS / DirectML בלבד ונופל בשקט למעבד על Arc | `AudioSeparatorBackend._apply_device` |
| ביטול נרמול העוצמה | ברירת המחדל מנרמלת כל שכבה לשיא. אחרי זה סכום השכבות אינו המיקס המקורי | `AudioSeparatorBackend._build_separator` |
| כתיבה ב-float | ברירת המחדל כותבת 16-bit. השכבות עוברות עוד כמה עיבודים | אותו מקום |
| תרגום `overlap` | ב-Roformer השדה הוא אורך צעד בשניות, במקומות אחרים מחלק. ראה [phase-2.md §4](phase-reports/phase-2.md) | `AudioSeparatorBackend._apply_overlap` |

---

## 2. משקולות מודלים

**כל שורה אומתה מול `huggingface.co/api/models/<id>` ב-17.8.2026** (שורת ה-RMVPE
נוספה ואומתה ב-18.8.2026), וכל כתובת נבדקה בפועל. מקור האמת התפעולי הוא
[`src/svc_engine/data/models.json`](../src/svc_engine/data/models.json);
הטבלה כאן היא הסיכום לקריאה אנושית.

| מזהה בקטלוג | מקור | רישיון | הפצה? |
|--------------|------|--------|--------|
| `sep_melband_kim` | [KimberleyJSN/melbandroformer](https://huggingface.co/KimberleyJSN/melbandroformer) | **MIT** | ✅ כן |
| `f0_rmvpe` | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) | **MIT** | ✅ כן |
| `sep_melband_kim_ft2` | [pcunwa/Kim-Mel-Band-Roformer-FT](https://huggingface.co/pcunwa/Kim-Mel-Band-Roformer-FT) | אין הצהרה | ⚠️ פרטי בלבד |
| `sep_bs_roformer_1297` | UVR model repo | אין הצהרה | ⚠️ פרטי בלבד |
| `karaoke_aufr33_viperx` | UVR model repo | אין הצהרה | ⚠️ פרטי בלבד |
| `dereverb_anvuew` | [anvuew/dereverb_mel_band_roformer](https://huggingface.co/anvuew/dereverb_mel_band_roformer) | **GPL-3.0** | ⛔ לא |
| `deecho_sucial_v2` | [Sucial/Dereverb-Echo_Mel_Band_Roformer](https://huggingface.co/Sucial/Dereverb-Echo_Mel_Band_Roformer) | **CC-BY-NC-SA-4.0** | ⛔ לא |
| `denoise_aufr33` | [poiqazwsx/melband-roformer-denoise](https://huggingface.co/poiqazwsx/melband-roformer-denoise) | אין הצהרה | ⚠️ פרטי בלבד |

**קובצי ה-`.yaml`** של המודלים מגיעים מריפו ההפצה של
[`python-audio-separator`](https://github.com/nomadkaraoke/python-audio-separator/releases/tag/model-configs) (MIT).

### איך זה נאכף

`ModelSpec.license.is_redistributable` הוא המקור היחיד לאמת בקוד.
`allow_private_models=False` מכבה בדיוק את השורות שאינן ✅ ומדווח בעברית
למה שלב דילג. `tools/check_model_catalogue.py` רץ ב-CI ונכשל אם נוספה
רשומה בלי audit רישוי.

---

## 3. מקורות ייחוס שנקראו אך לא נלקח מהם קוד

לפי [reuse-policy.md](reuse-policy.md), לפני כל רכיב נבדק מה כבר קיים.

| מקור | מה נבדק ב-Phase 2 | התוצאה |
|------|---------------------|---------|
| [mason369/AI-RVC](https://github.com/mason369/AI-RVC) (MIT) | סדר צינור ההפרדה, DeReverb, ניהול מודלים | הסדר אומץ כרעיון; הקוד לא — הוא עוטף את אותה `audio-separator` שאנחנו עוטפים ישירות |
| [IAHispano/Applio](https://github.com/IAHispano/Applio) (MIT) | ניהול והורדת מודלים | לא נדרש כאן; רלוונטי ל-Phase 9 (אימון) |
| [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT) | קוד ה-RMVPE (`infer/rmvpe.py`) | **נלקח ב-Phase 3** — ראה §1.3. קוד ה-RVC inference עצמו עדיין ל-Phase 5 |

---

## 4. מה ייכנס לכאן בשלבים הבאים

| שלב | מה צפוי | דרגה צפויה בסולם |
|------|----------|--------------------|
| Phase 3 | ✅ **נעשה** — משקולות RMVPE (§2) + קוד RMVPE (§1.3) + `torchfcpe` (§1.1) | 1 + 4 |
| Phase 5 | קוד ה-inference של RVC v2, ו-ContentVec/HuBERT | **4 — הכנסת תת-קבוצה.** מחייב נתיבי קבצים ו-commit מדויקים כאן |
| Phase 9 | מנגנון האימון של Applio, מודל בסיס TITAN (Apache-2.0) | 3–4 |
| Phase 10 | Seed-VC (GPL-3.0) | **סביבה מבודדת בלבד — לעולם לא בליבה** |
