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
| [`PySide6`](https://doc.qt.io/qtforpython-6/) | 6.11.1 | LGPL-3.0 (קישור דינמי) | ממשק Windows, RTL, drag/drop ו־Qt Multimedia | 1 — שימוש ישיר (Phase 8) |
| [`noisereduce`](https://github.com/timsainb/noisereduce) | 3.0.3 | MIT | תלות runtime של שלב ה-preprocess ב-Applio; SongVoice מבצע את הניקוי לפניו | 1 — שימוש ישיר (Phase 9) |
| [`tensorboard`](https://github.com/tensorflow/tensorboard) | 2.21.0 | Apache-2.0 | כתיבת מדדי האימון של Applio | 1 — שימוש ישיר (Phase 9) |
| `ffmpeg` | build חיצוני | LGPL | קליטה, ייצוא, `loudnorm`/`alimiter`/`acompressor` | 1 — קישור דינמי, נקרא כתהליך |

### 1.2 עטיפת מנגנון האימון (דרגה 3)

| מה | ריפו + commit | נקודות הכניסה שנצרכות | צורת שילוב | רישיון |
|----|----------------|-------------------------|------------|--------|
| אימון RVC, checkpoints ו-index | [IAHispano/Applio](https://github.com/IAHispano/Applio) @ `085197e738ce9dd4c0bae1e0a74df5de25b89444` | `rvc/train/preprocess/preprocess.py`, `rvc/train/extract/extract.py`, `rvc/train/train.py`, `rvc/train/process/extract_index.py` | ארכיון מקור של commit נעול יורד לנתוני האפליקציה, מאומת ב-SHA-256 `648c6322…9807`, ומופעל כתהליך חיצוני; אף קובץ Applio לא הועתק לריפו | **MIT** |

Applio נבחר כי הוא מקור הייחוס שנקבע מראש ב-[reuse-policy.md](reuse-policy.md) והוא
כבר מממש preprocess, חילוץ RMVPE/ContentVec, חידוש מ-checkpoint, export ו-index.
SongVoice מוסיף מסביבו את חוזה ההסכמה, בדיקת האיכות, ניקוי החומר, session עמיד,
התקדמות/ETA, פרסום אטומי לספריית הקולות וחישוב פרופיל המנעד. המאמן רץ על CUDA
כשזמין במסלול Applio; במטריצה הנוכחית Intel XPU משתמש בגיבוי CPU, משום שהמאמן
של Applio אינו מצהיר תמיכת XPU. לא שונתה ארכיטקטורת המודל ולא הוכנס קוד אימון עצמי.

### 1.3 קוד שהועתק לריפו (דרגה 4 — הכנסת תת-קבוצה)

| מה | ריפו + commit | קובץ מקור | היעד אצלנו | רישיון |
|----|----------------|-----------|-------------|--------|
| ארכיטקטורת RMVPE + inference | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) @ `81eed5e` | `infer/rmvpe.py` | `src/svc_engine/analysis/rmvpe_model.py` | **MIT** |
| רשת ה-inference של RVC v2 (Phase 5) | [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) @ `81eed5e` | `infer/module/{commons,transforms,modules,attentions,models}.py` | `src/svc_engine/conversion/rvc/infer_pack/` | **MIT** |

**למה הועתק ולא נצרך כתלות (RVC v2):** אין חבילת `pip` יציבה ל-inference של RVC;
מחלקות הרשת (`SynthesizerTrnMs256/768NSFsid`, `GeneratorNSF`, `TextEncoder` וכו')
חייבות להתאים בדיוק לשמות ולצורות שבהן אומנו קובצי ה-`.pth` של הקהילה, אחרת
`load_state_dict` נכשל — בדיוק כמו RMVPE. **מה שהשתנה מהמקור:** ה-import-ים כוונו
לחבילה שלנו, ומחלקות ה-Discriminator (אימון בלבד) הוסרו מ-`models.py`. הארכיטקטורה
עצמה מילה-במילה. הקבצים נושאים `# ruff: noqa` ו-`# type: ignore` ומוחרגים מ-mypy
(`pyproject.toml`) — קוד צד-שלישי נבדק במעלה הזרם; אצלנו הם עוברים `py_compile`
ו-ruff. **הלוגיקה שלנו** (כימות F0, blend של ה-index, התאמת RMS, הטוען, המתאם)
חיה מסביבם ב-`conversion/rvc/{f0,index,rms,model,hubert,infer,backend}.py` ונבדקת
ביחידות. **מה שלא רץ כאן:** ה-inference הנוירוני עצמו — דורש torch+transformers+faiss
(טרם ב-`constraints.txt`), משקולות HuBERT, ומודל קול אמיתי. ראה
[phase-reports/phase-5.md](phase-reports/phase-5.md).

**למה הועתק ולא נצרך כתלות:** ל-RMVPE אין חבילת `pip` עצמאית ויציבה; מחלקות הרשת
(`E2E`, `DeepUnet`, `MelSpectrogram` וכו') חייבות להתאים בדיוק לשמות ולצורות
שבהן אומן `rmvpe.pt`, אחרת `load_state_dict` נכשל. **מה שהוסר מהמקור:** מסלול
ה-fp16, ענף ה-DirectML/ONNX, לכידת CUDA-graph, והתלות ב-`configs.config` —
כולם קשרו את הקוד לאפליקציה המארחת. הארכיטקטורה עצמה הועתקה מילה-במילה.

טענת "אף שורת קוד לא הועתקה" מ-Phase 2 כבר אינה נכונה החל מ-Phase 3: RMVPE הייתה
ההעתקה הראשונה. **Phase 5 הוסיף את השנייה** — רשת ה-inference של RVC v2 (השורה
השנייה בטבלה למעלה). שתיהן דרגה 4, שתיהן מ-`81eed5e`, שתיהן מאותה סיבה: המשקולות
נעולות לשמות המחלקות.

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
| `content_hubert` | [lj1995/VoiceConversionWebUI](https://huggingface.co/lj1995/VoiceConversionWebUI) — `hubert_base.pt`, sha256 נעול (Phase 5) | **MIT** | ✅ כן |
| `rvc_training_runtime` | [IAHispano/Applio Resources](https://huggingface.co/IAHispano/Applio) @ `774d3d1f…f293` — ContentVec + RMVPE | **MIT** | ✅ כן |
| `titan_medium_48k` | [blaise-tk/TITAN](https://huggingface.co/blaise-tk/TITAN) @ `cb72bb5b…25f7` — generator + discriminator, 48kHz | **Apache-2.0** | ✅ כן |
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
| [IAHispano/Applio](https://github.com/IAHispano/Applio) (MIT) | ניהול מודלים ואימון | **נעטף ב-Phase 9** — commit וקבצים מדויקים בסעיף 1.2 |
| [RVC-Project](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) (MIT) | קוד ה-RMVPE (`infer/rmvpe.py`) | **נלקח ב-Phase 3** — ראה §1.3. קוד ה-RVC inference עצמו עדיין ל-Phase 5 |

---

## 4. מה ייכנס לכאן בשלבים הבאים

| שלב | מה צפוי | דרגה צפויה בסולם |
|------|----------|--------------------|
| Phase 3 | ✅ **נעשה** — משקולות RMVPE (§2) + קוד RMVPE (§1.3) + `torchfcpe` (§1.1) | 1 + 4 |
| Phase 5 | ✅ **נעשה** — רשת inference של RVC v2 (§1.3) + משקולות HuBERT (§2). התלויות `faiss-cpu`/`transformers` ננעלות ב-re-lock (ראה phase-5.md) | 4 + 1 |
| Phase 9 | ✅ **נעשה** — Applio במתאם תהליך נעול + TITAN Medium 48k + ContentVec/RMVPE נעולים | 3 + 1 |
| Phase 10 | Seed-VC (GPL-3.0) | **סביבה מבודדת בלבד — לעולם לא בליבה** |
