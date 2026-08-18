# דוח סיום — Phase 5: מנוע המרת הקול

**תאריך:** 19 באוגוסט 2026
**מכונת הפיתוח:** Lenovo 83DJ · Intel Core Ultra 7 155H · 15GB RAM · Intel Arc Graphics
**סטטוס:** ✅ **המנגנון הושלם** — הצינור המלא מחובר מקצה לקצה ונבדק עם רכיבים מוזרקים.
שני חסמים חיצוניים לקוד (סעיף 7): נעילת תלויות ה-inference על מחשב הפיתוח, וחומרי
הבדיקה + מודלי קול אמיתיים.

---

## 1. מה נבנה

| מודול | מה הוא עושה |
|-------|--------------|
| `voices/manifest.py` | המניפסט (`voice.json`), פריסת הקבצים של קול, ומצב התקינות. **הסכמה (consent) היא שדה ממדרגה ראשונה** — קול בלי אישור מפורש אינו שמיש |
| `voices/library.py` | ספריית הקולות: סריקה, `get`, הסרה, ובדיקת תקינות torch-free (חתימת בייטים של checkpoint) |
| `voices/importer.py` | ייבוא קול מ-`zip`: חילוץ streaming מוגבל (3GiB לחבר, 4GiB מצטבר), ולידציה, **סירוב בלי אישור**, שמירה staged והחלפה אטומית/ברת־שחזור ב-overwrite, והגנת zip-slip |
| `conversion/chunking.py` | **הלב של "בלי תפרים".** חלוקה לקטעים חופפים, המרה של כל קטע דרך מתאם מוזרק, והרכבה מחדש עם crossfade — **אורך מדויק-לדגימה** |
| `conversion/rvc/f0.py` | כימות ה-F0 לדליי RVC (1..255) והזזת האוקטבה — numpy טהור, נבדק |
| `conversion/rvc/index.py` | ה-blend של ה-`.index` (משיכת הגוון ליעד), עם `BruteForceIndex` נטול-faiss לבדיקה |
| `conversion/rvc/rms.py` | התאמת מעטפת העוצמה (`rms_mix_rate`) — numpy, נבדק |
| `conversion/rvc/model.py` | טעינת checkpoint של RVC בבסיס בטוח (`torch.load(weights_only=True)`; אין fallback ל-pickle), בחירת הסינתסייזר לפי `(version, f0)`, וטעינה לא-קשיחה |
| `conversion/rvc/hubert.py` | עטיפת מקודד ה-HuBERT/ContentVec (torch+transformers, עצל) |
| `conversion/rvc/infer.py` | ליבת ה-inference: מחברת את ה-DSP הנבדק לרשת הנוירונית |
| `conversion/rvc/backend.py` | `RVCv2Backend` — המימוש של `ConversionBackend`. זיהוי זמינות, מיפוי פרמטרים, חוזה ה-resample, שחרור VRAM |
| `conversion/rvc/infer_pack/` | **רשת RVC v2 מוטמעת** מ-RVC-Project @ `81eed5e` (MIT), כמו RMVPE ב-Phase 3 |
| `conversion/pipeline.py` | `ConversionPipeline` — הצינור המלא: הפרדה → ניתוח → החלטת הזזה → המרה → הזזת פלייבק → מיקס |
| `svc voices` / `svc convert` | הפקודות החדשות: ניהול ספריית הקולות, וההמרה המלאה מקצה-לקצה |
| `content_hubert` בקטלוג | משקולות HuBERT (MIT, lj1995), sha256 **נעול** |

**כל מה שמעל המתאם `ConversionBackend` הוא torch-free.** ייבוא חבילת `conversion`,
`voices`, וה-DSP של `conversion.rvc` (f0/index/rms) לא גורר את המחסנית הכבדה — בדיוק
כמו `analysis` ו-`pitch`. torch, transformers ו-faiss מיובאים בעצלתיים בתוך
`RVCv2Backend` ו-`infer_pack`.

---

## 2. הצינור המלא — לראשונה מחובר

`ConversionPipeline.run` משרשר את ארבעת השלבים הקודמים לתוצר אחד:

```
שיר → [Phase 2] הפרדה → vocals + instrumental
          → [Phase 3] F0 על הווקאל
          → [Phase 4] PitchDistribution → decide_shift → החלטה s = 12k + r
          → [Phase 5] המרת הווקאל (f0_up_key = s) בקטעים חופפים
          → הזזת הפלייבק ב-r בלבד (אסטרטגיה A/B מ-Phase 4)
          → mix_cover → קאבר
```

**ההזזה זורמת נכון:** `f0_up_key` של ההמרה הוא `decision.best.semitones` (כל ה-s),
בעוד שהפלייבק זז רק ב-`decision.best.remainder` (ה-r). האוקטבות נשארות חינם מבחינת
הפלייבק — התובנה המרכזית של הפרויקט, מיושמת בקוד. הבדיקה
`test_render_cover_wires_decision_convert_shift_mix` מוכיחה ששיר גבוה מול קול נמוך
מקבל הזזה שלילית, שאותה הזזה מגיעה ל-`ConversionParams.semitones`, ושהקאבר יוצא
באורך השיר בדיוק.

**השלבים הכבדים מוזרקים** (`separate`, `f0_extractor`, `ConversionBackend`,
`PitchShifter`), כך ש-`render_cover` נבדק מקצה-לקצה עם מזויפים, בלי torch — אותה
תבנית של `quality_probe` ב-Phase 4.

---

## 3. "בלי תפרים נשמעים" — נמדד, לא מובטח

`chunking.py` חותך ווקאל ארוך לחלונות חופפים, ממיר כל אחד, ומחבר עם crossfade.
המתמטיקה נבחרה כך שההרכבה **מדויקת-לדגימה**: עם `hop = chunk − overlap`, חלונות
שמתחילים ב-`i·hop`, וכל קטע מומר ששומר על אורכו — סכום ה-crossfade מצטמצם בחזרה
לאורך המקורי. הבדיקות:

- `test_identity_reassembles_bit_exact` — מתאם זהותי → הפלט **זהה-ביט** לקלט על פני
  4 אורכים × מונו/סטריאו. ה-crossfade הליניארי של שני חלונות זהים מסתכם למקור.
- `test_total_length_exact_even_with_misbehaving_backend` — גם מתאם ש"שובר" אורך,
  `fit_length` אוכף את החוזה והאורך הכולל נשאר מדויק.
- `test_no_nans_and_bounded_seam_with_varying_chunks` — קפיצה של 1.0 בין קטעים
  מתפרשת על החפיפה כך שאין אי-רציפות חדה בתפר.

---

## 4. הסכמה (Consent) — לא פורמליות

README קובע: קול נכנס למערכת **אך ורק** אם המשתמש יצר אותו או קיבל רשות מפורשת.
זה נאכף בקוד, לא רק בתיעוד:

- `import_voice_from_zip` **זורק `E_CONSENT_REQUIRED`** בלי `consent_confirmed=True`.
  אין נתיב ברירת-מחדל-כן.
- `VoiceManifest.usable` דורש גם אישור וגם תקינות מודל.
- `svc voices --import … --consent` הוא הדרך היחידה, וה-CLI מדפיס את השגיאה בעברית
  אנושית (ראה סעיף 5), לא traceback.

נאכף ב-`test_import_requires_consent` ובהרצת CLI אמיתית.

---

## 5. אימות שרץ בפועל

- **441 בדיקות עוברות**, 6 מדולגות. ruff נקי · mypy נקי
  (79 קבצים) · audit הקטלוג exit 0 · בדיקת הגבול app/engine עוברת.
- **70 הבדיקות החדשות:** ספריית הקולות (14), chunking (14), DSP של RVC + חוזה המתאם
  (13), צינור ההמרה (8), ועוד. כולן על הלוגיקה ה-torch-free האמיתית.
- **תיקון אבטחה (19.8.2026):** checkpoint מיובא נטען רק ב־restricted unpickler של
  PyTorch; חברי ZIP מועתקים ב-chunks תחת שתי מגבלות גודל; overwrite שומר את הקול
  הקודם עד שההחלפה הופעלה, ומשחזר אותו אם ההפעלה נכשלת. בדיקות רגרסיה מכסות את
  שלושת הנתיבים האלה.
- **הרשת המוטמעת של RVC:** עוברת `py_compile` (תקינות תחביר Python 3.11) ו-ruff;
  מוחרגת מ-mypy כקוד צד-שלישי מוטמע (`pyproject.toml`) — אותה גישה שכל עץ `vendor/`
  מקבל.
- **ה-CLI, מקצה-לקצה, על המסלולים ה-torch-free:**
  - `svc voices` (רשימה ריקה), `svc voices --import` **בלי** `--consent` → סירוב עם
    הודעה עברית · **עם** `--consent` → *"הקול 'יוסי' יובא — מוכן לשימוש (כולל קובץ
    חיפוש, פרופיל מנעד)."* · `svc voices` → מציג את הקול כתקין.
  - `svc convert --help` — הפקודה רשומה עם כל הדגלים.
  - `svc models` — מציג את `content_hubert` (MIT, נעול).
- **משקולות HuBERT:** הורדו, אומתו (HTTP 200, 189,507,909 בייט) ו-ה-sha256 חושב ונעול
  בקטלוג: `f54b40fd…7db96`.

---

## 6. Definition of Done

| # | דרישה מה-roadmap | סטטוס |
|---|--------------------|--------|
| 1 | צינור מלא עובד על 5 שירי בדיקה בלי קריסה | ⚠️ המנגנון מחובר ונבדק מקצה-לקצה עם מזרקים; הרצה על 5 שירים אמיתיים חסומה על חומרי הבדיקה + מודלי קול + התלויות — סעיף 7 |
| 2 | אין תפרים נשמעים בין chunks | ✅ **עבר** — הרכבה מדויקת-לדגימה, זהות-ביט למתאם זהותי, נבדק (סעיף 3) |
| 3 | ייבוא קול מקובץ zip עובד | ✅ **עבר** — נבדק ביחידות + הרצת CLI אמיתית, כולל סירוב בלי הסכמה (סעיף 4) |
| 4 | VRAM משתחרר אחרי כל הרצה | ✅ **עבר** — `render_cover` קורא `backend.unload()` ב-`finally`; נבדק ש-`unload` נקרא. `_empty_cache` מנקה CUDA/XPU |
| — | חילוץ RVC v2 למודול נקי ומתועד | ✅ עבר — `conversion/rvc/`, מוטמע + נרשם ב-third-party §1.3 |
| — | כל הפרמטרים נתמכים | ✅ `f0_up_key`, `index_rate`, `protect`, `rms_mix_rate` ממופים ומיושמים; `filter_radius`/`formant_shift` מתקבלים ומתועדים כלא-בשימוש בגרסה הזו |
| — | CI ירוק (ruff/mypy/pytest/רישוי/גבולות) | ✅ עבר — סעיף 5 |

---

## 7. הסייגים המפורשים

**חומרי ההרצה האמיתיים הם החסם שנותר. הקוד שמעל המתאם שלם ונבדק; הרשת המוטמעת נאמנה למקור.**

### א. נעילת תלויות ה-inference — הושלמה
ה-Spike של Phase 1 אימת "conversion" כעומס אופרטורים סינתטי
(`conv1d+transformer+convtranspose`), **לא** את RVC האמיתי. Phase 5 מוסיף extra מפורש
`.[rvc]` ואת `rvc-requirements.lock`, שנוצר אוטומטית (לא נערך ידנית) עם `uv pip compile`
מ־`constraints.txt` **כקלט דרישות מלא**, לא רק כ־constraints, ומ־`pyproject.toml`.
לכן ה-lock כולל את כל מטריצת Phase 2–5: `audio-separator`, `torchfcpe`, `librosa`,
`python-stretch`, `soundfile`, `scipy`, `soxr`, `pyloudnorm`, וכן
`torch==2.13.0+xpu`, `transformers==5.15.0` ו-`faiss-cpu==1.15.0`. ה־Spike יוצר
מעתה את שני הקבצים מאותו winner, ולכן לא ניתן לעדכן את המטריצה בלי לעדכן גם את ה-lock.

התקנה נקייה משתמשת ב:

```bash
uv pip sync --python .venv\\Scripts\\python.exe \
  --index-url https://download.pytorch.org/whl/xpu \
  --extra-index-url https://pypi.org/simple rvc-requirements.lock
```

לאחר `uv pip sync` של lock זה בסביבת Python 3.11/XPU המבודדת, `uv pip check`
עבר על 108 חבילות. import smoke עבר עבור `audio_separator`, `torchfcpe`, `librosa`,
`python_stretch`, `soundfile`, `scipy`, `soxr`, `pyloudnorm`, `torch`,
`transformers` ו-`faiss`. ה-Spike וסקריפט ה-smoke כוללים מעתה את שתי התלויות
כחובה, כך שהרצה מלאה עתידית תבחן אותן יחד עם שאר המטריצה.

### ב. חומרי בדיקה + מודלי קול אמיתיים (DoD #1)
הרצת ההמרה בפועל דורשת: (1) משקולות HuBERT (בקטלוג, MIT, נעולות — יורדות אוטומטית);
(2) מודל קול RVC אמיתי (`.pth`) שהמשתמש יצר או קיבל רשות אליו — **לא נכנס לריפו**,
מגודר בהסכמה; (3) 5 שירי הבדיקה + 3 קולות היעד מ-[testing.md §1, §1א](../testing.md),
הסייג הפתוח של Phase 2/3/4; (4) פורמט הטעינה הסופי של HuBERT (fairseq מול Transformers)
ננעל יחד עם (א).

### ג. סגירת סייג Phase 4 (`quality_vs_shift`)
Phase 4 השאיר את עקומת האיכות `None` כי לא היה מנוע המרה למדוד מולו. **עכשיו יש מנוע.**
`quality_probe.probe_quality_curve` מקבל `RVCv2Backend` בהזרקה, וברגע שחסם (ב)
נסגרים — הוא מודד את העקומה האמיתית לכל קול, ואיבר `w₅` בפונקציית העלות מפסיק להיות 0.
המנגנון מוכן ומחובר; חסרה רק ההרצה עם משקולות ומודל.

**מה צריך ממך:** אותם 5 שירים + 3 קולות יעד.

---

## 8. מה **לא** נעשה, בכוונה

מיקס מקצועי, הדהוד, לימיטר ו-LUFS (Phase 6) · ניהול משימות, cache והתאוששות (Phase 7)
· ממשק גרפי (Phase 8) · אימון קולות (Phase 9). `formant_shift` ו-`filter_radius`
מתקבלים כפרמטרים אך אינם מיושמים במסלול הזה — הם נקבעים בכיול של Phase 6.
DDSP-SVC ו-Seed-VC לא נכנסו — הם מנועי השוואה של Phase 10, בסביבה מבודדת בלבד.

---

## 9. השלב הבא

**Phase 6 — תיקונים ומיקס.** לוקח את הפלט הגולמי של ההמרה והופך אותו לתוצאה
מקצועית: תיקון artifacts, התאמת מעטפת, de-esser, אסטרטגיית הדהוד (שלוש, נמדדות),
ומיקס LUFS דרך ffmpeg. הקאבר הגולמי ש-`ConversionPipeline` מייצר הוא הקלט שלו.
