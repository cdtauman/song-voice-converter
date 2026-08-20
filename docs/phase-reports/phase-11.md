# דוח סיום — Phase 11: אריזה, installer ועדכונים

**תאריך:** 19 באוגוסט 2026  
**ענף:** `phase/11-packaging`  
**בסיס:** `phase/10-benchmark` (`e144712`)  
**סטטוס:** ✅ המימוש, הבנייה והשערים המקומיים הושלמו; קבלת clean-machine מלאה
ושני שירותים חיצוניים נשארו חסומים כמפורט בסעיף 6.

---

## 1. מה נבנה

| אזור | תוצאה |
|---|---|
| runtime | build מסוג PyInstaller `onedir` על Python 3.11.15 הנעול של הפרויקט, ללא תלות ב־Python/CUDA מערכתיים |
| הפעלה | launcher יציב מפעיל עדכון ממתין ואז את `SongVoice.exe`; מנוע ו־training worker רצים מתוך אותו executable בלי חלון console |
| installer | Inno Setup 6 בעברית, התקנה per-user, קיצורי דרך, הסרה של האפליקציה ושל `%LOCALAPPDATA%\SongVoice` |
| הפעלה ראשונה | אשף RTL מזהה CPU/CUDA/XPU, מאמת את Torch הארוז ואת ffmpeg, מוריד רק שלושה מודלי ליבה מתירי הפצה ומאומתים, ואז כותב marker אטומי |
| offline | installer מלא כולל את ארבעת קובצי המודל הנדרשים לקאבר ראשון ומעתיק אותם לנתוני המשתמש |
| עדכונים | manifest ב־HTTPS, בדיקת SemVer/גודל/SHA-256, staging, מניעת Zip Slip, החלפה אטומית, backup ו־rollback אוטומטי גם אם כתיבת receipt נכשלת |
| CI | workflow ל־Windows 2025 על tag `v*`: lint/types/tests, גבולות ורישוי, build, online/offline installers, clean-machine install/smoke/uninstall, checksums ו־release artifacts |
| mirror | כלי audit/upload מסרב למודל ללא רישיון הפצה או SHA; כתיבה ל־Hugging Face דורשת במפורש `HF_TOKEN` ו־`SONGVOICE_MODEL_REPO` |

`env-bench`,‏ Seed-VC,‏ DDSP-SVC ו־`benchmark/` מוחרגים מפורשות מה־spec ומה־installer.

---

## 2. ארטיפקטים שנבנו בפועל

| ארטיפקט | גודל | SHA-256 |
|---|---:|---|
| `SongVoice-0.1.0-Setup.exe` | 1,676,784,496 bytes | `2548283f88821c9d20c4aae7e6aaae4e91de8493942935f18a82d00b55fdd5d1` |
| `SongVoice-0.1.0-Offline-Setup.exe` | 2,865,519,090 bytes | `c85bd2aa3b7eee532f953d23e2665195a6ed9cec8564dad8d91f00a06a0eb67e` |

הארטיפקטים נמצאים מקומית תחת `packaging/output/` ומוחרגים מ־Git. שני לוגי Inno
מאשרים טעינת `Hebrew.isl` ו־`Successful compile`. תיקיית ה־distribution כוללת
גם `SHA256SUMS.txt` לכל קובץ שנארז.

---

## 3. שלמות, רישוי ובידוד

- ffmpeg הוא BtbN `win64-lgpl-shared`, asset ‏`519628215` מתוך release
  `372453964`, בגודל 67,704,853 bytes וב־SHA-256
  `df45248120867ad9b2a17ba80633c642dff8cbabd5156ab400c87f8dc3d3f4ca`.
  הבנייה נכשלת אם מופיעים `--enable-gpl`/`--enable-nonfree` או אם חסרים
  `alimiter`,‏ `acompressor`,‏ `loudnorm`.
- כל מודל offline נבדק לפי גודל ו־SHA לפני האריזה. במהלך הבנייה התגלה שמנעול
  `vocals_mel_band_roformer.yaml` הישן היה שגוי. אותו קובץ בן 944 bytes התקבל
  משני מקורות בלתי תלויים וננעל ל־
  `b958b29c8f7195f0d86bee6759a33980db675c4ecaf2fcaa80fa125828e6cd38`.
- mirror כולל רק תשעה קבצים בעלי רישיון הפצה מאומת ומנעול SHA. מודלים פרטיים,
  ללא רישיון או benchmark-only אינם מועלים ואינם נארזים.
- build/runtime אינם כוללים checkout, dependency או import מ־`env-bench`.
- updater אינו מחליף את ה־launcher הפעיל. אם העתקה או כתיבת transaction receipt
  נכשלות, קבצים שהוחלפו משוחזרים וקבצים חדשים נמחקים; ה־pending נשמר לניסיון הבא.

---

## 4. אימותים שרצו בפועל

| אימות | תוצאה |
|---|---|
| `packaging/fetch-dependencies.ps1` | ✅ SHA/גודל, LGPL ושלושת מסנני ffmpeg אומתו |
| `packaging/build-offline-models.py` | ✅ 4 קבצים, 1,283,800,025 bytes, manifest מאומת |
| PyInstaller דרך runtime Python 3.11 הנעול | ✅ `SongVoice.exe` ו־launcher נבנו; dist בגודל 3,855,849,712 bytes |
| `packaging/validate-dist.ps1` | ✅ GUI smoke ללא terminal, LGPL, checksums והיעדר `env-bench` |
| Inno Setup online + offline | ✅ שני מתקינים עבריים נבנו וה־hashes חושבו מחדש |
| התקנה מבודדת מקומית + `SongVoice.exe --smoke-test` | ✅ עבדה בלי Python/CUDA מערכתיים |
| `tools/mirror_models.py --audit` | ✅ 9/9 קבצים זכאים נעולים; אין כתיבה חיצונית ללא credentials |
| `python -m pytest` | ✅ כל הסוויטה עברה; 6 דילוגי חומרה קיימים |
| `python -m ruff check src tests tools env-bench` | ✅ עבר |
| `python -m mypy src` | ✅ עבר |
| `tools/check_model_catalogue.py` + `tools/audit_constraints_licenses.py` | ✅ קטלוג ורישוי הליבה עברו |
| `env-bench/verify_isolation.py` | ✅ סביבת benchmark אינה נכנסת לליבת הייצור או ל־build |

---

## 5. Definition of Done

| דרישה | סטטוס |
|---|---|
| Download → Install → Open בלי Python/CUDA מותקנים | ✅ installer ו־runtime עצמאי נבנו; התקנה מבודדת ו־GUI smoke עברו |
| installer עברי, קיצורי דרך והסרה נקייה | ✅ מימוש ובניית Inno הושלמו; clean-runner gate קיים ב־CI |
| הפעלה ראשונה: GPU → runtime → מודלים → health | ✅ האשף מזהה backend, מאמת Torch ארוז, מוריד/קולט מודלים ומריץ health |
| עדכון בתוך האפליקציה עם checksum ו־rollback | ✅ בדיקות stage/apply/explicit rollback/failure rollback/Zip Slip עברו |
| installer מלא offline | ✅ נבנה בפועל עם כל מודלי הליבה המאומתים |
| clean Windows 11 → קאבר ראשון | ⚠️ מסלול ההתקנה וה־smoke מוכנים; קאבר שמע אמיתי חסום בחומרים החיצוניים בסעיף 6 |
| הסרה מנקה הכול על מכונה נקייה | ⚠️ סקריפט CI מאמת זאת; לא הורץ מקומית כדי לא למחוק נתוני SongVoice קיימים ובהיעדר Windows Sandbox |
| שיקוף פרטי של כל מודל חוקי להפצה | ⚠️ כלי ההעברה ושער הרישוי מוכנים; הכתיבה עצמה חסומה בסעיף 6 |

---

## 6. חסמים חיצוניים שנשארו לסוף

לא נעשתה פנייה למשתמש במהלך השלב.

1. **חתימת קוד:** אין תעודת Authenticode או secrets. `packaging/sign.ps1` וה־CI
   תומכים ב־PFX, סיסמה ו־timestamp ומאמתים את החתימה; המתקינים המקומיים אינם חתומים.
2. **Hugging Face פרטי:** אין `HF_TOKEN` ואין `SONGVOICE_MODEL_REPO`. בנוסף, לפי
   המדיניות אין לשקף checkpoints שאין להם רישיון הפצה מפורש. לכן בוצע audit מלא
   ולא בוצעה כתיבה חיצונית.
3. **קבלת clean-machine וקאבר ראשון:** אין Windows 11 VM/Sandbox זמין ואין בריפו
   שיר, מודל קול יעד והסכמה מורשים. ה־workflow מריץ install/smoke/uninstall על runner
   נקי, אבל קאבר אמיתי דורש לספק את שלושת חומרי הקבלה האלה ולתייג build.

---

## 7. חריגות והחלטות ממוקדות

- Torch אינו יורד לאחר זיהוי GPU: ה־runtime הנעול `2.9.1+xpu` נארז מראש ומאומת
  בהפעלה הראשונה. כך online ו־offline עובדים בלי Python/CUDA מערכתיים ובלי החלפת
  סביבת Python לאחר ההתקנה; CUDA/XPU נבחרים בזמן ריצה ו־CPU נשאר fallback. זו
  סטייה מילולית מסדר החצים בתוכנית, לא החסרה של שלב runtime.
- `onedir` נבחר במקום executable יחיד משום ש־Torch/Intel/Qt הם runtime גדול,
  דינמי ורב־DLL; Inno מספק למשתמש installer יחיד.
- ה־launcher נשאר מחוץ לחבילת העדכון כדי לא להחליף executable פעיל. תוכן האפליקציה
  מתעדכן מאחוריו באופן טרנזקציוני.
- לא התחיל שום פריט של Phase 12. `_MOVED.md` לא נקרא, לא שונה ולא צורף.
