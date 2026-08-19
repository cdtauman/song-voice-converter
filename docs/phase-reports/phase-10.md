# דוח סיום — Phase 10: מצב מתקדם ומעבדת הבנצ'מרק

**תאריך:** 19 באוגוסט 2026

**ענף:** `phase/10-benchmark`
**סטטוס:** ✅ המימוש ושערי הבידוד הושלמו; שתי קבלות שמע אמיתיות חסומות חיצונית בלבד.

---

## 1. מה נבנה

| אזור | תוצאה |
|---|---|
| `tuning/config.py` | 11 בקרות advanced מאומתות: RVC, F0, playback, ambience, loudness ותיקוני PostFX; הסבר עברי ⓘ לכל פרמטר |
| `tuning/optimizer.py` | חיפוש bounded של ארבעה מועמדים על Preview, כולל baseline ידני, score שקוף ושמירת כל התוצאות |
| `benchmark/` | schema TOML/JSON, runner עם Windows Job Object, timeout לכל עץ התהליכים, RAM/VRAM מצטברים, CSV, HTML, audio/logs ו־manifest שחזורי |
| GUI | פאנל מתקדם באשף; כרטיס tuning בתצוגה המקדימה; מסך “מעבדת השוואה” עם טבלה ונגן רב־גרסאות מסונכרן במצב עיוור |
| workflow/RPC | advanced config נכנס לבקשת job, ל־cache key ול־metadata; full cover ממשיך עם המנצח שנבחר ב־Preview |
| `env-bench/` | coordinator מבודד, source pins, bootstrap פר־מנוע, adapter WAV אחיד ושער isolation |

`svc-bench run experiment.toml --out results/...` מסרב לכתוב לתיקייה לא ריקה.
`manifest.json` כולל SHA-256 וגודל קלט, host/Python, seed, פקודות, הגדרות,
רישיון וסביבה לכל variant ואת מיפוי הזהויות העיוור. `report.html` עצמאי ומחליף
גרסה באותה נקודת זמן.

---

## 2. Auto-tuning ומצב מתקדם

ה־baseline הוא ההגדרה הידנית/המומלצת של הקול. שלושת המועמדים הנוספים משנים
באופן שמרני `index_rate`, `protect`, `rms_mix_rate` ו־`filter_radius`; אין חיפוש
פתוח או זמן בלתי מוגבל. המדד מעניש clipping, peak, discontinuity, השטחה וסילוק
יתר של תוכן. הוא מוגדר במפורש כ־proxy ולא כ־MOS.

כל ארבעת קובצי ה־Preview נשמרים ב־StepCache ומוצגים עיוור. אחרי הבחירה האוטומטית
ה־winner config עובר לקאבר המלא עם `auto_tune=false`, ולכן אין הכפלת ארבע ריצות
על השיר המלא. progress מחולק לארבעה מקטעים מונוטוניים.

---

## 3. בידוד ורישוי

- ליבת `src/` אינה מכילה import או dependency של Seed-VC/DDSP-SVC.
- Hatch אורז במפורש רק `src/svc_engine` ו־`src/svc_app`; `env-bench` אינה package.
- coordinator מקומי נוצר ב־`env-bench/.venv` עם Python 3.10.20 ו־`pip check` נקי.
- כל מנוע מקבל checkout ו־venv נפרדים תחת `env-bench/runtimes/<engine>/`, המוחרגים
  מ־Git. bootstrap מאמת commit אחרי checkout ושומר receipt עם hash מנעול הליבה.
- Seed-VC נעול ל־`51383efd921027683c89e5348211d93ff12ac2a8`, GPL-3.0-only,
  archived ו־`production_allowed=false`. DDSP-SVC נעול ל־
  `3635301027473c6662d05a1c73ef34fba7f15f90`, MIT, וגם הוא benchmark-only.
- לא הועתק קוד ולא הורדו checkpoints משני המנועים. רישיון ריפו אינו מעניק
  אוטומטית רישיון למשקולות; כל מודל אמיתי דורש audit נפרד.
- `constraints.txt`, `rvc-requirements.lock` ומטריצת הליבה לא השתנו.

---

## 4. אימותים שרצו בפועל

| אימות | תוצאה |
|---|---|
| `python -m pytest -q` | ✅ 561 נאספו · 555 עברו · 6 דולגו לפי תנאי חומרה קיימים |
| בדיקת timeout לעץ variant | ✅ parent הוליד child שהקצה 96 MiB; timeout חיסל והמתין לילד, ו־peak RAM המצטבר כלל אותו |
| `python -m ruff check src tests tools env-bench` | ✅ עבר |
| `python -m mypy src` | ✅ 132 קובצי מקור, 0 שגיאות |
| `python tools/bench_phase10.py` | ✅ 4/4 ריצות; CSV/HTML/manifest/audio; telemetry ומיפוי עיוור; objective selector 5/5 מול baseline |
| `python tools/bench_gui.py` | ✅ 17/17 מסכים, כולל advanced ו־benchmark; `all_rtl=true`; contact sheet נבדק חזותית ללא חיתוך |
| `python env-bench/verify_isolation.py` | ✅ אין אזכור מנוע benchmark ב־`src`; build מבודד; source pins מוכנים |
| `uv pip check --python env-bench/.venv/Scripts/python.exe` | ✅ coordinator Python 3.10, אפס התנגשויות |
| `uv pip check --python .venv/Scripts/python.exe` | ✅ ליבת Python 3.11 נקייה |
| `python tools/check_model_catalogue.py` | ✅ 11 מודלים; 5 מותרים להפצה; Seed/DDSP לא הוכנסו לקטלוג production |
| `python tools/audit_constraints_licenses.py` | ✅ 95 חבילות; 0 הפרות GPL/AGPL; 5 LGPL דינמיות |
| `git diff --check` | ✅ ללא שגיאות whitespace (אזהרות CRLF של Git בלבד) |

---

## 5. Definition of Done

| דרישה | סטטוס |
|---|---|
| מטריצת השוואה וטבלת תוצאות | ✅ runner, CSV, HTML, logs, manifest ו־framework gate אמיתי בן ארבע ריצות |
| האזנת A/B עיוורת | ✅ GUI ו־HTML מסתירים שמות ושומרים נקודת זמן; reveal מפורש בלבד |
| auto-tuning לפחות כמו ידני ב־4/5 | ⚠️ selector objective עבר 5/5 וה־baseline תמיד במועמדים; קבלת האזנה אנושית חסומה בסעיף 6 |
| `env-bench` מבודדת | ✅ source/build/import/process/venv boundaries מאומתים; checkouts כבדים ממתינים לחומרי הקבלה |
| `pip check` של הליבה נשאר נקי | ✅ עבר; מנעולי הליבה לא השתנו |

---

## 6. החסמים החיצוניים שנשארו לסוף

לא פניתי למשתמש במהלך השלב, לפי ההנחיה. אי אפשר להשלים מתוך הריפו את שתי קבלות
השמע הבאות בלי להמציא תוצאה:

1. **חמש האזנות tuning:** נדרשים חמשת השירים המורשים, לפחות קול יעד מורשה אחד
   ומאזין. מריצים Preview עם auto-tuning, מצביעים עיוור `automatic|manual|tie`
   בתבנית `benchmark/experiments/tuning-ballots.example.json`, ואז מפעילים
   `python tools/check_tuning_acceptance.py ballots.json`. היעד הוא 4 מתוך 5.
2. **מטריצת מנועים אמיתית:** נדרשים reference voice בהסכמה ל־Seed-VC, checkpoint
   DDSP עם רישיון משקולות מאומת, קול RVC מקביל, NVIDIA GPU וזמן/שטח להורדות.
   לאחר audit מריצים את `env-bench/bootstrap.ps1` לכל מנוע ואת
   `benchmark/experiments/conversion_engines.example.toml`. ה־checkouts,
   המשקולות והשמע אינם נכנסים לריפו או למוצר.

`env-bench/runtimes/seed` ו־`ddsp` לא bootstrap-ו בכוונה: בלי חומרי הקבלה אין
הרצה חוקית, והתקנת stacks כבדים/הורדת מודלים אינה ראיה לאיכות או לרישיון משקולות.

---

## 7. חריגות והחלטות ממוקדות

- מפרט מוקדם הציג YAML, אך ליבה ללא PyYAML היא בטוחה וקטנה יותר. נבחר TOML/JSON
  תקני Python 3.11; התוכן והמטריצה זהים ואין dependency חדש.
- `env-bench` מפוצלת פנימית ל־venv פר־מנוע. זו אינה הרחבת scope אלא הדרך היחידה
  לשמור על בידוד כשמטריצות Torch ההיסטוריות סותרות זו את זו.
- VRAM נאסף דרך `nvidia-smi` כשזמין ונשמר `null` בחומרה אחרת; RAM נאסף ללא psutil.
- המדד האובייקטיבי אינו מוצג כהוכחת 4/5 אנושית. כלי ballots הופך את שער הקבלה
  לחוזה אוטומטי ברגע שחומרי הבדיקה קיימים.

---

## 8. מה לא נעשה, בכוונה

לא נבנו installer, updater, bundling, model mirror או code signing; לא הוכנס
ffmpeg להפצה ולא שונה שום checkpoint production. אלה משימות Phase 11, והיא לא
התחילה. `_MOVED.md` לא נקרא, לא שונה ולא צורף.
