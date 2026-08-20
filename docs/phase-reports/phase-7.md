# דוח סיום — Phase 7: ניהול משימות ושרידות

**תאריך:** 19 באוגוסט 2026

**ענף:** `phase/7-jobs`

**סטטוס:** ✅ הושלם מכנית במלואו; Phase 8 לא התחילה.

---

## 1. סקירת שימוש חוזר

הרכיב הוא orchestration ואחסון מקומי ייחודי ל־SongVoice, לא inference. רכיבי
AI-RVC, ‏Applio ו־RVC-Project שכבר נבדקו במדיניות השימוש החוזר מספקים זרימות inference
אך לא חוזה DAG/cache/recovery שמתאים לגבולות ולבטיחות הנתונים כאן. לכן לא הוכנס
קוד צד שלישי חדש. המימוש משתמש רק בספרייה הסטנדרטית: `hashlib`, `sqlite3`,
`threading`, `subprocess` ופעולות קבצים אטומיות של מערכת ההפעלה.

---

## 2. מה נבנה

| מודול | אחריות |
|-------|--------|
| `jobs/runner.py` | אימות ומיון DAG, התקדמות משוקללת, cache hits, ביצוע, resume וניקוי |
| `jobs/cache.py` | מפתח תוכן קנוני, SHA-256 לפלטים, publication אטומי ו־LRU quota |
| `jobs/cancel.py` | `CancellationToken` שיתופי ו־terminate/kill עם deadline כולל |
| `jobs/recovery.py` | snapshot גרסאי אטומי אחרי כל מעבר מצב וגילוי עבודות להמשך |
| `projects/store.py` | שמירה/טעינה אטומית, schema ועותק last-known-good |
| `history/store.py` | היסטוריית עבודות והתקדמות ב־SQLite WAL |
| `rpc/server.py` | recovery/history/cleanup/cache ו־project list/load/save כ־JSON בלבד |
| `svc_app/engine_client.py` | wrapper דק ל־RPC וביטול בגבול process, בלי AI/Qt |
| `tools/bench_jobs.py` | מטריצת hard-crash אמיתית ו־force-cancel timing |

### חוזה cache

המפתח כולל גרסת חוזה, מזהה וגרסת צעד, פרמטרים קנוניים, SHA-256+גודל של קובצי
קלט ומפתחות התלויות בסדרם. כל output מועתק תחילה לתיקייה זמנית, עובר hash ונרשם
ב־manifest, ורק אז התיקייה מתפרסמת ב־`os.replace`. lookup מאמת מחדש גודל ו־hash;
פלט חלקי או מושחת הוא miss, לעולם לא ראיה להשלמת צעד.

### recovery ובטיחות נתונים

snapshot נשמר לפני כניסה לצעד ואחרי השלמה. hard-kill משאיר לכל היותר scratch
חלקי ומצב `running`; בהפעלה הבאה scratch זה נמחק, קודמים תקינים נקראים מה־cache,
והצעד שנקטע רץ מחדש. כשל רגיל נשמר כ־`failed` וניתן להמשך; ביטול הוא terminal.
קובץ פרויקט קיים נשאר ראשי עד שגרסתו החדשה נכתבה ופורסמה אטומית, והעותק הקודם
נכתב גם הוא אטומית. JSON פגום נשמר לאבחון ולא נמחק אוטומטית.

### Windows וגבולות app/engine

מזהי job/project/step מוגבלים לרכיב ASCII יחיד ודוחים שמות שמורים כמו `CON`,
מעברי `..` ומפרידי נתיב. scratch נמצא תחת `work/jobs` בנתיב ASCII של האפליקציה;
קלט ופרויקטים תומכים בתוכן ובנתיבים עבריים. כל מחיקה מאמתת שה־resolved parent הוא
ה־root הצפוי. `svc_app` אינו מייבא torch/numpy/AI; הוא מחליף רק JSON ותהליך.

---

## 3. ביטול, המשך וניקוי

- לולאות ארוכות קוראות `context.check_cancelled()` או `context.progress()`, שבודקת
  token לפני דיווח. runner בודק גם בין כל שני צעדים ולפני publication.
- worker משתף פעולה נעצר מיד ומסומן `cancelled`; scratch ו־cache temp מנוקים.
- worker תקוע מקבל חלון שיתופי ששומר זמן ל־terminate ול־kill בתוך אותו deadline.
  `EngineClient.cancel_current()` סוגר stdin, מחכה 2.5 שניות לכל היותר, ואז מסיים
  בכוח בתוך יתרת שלוש השניות.
- תחזוקת cache אינה יכולה להפוך job שהושלם לכישלון רק כי Windows/אנטי־וירוס נעל
  entry ישן. היא נרשמת כאזהרה ותנסה שוב בניקוי הבא.
- `ENOSPC`/WinError 112 מתורגם ל־`E_DISK_FULL`; temp נמחק וה־snapshot הקודם נשאר
  recovery point גם אם אין מקום לכתוב עדכון נוסף.

---

## 4. אימות שרץ בפועל

`python tools/bench_jobs.py` רץ בתהליך אב אמיתי והפעיל worker נפרד לכל מיקום:

| kill בתוך | קוד יציאה | קודמים הגיעו מ־cache | הצעד שנקטע רץ מחדש | הרצה זהה | scratch |
|-----------|-----------|----------------------|---------------------|------------|---------|
| `separate` | 91 | — | ✅ | 100% hits | נקי |
| `analyze` | 91 | `separate` | ✅ | 100% hits | נקי |
| `convert` | 91 | `separate`, `analyze` | ✅ | 100% hits | נקי |
| `master` | 91 | שלושת הקודמים | ✅ | 100% hits | נקי |

ה־worker הלא־משתף הופסק בכ־0.08 שניות בהרצה המתועדת, מתחת ל־3 שניות. המטריצה
רצה גם תחת נתיב עברי עם רווחים. הפלט המלא ב־`benchmark/results/jobs/results.json`.

בדיקות יחידה חדשות מכסות: שינוי תוכן/פרמטר/תלות, שחיתות cache, LRU והגנה על entry
פעיל, DAG מחזורי/חסר, progress מונוטוני, cache חלקי, resume אחרי exception,
cooperative cancel, force cancel, disk-full, ניקוי orphan, JSON אטומי, פרויקט
עברית+backup, SQLite ו־RPC.

האימות הסופי: **501 בדיקות נאספו; 495 עברו ו־6 דולגו** (אותם extras כבדים שאינם
בסביבה הקלה). ‏ruff נקי, ‏mypy נקי על 98 קובצי מקור, בדיקת הגבולות כלולה ועברה,
קטלוג המודלים עבר, ו־audit של 89 החבילות ב־`constraints.txt` מצא **0 הפרות
GPL/AGPL**. סריקת *כל* סביבת Codex Desktop דיווחה כצפוי על חבילות GPL של כלי host
שאינן ב־`pyproject.toml` או ב־lock של SongVoice; היא אינה audit של המוצר ולא נוספה
אף תלות ב־Phase 7.

---

## 5. Definition of Done

| דרישה | סטטוס | ראיה |
|-------|-------|------|
| הריגה באמצע כל שלב → התאוששות | ✅ | 4/4 hard exits אמיתיים + process חדש |
| ביטול תוך פחות מ־3 שניות | ✅ | cooperative unit + force-cancel benchmark |
| cache חוסך זמן בהרצה שנייה | ✅ | 100% hits; action לא נקראה שוב |
| שינוי פרמטר לא מריץ שלבים לא מושפעים | ✅ | `separate` ו־`convert` hits, רק `mix` רץ |
| אין קבצים זמניים יתומים | ✅ | cleanup אחרי success/failure/cancel/resume + orphan sweep |
| פרויקטים והיסטוריה נשמרים | ✅ | atomic project+backup ו־SQLite WAL דרך API/RPC |

---

## 6. חסמים ומה לא נעשה

אין חסם חיצוני ל־Phase 7 עצמה. חסמי האיכות הקודמים נשארו ללא שינוי: חמשת השירים,
שלושת קולות היעד והאזנה אנושית; מודל DeReverb מתירני; RVC אמיתי; ובינארי ffmpeg
LGPL להפצה. הם אינם משפיעים על נכונות orchestration/שרידות ונשארים מתועדים בדוחות
Phase 5–6 וב־backlog.

לא נבנה ממשק גרפי, לא נוספו מסכים ולא בוצע RTL — אלה Phase 8, והיא לא התחילה.
