# דוח סיום — Phase 8: הממשק הגרפי במצב פשוט

**תאריך:** 19 באוגוסט 2026

**ענף:** `phase/8-gui`
**סטטוס:** ✅ המימוש הושלם; אימות שמיעה ושימושיות על חומר אמיתי חסום חיצונית בלבד.

---

## 1. סקירת שימוש חוזר

זהו שלב ממשק, לא inference. לא הועתק קוד מ־AI-RVC, ‏Applio או RVC-Project.
נצרכה ישירות `PySide6 6.11.1` תחת LGPL בקישור דינמי: Qt מספקת RTL, drag/drop,
נגישות מקלדת, thread/event loop ו־Qt Multimedia. מעליה נכתבו רק הזרימה, העיצוב
והחוזים הייחודיים ל־SongVoice. החבילה ותלויות Qt ננעלו ב־`constraints.txt` ונרשמו
ב־[third-party.md](../third-party.md).

---

## 2. מה נבנה

| אזור | תוצאה |
|------|-------|
| `svc_app/design/` | QSS, טיפוגרפיה, צבעים וערכות system/light/dark |
| `svc_app/screens/wizard.py` | אשף שבעה מצבים: שיר → קול → איכות → עיבוד → המלצה → Preview → תוצאה |
| `svc_app/screens/` | ספריית קולות, פרויקטים והגדרות פשוטות |
| `svc_app/widgets/` | DropZone נגיש, VoiceCard, Progress ו־ABPlayer מסונכרן |
| `svc_app/i18n/` | מיפוי כל `ErrorCode` לכותרת בעברית ולפעולה מוצעת |
| `svc_app/engine_client.py` | תהליך מנוע נפרד, streaming events, שגיאות מובנות וביטול |
| `svc_engine/rpc/` | Event לפני Response, פעולות קולות/הגדרות/Preview/קאבר |
| `svc_engine/workflows/cover.py` | גרף production שריד (durable) מעל שרשרת Phases 2–7 |
| `tools/bench_gui.py` | שער חזותי שחזורי לעשרת מצבי הממשק + contact sheet |

הפקת Preview מוגבלת ל־30 שניות ונכתבת ל־scratch של האפליקציה. הפקה מלאה שומרת
WAV ליד השיר בשם `*-SongVoice.wav` ומוסיפה מספר אם השם תפוס; מקור לעולם אינו
נדרס. בחירת השיר והקול נשמרת אוטומטית כפרויקט Phase 7.

---

## 3. גבול התהליך, התקדמות וביטול

ה־GUI מפעיל `svc serve` ומדבר איתו ב־JSON שורה־אחר־שורה. בקשה ארוכה שולחת אפס או
יותר אירועי `progress`, ואז Response סופי אחד. כל אירוע נושא request id, אחוז
והודעה עברית. הקריאה מתבצעת ב־`QThread`; ה־event loop של החלון אינו נחסם.

Preview וקאבר בנויים כגרף `separate` → `analyze` → `render` → `deliver`. תוצרי כל
צעד הם קבצים שמפורסמים אטומית ל־`StepCache`; בקשת העבודה נשמרת תחת
`jobs/requests/<job_id>.json`. אחרי hard-kill, `jobs.recoverable` מחזיר את העבודה
ו־`covers.resume` משחזר את אותו גרף. פתיחת הממשק מציעה בעברית להמשיך מהשלב האחרון.

לחיצה על “בטל עיבוד” קוראת `EngineClient.cancel_current`: stdin נסגר, לאחר 2.5
שניות לכל היותר נעשה terminate ולאחר מכן kill. שער ה־hard-crash של Phase 7 הורץ
שוב: ביטול worker לא משתף פעולה ארך **0.078 שניות**, פחות מ־3 שניות, recovery
עבר בארבעת הצעדים ולא נשאר scratch.

---

## 4. RTL ו־QA חזותי

`QApplication`, החלון וכל עץ הווידג'טים מוגדרים RTL. טקסט מעורב (שמות פורמטים,
מספרי חצאי טונים ונתיבים) נשמר בווידג'טים ייעודיים ולא מורכב כמחרוזת כיוונית אחת.
`tools/bench_gui.py` צילם ובדק:

1. שבעת מצבי האשף;
2. ספריית הקולות;
3. פרויקטים;
4. הגדרות.

כל **10/10** המצבים ירשו RTL, ברזולוציית בדיקה 1770×1140. התוצאות נמצאות
ב־`benchmark/results/gui/results.json`; ה־contact sheet המאוחד נבדק חזותית ולא נמצאו
חיתוכים, גלישה או היפוך סדר פעולות. מצב `offscreen` של Qt ב־Windows לא טוען פונטים
מערכתיים ולכן השער רץ בכוונה עם platform plugin הרגיל של Windows.

---

## 5. אימותים שרצו בפועל

| אימות | תוצאה |
|-------|-------|
| `python -m pytest -q` | ✅ 527 נאספו · 521 עברו · 6 דולגו לפי תנאי חומרה קיימים |
| בדיקות Phase 8 ממוקדות | ✅ 19/19 — GUI, recovery דרך RPC, streaming, שגיאות ו־RTL |
| `python -m ruff check src tests tools/bench_gui.py` | ✅ עבר |
| `python -m mypy src` | ✅ 114 קובצי מקור, 0 שגיאות |
| `python tools/bench_gui.py` | ✅ 10/10 מסכים, `all_rtl=true` |
| `python tools/bench_jobs.py` | ✅ 4/4 hard-crash, cache/recovery וניקוי; cancel <3s |
| `tests/integration/test_cover_rpc_recovery.py` | ✅ RPC קאבר אמיתי יצר snapshot; אחרי hard-kill ההפרדה הגיעה מה־cache ו־scratch נוקה |
| `python tools/check_model_catalogue.py` | ✅ 9 מודלים, 3 מותרים להפצה |
| `python tools/audit_constraints_licenses.py` | ✅ 93 חבילות; 0 GPL/AGPL; Qt מסווגת LGPL דינמית |
| `python -m pip check` | ✅ אין תלויות שבורות |
| `git diff --check` | ✅ עבר |

`tools/check_licenses.py` על **כל סביבת Codex המשותפת** החזיר exit 1 בגלל חבילות
שאינן ב־SongVoice (`bgutil-ytdlp-pot-provider`, ‏`MouseInfo`, ‏`mutagen`,
`PyMsgBox`, ‏`PySide6-Fluent-Widgets`). הן אינן תלויות ב־`pyproject.toml` ואינן
מופיעות ב־`constraints.txt`. שער ההפצה הרלוונטי — audit של הקובץ הנעול — עבר עם
0 הפרות. לא נוספו exemptions לחבילות הזרות.

---

## 6. Definition of Done

| דרישה | סטטוס |
|-------|-------|
| מסלול מלא בממשק בלי טרמינל | ✅ קיים ומחובר לשרשרת production דרך RPC; בדיקת שמע אמיתית חסומה בסעיף 7 |
| שגיאות בעברית עם פעולה מוצעת | ✅ כל 14 קודי `ErrorCode` מכוסים; אין stack trace ב־GUI |
| ביטול מהממשק | ✅ גבול process מחובר; fallback קשיח נמדד ב־0.078 שניות |
| המשך עבודת קאבר אחרי hard-kill | ✅ `covers.resume`; קודמים לא רצים שוב וה־GUI מציע המשך בפתיחה |
| Preview | ✅ מסלול 30 שניות + נגן A/B מסונכרן; האזנה אמיתית חסומה בסעיף 7 |
| RTL בכל מסך | ✅ 10/10 מצבים עברו שער אוטומטי ו־QA חזותי |
| משתמש חדש מצליח בלי הסבר | ⚠️ ההרצה המונחית של האשף שלמה; מבחן עם אדם וחומר אמיתי דורש את סעיף 7 |

---

## 7. החסמים החיצוניים שנשארו לסוף

לא פניתי למשתמש במהלך השלב, לפי ההנחיה. שני האימותים הבאים אינם ניתנים להשלמה
מתוך הריפו:

1. **Preview וקאבר אמיתיים:** בספרייה המקומית אין חבילת קול RVC שמכילה `.pth`,
   פרופיל מנעד והרשאת שימוש, ואין את חמשת השירים המורשים מ־`testing.md`. לכן אי
   אפשר להפעיל ולהאזין לשרשרת האמיתית. חוזה ה־RPC, הזרימה וה־UI נבדקו עם fakes;
   המנוע עצמו נשאר מכוסה בבדיקות Phases 2–6.
2. **מבחן שימושיות:** דרישת “משתמש שלא ראה את התוכנה” מחייבת אדם חיצוני ואת אותם
   חומרים. יש לתת לו שיר וקול, בלי הסבר, ולתעד אם הגיע לתוצאה ושמע A/B.

עם קבלת החומרים: מייבאים את הקול דרך המסך (כולל אישור), מפעילים `songvoice`,
מבצעים Preview וקאבר מלא, מאזינים A/B ומתעדים את מבחן המשתמש. אין צורך בשינוי קוד
או בהתחלת Phase 9 כדי לבצע זאת.

---

## 8. מה לא נעשה, בכוונה

אימון קולות, בדיקת איכות חומר אימון והמשך אימון הם **Phase 9** ולא התחילו. מצב
מתקדם ומעבדת benchmark הם Phase 10; installer ועדכונים הם Phase 11. לא נוסף אף
קוד או מסך מהשלבים האלה.
