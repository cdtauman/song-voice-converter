# המרת קול בשירים — תוכנת Desktop ל-Windows

תוכנה מקומית (רצה על המחשב, בלי ענן) שממירה שיר שמבוצע בקול אחד לקול אחר,
תוך שמירה על המנגינה, ההגייה, התזמון, הסלסולים והרגש של הביצוע המקורי.

> **מצב הפרויקט:** Phase 11 הושלם ברמת המימוש והאריזה — ניתן להתקין ולהפעיל בלי Python.
> יש מנוע שמפריד שיר לשכבות (Phase 2), מנוע שמנתח את השירה (Phase 3: F0, מנעד,
> סולם, מקטעים, Preview), מנוע שמחליט **כמה להזיז** את השיר (Phase 4: `s = 12k + r`,
> פונקציית העלות, הסבר בעברית, מזיז גובה מדויק-אורך), ו**מנוע המרת הקול** (Phase 5:
> ספריית קולות עם הסכמה, המרה בקטעים חופפים בלי תפרים, רשת RVC v2 מוטמעת, וצינור
> `svc convert` מקצה-לקצה), ו**שרשרת Phase 6**: repair, בדיקת F0, מעטפת, de-esser,
> מרחב A/B/C, `loudnorm`/`alimiter` דרך ffmpeg וייצוא WAV/MP3. **Phase 7** מוסיף
> גרף צעדים, התקדמות, cache מבוסס תוכן, ביטול, המשך אחרי קריסה, פרויקטים והיסטוריה.
> **Phase 8** מוסיף אשף מלא, ספריית קולות, Preview A/B, המלצה, הגדרות ושגיאות
> עבריות עם פעולה מוצעת — בלי לגעת בטרמינל.
> **Phase 9** מוסיף בדיקת איכות חומר, ניקוי וחיתוך אוטומטיים, מעטפת אימון
> Applio+TITAN עמידה לעצירה/המשך, ETA, פרופיל מנעד אוטומטי וניהול מלא של כרטיסי קול.
> **Phase 10** מוסיף מצב מתקדם, כוונון Preview בן ארבעה מועמדים, מטריצת benchmark
> שחזורית ומסך A/B עיוור. Seed-VC ו-DDSP-SVC נשארים מחוץ לליבה תחת `env-bench`.
> **Phase 11** מוסיף runtime עצמאי, מתקיני Windows עבריים online/offline, אשף
> הפעלה ראשונה, עדכונים עם checksum/rollback ו־release CI. קבלת clean-machine
> עם קאבר אמיתי, חתימה ו־mirror פרטי ממתינות לחומרים/credentials חיצוניים.
> הכרעת האיכות הסופית
> עדיין ממתינה למודלי קול, לחמשת השירים המורשים ולהאזנה עיוורת — ראה
> [phase-6.md](docs/phase-reports/phase-6.md). אימות GUI מקצה־לקצה על שמע אמיתי
> ממתין לאותם חומרים; דוח השלב נמצא ב־[phase-8.md](docs/phase-reports/phase-8.md).

במסלול `svc convert`, מודלים שאינם מורשים להפצה חסומים כברירת מחדל: המערכת לא
מורידה ולא מפעילה את מודל ה-DeReverb מסוג GPL, ואם אין חלופה מתירנית היא ממשיכה
בבטחה בלי שכבת ambience. הדגל `--allow-private-models` הוא opt-in מפורש לשימוש
פרטי/פיתוחי בלבד ומציג אזהרה; הוא אינו תנאי שקט להפעלת אסטרטגיות B/C.

## מה כבר עובד

```bash
pip install -e ".[gui]"
songvoice                                    # הממשק הגרפי — ללא צורך בטרמינל לאחר ההפעלה
```

ממשק שורת הפקודה נשאר זמין לפיתוח ולאוטומציה:

```bash
svc doctor                                  # בדיקת מערכת
svc models                                  # קטלוג המודלים והרישיונות שלהם
svc separate "song.mp3" --quality balanced --out ./out
svc analyze "vocals.wav" --report report.json --plot report.png
svc profile "voice_sample.wav" --name yossi --out yossi.json
svc pitch "vocals.wav" --voice yossi.json --report pitch.json
svc convert "song.mp3" --voice yossi             # חוסם מודלים פרטיים כברירת מחדל
svc-bench run benchmark/experiments/my-test.toml --out benchmark/results/my-test
```

| שלב | מצב | דוח |
|------|-----|-----|
| Phase 0 — מחקר ותכנון | ✅ | — |
| Phase 1 — Spike תאימות + שלד | ✅ | [phase-1.md](docs/phase-reports/phase-1.md) |
| Phase 2 — מנוע ההפרדה | ✅ | [phase-2.md](docs/phase-reports/phase-2.md) |
| Phase 3 — מנוע הניתוח | ✅ | [phase-3.md](docs/phase-reports/phase-3.md) |
| **Phase 4 — מנוע המנעד וההזזה** | ✅ | [phase-4.md](docs/phase-reports/phase-4.md) |
| **Phase 5 — מנוע המרת הקול** | ✅ מנגנון הושלם; אימות אמיתי ממתין לחסמים | [phase-5.md](docs/phase-reports/phase-5.md) |
| **Phase 6 — תיקונים ומיקס** | ✅ מימוש ואימות מכני; הכרעת האזנה חסומה חיצונית | [phase-6.md](docs/phase-reports/phase-6.md) |
| **Phase 7 — ניהול משימות ושרידות** | ✅ מימוש + מטריצת hard-crash מלאה | [phase-7.md](docs/phase-reports/phase-7.md) |
| **Phase 8 — ממשק גרפי במצב פשוט** | ✅ מימוש + GUI/RTL gate; בדיקת שמע אנושית חסומה חיצונית | [phase-8.md](docs/phase-reports/phase-8.md) |
| **Phase 9 — ניהול קולות ואשף אימון** | ✅ מימוש + session/quality/GUI; אימון קבלה על 15 דקות חסום על חומר חיצוני | [phase-9.md](docs/phase-reports/phase-9.md) |
| **Phase 10 — מצב מתקדם ומעבדת benchmark** | ✅ מימוש ושערים מכניים; האזנת 5 מקרים ומטריצת מנועים חסומות על חומר חיצוני | [phase-10.md](docs/phase-reports/phase-10.md) |
| **Phase 11 — אריזה, installer ועדכונים** | ✅ runtime ומתקינים נבנו; קבלת קאבר נקי, חתימה ו־mirror חסומים חיצונית | [phase-11.md](docs/phase-reports/phase-11.md) |

## המסמכים

| קובץ | מה יש בו |
|------|----------|
| [docs/research.md](docs/research.md) | המחקר המלא: השוואת טכנולוגיות, בחירת ה-pipeline, הפתרון לבעיית המנעד |
| [docs/architecture.md](docs/architecture.md) | ארכיטקטורת התוכנה, המתאמים, מבנה הקוד, תכנון ה-UX |
| [docs/roadmap.md](docs/roadmap.md) | תוכנית הפיתוח המלאה לפי שלבים (Phase 0 עד Phase 12) |
| [docs/testing.md](docs/testing.md) | תוכנית הבדיקות ומערכת ה-Benchmark להשוואת איכות |
| [docs/decisions.md](docs/decisions.md) | החלטות שכבר התקבלו + החלטות שממתינות להכרעה שלך |
| [docs/models.md](docs/models.md) | טבלת audit רישוי מלאה למודלים ולספריות, כולל גדלים |
| [docs/reuse-policy.md](docs/reuse-policy.md) | **מדיניות שימוש חוזר** — מתי לוקחים קוד קיים ומתי כותבים בעצמנו |
| [docs/third-party.md](docs/third-party.md) | מרשם כל הקוד החיצוני: ריפו, קבצים, commit, רישיון |
| [docs/backlog.md](docs/backlog.md) | רעיונות שנדחו מהשלב הנוכחי |
| [docs/changelog.md](docs/changelog.md) | יומן השינויים במסמכי התכנון, כולל תיקונים עובדתיים שאומתו |
| [docs/phase-reports/](docs/phase-reports/) | דוח סיום לכל שלב שהושלם |
| [PLAN-FULL.md](PLAN-FULL.md) | כל מסמכי התכנון בקובץ אחד |

## איך מתקדמים מכאן

1. **לאסוף את חומרי הבדיקה** — 5 שירים, 2 קטעי ווקאל יבשים, ו-3 קולות יעד
   (בס/בריטון, טנור, אלט), לפי [docs/testing.md §1, §1א](docs/testing.md). בלעדיהם
   אי אפשר לכייל את משקלי ההזזה ולשפוט את ההמלצה על מוזיקה אמיתית — הסייג הפתוח
   של Phase 2, 3 ו-4.
2. לאמן דרך אשף Phase 9 את שלושת קולות היעד מדגימות מורשות של 15–30 דקות, ולאמת
   עצירה/המשך וקאבר ראשון בכל קול.
3. לאמת את Phase 5 ואת מסלול ה־GUI על החומרים שנאספו, ואז להריץ את מטריצת Raw/A/B/C של
   [דוח Phase 6](docs/phase-reports/phase-6.md), למלא את יומן ההאזנה העיוורת ולנעול
   את אסטרטגיית המרחב ו-`rms_mix_rate`.
4. להריץ את דוגמת Phase 10 על RVC/DDSP/Seed-VC, למלא חמישה ballots עיוורים ולאמת
   אותם ב־`tools/check_tuning_acceptance.py`; אף runtime של `env-bench` אינו נארז.
5. למסור תעודת Authenticode ו־Hugging Face repo/token, ואז להריץ build מתויג ואת
   קבלת Windows 11 הנקייה עם שיר וקול יעד מורשים לפי דוח Phase 11.

## מדיניות שימוש בקולות

התוכנה מיועדת לשימוש **אך ורק** בקולות שהמשתמש יצר בעצמו או שיש לו הרשאה מפורשת
להשתמש בהם. כל הוספת קול חדש למערכת תדרוש אישור מפורש של המשתמש על כך.
