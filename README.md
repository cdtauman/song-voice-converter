# המרת קול בשירים — תוכנת Desktop ל-Windows

תוכנה מקומית (רצה על המחשב, בלי ענן) שממירה שיר שמבוצע בקול אחד לקול אחר,
תוך שמירה על המנגינה, ההגייה, התזמון, הסלסולים והרגש של הביצוע המקורי.

> **מצב הפרויקט:** Phase 5 הושלם — הצינור המלא מחובר.
> יש מנוע שמפריד שיר לשכבות (Phase 2), מנוע שמנתח את השירה (Phase 3: F0, מנעד,
> סולם, מקטעים, Preview), מנוע שמחליט **כמה להזיז** את השיר (Phase 4: `s = 12k + r`,
> פונקציית העלות, הסבר בעברית, מזיז גובה מדויק-אורך), ו**מנוע המרת הקול** (Phase 5:
> ספריית קולות עם הסכמה, המרה בקטעים חופפים בלי תפרים, רשת RVC v2 מוטמעת, וצינור
> `svc convert` מקצה-לקצה). הרצת ההמרה בפועל ממתינה למודלי קול אמיתיים ולחומרי
> הבדיקה המורשים — ראה [phase-5.md](docs/phase-reports/phase-5.md). אין עדיין
> מיקס מקצועי או ממשק גרפי.

## מה כבר עובד

```bash
svc doctor                                  # בדיקת מערכת
svc models                                  # קטלוג המודלים והרישיונות שלהם
svc separate "song.mp3" --quality balanced --out ./out
svc analyze "vocals.wav" --report report.json --plot report.png
svc profile "voice_sample.wav" --name yossi --out yossi.json
svc pitch "vocals.wav" --voice yossi.json --report pitch.json
```

| שלב | מצב | דוח |
|------|-----|-----|
| Phase 0 — מחקר ותכנון | ✅ | — |
| Phase 1 — Spike תאימות + שלד | ✅ | [phase-1.md](docs/phase-reports/phase-1.md) |
| Phase 2 — מנוע ההפרדה | ✅ | [phase-2.md](docs/phase-reports/phase-2.md) |
| Phase 3 — מנוע הניתוח | ✅ | [phase-3.md](docs/phase-reports/phase-3.md) |
| **Phase 4 — מנוע המנעד וההזזה** | ✅ | [phase-4.md](docs/phase-reports/phase-4.md) |
| **Phase 5 — מנוע המרת הקול** | ✅ מנגנון הושלם; אימות אמיתי ממתין לחסמים | [phase-5.md](docs/phase-reports/phase-5.md) |

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
2. לנעול מחדש את תלויות ה-inference ולאמת את Phase 5 על החומרים שנאספו, לפי
   [דוח Phase 5](docs/phase-reports/phase-5.md). לאחר מכן אפשר לכייל את עקומת
   האיכות (`quality_vs_shift`) שנשארה `null` ב-Phase 4.

## מדיניות שימוש בקולות

התוכנה מיועדת לשימוש **אך ורק** בקולות שהמשתמש יצר בעצמו או שיש לו הרשאה מפורשת
להשתמש בהם. כל הוספת קול חדש למערכת תדרוש אישור מפורש של המשתמש על כך.
