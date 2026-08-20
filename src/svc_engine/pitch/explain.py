"""Turn a shift decision into plain Hebrew.

Research section 4.3 step 5: the user never sees ``s = 12k + r``. They see two or
three sentences that say what will happen to their song and what it costs. The
distinction that matters to them is the one the whole project is built on -- an
octave shift leaves the music untouched, anything else moves the music too -- so
that is exactly what these sentences make plain.

Pure strings. No numpy, no audio. Every message is Hebrew, matching the rest of
the CLI (``svc_engine/cli.py``) and ``docs/architecture.md`` section 8.
"""

from __future__ import annotations

from svc_engine.pitch.strategy import ShiftCandidate, ShiftDecision

__all__ = ["explain_candidate", "explain_decision"]


def _direction(semitones: int) -> str:
    return "להוריד" if semitones < 0 else "להעלות"


def _semitone_word(n: int) -> str:
    n = abs(n)
    if n == 1:
        return "חצי טון"
    if n == 2:
        return "טון שלם"
    return f"{n} חצאי טונים"


def _bprefix(word: str) -> str:
    """Hebrew "by X": ``ב-3 חצאי טונים`` before a digit, ``בחצי טון`` before a word.

    The hyphen only belongs before a numeral; gluing ``ב`` straight onto a word
    is the correct spelling and reads far better.
    """
    return f"ב-{word}" if word[:1].isdigit() else f"ב{word}"


def explain_candidate(candidate: ShiftCandidate) -> str:
    """One human sentence for a single shift option."""
    s = candidate.semitones
    r = candidate.remainder

    if s == 0:
        return "אין צורך בהזזה — הקול כבר יושב בטווח הנוח של הזמר."

    verb = _direction(s)
    amount = _semitone_word(s)

    if r == 0:
        octaves = abs(candidate.octaves)
        octave_word = "אוקטבה שלמה" if octaves == 1 else f"{octaves} אוקטבות שלמות"
        return (
            f"{verb} את השירה ב{octave_word} ({amount}). "
            "המוזיקה נשארת בדיוק כמו שהיא."
        )

    playback_verb = _direction(r)
    playback_amount = _semitone_word(r)
    return (
        f"{verb} את השירה {_bprefix(amount)} — "
        f"אז צריך {playback_verb} גם את המוזיקה {_bprefix(playback_amount)}."
    )


def explain_decision(decision: ShiftDecision) -> str:
    """The recommendation block: the winner, then any distinct runners-up.

    Mirrors the example in research 4.3 step 5 -- a checked recommendation line
    and one or two "another option" lines, each with its playback consequence.
    """
    lines = [f"✅ מומלץ: {explain_candidate(decision.best)}"]
    for alt in decision.alternatives:
        lines.append(f"אפשרות נוספת: {explain_candidate(alt)}")
    if not decision.quality_measured:
        lines.append(
            "ℹ️ עקומת האיכות של הקול הזה עדיין לא נמדדה — ההמלצה מתבססת על "
            "המנעד בלבד. היא תדויק אחרי מדידת האיכות (שלב ההמרה)."
        )
    return "\n".join(lines)
