"""Phase 4 benchmark harness -- the two gates the pitch engine must pass.

``docs/roadmap.md`` Phase 4 says the phase does not end on my say-so but on
measurement. Two questions close it, and this is the harness for both:

  --calibrate   The cost-function recommendation, run over THREE reference
                voice profiles (bass/baritone, tenor, alto), exactly the spread
                ``docs/testing.md`` 1a demands so the weights are not tuned on one
                case. This side is pure maths and runs fully offline, right now.

  --playback    The strategy-A-vs-B x |r| matrix (``docs/testing.md`` 5a): for
                |r| in {1,2,3,4,6}, shift a real instrumental both ways and
                record what is objectively measurable plus a sample-exact length
                check. The *winner* column is decided by blind listening, which
                needs the user's own material and a human ear -- the harness
                leaves it empty rather than inventing it.

    python tools/bench_pitch.py --calibrate
    python tools/bench_pitch.py --playback song_instrumental.wav
    python tools/bench_pitch.py                      # both, synthetic inputs

Judging the recommendation on real songs (DoD #3) and measuring the quality
curve (DoD #4) both need material this repo does not carry; see the Phase 4
report. What runs here proves the maths and the mechanics are sound and
reproducible -- the same division of labour as tools/bench_analysis.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.analysis.midi import midi_to_note_name  # noqa: E402
from svc_engine.backends.base import AudioBuffer  # noqa: E402
from svc_engine.backends.separation import StemKind  # noqa: E402
from svc_engine.pitch import (  # noqa: E402
    CostWeights,
    PitchDistribution,
    PlaybackStrategy,
    PythonStretchShifter,
    decide_shift,
    explain_decision,
    measure_artifacts,
    shift_playback,
)
from svc_engine.profiles import VoiceProfile  # noqa: E402

SR = 44100
RESULTS_DIR = Path(__file__).resolve().parents[1] / "benchmark" / "results" / "pitch"

#: Three reference target profiles spanning the cases testing.md 1a requires:
#: a big drop down, a small drop, and a shift that may go up. These are
#: hand-built reference points for calibration, NOT measured real voices -- real
#: voices and their measured quality curves are the user-supplied material.
REFERENCE_PROFILES: tuple[VoiceProfile, ...] = (
    VoiceProfile("bass_baritone", 45, 57, 40, 60, 50, "reference", 0.0, 0),   # A2..A3
    VoiceProfile("tenor", 48, 64, 45, 69, 55, "reference", 0.0, 0),           # C3..E4
    VoiceProfile("alto_female", 57, 72, 53, 76, 64, "reference", 0.0, 0),     # A3..C5
)


def synth_song_distribution() -> PitchDistribution:
    """A female pop vocal: mostly around E4, up to A4, a few low A3s. Median ~E4."""
    midi = np.concatenate([np.full(600, 64.0), np.full(250, 69.0), np.full(150, 57.0)])
    hz = 440.0 * 2.0 ** ((midi - 69.0) / 12.0)
    energy = np.concatenate([np.full(600, 1.0), np.full(250, 0.8), np.full(150, 0.5)])
    return PitchDistribution.from_f0(hz, energy)


def synth_instrumental(seconds: float = 6.0) -> dict[StemKind, AudioBuffer]:
    """A crude three-stem backing: a low bass tone, noisy 'drums', a mid 'other'."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    bass = 0.3 * np.sin(2 * np.pi * 110.0 * t)
    other = 0.2 * (np.sin(2 * np.pi * 330.0 * t) + np.sin(2 * np.pi * 440.0 * t))
    rng = np.random.default_rng(0)
    drums = 0.25 * rng.standard_normal(n) * (np.sin(2 * np.pi * 2.0 * t) > 0.9)
    mk = lambda x: AudioBuffer(samples=x.astype(np.float32)[None, :], sample_rate=SR)  # noqa: E731
    return {StemKind.BASS: mk(bass), StemKind.OTHER: mk(other), StemKind.DRUMS: mk(drums)}


def run_calibrate(weights: CostWeights) -> list[dict[str, object]]:
    """Recommendation for each reference profile. The core DoD-#2 evidence table."""
    song = synth_song_distribution()
    rows: list[dict[str, object]] = []
    print(f"\nשיר מבחן: חציון {midi_to_note_name(song.median)} (MIDI {song.median:.1f})")
    print(f"משקלים: {weights.to_dict()}\n")
    header = f"{'profile':14s} {'best s':>6s} {'k':>3s} {'r':>3s} {'playback':>9s}  recommendation"
    print(header)
    print("-" * len(header))
    for profile in REFERENCE_PROFILES:
        decision = decide_shift(song, profile, weights)
        b = decision.best
        print(
            f"{profile.name:14s} {b.semitones:>6d} {b.octaves:>3d} {b.remainder:>3d} "
            f"{('כן' if b.needs_playback_shift else 'לא'):>9s}  "
            f"→ {midi_to_note_name(profile.median)} voice"
        )
        rows.append(
            {
                "profile": profile.name,
                "target_median_note": midi_to_note_name(profile.median),
                "decision": decision.to_dict(),
                "explanation_he": explain_decision(decision),
            }
        )
    return rows


def run_playback(instrumental_path: str | None) -> dict[str, object]:
    """Strategy A vs B across |r|. Needs python-stretch for real audio numbers."""
    shifter = PythonStretchShifter()
    available = shifter.info().available

    if instrumental_path:
        from svc_engine.audio import io as audio_io

        whole = audio_io.load_audio(instrumental_path)
        stems = None  # a real split needs Phase 2 separation of THIS instrumental
        source = Path(instrumental_path).name
    else:
        stems = synth_instrumental()
        # For strategy A we need the summed instrumental.
        from svc_engine.audio.buffers import add

        whole = stems[StemKind.BASS]
        for kind in (StemKind.OTHER, StemKind.DRUMS):
            whole = add(whole, stems[kind])
        source = "synthetic 3-stem"

    print(f"\nמטריצת פלייבק (A מול B) על: {source}")
    if not available:
        print(
            "⚠️ python-stretch לא מותקן — לא ניתן להזיז אודיו אמיתי. "
            "מריצים בדיקת מנגנון בלבד (אורך), לא איכות."
        )

    rows: list[dict[str, object]] = []
    header = (
        f"{'|r|':>3s}  {'A len ok':>8s} {'A artf':>7s}  "
        f"{'B len ok':>8s} {'B artf':>7s}  winner"
    )
    print(header)
    print("-" * len(header))
    for abs_r in (1, 2, 3, 4, 6):
        r = -abs_r  # measure the downward case; symmetric by construction
        row: dict[str, object] = {"abs_r": abs_r, "r": r}
        if available:
            a_out = shift_playback(whole, r, shifter, PlaybackStrategy.WHOLE)
            row["A_length_ok"] = bool(a_out.frames == whole.frames)
            row["A_artifacts"] = round(measure_artifacts(a_out).penalty, 5)
            if stems is not None:
                b_out = shift_playback(stems, r, shifter, PlaybackStrategy.SPLIT)
                row["B_length_ok"] = bool(b_out.frames == whole.frames)
                row["B_artifacts"] = round(measure_artifacts(b_out).penalty, 5)
            else:
                row["B_length_ok"] = None
                row["B_artifacts"] = None
        else:
            row["A_length_ok"] = row["B_length_ok"] = None
            row["A_artifacts"] = row["B_artifacts"] = None
        # The winner is a blind-listening decision -- never filled by a machine.
        row["winner"] = None
        rows.append(row)
        print(
            f"{abs_r:>3d}  {str(row['A_length_ok']):>8s} {str(row['A_artifacts']):>7s}  "
            f"{str(row['B_length_ok']):>8s} {str(row['B_artifacts']):>7s}  (blind)"
        )

    print(
        "\nעמודת ה-winner נסגרת בהאזנה עיוורת על חומר אמיתי — "
        "אין מדד אובייקטיבי ל'התופים איבדו חדות' (testing.md 5a)."
    )
    return {
        "source": source,
        "shifter_available": available,
        "note": "winner column is decided by blind listening on real material",
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("instrumental", nargs="?", help="instrumental file for the playback matrix")
    ap.add_argument("--calibrate", action="store_true", help="run the cost-function table only")
    ap.add_argument("--playback", action="store_true", help="run the A/B x |r| matrix only")
    ap.add_argument("--out", metavar="DIR", help="write JSON results under this directory")
    args = ap.parse_args(argv)

    do_cal = args.calibrate or not args.playback
    do_pb = args.playback or not args.calibrate

    payload: dict[str, object] = {}
    if do_cal:
        payload["calibration"] = run_calibrate(CostWeights())
    if do_pb:
        payload["playback_matrix"] = run_playback(args.instrumental)

    out_dir = Path(args.out) if args.out else RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nנשמר: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
