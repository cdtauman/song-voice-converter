"""SongVoice engine CLI.

    svc doctor           system check
    svc verify-backends  compatibility / production-proof check
    svc models           inspect / fetch the model catalogue
    svc separate         split a song into stems
    svc analyze          F0 / range / key / segments / preview of a vocal
    svc profile          compute a target voice's range profile
    svc pitch            decide the shift to seat a song in a target voice
    svc voices           list / import / remove target voices
    svc convert          full pipeline: song + voice -> cover
    svc serve            run the RPC engine on stdin/stdout
    svc version
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

from svc_engine import __version__
from svc_engine.config import paths
from svc_engine.diag import Status, render_json, render_text, run_all_checks
from svc_engine.diag.report import overall_status
from svc_engine.errors import EngineError
from svc_engine.logging_setup import setup_logging

__all__ = ["main"]

_EXIT_OK = 0
_EXIT_WARN = 0  # warnings must not fail scripts or CI
_EXIT_FAIL = 2


def _lufs_value(value: str) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("LUFS חייב להיות מספר") from exc
    if not -70.0 <= parsed <= -5.0:
        raise argparse.ArgumentTypeError("יעד LUFS חייב להיות בין ‎-70 ל-‎-5")
    return parsed


def _cmd_doctor(args: argparse.Namespace) -> int:
    p = paths()
    p.ensure()
    results = run_all_checks(p.work)
    if args.json:
        print(render_json(results))
    else:
        print(render_text(results, verbose=args.verbose))
    status = overall_status(results)
    if status is Status.FAIL:
        return _EXIT_FAIL
    return _EXIT_OK if status is Status.OK else _EXIT_WARN


def _cmd_verify_backends(args: argparse.Namespace) -> int:
    """Probe compatibility and record only qualified production-routing proof."""
    import json
    import platform

    from svc_engine.compute.support import DATA_FILE
    from svc_engine.compute.verify import available_devices, build_support_payload, verify_all

    devices = available_devices()
    if not args.json:
        print("בודקים תאימות מאיצים והאם קיים אימות מלא למסלול הייצור…")
        print(f"מכשירים שנבדקים: {', '.join(devices)}\n")

    matrix = verify_all(devices)
    payload = build_support_payload(
        matrix, source=f"verify-backends on {platform.node()}", machine=platform.platform()
    )

    if args.json:
        print(json.dumps({"matrix": matrix, "support": payload}, ensure_ascii=False, indent=2))
    else:
        names_he = {
            "separation": "הפרדה", "f0": "זיהוי גובה",
            "conversion": "המרת קול", "pitch_shift": "הזזת גובה",
        }
        width = max(len(v) for v in names_he.values())
        header = "".ljust(width + 2) + "  ".join(d.ljust(6) for d in devices)
        print(header)
        for component, per_device in matrix.items():
            cells = []
            for device in devices:
                entry = per_device.get(device)
                if entry is None:
                    mark = "—"
                elif not entry.get("ok"):
                    mark = "❌"
                elif device == "cpu" or (
                    entry.get("end_to_end") and entry.get("production_eligible")
                ):
                    mark = "✅"
                else:
                    mark = "🟡"
                cells.append(mark.ljust(6))
            print(names_he.get(component, component).ljust(width + 2) + "  ".join(cells))
        print("\n✅ מסלול ייצור מאומת · 🟡 תאימות/מימוש מסוים בלבד · ❌ נכשל")
        for data in payload["components"].values():
            print(f"  {data['note_he']}")

    if args.write:
        DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        DATA_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not args.json:
            print(f"\nנשמר: {DATA_FILE}")

    accelerated = [d for d in devices if d != "cpu"]
    if not accelerated:
        return _EXIT_WARN
    any_evidence = any(data["proofs"] for data in payload["components"].values())
    return _EXIT_OK if any_evidence else _EXIT_WARN


def _cmd_models(args: argparse.Namespace) -> int:
    """List the catalogue, or fetch entries into the models directory."""
    from svc_engine.resources import DownloadManager, load_registry

    p = paths()
    p.ensure()
    registry = load_registry()

    if args.download:
        wanted = (
            list(registry.models.values())
            if "all" in args.download
            else [registry.get(model_id) for model_id in args.download]
        )
        downloader = DownloadManager(p.models)
        downloader.check_space_for(wanted)
        last = ""
        for spec in wanted:
            def show(progress: object, spec_id: str = spec.id) -> None:
                nonlocal last
                text = f"{spec_id}: {progress.message_he}"  # type: ignore[attr-defined]
                if text != last:
                    print(text, end="\r", flush=True)
                    last = text

            downloader.ensure_model(spec, on_progress=show)
            print(f"✅ {spec.display_name_he:40s} {spec.id}")
        return _EXIT_OK

    private = registry.unlicensed()
    for spec in sorted(registry.models.values(), key=lambda m: (m.kind.value, m.id)):
        present = "✅" if spec.is_present(p.models) else "  "
        licence = spec.license.spdx or "אין הצהרה"
        sdr = f"SDR {spec.sdr:.2f}" if spec.sdr is not None else ""
        print(
            f"{present} {spec.id:24s} {spec.kind.value:11s} "
            f"{spec.size_mb:7.0f}MB  {licence:16s} {sdr}"
        )
    if private:
        print(
            f"\n⚠️  {len(private)} מודלים ללא רישיון מתירני מאומת — "
            "מותרים לשימוש פרטי, לא ייכללו בגרסה מופצת."
        )
    unpinned = registry.unpinned()
    if unpinned:
        print(f"ℹ️  {len(unpinned)} קבצים עדיין בלי sha256 נעול (מאומתים לפי גודל).")
    return _EXIT_OK


def _cmd_separate(args: argparse.Namespace) -> int:
    from svc_engine.separation import CleanupStep, QualityLevel, SeparationPipeline

    pipeline = SeparationPipeline(paths=paths(), allow_downloads=not args.no_download)
    steps = tuple(CleanupStep(s) for s in (args.cleanup or ()))

    last = ""

    def show(progress: object) -> None:
        nonlocal last
        text = progress.message_he  # type: ignore[attr-defined]
        if text != last:
            print(text)
            last = text

    outcome = pipeline.run(
        args.input,
        level=QualityLevel(args.quality),
        cleanup=steps,
        on_progress=None if args.quiet else show,
        duration_seconds=args.seconds,
    )
    written = pipeline.write(outcome, args.out)

    print()
    print(outcome.summary_he())
    for kind, path in sorted(written.items(), key=lambda kv: kv[0].value):
        print(f"  {kind.value:13s} {path}")
    for step, seconds in outcome.timings.items():
        print(f"  ⏱ {step:34s} {seconds:6.1f}s")
    for note in outcome.notes_he:
        print(f"  ℹ️ {note}")
    for fallback in outcome.fallback_steps:
        print(f"  ⚠️ נסיגה בגלל זיכרון: {fallback}")
    return _EXIT_OK


def _f0_extractor_and_device(method: str, no_download: bool):  # type: ignore[no-untyped-def]
    """Build the requested F0 extractor and pick the device it is proven on."""
    from svc_engine.analysis.f0 import RMVPE_MODEL_ID, FcpeExtractor, RmvpeExtractor
    from svc_engine.backends.base import DeviceHint
    from svc_engine.compute import Component, DeviceManager, load_matrix
    from svc_engine.resources import DownloadManager, load_registry

    p = paths()
    if method == "rmvpe":
        registry = load_registry()
        spec = registry.get(RMVPE_MODEL_ID)
        downloader = DownloadManager(p.models, allow_downloads=not no_download)
        downloader.check_space_for([spec])
        last = ""

        def show(progress: object) -> None:
            nonlocal last
            text = progress.message_he  # type: ignore[attr-defined]
            if text != last:
                print(text, end="\r", flush=True)
                last = text

        downloader.ensure_model(spec, on_progress=show)
        weights = spec.files[0].path_in(p.models)
        # RMVPE has no accelerator proof yet: it runs on CPU by design.
        return RmvpeExtractor(weights), DeviceHint()

    # FCPE: routed only onto a backend where torchfcpe itself was proven.
    manager = DeviceManager()
    matrix = load_matrix()
    device = matrix.device_for_implementation(Component.F0, "torchfcpe", manager)
    return FcpeExtractor(), DeviceHint.from_device(device)


def _cmd_analyze(args: argparse.Namespace) -> int:
    import json

    from svc_engine.analysis import analyze_vocal, plot_report
    from svc_engine.audio import io as audio_io

    p = paths()
    p.ensure()
    extractor, device = _f0_extractor_and_device(args.method, args.no_download)
    audio = audio_io.load_audio(args.input, duration_seconds=args.seconds)

    if not args.quiet:
        print("מנתחים את השירה — גובה, מנעד, סולם, מקטעים…")
    report = analyze_vocal(
        audio, extractor, device, source=Path(args.input).name
    )
    extractor.unload()

    print(report.summary_he())
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  📄 דוח: {args.report}")
    if args.plot:
        try:
            plot_report(report, args.plot)
            print(f"  📈 גרף: {args.plot}")
        except RuntimeError as exc:
            print(f"  ⚠️ {exc}")
    return _EXIT_OK


def _cmd_profile(args: argparse.Namespace) -> int:
    from svc_engine.audio import io as audio_io
    from svc_engine.profiles import compute_profile

    p = paths()
    p.ensure()
    extractor, device = _f0_extractor_and_device(args.method, args.no_download)
    audio = audio_io.load_audio(args.input, duration_seconds=args.seconds)

    if not args.quiet:
        print("מחשבים פרופיל מנעד לקול היעד…")
    name = args.name or Path(args.input).stem
    profile = compute_profile(audio, extractor, name=name, device=device)
    extractor.unload()

    print(
        f"{profile.name}: נוח {profile.to_dict()['comfort_low_note']}–"
        f"{profile.to_dict()['comfort_high_note']} · "
        f"טווח {profile.comfortable_span:.1f} חצאי-טונים"
    )
    if args.out:
        profile.save(args.out)
        print(f"  📄 פרופיל: {args.out}")
    return _EXIT_OK


def _cmd_pitch(args: argparse.Namespace) -> int:
    """Decide how far to shift a song to seat it in a target voice."""
    import json

    import numpy as np

    from svc_engine.analysis.features import align_to, frame_rms
    from svc_engine.audio import io as audio_io
    from svc_engine.pitch import PitchDistribution, decide_shift, explain_decision
    from svc_engine.profiles import VoiceProfile

    p = paths()
    p.ensure()
    profile = VoiceProfile.load(args.voice)
    extractor, device = _f0_extractor_and_device(args.method, args.no_download)
    audio = audio_io.load_audio(args.input, duration_seconds=args.seconds)

    if not args.quiet:
        print("מודדים את גובה השירה ומחליטים על ההזזה…")
    curve = extractor.extract(audio, device, 0.01)
    extractor.unload()

    f0_hz = np.asarray(curve.hz, dtype=np.float64).ravel()
    mono = audio.samples.mean(axis=0) if audio.samples.ndim == 2 else audio.samples
    energy = align_to(frame_rms(mono, audio.sample_rate, curve.hop_seconds), f0_hz.size)
    dist = PitchDistribution.from_f0(f0_hz, energy)

    decision = decide_shift(dist, profile)

    print()
    print(explain_decision(decision))
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(
            json.dumps(decision.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  📄 דוח: {args.report}")
    return _EXIT_OK


def _cmd_convert(args: argparse.Namespace) -> int:
    """The full pipeline: song + voice -> repaired and mastered cover."""
    from svc_engine.backends.separation import StemKind
    from svc_engine.conversion import ConversionPipeline
    from svc_engine.pitch import PlaybackStrategy, explain_decision
    from svc_engine.postfx import AmbienceStrategy, PostFxConfig
    from svc_engine.separation import CleanupStep, QualityLevel, SeparationPipeline

    p = paths()
    p.ensure()

    last = ""

    def show(fraction: float, message: str) -> None:
        nonlocal last
        if message != last:
            print(message)
            last = message

    separator = SeparationPipeline(paths=p, allow_downloads=not args.no_download)

    def separate(song: Path):  # type: ignore[no-untyped-def]
        outcome = separator.run(
            song,
            level=QualityLevel(args.quality),
            # The removed room layer is evidence for Phase 6 A/B/C. If the
            # private-only model is unavailable, cleanup records the skip and
            # postfx safely leaves ambience untouched.
            cleanup=(CleanupStep.DEREVERB,),
            on_progress=None if args.quiet else (lambda pr: show(pr.fraction, pr.message_he)),
            duration_seconds=args.seconds,
        )
        return outcome.stems[StemKind.VOCALS], outcome.stems

    extractor, device = _f0_extractor_and_device(args.method, args.no_download)
    pipeline = ConversionPipeline(
        paths=p,
        postfx_config=PostFxConfig(
            ambience_strategy=AmbienceStrategy(args.ambience),
            target_lufs=args.target_lufs,
        ),
    )

    outcome = pipeline.run(
        args.input,
        args.voice,
        f0_extractor=extractor,
        device=device,
        separate=separate,
        strategy=PlaybackStrategy(args.playback),
        on_progress=None if args.quiet else show,
    )
    out_path = pipeline.write(outcome, args.out)

    print()
    print(explain_decision(outcome.decision))
    print()
    print(outcome.summary_he())
    print(f"  🎵 {out_path}")
    if outcome.postfx is not None:
        report = outcome.postfx
        print(
            f"  🔊 {report.mix.integrated_lufs:.2f} LUFS · "
            f"שיא {report.mix.peak_dbfs:.2f} dBFS · "
            f"ללא clipping: {'כן' if not report.mix.clipped else 'לא'}"
        )
        if not report.ambience.measurement.available:
            print("  ℹ️ לא נמצאה שכבת חדר נפרדת; לא נוסף הדהוד משוער.")
    for step, seconds in outcome.timings.items():
        print(f"  ⏱ {step:12s} {seconds:6.1f}s")
    return _EXIT_OK


def _cmd_voices(args: argparse.Namespace) -> int:
    """List the voice library, or import a voice from a zip."""
    from svc_engine.voices import VoiceLibrary, import_voice_from_zip

    p = paths()
    p.ensure()
    library = VoiceLibrary(p)

    if args.import_zip:
        result = import_voice_from_zip(
            args.import_zip,
            display_name=args.name or Path(args.import_zip).stem,
            consent_confirmed=args.consent,
            consent_note=args.consent_note or "",
            voice_id=args.id,
            library=library,
            overwrite=args.overwrite,
        )
        print(result.summary_he())
        return _EXIT_OK

    if args.remove:
        library.remove(args.remove)
        print(f"הוסר: {args.remove}")
        return _EXIT_OK

    entries = library.list()
    if not entries:
        print(
            "אין קולות בספרייה. הוסף קול עם: "
            "svc voices --import <voice.zip> --name <שם> --consent"
        )
        return _EXIT_OK
    for entry in entries:
        m = entry.manifest
        mark = "✅" if m.usable else "⚠️"
        extras = []
        if entry.index_path is not None:
            extras.append("index")
        if entry.profile() is not None:
            extras.append("פרופיל")
        tail = f" · {', '.join(extras)}" if extras else ""
        print(
            f"{mark} {m.voice_id:20s} {m.display_name:20s} "
            f"{m.rvc_version}  {m.health.note_he}{tail}"
        )
    return _EXIT_OK


def _cmd_serve(_: argparse.Namespace) -> int:
    from svc_engine.rpc import serve_stdio

    serve_stdio()
    return _EXIT_OK


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"SongVoice engine {__version__}")
    return _EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="svc", description="SongVoice engine")
    sub = parser.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="בדיקת מערכת")
    d.add_argument("--json", action="store_true", help="פלט JSON במקום טקסט")
    d.add_argument("-v", "--verbose", action="store_true", help="הצג פרטים טכניים")
    d.set_defaults(func=_cmd_doctor)

    b = sub.add_parser(
        "verify-backends",
        help="בדיקת תאימות ואימות מלא של מסלולי האצה",
    )
    b.add_argument("--json", action="store_true", help="פלט JSON במקום טקסט")
    b.add_argument(
        "--write",
        action="store_true",
        help="לשמור את התוצאה כמטריצת התמיכה של האפליקציה",
    )
    b.set_defaults(func=_cmd_verify_backends)

    m = sub.add_parser("models", help="קטלוג המודלים")
    m.add_argument(
        "--download", nargs="+", metavar="ID",
        help="להוריד מודלים לפי מזהה, או 'all' להכל",
    )
    m.set_defaults(func=_cmd_models)

    sep = sub.add_parser("separate", help="הפרדת שיר לשכבות")
    sep.add_argument("input", help="קובץ השיר")
    sep.add_argument("--out", default="./out", help="תיקיית הפלט")
    sep.add_argument(
        "--quality", choices=["fast", "balanced", "max"], default="balanced",
        help="רמת האיכות",
    )
    sep.add_argument(
        "--cleanup", nargs="*", choices=["denoise", "dereverb", "deecho", "karaoke"],
        help="שלבי ניקוי נוספים על הווקאל",
    )
    sep.add_argument(
        "--seconds", type=float, default=None,
        help="לעבד רק את X השניות הראשונות (לבדיקות)",
    )
    sep.add_argument("--no-download", action="store_true", help="לא להוריד מודלים חסרים")
    sep.add_argument("-q", "--quiet", action="store_true", help="בלי הודעות התקדמות")
    sep.set_defaults(func=_cmd_separate)

    an = sub.add_parser("analyze", help="ניתוח שירה: גובה, מנעד, סולם, מקטעים, Preview")
    an.add_argument("input", help="קובץ הווקאל (רצוי שירה מופרדת)")
    an.add_argument(
        "--method", choices=["fcpe", "rmvpe"], default="fcpe",
        help="שיטת זיהוי הגובה: fcpe (מהיר, ברירת מחדל) או rmvpe (מדויק)",
    )
    an.add_argument("--report", metavar="FILE", help="לכתוב דוח JSON לקובץ")
    an.add_argument("--plot", metavar="FILE", help="לצייר גרף F0 לקובץ (דורש matplotlib)")
    an.add_argument(
        "--seconds", type=float, default=None,
        help="לנתח רק את X השניות הראשונות (לבדיקות)",
    )
    an.add_argument("--no-download", action="store_true", help="לא להוריד מודלים חסרים")
    an.add_argument("-q", "--quiet", action="store_true", help="בלי הודעות התקדמות")
    an.set_defaults(func=_cmd_analyze)

    pr = sub.add_parser("profile", help="חישוב פרופיל מנעד לקול יעד מדגימת אודיו")
    pr.add_argument("input", help="דגימת אודיו של הקול")
    pr.add_argument("--name", help="שם הקול (ברירת מחדל: שם הקובץ)")
    pr.add_argument("--out", metavar="FILE", help="לשמור את הפרופיל כ-JSON")
    pr.add_argument(
        "--method", choices=["fcpe", "rmvpe"], default="fcpe",
        help="שיטת זיהוי הגובה",
    )
    pr.add_argument(
        "--seconds", type=float, default=None, help="להשתמש רק ב-X השניות הראשונות"
    )
    pr.add_argument("--no-download", action="store_true", help="לא להוריד מודלים חסרים")
    pr.add_argument("-q", "--quiet", action="store_true", help="בלי הודעות התקדמות")
    pr.set_defaults(func=_cmd_profile)

    pt = sub.add_parser("pitch", help="החלטת הזזה: כמה להזיז שיר כדי שיתאים לקול היעד")
    pt.add_argument("input", help="קובץ הווקאל (רצוי שירה מופרדת)")
    pt.add_argument("--voice", required=True, metavar="FILE", help="קובץ פרופיל הקול (JSON)")
    pt.add_argument(
        "--method", choices=["fcpe", "rmvpe"], default="fcpe",
        help="שיטת זיהוי הגובה: fcpe (מהיר, ברירת מחדל) או rmvpe (מדויק)",
    )
    pt.add_argument("--report", metavar="FILE", help="לכתוב את ההחלטה כ-JSON לקובץ")
    pt.add_argument(
        "--seconds", type=float, default=None,
        help="לנתח רק את X השניות הראשונות (לבדיקות)",
    )
    pt.add_argument("--no-download", action="store_true", help="לא להוריד מודלים חסרים")
    pt.add_argument("-q", "--quiet", action="store_true", help="בלי הודעות התקדמות")
    pt.set_defaults(func=_cmd_pitch)

    cv = sub.add_parser("convert", help="המרת שיר לקול היעד — הצינור המלא")
    cv.add_argument("input", help="קובץ השיר")
    cv.add_argument("--voice", required=True, metavar="ID", help="מזהה הקול בספרייה")
    cv.add_argument("--out", default="cover.wav", help="קובץ הפלט")
    cv.add_argument(
        "--quality", choices=["fast", "balanced", "max"], default="balanced",
        help="רמת איכות ההפרדה",
    )
    cv.add_argument(
        "--method", choices=["fcpe", "rmvpe"], default="rmvpe",
        help="שיטת זיהוי הגובה להמרה (rmvpe מדויק, ברירת מחדל)",
    )
    cv.add_argument(
        "--playback", choices=["A", "B"], default="A",
        help="אסטרטגיית פלייבק: A (הזזת המיקס) או B (פיצול, בלי לגעת בתופים)",
    )
    cv.add_argument(
        "--ambience", choices=["A", "B", "C"], default="B",
        help="מרחב אקוסטי: A מקורי, B שחזור חדש (זמני), C שילוב",
    )
    cv.add_argument(
        "--target-lufs", type=_lufs_value, default=-14.0,
        help="יעד עוצמה סופי ב-LUFS (ברירת מחדל: ‎-14)",
    )
    cv.add_argument(
        "--seconds", type=float, default=None, help="לעבד רק את X השניות הראשונות (לבדיקות)"
    )
    cv.add_argument("--no-download", action="store_true", help="לא להוריד מודלים חסרים")
    cv.add_argument("-q", "--quiet", action="store_true", help="בלי הודעות התקדמות")
    cv.set_defaults(func=_cmd_convert)

    vo = sub.add_parser("voices", help="ספריית הקולות: רשימה, ייבוא, הסרה")
    vo.add_argument("--import", dest="import_zip", metavar="ZIP", help="לייבא קול מקובץ zip")
    vo.add_argument("--name", help="שם התצוגה של הקול")
    vo.add_argument("--id", help="מזהה הקול (ברירת מחדל: נגזר מהשם)")
    vo.add_argument(
        "--consent", action="store_true",
        help="אישור מפורש שיש לך הרשאה להשתמש בקול הזה (חובה לייבוא)",
    )
    vo.add_argument("--consent-note", help="הערת הרשאה (למשל 'הקלטות שלי')")
    vo.add_argument("--overwrite", action="store_true", help="לדרוס קול קיים באותו מזהה")
    vo.add_argument("--remove", metavar="ID", help="להסיר קול מהספרייה")
    vo.set_defaults(func=_cmd_voices)

    s = sub.add_parser("serve", help="הרצת המנוע במצב RPC")
    s.set_defaults(func=_cmd_serve)

    v = sub.add_parser("version", help="גרסה")
    v.set_defaults(func=_cmd_version)

    return parser


def _force_utf8_console() -> None:
    """Make the console able to print Hebrew and status glyphs.

    A Windows console inherits the system codepage -- cp1255 on a Hebrew
    machine, which has no '✅'. Every message this CLI prints is Hebrew, so
    without this the program dies on its own success message. Errors are
    replaced rather than raised: an undisplayable character is a cosmetic
    problem, not a reason to fail a two-hour separation run.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        with contextlib.suppress(OSError, ValueError):  # a redirected stream
            reconfigure(encoding="utf-8", errors="replace")


def main(argv: list[str] | None = None) -> int:
    _force_utf8_console()
    args = build_parser().parse_args(argv)
    p = paths()
    p.ensure()
    # `serve` speaks JSON on stdout -- logs must not pollute it.
    setup_logging(p.logs, console=(args.command != "serve"))
    try:
        return int(args.func(args))
    except EngineError as exc:
        # architecture.md section 8: the user sees a Hebrew what+action line,
        # never a stack trace. The detail is in the log for us.
        import logging

        logging.getLogger(__name__).error("%s: %s", exc.code.value, exc.detail)
        print(f"\n❌ {exc.user_message.render()}", file=sys.stderr)
        return _EXIT_FAIL


if __name__ == "__main__":
    sys.exit(main())
