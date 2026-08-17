"""SongVoice engine CLI.

    svc doctor           system check
    svc verify-backends  prove which stages run on which accelerator
    svc models           inspect / fetch the model catalogue
    svc separate         split a song into stems
    svc serve            run the RPC engine on stdin/stdout
    svc version
"""

from __future__ import annotations

import argparse
import contextlib
import sys

from svc_engine import __version__
from svc_engine.config import paths
from svc_engine.diag import Status, render_json, render_text, run_all_checks
from svc_engine.diag.report import overall_status
from svc_engine.logging_setup import setup_logging

__all__ = ["main"]

_EXIT_OK = 0
_EXIT_WARN = 0  # warnings must not fail scripts or CI
_EXIT_FAIL = 2


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
    """Prove, on this machine, which components can run on which accelerator."""
    import json
    import platform

    from svc_engine.compute.support import DATA_FILE
    from svc_engine.compute.verify import available_devices, build_support_payload, verify_all

    devices = available_devices()
    if not args.json:
        print("בודקים אילו שלבים יכולים לרוץ על כל מאיץ…")
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
                    cells.append("—".ljust(6))
                else:
                    cells.append(("✅" if entry.get("ok") else "❌").ljust(6))
            print(names_he.get(component, component).ljust(width + 2) + "  ".join(cells))
        print()
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
    any_proof = any(data["proofs"] for data in payload["components"].values())
    return _EXIT_OK if any_proof else _EXIT_WARN


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

    b = sub.add_parser("verify-backends", help="אימות אילו שלבים רצים על כל מאיץ")
    b.add_argument("--json", action="store_true", help="פלט JSON במקום טקסט")
    b.add_argument("--write", action="store_true",
                   help="לשמור את התוצאה כמטריצת התמיכה של האפליקציה")
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
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
