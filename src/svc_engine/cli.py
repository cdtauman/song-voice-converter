"""SongVoice engine CLI.

Phase 1 commands:
    svc doctor        system check
    svc serve         run the RPC engine on stdin/stdout
    svc version
"""

from __future__ import annotations

import argparse
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

    s = sub.add_parser("serve", help="הרצת המנוע במצב RPC")
    s.set_defaults(func=_cmd_serve)

    v = sub.add_parser("version", help="גרסה")
    v.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    p = paths()
    p.ensure()
    # `serve` speaks JSON on stdout -- logs must not pollute it.
    setup_logging(p.logs, console=(args.command != "serve"))
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
