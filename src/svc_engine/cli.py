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
