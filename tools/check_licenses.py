"""Fail if a copyleft dependency enters the distributable core.

Approved policy (docs/decisions.md #10): the core is built as if the app will be
distributed. GPL/AGPL is allowed only in build tools and in the isolated
benchmark environment -- never in packages we ship.

Usage:
    python tools/check_licenses.py            # check the current environment
    python tools/check_licenses.py --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from importlib import metadata

#: Substrings that mark a copyleft licence. LGPL is handled separately.
DENY_MARKERS = (
    "agpl",
    "affero",
    "gpl-3",
    "gplv3",
    "gpl-2",
    "gplv2",
    "gnu general public",
)

#: LGPL is acceptable for dynamically linked libraries (PySide6, ffmpeg, soxr).
ALLOW_LGPL_MARKERS = ("lgpl", "lesser general public")

#: Packages exempt with a written reason. Build tools are never shipped.
EXEMPT: dict[str, str] = {
    "pyinstaller": "build tool only; GPL with an explicit commercial exception",
    "pyinstaller-hooks-contrib": "build tool only",
    "pyside6": "LGPL-3.0, dynamically linked",
    "pyside6-essentials": "LGPL-3.0, dynamically linked",
    "pyside6-addons": "LGPL-3.0, dynamically linked",
    "shiboken6": "LGPL-3.0, dynamically linked",
    "chardet": "LGPL, dynamically linked, optional transitive dep",
}


@dataclass(frozen=True)
class Finding:
    name: str
    version: str
    license: str
    reason: str


def _license_text(dist: metadata.Distribution) -> str:
    md = dist.metadata
    parts: list[str] = []
    for key in ("License", "License-Expression"):
        value = md.get(key)
        if value and value != "UNKNOWN":
            parts.append(str(value))
    parts.extend(c for c in md.get_all("Classifier") or [] if c.startswith("License ::"))
    return " | ".join(parts) or "UNKNOWN"


def scan() -> tuple[list[Finding], list[Finding]]:
    """Returns (violations, unknown_licences)."""
    violations: list[Finding] = []
    unknown: list[Finding] = []

    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name:
            continue
        key = name.lower()
        version = dist.version or "?"
        lic = _license_text(dist)
        low = lic.lower()

        if key in EXEMPT:
            continue

        if lic == "UNKNOWN":
            unknown.append(Finding(name, version, lic, "no licence metadata"))
            continue

        if any(m in low for m in ALLOW_LGPL_MARKERS):
            continue

        hit = next((m for m in DENY_MARKERS if m in low), None)
        if hit:
            violations.append(Finding(name, version, lic, f"copyleft marker '{hit}'"))

    violations.sort(key=lambda f: f.name.lower())
    unknown.sort(key=lambda f: f.name.lower())
    return violations, unknown


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Copyleft dependency check")
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--strict-unknown",
        action="store_true",
        help="also fail on packages with no licence metadata",
    )
    args = ap.parse_args(argv)

    violations, unknown = scan()

    if args.json:
        print(json.dumps(
            {
                "violations": [f.__dict__ for f in violations],
                "unknown": [f.__dict__ for f in unknown],
            },
            indent=2,
        ))
    else:
        if violations:
            print("Copyleft dependencies found in the core environment:\n")
            for f in violations:
                print(f"  {f.name} {f.version}\n      {f.license}\n      -> {f.reason}")
        else:
            print("No copyleft dependencies in the core environment.")
        if unknown:
            print(f"\n{len(unknown)} package(s) without licence metadata:")
            for f in unknown:
                print(f"  {f.name} {f.version}")

    if violations:
        return 1
    if unknown and args.strict_unknown:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
