"""Audit the licences of every package in constraints.txt, straight from PyPI.

`tools/check_licenses.py` inspects an installed environment. This script checks
the *locked matrix* without having to install it, so the copyleft policy can be
verified on any machine.

Usage:
    python tools/audit_constraints_licenses.py [--constraints constraints.txt] [--json]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

DENY_MARKERS = ("agpl", "affero", "gpl-3", "gplv3", "gpl-2", "gplv2", "gnu general public")
ALLOW_LGPL_MARKERS = ("lgpl", "lesser general public")

ROOT = Path(__file__).resolve().parent.parent


def package_names(path: Path) -> list[str]:
    names = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "==" in line:
            names.append(line.split("==")[0].strip())
    return names


def licence_of(name: str) -> str | None:
    url = f"https://pypi.org/pypi/{name}/json"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
            info = json.load(resp)["info"]
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError):
        return None
    # Modern packages (PEP 639) use `license_expression`; older ones the free-text
    # `license` field or a trove classifier. Read all three.
    parts = [info.get("license_expression") or "", info.get("license") or ""]
    parts += [c for c in (info.get("classifiers") or []) if c.startswith("License ::")]
    text = re.sub(r"\s+", " ", " ".join(parts)).strip()
    # A pasted full licence body is noise, not a declaration -- keep the head only.
    if len(text) > 400:
        text = text[:400]
    return text or None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--constraints", default=str(ROOT / "constraints.txt"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    path = Path(args.constraints)
    if not path.exists():
        print(f"{path} not found -- run the spike first.", file=sys.stderr)
        return 2

    names = package_names(path)
    violations: list[tuple[str, str]] = []
    lgpl: list[tuple[str, str]] = []
    unknown: list[str] = []

    for name in names:
        lic = licence_of(name)
        if lic is None:
            unknown.append(name)
            continue
        low = lic.lower()
        if any(m in low for m in ALLOW_LGPL_MARKERS):
            lgpl.append((name, lic))
            continue
        if any(m in low for m in DENY_MARKERS):
            violations.append((name, lic))

    if args.json:
        print(json.dumps(
            {"checked": len(names), "violations": violations, "lgpl": lgpl, "unknown": unknown},
            ensure_ascii=False, indent=2,
        ))
    else:
        print(f"checked {len(names)} packages from {path.name}\n")
        print(f"GPL/AGPL violations: {len(violations)}")
        for n, lic in violations:
            print(f"   {n:<26} {lic[:80]}")
        print(f"\nLGPL (allowed, dynamically linked): {len(lgpl)}")
        for n, lic in lgpl:
            print(f"   {n:<26} {lic[:80]}")
        print(f"\nno licence metadata: {len(unknown)}")
        for n in unknown:
            print(f"   {n}")

    return 1 if violations else 0


if __name__ == "__main__":
    sys.exit(main())
