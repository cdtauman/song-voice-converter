"""Audit the model catalogue: licences, and whether the downloads still exist.

Two separate jobs, deliberately in one place because they are the two ways a
catalogue entry goes bad.

**Licences** (offline, runs in CI). Every entry must record what its licence is,
when that was checked, and against which source. An entry with no permissive
licence is not an error -- it is usable privately -- but it must be *declared*,
so that packaging in Phase 11 can exclude it instead of discovering it.

**Reachability** (network, run by hand). This is not hypothetical. When this
phase was written, five of the seven checkpoints the plan named had already
disappeared from the UVR release repository that most tools fetch them from.
Every URL here is a mirror found after that, and the same thing will happen to
some of them. Running this is how we find out before a user does.

    python tools/check_model_catalogue.py                 # licences only
    python tools/check_model_catalogue.py --check-urls    # + reachability
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.resources import load_registry  # noqa: E402


def check_licences(strict: bool) -> tuple[int, list[dict]]:
    registry = load_registry()
    rows: list[dict] = []
    undeclared = 0

    for spec in sorted(registry.models.values(), key=lambda m: (m.kind.value, m.id)):
        if not spec.license.verified_at or not spec.license.source:
            undeclared += 1
        rows.append(
            {
                "id": spec.id,
                "kind": spec.kind.value,
                "licence": spec.license.spdx,
                "verified_at": spec.license.verified_at,
                "redistributable": spec.license.is_redistributable,
                "size_mb": round(spec.size_mb, 1),
            }
        )

    shippable = [r for r in rows if r["redistributable"]]
    private = [r for r in rows if not r["redistributable"]]

    print(f"{len(rows)} models in the catalogue\n")
    for row in rows:
        mark = "ship" if row["redistributable"] else "priv"
        licence = row["licence"] or "(no declaration)"
        print(f"  [{mark}] {row['id']:24s} {row['kind']:11s} {licence}")

    print(f"\n  redistributable: {len(shippable)}")
    print(f"  private use only: {len(private)}")

    if not shippable:
        print("\nFAIL: no model may be shipped at all -- the app would be unusable.")
        return 1, rows

    kinds = {r["kind"] for r in shippable}
    if "separation" not in kinds:
        print("\nFAIL: no redistributable separation model.")
        return 1, rows

    if undeclared:
        print(f"\nFAIL: {undeclared} entries have no licence audit trail.")
        return 1, rows

    if strict and private:
        print("\nFAIL (--strict): the catalogue contains non-redistributable models.")
        return 1, rows

    return 0, rows


def check_urls(rows: list[dict]) -> int:
    import requests

    registry = load_registry()
    session = requests.Session()
    dead: list[str] = []

    print("\nchecking every download URL…\n")
    for spec in registry.models.values():
        for file_spec in spec.files:
            reachable = False
            for url in file_spec.urls:
                try:
                    response = session.get(
                        url, stream=True, timeout=(20, 60), allow_redirects=True
                    )
                    size = response.headers.get("content-length")
                    response.close()
                except requests.RequestException as exc:
                    print(f"  ERR  {file_spec.name}  {exc}")
                    continue

                if response.status_code == 200:
                    reachable = True
                    declared = file_spec.size_bytes
                    if size and declared and int(size) != declared:
                        print(
                            f"  SIZE {file_spec.name}: catalogue says {declared}, "
                            f"server says {size}"
                        )
                    else:
                        print(f"  ok   {file_spec.name}")
                    break
                print(f"  {response.status_code}  {url}")

            if not reachable:
                dead.append(f"{spec.id}/{file_spec.name}")

    if dead:
        print(f"\nFAIL: {len(dead)} file(s) have no working mirror:")
        for name in dead:
            print(f"  {name}")
        return 1

    print("\nevery catalogue file is reachable from at least one mirror.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-urls", action="store_true", help="also probe every URL")
    parser.add_argument(
        "--strict", action="store_true",
        help="fail if any non-redistributable model is listed (for packaging)",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    code, rows = check_licences(args.strict)
    if args.json:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    if args.check_urls:
        code = max(code, check_urls(rows))
    return code


if __name__ == "__main__":
    sys.exit(main())
