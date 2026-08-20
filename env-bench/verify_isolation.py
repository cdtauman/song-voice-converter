"""Static isolation and source-pin gate; safe before heavyweight bootstrap."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path


def main() -> int:
    bench = Path(__file__).resolve().parent
    repo = bench.parent
    source_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (repo / "src").rglob("*.py")
    ).lower()
    violations = [token for token in ("seed-vc", "seed_vc", "ddsp-svc") if token in source_text]
    hatch = (repo / "pyproject.toml").read_text(encoding="utf-8")
    build_isolated = 'packages = ["src/svc_engine", "src/svc_app"]' in hatch
    sources = json.loads((bench / "sources.json").read_text(encoding="utf-8"))["engines"]
    receipts: dict[str, object] = {}
    for engine, spec in sources.items():
        runtime = bench / "runtimes" / engine
        if not runtime.exists():
            receipts[engine] = "not bootstrapped"
            continue
        receipt = json.loads((runtime / "receipt.json").read_text(encoding="utf-8-sig"))
        actual = subprocess.run(
            ["git", "-C", str(runtime / "source"), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        receipts[engine] = {"expected": spec["commit"], "actual": actual}
        if actual != spec["commit"] or receipt["commit"] != actual:
            violations.append(f"{engine} source pin")
    constraints_hash = hashlib.sha256((repo / "constraints.txt").read_bytes()).hexdigest()
    result = {
        "core_import_violations": violations,
        "hatch_build_isolated": build_isolated,
        "constraints_sha256": constraints_hash,
        "runtimes": receipts,
        "result": "pass" if not violations and build_isolated else "fail",
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["result"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
