"""Validate five human blind comparisons between manual and automatic tuning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ballots", type=Path)
    args = parser.parse_args()
    raw = json.loads(args.ballots.read_text(encoding="utf-8"))
    cases = raw.get("cases") if isinstance(raw, dict) else None
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("ballot file must contain exactly five blind cases")
    wins = 0
    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict) or case.get("preferred") not in {
            "automatic", "manual", "tie"
        }:
            raise ValueError(f"case {index} requires preferred=automatic|manual|tie")
        if not case.get("licensed_audio_confirmed"):
            raise ValueError(f"case {index} lacks licensed_audio_confirmed")
        if case["preferred"] in {"automatic", "tie"}:
            wins += 1
    result = {"cases": 5, "automatic_at_least_manual": wins, "required": 4}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if wins >= 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
