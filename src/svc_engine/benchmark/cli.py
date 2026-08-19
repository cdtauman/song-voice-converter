from __future__ import annotations

import argparse
from pathlib import Path

from svc_engine.benchmark import BenchmarkRunner, load_experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="svc-bench")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a TOML/JSON experiment matrix")
    run.add_argument("experiment", type=Path)
    run.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = BenchmarkRunner().run(load_experiment(args.experiment), args.out)
    print(f"results: {output / 'results.csv'}")
    print(f"report:  {output / 'report.html'}")
    print(f"manifest:{output / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
