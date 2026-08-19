"""Uniform subprocess adapter for isolated third-party benchmark engines.

This module intentionally lives outside ``src`` and imports no SongVoice code.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


def _runtime(root: Path, engine: str) -> tuple[Path, Path]:
    runtime = root / "runtimes" / engine
    python = runtime / ".venv" / "Scripts" / "python.exe"
    source = runtime / "source"
    receipt = runtime / "receipt.json"
    if not python.is_file() or not source.is_dir() or not receipt.is_file():
        raise RuntimeError(f"{engine} is not bootstrapped under env-bench")
    metadata = json.loads(receipt.read_text(encoding="utf-8-sig"))
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    ).stdout.strip()
    if actual != metadata["commit"]:
        raise RuntimeError(f"{engine} checkout no longer matches its receipt")
    return python, source


def _seed(args: argparse.Namespace, root: Path) -> None:
    python, source = _runtime(root, "seed")
    generated = args.output.parent / f".{args.output.stem}-seed"
    generated.mkdir(parents=True, exist_ok=True)
    command = [
        str(python), str(source / "inference.py"),
        "--source", str(args.input), "--target", str(args.reference),
        "--output", str(generated), "--diffusion-steps", str(args.diffusion_steps),
        "--f0-condition", "True", "--auto-f0-adjust", "False",
        "--semi-tone-shift", str(args.semitones), "--fp16", str(args.fp16),
    ]
    subprocess.run(command, cwd=source, timeout=args.timeout, check=True)
    produced = sorted(generated.glob("*.wav"), key=lambda item: item.stat().st_mtime_ns)
    if not produced:
        raise RuntimeError("Seed-VC produced no WAV")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(produced[-1], args.output)


def _ddsp(args: argparse.Namespace, root: Path) -> None:
    python, source = _runtime(root, "ddsp")
    if not args.model:
        raise ValueError("DDSP-SVC requires --model")
    command = [
        str(python), str(source / "main_reflow.py"),
        "-i", str(args.input), "-m", str(args.model), "-o", str(args.output),
        "-k", str(args.semitones), "-eak", "0",
    ]
    subprocess.run(command, cwd=source, timeout=args.timeout, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", choices=["seed", "ddsp"])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--semitones", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=30)
    parser.add_argument("--fp16", choices=["True", "False"], default="True")
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error("--input must be an existing audio file")
    if args.engine == "seed":
        if args.reference is None or not args.reference.is_file():
            parser.error("Seed-VC requires an existing --reference audio file")
        _seed(args, Path(__file__).resolve().parent)
    else:
        _ddsp(args, Path(__file__).resolve().parent)
    if not args.output.is_file() or args.output.stat().st_size == 0:
        raise RuntimeError("adapter output contract was not satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
