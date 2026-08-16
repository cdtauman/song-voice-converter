"""Compatibility Spike runner (Phase 1).

Answers one question: is there a single dependency matrix that runs RoFormer
separation, RMVPE/F0, RVC inference and pitch shifting together, reliably, on
this machine?

For every candidate it creates an isolated uv venv, installs the stack, runs
spike/smoke_test.py inside it, and records what happened. Output:

    spike/results/<timestamp>/results.json
    docs/compat-matrix.md
    constraints.txt        (only when a candidate fully passes)

Usage:
    python spike/run_spike.py                 # auto-detect GPU, run what applies
    python spike/run_spike.py --only cpu-311
    python spike/run_spike.py --keep-venvs
"""

from __future__ import annotations

import argparse
import ctypes
import json
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPIKE = ROOT / "spike"
RESULTS = SPIKE / "results"
VENVS = SPIKE / ".venvs"

TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_CU128_INDEX = "https://download.pytorch.org/whl/cu128"
TORCH_CU126_INDEX = "https://download.pytorch.org/whl/cu126"

#: Everything the Phase 2-6 pipeline needs, installed together on purpose --
#: the whole point is to surface resolver conflicts now rather than in Phase 5.
PIPELINE = [
    "numpy",
    "scipy",
    "soundfile",
    "soxr",
    "librosa",
    "pyloudnorm",
    "python-stretch",
    "audio-separator[cpu]",
    "torchfcpe",
]

PIPELINE_GPU = [
    "numpy",
    "scipy",
    "soundfile",
    "soxr",
    "librosa",
    "pyloudnorm",
    "python-stretch",
    "audio-separator[gpu]",
    "torchfcpe",
]


@dataclass
class Candidate:
    id: str
    python: str
    torch_index: str | None
    needs_nvidia: bool
    note: str


CANDIDATES: list[Candidate] = [
    Candidate("cpu-311", "3.11", TORCH_CPU_INDEX, False, "Python 3.11 + torch CPU"),
    Candidate("cpu-312", "3.12", TORCH_CPU_INDEX, False, "Python 3.12 + torch CPU"),
    Candidate("cu128-311", "3.11", TORCH_CU128_INDEX, True, "Python 3.11 + torch CUDA 12.8"),
    Candidate("cu126-311", "3.11", TORCH_CU126_INDEX, True, "Python 3.11 + torch CUDA 12.6"),
    Candidate("cu128-312", "3.12", TORCH_CU128_INDEX, True, "Python 3.12 + torch CUDA 12.8"),
]


@dataclass
class Outcome:
    candidate: str
    note: str
    status: str  # pass | fail | skipped
    reason: str = ""
    install_seconds: float = 0.0
    resolved: dict[str, str] = field(default_factory=dict)
    smoke: dict = field(default_factory=dict)
    freeze: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #

def has_nvidia_driver() -> bool:
    if sys.platform != "win32":
        return Path("/proc/driver/nvidia/version").exists()
    try:
        ctypes.WinDLL("nvcuda.dll")
        return True
    except OSError:
        return False


def run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout,
        encoding="utf-8", errors="replace",
    )


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def build_candidate(c: Candidate, uv: str) -> Outcome:
    out = Outcome(candidate=c.id, note=c.note, status="fail")
    venv = VENVS / c.id
    if venv.exists():
        shutil.rmtree(venv, ignore_errors=True)
    venv.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()

    r = run([uv, "venv", "--python", c.python, str(venv)], timeout=900)
    if r.returncode != 0:
        out.reason = f"venv creation failed: {r.stderr.strip()[:400]}"
        return out

    py = str(venv_python(venv))

    torch_cmd = [uv, "pip", "install", "--python", py, "torch"]
    if c.torch_index:
        torch_cmd += ["--index-url", c.torch_index]
    r = run(torch_cmd, timeout=3600)
    if r.returncode != 0:
        out.reason = f"torch install failed: {r.stderr.strip()[:600]}"
        out.install_seconds = round(time.perf_counter() - started, 1)
        return out

    pkgs = PIPELINE_GPU if c.needs_nvidia else PIPELINE
    r = run([uv, "pip", "install", "--python", py, *pkgs], timeout=3600)
    if r.returncode != 0:
        out.reason = f"pipeline install failed: {r.stderr.strip()[:1200]}"
        out.install_seconds = round(time.perf_counter() - started, 1)
        return out

    out.install_seconds = round(time.perf_counter() - started, 1)

    freeze = run([uv, "pip", "freeze", "--python", py], timeout=300)
    out.freeze = sorted(line.strip() for line in freeze.stdout.splitlines() if line.strip())
    for line in out.freeze:
        if "==" in line:
            name, _, ver = line.partition("==")
            if name.lower() in {
                "torch", "numpy", "scipy", "librosa", "soundfile", "soxr",
                "pyloudnorm", "python-stretch", "audio-separator", "torchfcpe",
                "onnxruntime", "onnxruntime-gpu",
            }:
                out.resolved[name] = ver

    smoke = run([py, str(SPIKE / "smoke_test.py")], timeout=1800)
    try:
        out.smoke = json.loads(smoke.stdout)
    except json.JSONDecodeError:
        out.reason = f"smoke test produced no JSON: {(smoke.stderr or smoke.stdout)[:600]}"
        return out

    required = ("torch", "numpy", "soundfile", "librosa", "pyloudnorm",
                "pitch_shift", "audio_separator", "ffmpeg")
    failed = [k for k in required if not out.smoke.get("probes", {}).get(k, {}).get("ok")]
    if failed:
        out.status = "fail"
        out.reason = "failed probes: " + ", ".join(failed)
    else:
        out.status = "pass"
    return out


# --------------------------------------------------------------------------- #

def write_constraints(out: Outcome, dest: Path) -> None:
    header = [
        "# SongVoice -- locked dependency matrix",
        "# Produced by spike/run_spike.py. Do not hand-edit.",
        f"# candidate : {out.candidate}  ({out.note})",
        f"# generated : {datetime.now(UTC).isoformat(timespec='seconds')}",
        f"# machine   : {platform.platform()}",
        "",
    ]
    dest.write_text("\n".join(header + out.freeze) + "\n", encoding="utf-8")


def write_matrix(outcomes: list[Outcome], dest: Path, gpu_present: bool) -> None:
    icon = {"pass": "✅", "fail": "❌", "skipped": "⏭️"}
    lines = [
        "# מטריצת תאימות — Compatibility Spike",
        "",
        f"**נוצר:** {datetime.now(UTC).astimezone().strftime('%d.%m.%Y %H:%M')}",
        f"**מכונה:** {platform.platform()}",
        f"**כרטיס NVIDIA:** {'נמצא' if gpu_present else 'לא נמצא'}",
        "",
        "> נוצר אוטומטית ע\"י `spike/run_spike.py`. לא לערוך ביד.",
        "",
        "## תוצאות",
        "",
        "| מועמד | תיאור | תוצאה | זמן התקנה | הערה |",
        "|-------|-------|-------|-----------|------|",
    ]
    for o in outcomes:
        secs = f"{o.install_seconds:.0f}s" if o.install_seconds else "—"
        reason = (o.reason or "").replace("|", "/").replace("\n", " ")[:120] or "—"
        mark = icon.get(o.status, "?")
        lines.append(
            f"| `{o.candidate}` | {o.note} | {mark} {o.status} | {secs} | {reason} |"
        )

    for o in outcomes:
        if o.status == "skipped":
            continue
        lines += ["", f"### `{o.candidate}` — {o.note}", ""]
        if o.resolved:
            lines += ["| חבילה | גרסה שנפתרה |", "|-------|--------------|"]
            lines += [f"| {k} | {v} |" for k, v in sorted(o.resolved.items())]
            lines.append("")
        probes = o.smoke.get("probes", {})
        if probes:
            lines += ["| בדיקה | תוצאה | פרטים |", "|-------|--------|--------|"]
            for name, data in probes.items():
                mark = "✅" if data.get("ok") else "❌"
                detail = data.get("error") or ", ".join(
                    f"{k}={v}" for k, v in data.items()
                    if k not in {"ok", "seconds", "error", "trace"}
                )
                lines.append(f"| {name} | {mark} | {str(detail).replace('|', '/')[:160]} |")
            lines.append("")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 compatibility spike")
    ap.add_argument("--only", action="append", help="run only these candidate ids")
    ap.add_argument("--keep-venvs", action="store_true")
    ap.add_argument("--force-gpu", action="store_true",
                    help="run CUDA candidates even if no NVIDIA driver is detected")
    args = ap.parse_args(argv)

    uv = shutil.which("uv")
    if not uv:
        print("uv is not installed. See https://docs.astral.sh/uv/", file=sys.stderr)
        return 2

    gpu = has_nvidia_driver() or args.force_gpu
    print(f"NVIDIA driver detected: {gpu}")

    selected = [c for c in CANDIDATES if not args.only or c.id in args.only]
    outcomes: list[Outcome] = []

    for c in selected:
        if c.needs_nvidia and not gpu:
            outcomes.append(Outcome(
                candidate=c.id, note=c.note, status="skipped",
                reason="לא נמצא כרטיס NVIDIA במכונה הזו — חייב לרוץ על מחשב היעד",
            ))
            print(f"[skip] {c.id}: no NVIDIA GPU on this machine")
            continue
        print(f"[run ] {c.id}: {c.note}")
        o = build_candidate(c, uv)
        print(f"[{o.status:>4}] {c.id}  {o.reason[:120]}")
        outcomes.append(o)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = RESULTS / stamp
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "results.json").write_text(
        json.dumps(
            {
                "machine": platform.platform(),
                "gpu_detected": gpu,
                "outcomes": [asdict(o) for o in outcomes],
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    write_matrix(outcomes, ROOT / "docs" / "compat-matrix.md", gpu)

    winner = next(
        (o for o in outcomes
         if o.status == "pass"
         and o.smoke.get("probes", {}).get("torch", {}).get("cuda_available")),
        None,
    ) or next((o for o in outcomes if o.status == "pass"), None)

    if winner:
        write_constraints(winner, ROOT / "constraints.txt")
        print(f"\nconstraints.txt written from candidate '{winner.candidate}'")
    else:
        print("\nNo candidate passed. constraints.txt not written.")

    if not args.keep_venvs:
        shutil.rmtree(VENVS, ignore_errors=True)

    print(f"results: {outdir / 'results.json'}")
    print(f"matrix : {ROOT / 'docs' / 'compat-matrix.md'}")
    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
