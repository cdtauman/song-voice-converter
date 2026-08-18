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
#: Official PyTorch wheels for Intel GPUs (Arc, Core Ultra integrated Arc).
TORCH_XPU_INDEX = "https://download.pytorch.org/whl/xpu"

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
    "transformers",
    "faiss-cpu",
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
    "transformers",
    "faiss-cpu",
]


@dataclass
class Candidate:
    id: str
    python: str
    torch_index: str | None
    note: str
    needs_nvidia: bool = False
    needs_intel: bool = False
    #: Accelerator this candidate is meant to prove, if any.
    accelerator: str | None = None


CANDIDATES: list[Candidate] = [
    Candidate("cpu-311", "3.11", TORCH_CPU_INDEX, "Python 3.11 + torch CPU"),
    Candidate("cpu-312", "3.12", TORCH_CPU_INDEX, "Python 3.12 + torch CPU"),
    Candidate("xpu-311", "3.11", TORCH_XPU_INDEX, "Python 3.11 + torch XPU (Intel)",
              needs_intel=True, accelerator="xpu"),
    Candidate("cu128-311", "3.11", TORCH_CU128_INDEX, "Python 3.11 + torch CUDA 12.8",
              needs_nvidia=True, accelerator="cuda"),
    Candidate("cu126-311", "3.11", TORCH_CU126_INDEX, "Python 3.11 + torch CUDA 12.6",
              needs_nvidia=True, accelerator="cuda"),
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


def display_adapters() -> list[str]:
    """Installed display adapters, read from the Windows registry (no deps)."""
    if sys.platform != "win32":
        return []
    import winreg

    key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as root:
            i = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, i)
                except OSError:
                    break
                i += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as adapter:
                        desc, _ = winreg.QueryValueEx(adapter, "DriverDesc")
                except OSError:
                    continue
                if isinstance(desc, str) and desc not in names:
                    names.append(desc)
    except OSError:
        return []
    return names


def has_intel_gpu() -> bool:
    return any("intel" in a.lower() for a in display_adapters())


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
    if r.returncode != 0 and c.accelerator == "xpu":
        # The XPU index does not mirror every package; retry those from PyPI so a
        # missing mirror never masquerades as an Intel incompatibility.
        r = run([uv, "pip", "install", "--python", py, "--index-strategy",
                 "unsafe-best-match", *pkgs], timeout=3600)
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
                "onnxruntime", "onnxruntime-gpu", "transformers", "faiss-cpu",
            }:
                out.resolved[name] = ver

    # Read the report from a file: dependencies print banners to stdout and would
    # otherwise corrupt it.
    report = venv.parent / f"{c.id}-smoke.json"
    report.unlink(missing_ok=True)
    smoke = run([py, str(SPIKE / "smoke_test.py"), "--out", str(report)], timeout=1800)
    if not report.exists():
        out.reason = f"smoke test wrote no report: {(smoke.stderr or smoke.stdout)[-600:]}"
        return out
    try:
        out.smoke = json.loads(report.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        out.reason = f"smoke report is not valid JSON: {exc}"
        return out

    required = ("torch", "numpy", "soundfile", "librosa", "pyloudnorm",
                "pitch_shift", "audio_separator", "transformers", "faiss", "ffmpeg")
    failed = [k for k in required if not out.smoke.get("probes", {}).get(k, {}).get("ok")]
    if failed:
        out.status = "fail"
        out.reason = "failed probes: " + ", ".join(failed)
        return out

    # An accelerator candidate that cannot actually use its accelerator has not
    # proven anything, even if every library imports cleanly.
    if c.accelerator:
        acc = out.smoke.get("probes", {}).get(c.accelerator, {})
        if c.accelerator == "xpu" and not acc.get("ok"):
            out.status = "fail"
            out.reason = f"XPU לא זמין בפועל: {str(acc.get('error', 'unknown'))[:200]}"
            return out
        if c.accelerator == "cuda" and not out.smoke["probes"]["torch"].get("cuda_available"):
            out.status = "fail"
            out.reason = "CUDA לא זמין בפועל אחרי ההתקנה"
            return out

    out.status = "pass"
    return out


def build_support_matrix(outcomes: list[Outcome]) -> dict:
    """Collapse the per-candidate evidence into the per-component support matrix.

    Only a workload that actually ran counts. Anything else stays CPU.
    """
    notes = {
        "separation": "הפרדה",
        "f0": "זיהוי גובה",
        "conversion": "המרת קול",
        "pitch_shift": "הזזת גובה",
    }
    components: dict[str, dict] = {
        name: {"proofs": {}, "note_he": ""} for name in notes
    }

    for o in outcomes:
        if o.status != "pass":
            continue
        recorded = o.smoke.get("probes", {}).get("component_backends", {})
        if not recorded.get("ok"):
            continue
        for component, per_device in (recorded.get("matrix") or {}).items():
            if component not in components:
                continue
            for device, data in per_device.items():
                if device == "cpu" or not data.get("ok"):
                    continue
                level = "end_to_end" if data.get("end_to_end") else "ops"
                current = components[component]["proofs"].get(device)
                if current != "end_to_end":
                    components[component]["proofs"][device] = level

    for component, label in notes.items():
        proofs = components[component]["proofs"]
        if component == "pitch_shift":
            components[component]["note_he"] = (
                "רץ על המעבד מעצם טבעו — ספריית DSP ב-C++ בלי מסלול GPU."
            )
        elif proofs:
            devices = ", ".join(sorted(proofs))
            e2e = [d for d, lvl in proofs.items() if lvl == "end_to_end"]
            kind = "מודל מלא" if e2e else "עומס אמיתי ברמת האופרטורים"
            components[component]["note_he"] = f"{label}: אומת על {devices} ({kind})."
        else:
            components[component]["note_he"] = (
                f"{label}: לא אומת על שום מאיץ — רץ על המעבד."
            )

    return {
        "source": f"spike {datetime.now(UTC).isoformat(timespec='seconds')}",
        "machine": platform.platform(),
        "components": components,
    }


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


def write_pipeline_lock(uv: str, candidate: Candidate, constraints: Path, dest: Path) -> None:
    """Generate the installable Phase 2-5 lock from the Spike winner's matrix."""
    cmd = [
        uv, "pip", "compile", "--python-version", candidate.python,
        "--index-strategy", "unsafe-best-match", "--extra", "rvc",
        str(constraints), str(ROOT / "pyproject.toml"), "-o", str(dest),
    ]
    if candidate.torch_index:
        cmd += ["--index-url", candidate.torch_index, "--extra-index-url", "https://pypi.org/simple"]
    result = run(cmd, timeout=1800)
    if result.returncode != 0:
        raise RuntimeError(f"RVC lock generation failed: {result.stderr.strip()[:600]}")


def write_matrix(
    outcomes: list[Outcome],
    dest: Path,
    gpu_present: bool,
    intel_present: bool = False,
    adapters: list[str] | None = None,
) -> None:
    icon = {"pass": "✅", "fail": "❌", "skipped": "⏭️"}
    lines = [
        "# מטריצת תאימות — Compatibility Spike",
        "",
        f"**נוצר:** {datetime.now(UTC).astimezone().strftime('%d.%m.%Y %H:%M')}",
        f"**מכונה:** {platform.platform()}",
        f"**מתאמים גרפיים:** {', '.join(adapters or []) or 'לא זוהו'}",
        f"**כרטיס NVIDIA:** {'נמצא' if gpu_present else 'לא נמצא'}",
        f"**כרטיס Intel:** {'נמצא' if intel_present else 'לא נמצא'}",
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
                if name == "component_backends":
                    continue
                mark = "✅" if data.get("ok") else "❌"
                detail = data.get("error") or ", ".join(
                    f"{k}={v}" for k, v in data.items()
                    if k not in {"ok", "seconds", "error", "trace"}
                )
                lines.append(f"| {name} | {mark} | {str(detail).replace('|', '/')[:160]} |")
            lines.append("")

        backends = probes.get("component_backends", {})
        if backends.get("ok"):
            devices = backends.get("devices", [])
            lines += [
                f"**עומס אמיתי לכל רכיב** (מכשירים שנבדקו: {', '.join(devices)})",
                "",
                "| רכיב | " + " | ".join(devices) + " | מה נבדק |",
                "|------|" + "|".join(["------"] * len(devices)) + "|---------|",
            ]
            for component, per_device in (backends.get("matrix") or {}).items():
                cells = []
                ops = ""
                for device in devices:
                    data = per_device.get(device)
                    if data is None:
                        cells.append("—")
                        continue
                    cells.append("✅" if data.get("ok") else "❌")
                    ops = ops or str(data.get("ops", ""))
                lines.append(f"| {component} | " + " | ".join(cells) + f" | {ops} |")
            lines.append("")

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Phase 1 compatibility spike")
    ap.add_argument("--only", action="append", help="run only these candidate ids")
    ap.add_argument("--keep-venvs", action="store_true")
    ap.add_argument("--force-gpu", action="store_true",
                    help="run CUDA candidates even if no NVIDIA driver is detected")
    ap.add_argument("--force-intel", action="store_true",
                    help="run the XPU candidate even if no Intel GPU is detected")
    args = ap.parse_args(argv)

    uv = shutil.which("uv")
    if not uv:
        print("uv is not installed. See https://docs.astral.sh/uv/", file=sys.stderr)
        return 2

    adapters = display_adapters()
    gpu = has_nvidia_driver() or args.force_gpu
    intel = has_intel_gpu() or args.force_intel
    print(f"display adapters : {', '.join(adapters) or 'none detected'}")
    print(f"NVIDIA driver    : {gpu}")
    print(f"Intel GPU        : {intel}")

    selected = [c for c in CANDIDATES if not args.only or c.id in args.only]
    outcomes: list[Outcome] = []

    for c in selected:
        if c.needs_nvidia and not gpu:
            outcomes.append(Outcome(
                candidate=c.id, note=c.note, status="skipped",
                reason="לא נמצא כרטיס NVIDIA במכונה הזו — חייב לרוץ על מחשב עם NVIDIA",
            ))
            print(f"[skip] {c.id}: no NVIDIA GPU on this machine")
            continue
        if c.needs_intel and not intel:
            outcomes.append(Outcome(
                candidate=c.id, note=c.note, status="skipped",
                reason="לא נמצא כרטיס Intel במכונה הזו",
            ))
            print(f"[skip] {c.id}: no Intel GPU on this machine")
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

    write_matrix(outcomes, ROOT / "docs" / "compat-matrix.md", gpu, intel, adapters)

    support = build_support_matrix(outcomes)
    support_path = ROOT / "src" / "svc_engine" / "data" / "compute-support.json"
    support_path.parent.mkdir(parents=True, exist_ok=True)
    support_path.write_text(
        json.dumps(support, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"support matrix written: {support_path}")

    # Prefer an accelerated candidate that actually proved its accelerator.
    def rank(o: Outcome) -> int:
        probes = o.smoke.get("probes", {})
        if probes.get("torch", {}).get("cuda_available"):
            return 0
        if probes.get("xpu", {}).get("ok"):
            return 1
        return 2

    passed = sorted((o for o in outcomes if o.status == "pass"), key=rank)
    winner = passed[0] if passed else None

    if winner:
        constraints_path = ROOT / "constraints.txt"
        write_constraints(winner, constraints_path)
        write_pipeline_lock(uv, next(c for c in CANDIDATES if c.id == winner.candidate),
                            constraints_path, ROOT / "rvc-requirements.lock")
        print(f"\nconstraints.txt written from candidate '{winner.candidate}'")
        print("rvc-requirements.lock written from the same pipeline matrix")
    else:
        print("\nNo candidate passed. constraints.txt not written.")

    if not args.keep_venvs:
        shutil.rmtree(VENVS, ignore_errors=True)

    print(f"results: {outdir / 'results.json'}")
    print(f"matrix : {ROOT / 'docs' / 'compat-matrix.md'}")
    return 0 if winner else 1


if __name__ == "__main__":
    sys.exit(main())
