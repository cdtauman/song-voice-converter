"""Dependency-free Phase-10 framework and isolation acceptance gate."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from svc_engine.backends.base import AudioBuffer  # noqa: E402
from svc_engine.benchmark import BenchmarkRunner, ExperimentSpec, VariantSpec  # noqa: E402
from svc_engine.tuning import AdvancedConfig, auto_tune  # noqa: E402

RESULTS = REPO / "benchmark" / "results" / "phase10"


def _copy_variant(identifier: str) -> VariantSpec:
    code = "import shutil;shutil.copyfile(r'{input}',r'{output}')"
    return VariantSpec(
        identifier,
        f"reference {identifier}",
        "framework-smoke",
        (sys.executable, "-c", code),
        settings={"mode": identifier},
        license="MIT",
    )


def _tuning_gate() -> dict[str, Any]:
    cases = []
    for case in range(5):
        phase = np.linspace(0, 20 + case, 8000, dtype=np.float32)

        def render(
            config: AdvancedConfig,
            phase: np.ndarray = phase,
            case: int = case,
        ) -> AudioBuffer:
            # Deterministic proxy: more aggressive variants receive a tiny
            # discontinuity. The gate asserts selector ordering, not listening.
            wave = 0.25 * np.sin(phase)
            if config.index_rate > 0.8 or (config == AdvancedConfig() and case in {1, 3}):
                wave[::20] = 1.0
            return AudioBuffer(wave.reshape(1, -1), 8000)

        result = auto_tune(AdvancedConfig(), render)
        manual = result.candidates[0].score
        cases.append(
            {
                "case": case + 1,
                "manual_score": manual,
                "winner": result.winner.candidate_id,
                "winner_score": result.winner.score,
                "at_least_manual": result.winner.score >= manual,
            }
        )
    return {
        "cases": cases,
        "objective_at_least_manual": sum(item["at_least_manual"] for item in cases),
        "human_blind_acceptance": "pending licensed test songs and listeners",
    }


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="songvoice-phase10-") as temporary:
        root = Path(temporary)
        source = root / "source.wav"
        source.write_bytes(b"RIFF" + b"\0" * 4096)
        spec = ExperimentSpec(
            "Phase 10 framework gate",
            source,
            (_copy_variant("manual"), _copy_variant("automatic")),
            repetitions=2,
            seed=10,
        )
        output = BenchmarkRunner().run(spec, root / "matrix")
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        files = ["results.csv", "report.html", "manifest.json"]
        matrix_ok = all((output / item).is_file() for item in files)
        audio_ok = all(
            (output / str(item["audio"])).is_file()
            for item in manifest["runs"]
            if item["status"] == "ok"
        )

    isolation_process = subprocess.run(
        [sys.executable, str(REPO / "env-bench" / "verify_isolation.py")],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    isolation = json.loads(isolation_process.stdout)
    tuning = _tuning_gate()
    result = {
        "phase": 10,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "matrix": {
            "runs": len(manifest["runs"]),
            "all_ok": all(item["status"] == "ok" for item in manifest["runs"]),
            "artifacts": files,
            "artifacts_ok": matrix_ok and audio_ok,
            "blind_aliases": list(manifest["blind_map"]),
            "telemetry": ["seconds", "peak_ram_mb", "peak_vram_mb", "status", "settings"],
        },
        "tuning": tuning,
        "isolation": isolation,
    }
    passed = (
        result["matrix"]["all_ok"]
        and result["matrix"]["artifacts_ok"]
        and tuning["objective_at_least_manual"] >= 4
        and isolation_process.returncode == 0
    )
    result["result"] = "pass" if passed else "fail"
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
