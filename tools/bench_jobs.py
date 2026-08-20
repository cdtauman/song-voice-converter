"""Phase 7 resilience gate: hard-crash every representative job stage.

This harness deliberately exits the worker process from inside each stage,
then reconstructs the same graph in a fresh process. It verifies that completed
predecessors come from the content cache, the interrupted stage reruns, no
scratch files remain, and a third identical job is a 100% cache hit.

    python tools/bench_jobs.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.config import paths  # noqa: E402
from svc_engine.jobs import Job, JobRunner, Step, cancel_process  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
RESULTS = REPO / "benchmark" / "results" / "jobs"
STAGES = ("separate", "analyze", "convert", "master")


def _steps(root: Path, crash_stage: str | None) -> tuple[Step, ...]:
    log = root / "calls.log"
    marker = root / f"crashed-{crash_stage}"
    steps: list[Step] = []
    previous: str | None = None
    for stage in STAGES:

        def action(context, *, _stage: str = stage, _previous: str | None = previous):  # type: ignore[no-untyped-def]
            with log.open("a", encoding="utf-8") as stream:
                stream.write(f"{_stage}\n")
                stream.flush()
                os.fsync(stream.fileno())
            target = context.output_dir / f"{_stage}.bin"
            target.write_bytes(f"complete:{_stage}".encode())
            if _stage == crash_stage and not marker.exists():
                marker.write_text("crashed once", encoding="utf-8")
                os._exit(91)
            if _previous is not None:
                dependency = context.dependencies[_previous]["artifact"]
                if dependency.read_bytes() != f"complete:{_previous}".encode():
                    raise RuntimeError("dependency cache returned wrong bytes")
            context.progress(1.0, f"{_stage} complete")
            return {"artifact": target}

        steps.append(
            Step(
                stage,
                action,
                needs=(previous,) if previous else (),
                parameters={"stage": stage},
                version="phase7-gate-v1",
            )
        )
        previous = stage
    return tuple(steps)


def worker(root: Path, crash_stage: str | None, job_id: str) -> int:
    runner = JobRunner(paths(root / "data"))
    result = runner.run(
        Job("Phase 7 crash matrix", _steps(root, crash_stage), job_id=job_id)
    )
    print(
        json.dumps(
            {
                "status": result.status.value,
                "resumed": result.resumed,
                "cache_hits": {
                    name: step.cache_hit for name, step in result.steps.items()
                },
                "scratch_exists": (runner.work_root / job_id).exists(),
            }
        )
    )
    return 0


def _run_worker(
    root: Path, crash_stage: str | None, job_id: str
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--root",
        str(root),
        "--job-id",
        job_id,
    ]
    if crash_stage:
        command.extend(["--crash-stage", crash_stage])
    return subprocess.run(command, text=True, capture_output=True, timeout=30, check=False)


def _json_stdout(process: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    if process.returncode != 0:
        raise RuntimeError(f"worker failed ({process.returncode}): {process.stderr}")
    return dict(json.loads(process.stdout.strip().splitlines()[-1]))


def run_gate() -> dict[str, Any]:
    matrix: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="songvoice-phase7-") as temporary:
        base = Path(temporary) / "נתיב עם רווחים"
        base.mkdir()
        for crash_index, crash_stage in enumerate(STAGES):
            root = base / crash_stage
            root.mkdir()
            job_id = f"crash-{crash_stage}"
            crashed = _run_worker(root, crash_stage, job_id)
            if crashed.returncode != 91:
                raise RuntimeError(
                    f"{crash_stage} did not hard-crash as expected: "
                    f"{crashed.returncode} {crashed.stderr}"
                )
            snapshot = json.loads(
                (root / "data" / "jobs" / f"{job_id}.json").read_text(encoding="utf-8")
            )
            resumed = _json_stdout(_run_worker(root, crash_stage, job_id))
            cached = _json_stdout(_run_worker(root, crash_stage, f"cache-{crash_stage}"))
            calls = (root / "calls.log").read_text(encoding="utf-8").splitlines()
            expected = {
                stage: (2 if stage == crash_stage else 1)
                for stage in STAGES
            }
            counts = {stage: calls.count(stage) for stage in STAGES}
            predecessors = STAGES[:crash_index]
            if snapshot["status"] != "running":
                raise RuntimeError(f"{crash_stage}: hard crash was not recoverable")
            if not resumed["resumed"] or resumed["scratch_exists"]:
                raise RuntimeError(f"{crash_stage}: resume/scratch invariant failed")
            if not all(resumed["cache_hits"][stage] for stage in predecessors):
                raise RuntimeError(f"{crash_stage}: a completed predecessor reran")
            if counts != expected:
                raise RuntimeError(f"{crash_stage}: call counts {counts}, expected {expected}")
            if not all(cached["cache_hits"].values()):
                raise RuntimeError(f"{crash_stage}: identical second job missed cache")
            matrix.append(
                {
                    "crash_stage": crash_stage,
                    "hard_exit_code": crashed.returncode,
                    "recovery_snapshot": snapshot["status"],
                    "predecessors_reused": list(predecessors),
                    "call_counts": counts,
                    "second_job_all_cache_hits": True,
                    "scratch_clean": True,
                }
            )

        sleeper = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--sleep-worker"]
        )
        started = time.monotonic()
        cooperative = cancel_process(sleeper, timeout=0.15)
        cancel_seconds = time.monotonic() - started
        if cooperative or sleeper.poll() is None or cancel_seconds >= 3.0:
            raise RuntimeError("forced cancellation missed the three-second deadline")

    return {
        "phase": 7,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stages": list(STAGES),
        "crash_matrix": matrix,
        "forced_cancellation": {
            "cooperative": cooperative,
            "seconds": cancel_seconds,
            "under_three_seconds": cancel_seconds < 3.0,
        },
        "cache_rule": "input content + parameters + dependency keys + step version",
        "result": "pass",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--sleep-worker", action="store_true")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--crash-stage", choices=STAGES)
    parser.add_argument("--job-id", default="phase7-worker")
    args = parser.parse_args()
    if args.sleep_worker:
        time.sleep(30)
        return 0
    if args.worker:
        if args.root is None:
            parser.error("--worker requires --root")
        return worker(args.root, args.crash_stage, args.job_id)

    report = run_gate()
    RESULTS.mkdir(parents=True, exist_ok=True)
    destination = RESULTS / "results.json"
    destination.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nWrote {destination.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
