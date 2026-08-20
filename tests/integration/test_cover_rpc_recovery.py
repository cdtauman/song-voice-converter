"""The real cover RPC path must use JobRunner recovery and StepCache."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from svc_engine.config import paths
from svc_engine.rpc import Request, Server

WORKER = Path(__file__).with_name("_cover_rpc_worker.py")


def _worker(mode: str, root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    return subprocess.run(
        [sys.executable, str(WORKER), mode, str(root)],
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=30,
        env=env,
        check=False,
    )


def test_cover_rpc_hard_kill_resumes_predecessors_from_cache_and_cleans_scratch(
    tmp_path: Path,
) -> None:
    (tmp_path / "song.wav").write_bytes(b"source-audio")
    crashed = _worker("crash", tmp_path)
    assert crashed.returncode == 91, crashed.stderr

    snapshot_path = tmp_path / "data" / "jobs" / "rpc-cover-recovery.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert snapshot["status"] == "running"
    assert snapshot["steps"]["separate"]["status"] == "completed"
    assert snapshot["steps"]["analyze"]["status"] == "running"
    scratch = tmp_path / "data" / "work" / "jobs" / "rpc-cover-recovery"
    assert scratch.is_dir()

    recoverable = Server(paths(tmp_path / "data")).handle(
        Request(id="jobs", method="jobs.recoverable")
    )
    assert recoverable.ok
    assert recoverable.result[0]["job_id"] == "rpc-cover-recovery"
    assert recoverable.result[0]["kind"] == "cover"

    resumed = _worker("resume", tmp_path)
    assert resumed.returncode == 0, resumed.stderr
    payload = json.loads(resumed.stdout.strip().splitlines()[-1])
    result = payload["response"]["result"]
    assert any(
        event.get("event") == "job"
        and event.get("data", {}).get("job_id") == "rpc-cover-recovery"
        for event in payload["events"]
    )
    assert result["resumed"] is True
    assert result["cache_hits"]["separate"] is True
    assert result["cache_hits"]["analyze"] is False
    assert (tmp_path / "cover.wav").read_bytes() == b"durable-cover"
    assert not scratch.exists()
    assert not (
        tmp_path / "data" / "jobs" / "requests" / "rpc-cover-recovery.json"
    ).exists()
    assert (tmp_path / "calls.log").read_text(encoding="utf-8").splitlines() == [
        "separate",
        "analyze",
        "analyze",
        "render",
        "deliver",
    ]
    after = Server(paths(tmp_path / "data")).handle(
        Request(id="jobs", method="jobs.recoverable")
    )
    assert after.ok and after.result == []
