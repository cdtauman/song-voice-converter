"""Phase 7 job graph, cache, cancellation and recovery tests."""

from __future__ import annotations

import errno
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from svc_engine.config import Settings, paths
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.jobs import (
    CancellationToken,
    Job,
    JobRunner,
    Step,
    StepCache,
    cache_key,
    cancel_process,
)
from svc_engine.jobs.recovery import JobState, RecoveryStore


def _write_action(name: str, calls: list[str], text: str = "ok"):  # type: ignore[no-untyped-def]
    def action(context):  # type: ignore[no-untyped-def]
        calls.append(name)
        target = context.output_dir / f"{name}.txt"
        target.write_text(text, encoding="utf-8")
        context.progress(1.0, f"{name} הושלם")
        return {"result": target}

    return action


def test_cache_key_tracks_contents_parameters_and_dependencies(tmp_path: Path) -> None:
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio-a")
    first = cache_key(
        step_id="separate",
        version="1",
        parameters={"quality": "balanced", "nested": {"b": 2, "a": 1}},
        input_files=(source,),
    )
    same = cache_key(
        step_id="separate",
        version="1",
        parameters={"nested": {"a": 1, "b": 2}, "quality": "balanced"},
        input_files=(source,),
    )
    changed_parameter = cache_key(
        step_id="separate",
        version="1",
        parameters={"quality": "max", "nested": {"a": 1, "b": 2}},
        input_files=(source,),
    )
    source.write_bytes(b"audio-b")
    changed_content = cache_key(
        step_id="separate",
        version="1",
        parameters={"nested": {"a": 1, "b": 2}, "quality": "balanced"},
        input_files=(source,),
    )
    dependent = cache_key(
        step_id="convert",
        version="1",
        parameters={},
        dependency_keys=(first,),
    )

    assert first == same
    assert len({first, changed_parameter, changed_content, dependent}) == 4


def test_cache_publishes_atomically_and_rejects_corruption(tmp_path: Path) -> None:
    cache = StepCache(tmp_path / "cache")
    source = tmp_path / "out.wav"
    source.write_bytes(b"complete output")
    key = "a" * 64
    entry = cache.publish(key, {"vocals": source})

    assert entry.outputs["vocals"].read_bytes() == b"complete output"
    entry.outputs["vocals"].write_bytes(b"corrupt")
    assert cache.lookup(key) is None


def test_cache_limit_evicts_oldest_and_can_protect_active_entry(tmp_path: Path) -> None:
    cache = StepCache(tmp_path / "cache")
    source = tmp_path / "out.bin"
    source.write_bytes(b"12345")
    first = cache.publish("1" * 64, {"out": source})
    time.sleep(0.01)
    second = cache.publish("2" * 64, {"out": source})

    stats = cache.enforce_limit(5, protected_keys={second.key})
    assert stats.entries == 1
    assert cache.lookup(first.key) is None
    assert cache.lookup(second.key) is not None


def test_runner_orders_graph_reports_progress_and_reuses_cache(tmp_path: Path) -> None:
    app_paths = paths(tmp_path)
    calls: list[str] = []

    def downstream(context):  # type: ignore[no-untyped-def]
        calls.append("mix")
        assert context.dependencies["separate"]["result"].read_text() == "stems"
        target = context.output_dir / "mix.wav"
        target.write_text("cover", encoding="utf-8")
        return {"cover": target}

    steps = (
        Step("mix", downstream, needs=("separate",), parameters={"lufs": -14}),
        Step("separate", _write_action("separate", calls, "stems")),
    )
    progress = []
    runner = JobRunner(app_paths)
    first = runner.run(Job("cover", steps, job_id="job-one"), on_progress=progress.append)
    second = runner.run(Job("cover", steps, job_id="job-two"))

    assert calls == ["separate", "mix"]
    assert list(first.steps) == ["separate", "mix"]
    assert not first.steps["separate"].cache_hit
    assert all(result.cache_hit for result in second.steps.values())
    assert progress[-1].overall_fraction == 1.0
    assert all(
        a.overall_fraction <= b.overall_fraction
        for a, b in zip(progress, progress[1:], strict=False)
    )
    assert not (app_paths.work / "jobs" / "job-one").exists()
    assert runner.history.get("job-one").status == "completed"  # type: ignore[union-attr]


def test_parameter_change_reruns_only_affected_step_and_dependents(tmp_path: Path) -> None:
    calls: list[str] = []
    runner = JobRunner(paths(tmp_path))

    def graph(target_lufs: float) -> tuple[Step, ...]:
        return (
            Step("separate", _write_action("separate", calls, "stems")),
            Step(
                "convert",
                _write_action("convert", calls, "voice"),
                needs=("separate",),
            ),
            Step(
                "mix",
                _write_action("mix", calls, "master"),
                needs=("convert",),
                parameters={"target_lufs": target_lufs},
            ),
        )

    runner.run(Job("cover", graph(-14.0), job_id="settings-a"))
    result = runner.run(Job("cover", graph(-12.0), job_id="settings-b"))

    assert calls == ["separate", "convert", "mix", "mix"]
    assert result.steps["separate"].cache_hit
    assert result.steps["convert"].cache_hit
    assert not result.steps["mix"].cache_hit


def test_failed_step_resumes_without_rerunning_completed_steps(tmp_path: Path) -> None:
    calls: list[str] = []
    fail_once = tmp_path / "fail-once"
    runner = JobRunner(paths(tmp_path / "data"))

    def unstable(context):  # type: ignore[no-untyped-def]
        calls.append("unstable")
        if not fail_once.exists():
            fail_once.write_text("failed", encoding="utf-8")
            raise RuntimeError("simulated crash")
        target = context.output_dir / "recovered.txt"
        target.write_text("recovered", encoding="utf-8")
        return {"out": target}

    job = Job(
        "recover",
        (
            Step("stable", _write_action("stable", calls)),
            Step("unstable", unstable, needs=("stable",)),
        ),
        job_id="resume-me",
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        runner.run(job)
    assert runner.recovery.load("resume-me").status is JobState.FAILED  # type: ignore[union-attr]

    result = runner.run(job)
    assert result.resumed
    assert result.steps["stable"].cache_hit
    assert calls == ["stable", "unstable", "unstable"]
    assert runner.recovery.discover() == []


def test_cooperative_cancellation_is_fast_and_cleans_scratch(tmp_path: Path) -> None:
    app_paths = paths(tmp_path)
    runner = JobRunner(app_paths)
    token = CancellationToken()
    caught: list[BaseException] = []

    def long_step(context):  # type: ignore[no-untyped-def]
        target = context.output_dir / "partial.wav"
        target.write_bytes(b"partial")
        while True:
            time.sleep(0.01)
            context.check_cancelled()

    def execute() -> None:
        try:
            runner.run(
                Job("cancel", (Step("long", long_step),), job_id="cancel-me"),
                cancellation=token,
            )
        except BaseException as exc:
            caught.append(exc)

    thread = threading.Thread(target=execute)
    started = time.monotonic()
    thread.start()
    time.sleep(0.05)
    token.cancel()
    thread.join(timeout=3.0)

    assert not thread.is_alive()
    assert time.monotonic() - started < 3.0
    assert isinstance(caught[0], EngineError)
    assert caught[0].code is ErrorCode.CANCELLED  # type: ignore[union-attr]
    assert not (app_paths.work / "jobs" / "cancel-me").exists()
    assert runner.recovery.discover() == []


def test_process_cancellation_forces_unresponsive_worker_within_deadline() -> None:
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    started = time.monotonic()
    cooperative = cancel_process(process, timeout=0.1)

    assert not cooperative
    assert process.poll() is not None
    assert time.monotonic() - started < 3.0


def test_disk_full_is_translated_and_partial_output_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app_paths = paths(tmp_path)
    runner = JobRunner(app_paths)

    def disk_full(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise OSError(errno.ENOSPC, "disk full")

    monkeypatch.setattr(runner.cache, "publish", disk_full)
    with pytest.raises(EngineError) as caught:
        runner.run(
            Job(
                "disk full",
                (Step("write", _write_action("write", [])),),
                job_id="disk-full",
            )
        )

    assert caught.value.code is ErrorCode.DISK_FULL
    assert not (app_paths.work / "jobs" / "disk-full").exists()
    assert not list((app_paths.cache / "steps").glob(".*.tmp"))


def test_invalid_graphs_are_rejected_before_execution(tmp_path: Path) -> None:
    runner = JobRunner(paths(tmp_path))
    action = _write_action("x", [])
    with pytest.raises(ValueError, match="cycle"):
        runner.run(
            Job(
                "cycle",
                (
                    Step("a", action, needs=("b",)),
                    Step("b", action, needs=("a",)),
                ),
            )
        )


def test_cleanup_removes_only_orphan_workspaces_and_cache_temporaries(tmp_path: Path) -> None:
    app_paths = paths(tmp_path)
    runner = JobRunner(app_paths, settings=Settings(keep_cache_gb=1.0))
    orphan = app_paths.work / "jobs" / "orphan"
    orphan.mkdir(parents=True)
    (orphan / "file.tmp").write_text("x", encoding="utf-8")
    cache_tmp = runner.cache.root / ".abandoned.tmp"
    cache_tmp.mkdir()

    report = runner.cleanup()
    assert report["workspaces_removed"] == 1
    assert report["cache_temporaries_removed"] == 1
    assert not orphan.exists()
    assert not cache_tmp.exists()


def test_recovery_discovery_preserves_but_skips_corrupt_state(tmp_path: Path) -> None:
    store = RecoveryStore(tmp_path)
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    snapshot = store.create(
        job_id="valid-job", name="valid", plan_signature="abc", step_ids=["one"]
    )
    snapshot.status = JobState.RUNNING
    store.save(snapshot)

    assert [item.job_id for item in store.discover()] == ["valid-job"]
    assert corrupt.read_text(encoding="utf-8") == "{not json"


def test_recovery_json_is_always_a_complete_object(tmp_path: Path) -> None:
    store = RecoveryStore(tmp_path)
    snapshot = store.create(job_id="atomic", name="atomic", plan_signature="x", step_ids=["a"])
    for _ in range(20):
        store.save(snapshot)
        saved = json.loads((tmp_path / "atomic.json").read_text(encoding="utf-8"))
        assert saved["job_id"] == "atomic"
