from __future__ import annotations

import csv
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import pytest

from svc_engine.benchmark import BenchmarkRunner, ExperimentSpec, VariantSpec, load_experiment


def _variant(identifier: str) -> VariantSpec:
    code = "import shutil;shutil.copyfile(r'{input}',r'{output}')"
    return VariantSpec(identifier, identifier, "fake", (sys.executable, "-c", code))


def test_runner_writes_table_manifest_html_audio_and_blind_mapping(tmp_path: Path) -> None:
    audio = tmp_path / "input.wav"
    audio.write_bytes(b"RIFF-test-audio")
    spec = ExperimentSpec("matrix", audio, (_variant("one"), _variant("two")), seed=19)
    output = BenchmarkRunner().run(spec, tmp_path / "result")
    assert (output / "results.csv").is_file()
    assert (output / "report.html").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["blind_map"].values()) == {"one", "two"}
    assert len(manifest["input"]["sha256"]) == 64
    with (output / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["status"] for row in rows] == ["ok", "ok"]
    assert all((output / str(row["audio"])).is_file() for row in rows)
    assert "השוואה עיוורת" in (output / "report.html").read_text(encoding="utf-8")


def test_experiment_loader_is_strict_and_resolves_relative_input(tmp_path: Path) -> None:
    (tmp_path / "source.wav").write_bytes(b"audio")
    path = tmp_path / "experiment.toml"
    path.write_text(
        "name='x'\ninput_audio='source.wav'\n"
        "[[variants]]\nid='a'\ncommand=['a']\n"
        "[[variants]]\nid='b'\ncommand=['b']\n",
        encoding="utf-8",
    )
    loaded = load_experiment(path)
    assert loaded.input_audio == (tmp_path / "source.wav").resolve()
    with pytest.raises(ValueError):
        load_experiment(tmp_path / "experiment.yaml")


def test_timeout_kills_descendant_and_aggregates_child_memory(tmp_path: Path) -> None:
    source = tmp_path / "input.wav"
    source.write_bytes(b"RIFF")
    pid_file = tmp_path / "child.pid"
    child = tmp_path / "child.py"
    child.write_text(
        "import os,sys,time\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text(str(os.getpid()), encoding='ascii')\n"
        "payload=bytearray(96*1024*1024)\n"
        "for offset in range(0,len(payload),4096): payload[offset]=1\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    parent = tmp_path / "parent.py"
    parent.write_text(
        "import subprocess,sys,time\n"
        "subprocess.Popen([sys.executable,sys.argv[1],sys.argv[2]])\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    variant = VariantSpec(
        "tree",
        "tree",
        "test",
        (sys.executable, str(parent), str(child), str(pid_file)),
    )
    output = BenchmarkRunner().run(
        ExperimentSpec("tree-timeout", source, (variant, _variant("control")), timeout_seconds=1.2),
        tmp_path / "result",
    )
    deadline = time.monotonic() + 2.0
    while not pid_file.is_file() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert pid_file.is_file(), "child never started"
    child_pid = int(pid_file.read_text(encoding="ascii"))
    assert not _pid_running(child_pid), "timeout left the inference child running"
    with (output / "results.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = {row["variant_id"]: row for row in csv.DictReader(stream)}
    assert rows["tree"]["status"] == "failed"
    assert "timeout" in rows["tree"]["error"]
    assert float(rows["tree"]["peak_ram_mb"]) >= 80.0


def _pid_running(pid: int) -> bool:
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = kernel32.OpenProcess(0x1000, False, pid)
    if not handle:
        return False
    exit_code = ctypes.c_ulong()
    try:
        return bool(kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))) and (
            exit_code.value == 259
        )
    finally:
        kernel32.CloseHandle(handle)
