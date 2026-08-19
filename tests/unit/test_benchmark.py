from __future__ import annotations

import csv
import json
import sys
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
