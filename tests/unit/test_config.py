from __future__ import annotations

import json
from pathlib import Path

import pytest

from svc_engine.config import Settings, load_settings, paths, save_settings


def test_convert_rejects_an_impossible_lufs_target() -> None:
    from svc_engine.cli import build_parser

    with pytest.raises(SystemExit):
        build_parser().parse_args(["convert", "song.wav", "--voice", "v", "--target-lufs", "2"])


def test_paths_are_all_under_the_root(tmp_path: Path) -> None:
    p = paths(tmp_path)
    for d in (p.models, p.voices, p.projects, p.cache, p.logs, p.work):
        assert str(d).startswith(str(tmp_path))


def test_ensure_creates_every_directory(tmp_path: Path) -> None:
    p = paths(tmp_path)
    p.ensure()
    for d in (p.root, p.models, p.voices, p.projects, p.cache, p.logs, p.work):
        assert d.is_dir()


def test_work_dir_is_ascii_only(tmp_path: Path) -> None:
    """Non-ASCII working paths break PyTorch and C++ tooling on Windows."""
    p = paths(tmp_path)
    assert p.work.name.isascii()


def test_settings_roundtrip(tmp_path: Path) -> None:
    p = paths(tmp_path)
    s = Settings(quality="max", target_lufs=-9.0, advanced_mode=True)
    save_settings(s, p)
    assert load_settings(p) == s


def test_defaults_when_file_missing(tmp_path: Path) -> None:
    s = load_settings(paths(tmp_path))
    assert s.quality == "balanced"
    assert s.language == "he"
    assert s.prefer_gpu is True


def test_corrupt_settings_file_does_not_crash(tmp_path: Path) -> None:
    p = paths(tmp_path)
    p.ensure()
    (p.root / "settings.json").write_text("{not json", encoding="utf-8")
    assert load_settings(p) == Settings()


def test_unknown_keys_are_ignored(tmp_path: Path) -> None:
    p = paths(tmp_path)
    p.ensure()
    (p.root / "settings.json").write_text(
        json.dumps({"quality": "fast", "from_a_future_version": 123}), encoding="utf-8"
    )
    assert load_settings(p).quality == "fast"


def test_env_override(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("SONGVOICE_HOME", str(tmp_path / "custom"))
    assert paths().root == tmp_path / "custom"
