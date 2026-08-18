"""Project persistence and SQLite job-history coverage."""

from __future__ import annotations

from pathlib import Path

import pytest

from svc_engine.history import HistoryStore
from svc_engine.projects import ProjectStore


def test_project_roundtrip_handles_hebrew_paths_and_preserves_created_time(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path / "פרויקטים")
    first = store.save("song-1", name="שיר שלי", data={"input": "C:/שירים/קול.wav"})
    second = store.save("song-1", name="שיר מעודכן", data={"quality": "max"})

    assert store.load("song-1") == second
    assert second.created_at == first.created_at
    assert (store.root / "song-1.json.bak").is_file()
    assert store.list() == [second]


def test_project_load_falls_back_to_last_known_good_backup(tmp_path: Path) -> None:
    store = ProjectStore(tmp_path)
    first = store.save("safe", name="first", data={"value": 1})
    store.save("safe", name="second", data={"value": 2})
    (tmp_path / "safe.json").write_text("{corrupt", encoding="utf-8")

    assert store.load("safe") == first


@pytest.mark.parametrize("project_id", ["../escape", "CON", "bad/name", "עברית"])
def test_project_ids_are_portable_single_path_components(
    tmp_path: Path, project_id: str
) -> None:
    with pytest.raises(ValueError):
        ProjectStore(tmp_path).save(project_id, name="x", data={})


def test_history_upsert_and_ordering(tmp_path: Path) -> None:
    store = HistoryStore(tmp_path / "db.sqlite")
    store.record(job_id="a", name="A", status="running", progress=0.25)
    store.record(job_id="a", name="A", status="completed", progress=1.0)
    store.record(job_id="b", name="B", status="failed", progress=0.5, error="boom")

    assert store.get("a").status == "completed"  # type: ignore[union-attr]
    assert store.get("a").progress == 1.0  # type: ignore[union-attr]
    assert [item.job_id for item in store.list(limit=2)] == ["b", "a"]
    assert store.get("b").error == "boom"  # type: ignore[union-attr]
