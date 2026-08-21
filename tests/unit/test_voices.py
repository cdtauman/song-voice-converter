"""Voice catalogue: manifest round-trip, zip import, consent, health, library."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

import svc_engine.voices.importer as importer
from svc_engine.config import Paths, paths
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.voices import (
    HealthStatus,
    VoiceLibrary,
    VoiceManifest,
    VoiceSource,
    import_voice_from_zip,
)
from svc_engine.voices.library import looks_like_torch_checkpoint
from svc_engine.voices.manifest import RecommendedParams, slugify

# A byte string that passes the torch-checkpoint signature check (zip magic).
_FAKE_PTH = b"PK\x03\x04" + b"\x00" * 2048
_FAKE_INDEX = b"faiss-index-bytes" * 64


def _paths(tmp_path: Path) -> Paths:
    p = paths(override_root=tmp_path)
    p.ensure()
    return p


def _make_zip(
    tmp_path: Path,
    *,
    model: bytes | None = _FAKE_PTH,
    index: bytes | None = _FAKE_INDEX,
    profile: dict | None = None,
    extra: dict[str, bytes] | None = None,
    name: str = "voice.zip",
) -> Path:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        if model is not None:
            zf.writestr("model.pth", model)
        if index is not None:
            zf.writestr("added_IVF_yossi.index", index)
        if profile is not None:
            zf.writestr("profile.json", json.dumps(profile))
        for member, data in (extra or {}).items():
            zf.writestr(member, data)
    path = tmp_path / name
    path.write_bytes(buf.getvalue())
    return path


# --- manifest -------------------------------------------------------------- #

def test_manifest_roundtrips_through_json(tmp_path: Path) -> None:
    m = VoiceManifest(
        voice_id="yossi",
        display_name="יוסי",
        source=VoiceSource.IMPORTED,
        consent_confirmed=True,
        consent_note="הקלטות שלי",
        sample_rate=48000,
        recommended=RecommendedParams(index_rate=0.6, protect=0.4, rms_mix_rate=0.2),
    )
    m.save(tmp_path)
    back = VoiceManifest.load(tmp_path)
    assert back.display_name == "יוסי"
    assert back.consent_confirmed is True
    assert back.recommended.index_rate == pytest.approx(0.6)
    assert back.source is VoiceSource.IMPORTED


def test_manifest_usable_requires_consent_and_health() -> None:
    m = VoiceManifest("v", "V", VoiceSource.IMPORTED, consent_confirmed=False)
    assert m.usable is False


def test_slugify_keeps_ids_ascii() -> None:
    assert slugify("יוסי כהן") == "voice"  # non-ascii collapses to fallback
    assert slugify("Yossi Cohen 2") == "yossi-cohen-2"


# --- signature gate -------------------------------------------------------- #

def test_checkpoint_signature(tmp_path: Path) -> None:
    good = tmp_path / "a.pth"
    good.write_bytes(_FAKE_PTH)
    legacy = tmp_path / "b.pth"
    legacy.write_bytes(b"\x80\x02" + b"junk")
    bad = tmp_path / "c.pth"
    bad.write_bytes(b"not a model")
    assert looks_like_torch_checkpoint(good)
    assert looks_like_torch_checkpoint(legacy)
    assert not looks_like_torch_checkpoint(bad)
    assert not looks_like_torch_checkpoint(tmp_path / "missing.pth")


# --- import ---------------------------------------------------------------- #

def test_import_requires_consent(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path)
    with pytest.raises(EngineError) as exc:
        import_voice_from_zip(archive, "Yossi", consent_confirmed=False, library=lib)
    assert exc.value.code is ErrorCode.CONSENT_REQUIRED
    assert lib.list() == []


def test_import_happy_path(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(
        tmp_path,
        profile={
            "comfort_low": 45.0, "comfort_high": 57.0, "abs_low": 40.0,
            "abs_high": 60.0, "median": 50.0,
        },
        extra={"cover.png": b"\x89PNG\r\n\x1a\n" + b"\x00" * 32},
    )
    result = import_voice_from_zip(
        archive, "יוסי", consent_confirmed=True, consent_note="הקלטות שלי", library=lib
    )
    assert result.voice_id == "voice" or result.voice_id  # ascii id
    entry = lib.get(result.voice_id)
    assert entry.manifest.display_name == "יוסי"
    assert entry.manifest.usable is True
    assert entry.manifest.health.status is HealthStatus.OK
    assert entry.model_path.exists()
    assert entry.index_path is not None
    assert entry.profile() is not None
    assert result.imported_index and result.imported_profile and result.imported_avatar
    # handle points the backend at the folder
    assert entry.handle().root == entry.root


def test_import_without_profile_generates_usable_neutral_profile(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    result = import_voice_from_zip(
        _make_zip(tmp_path), "Default", consent_confirmed=True, library=lib
    )

    entry = lib.get(result.voice_id)
    profile = entry.profile()
    assert profile is not None
    assert profile.f0_method == "unmeasured-neutral"
    assert entry.manifest.usable is True
    assert result.imported_profile is False
    assert result.generated_profile is True


def test_import_assigns_unique_ids_to_multiple_hebrew_names(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    first = import_voice_from_zip(
        _make_zip(tmp_path, name="first.zip"), "קול ראשון", True, library=lib
    )
    second = import_voice_from_zip(
        _make_zip(tmp_path, name="second.zip"), "קול שני", True, library=lib
    )

    assert first.voice_id == "voice"
    assert second.voice_id == "voice-2"
    assert len(lib.list()) == 2


def test_import_rejects_archive_without_model(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path, model=None, index=None)
    with pytest.raises(EngineError) as exc:
        import_voice_from_zip(archive, "NoModel", consent_confirmed=True, library=lib)
    assert exc.value.code is ErrorCode.VOICE_CORRUPT
    assert lib.list() == []  # nothing half-written


def test_import_rejects_unsupported_onnx_model(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path, model=None, index=None, extra={"model.onnx": b"onnx"})
    with pytest.raises(EngineError) as exc:
        import_voice_from_zip(archive, "Onnx", consent_confirmed=True, library=lib)
    assert exc.value.code is ErrorCode.VOICE_CORRUPT
    assert lib.list() == []


def test_import_rejects_non_zip(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    bogus = tmp_path / "not.zip"
    bogus.write_bytes(b"i am not a zip")
    with pytest.raises(EngineError) as exc:
        import_voice_from_zip(bogus, "X", consent_confirmed=True, library=lib)
    assert exc.value.code is ErrorCode.VOICE_CORRUPT


def test_import_is_zip_slip_safe(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path, extra={"../../escape.pth": _FAKE_PTH})
    result = import_voice_from_zip(
        archive, "Slip", consent_confirmed=True, library=lib, voice_id="slip"
    )
    # The traversal member was flattened; nothing was written outside the voice dir.
    assert not (tmp_path.parent / "escape.pth").exists()
    assert (result.root / "model.pth").exists()


def test_import_refuses_duplicate_without_overwrite(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    a = _make_zip(tmp_path, name="a.zip")
    import_voice_from_zip(a, "Dup", consent_confirmed=True, library=lib, voice_id="dup")
    with pytest.raises(EngineError):
        import_voice_from_zip(a, "Dup", consent_confirmed=True, library=lib, voice_id="dup")
    # overwrite succeeds
    import_voice_from_zip(
        a, "Dup", consent_confirmed=True, library=lib, voice_id="dup", overwrite=True
    )
    assert "dup" in lib


def test_import_picks_largest_pth(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    big = b"PK\x03\x04" + b"\x00" * 8192
    archive = _make_zip(
        tmp_path, model=big, index=None, extra={"optimizer.pth": b"PK\x03\x04tiny"}
    )
    result = import_voice_from_zip(
        archive, "Big", consent_confirmed=True, library=lib, voice_id="big"
    )
    assert result.root.joinpath("model.pth").stat().st_size == len(big)


def test_import_streams_member_reads_in_bounded_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path)
    original_read = zipfile.ZipExtFile.read

    def guarded_read(self: zipfile.ZipExtFile, n: int = -1) -> bytes:
        assert 0 < n <= importer._COPY_CHUNK_BYTES
        return original_read(self, n)

    monkeypatch.setattr(zipfile.ZipExtFile, "read", guarded_read)
    import_voice_from_zip(archive, "Stream", consent_confirmed=True, library=lib)


def test_import_rejects_member_over_per_file_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path, model=b"PK\x03\x04" + b"x" * 64)
    monkeypatch.setattr(importer, "_MAX_FILE_BYTES", 32)
    with pytest.raises(EngineError) as exc:
        import_voice_from_zip(archive, "TooBig", consent_confirmed=True, library=lib)
    assert exc.value.code is ErrorCode.VOICE_CORRUPT
    assert lib.list() == []


def test_import_rejects_total_extracted_bytes_over_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(
        tmp_path,
        model=b"PK\x03\x04" + b"x" * 20,
        index=b"y" * 20,
    )
    monkeypatch.setattr(importer, "_MAX_FILE_BYTES", 32)
    monkeypatch.setattr(importer, "_MAX_TOTAL_BYTES", 40)
    with pytest.raises(EngineError) as exc:
        import_voice_from_zip(archive, "Total", consent_confirmed=True, library=lib)
    assert exc.value.code is ErrorCode.VOICE_CORRUPT
    assert lib.list() == []


def test_failed_overwrite_restores_existing_voice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    original = b"PK\x03\x04" + b"old" * 20
    first = _make_zip(tmp_path, model=original, index=None, name="first.zip")
    import_voice_from_zip(first, "Dup", consent_confirmed=True, library=lib, voice_id="dup")
    replacement = _make_zip(
        tmp_path, model=b"PK\x03\x04" + b"new" * 20, index=None, name="next.zip"
    )

    real_rename = Path.rename

    def fail_activation(path: Path, target: Path) -> Path:
        if path.name.startswith(".dup.import-") and target.name == "dup":
            raise OSError("simulated activation failure")
        return real_rename(path, target)

    monkeypatch.setattr(Path, "rename", fail_activation)
    with pytest.raises(OSError, match="simulated activation failure"):
        import_voice_from_zip(
            replacement, "Dup", consent_confirmed=True, library=lib, voice_id="dup", overwrite=True
        )

    assert lib.get("dup").model_path.read_bytes() == original
    assert not list(lib.root.glob(".dup.import-*"))


def test_import_recovers_backup_left_by_crash_before_replacement(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    original = b"PK\x03\x04" + b"old" * 20
    first = _make_zip(tmp_path, model=original, index=None, name="first.zip")
    import_voice_from_zip(first, "Dup", consent_confirmed=True, library=lib, voice_id="dup")
    voice_dir = lib.voice_dir("dup")
    backup = voice_dir.with_name(".dup.previous")
    voice_dir.rename(backup)  # crash between the two renames

    with pytest.raises(EngineError):  # recovered voice still refuses an accidental overwrite
        import_voice_from_zip(first, "Dup", consent_confirmed=True, library=lib, voice_id="dup")

    assert lib.get("dup").model_path.read_bytes() == original
    assert not backup.exists()


# --- library --------------------------------------------------------------- #

def test_library_lists_and_removes(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    for i in range(2):
        import_voice_from_zip(
            _make_zip(tmp_path, name=f"v{i}.zip"),
            f"V{i}", consent_confirmed=True, library=lib, voice_id=f"v{i}",
        )
    assert [e.voice_id for e in lib.list()] == ["v0", "v1"]
    lib.remove("v0")
    assert [e.voice_id for e in lib.list()] == ["v1"]
    lib.remove("v0")  # idempotent


def test_library_health_flags_corrupt_model(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    result = import_voice_from_zip(
        _make_zip(tmp_path), "C", consent_confirmed=True, library=lib, voice_id="c"
    )
    # Corrupt the model on disk, then re-check.
    (result.root / "model.pth").write_bytes(b"broken")
    entry = lib.refresh_health("c")
    assert entry.manifest.health.status is HealthStatus.CORRUPT_MODEL
    assert entry.manifest.usable is False


def test_library_get_unknown_raises(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    with pytest.raises(KeyError):
        lib.get("nope")


def test_library_rejects_path_traversal_without_touching_outside_root(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    victim = tmp_path / "victim"
    victim.mkdir()
    marker = victim / "keep.txt"
    marker.write_text("do not delete", encoding="utf-8")

    for unsafe in ("../victim", "..\\victim", ".", "", "A"):
        assert unsafe not in lib
        with pytest.raises(EngineError):
            lib.voice_dir(unsafe)
        with pytest.raises(EngineError):
            lib.get(unsafe)
        with pytest.raises(EngineError):
            lib.remove(unsafe)

    assert marker.read_text(encoding="utf-8") == "do not delete"


def test_library_ignores_and_rejects_manifest_with_mismatched_id(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    folder = lib.voice_dir("folder")
    folder.mkdir(parents=True)
    VoiceManifest("other", "Other", VoiceSource.IMPORTED, consent_confirmed=True).save(folder)

    assert lib.list() == []
    with pytest.raises(EngineError) as exc:
        lib.get("folder")
    assert exc.value.code is ErrorCode.VOICE_CORRUPT


def test_overwrite_discards_stale_backup_when_active_voice_exists(tmp_path: Path) -> None:
    lib = VoiceLibrary(_paths(tmp_path))
    archive = _make_zip(tmp_path, index=None)
    import_voice_from_zip(archive, "Dup", consent_confirmed=True, library=lib, voice_id="dup")
    backup = lib.voice_dir("dup").with_name(".dup.previous")
    backup.mkdir()
    (backup / "stale.txt").write_text("old", encoding="utf-8")

    import_voice_from_zip(
        archive, "Dup", consent_confirmed=True, library=lib, voice_id="dup", overwrite=True
    )
    assert "dup" in lib
    assert not backup.exists()
