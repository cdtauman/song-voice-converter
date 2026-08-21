"""Phase 11 provisioning, packaging boundaries, and update transactions."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import pytest

from svc_engine.config import paths
from svc_engine.errors import EngineError
from svc_engine.provisioning import CORE_MODEL_IDS, Provisioner
from svc_engine.resources import FileSpec, LicenseInfo, ModelKind, ModelRegistry, ModelSpec
from svc_engine.updates import Release, UpdateManager, compare_versions


class StreamResponse:
    def __init__(
        self, body: bytes, json_body: dict[str, Any] | None = None, status_code: int = 200
    ) -> None:
        self.body = body
        self._json = json_body
        self.status_code = status_code

    def __enter__(self):  # type: ignore[no-untyped-def]
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_content(self, _size: int):  # type: ignore[no-untyped-def]
        yield self.body

    def json(self) -> dict[str, Any]:
        assert self._json is not None
        return self._json


class StreamSession:
    def __init__(self, body: bytes, manifest: dict[str, Any] | None = None) -> None:
        self.body = body
        self.manifest = manifest

    def get(self, url: str, **_kwargs: Any) -> StreamResponse:
        return StreamResponse(self.body, self.manifest if url.endswith(".json") else None)


def make_update_zip(files: dict[str, bytes]) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name, body in files.items():
            archive.writestr(name, body)
    return output.getvalue()


def release_for(body: bytes, version: str = "0.2.0") -> Release:
    return Release(
        version=version,
        url="https://updates.example/songvoice.zip",
        sha256=hashlib.sha256(body).hexdigest(),
        size_bytes=len(body),
    )


def test_version_comparison_is_numeric_not_lexicographic() -> None:
    assert compare_versions("0.10.0", "0.9.9") > 0
    assert compare_versions("v1.0", "1.0.0") == 0
    assert compare_versions("1.0.0", "2.0.0") < 0


def test_update_is_checksum_verified_staged_and_applied_with_backup(tmp_path: Path) -> None:
    body = make_update_zip({"app/core.txt": b"new", "app/extra.txt": b"created"})
    manager = UpdateManager(tmp_path / "updates", session=StreamSession(body))  # type: ignore[arg-type]
    staged = manager.stage(release_for(body))
    install = tmp_path / "install"
    (install / "app").mkdir(parents=True)
    (install / "app" / "core.txt").write_bytes(b"old")

    assert staged.is_dir()
    assert manager.apply_pending(install)
    assert (install / "app" / "core.txt").read_bytes() == b"new"
    assert (install / "app" / "extra.txt").read_bytes() == b"created"
    manager.rollback_version(install, "0.2.0")
    assert (install / "app" / "core.txt").read_bytes() == b"old"
    assert not (install / "app" / "extra.txt").exists()


def test_update_rolls_back_if_transaction_receipt_cannot_be_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = make_update_zip({"app/core.txt": b"new", "app/extra.txt": b"created"})
    manager = UpdateManager(tmp_path / "updates", session=StreamSession(body))  # type: ignore[arg-type]
    manager.stage(release_for(body))
    install = tmp_path / "install"
    (install / "app").mkdir(parents=True)
    (install / "app" / "core.txt").write_bytes(b"old")

    def fail_receipt(_path: Path, _payload: dict[str, Any]) -> None:
        raise OSError("simulated receipt failure")

    monkeypatch.setattr(manager, "_atomic_json", fail_receipt)
    with pytest.raises(OSError, match="receipt failure"):
        manager.apply_pending(install)

    assert (install / "app" / "core.txt").read_bytes() == b"old"
    assert not (install / "app" / "extra.txt").exists()
    assert manager.pending() is not None


def test_bad_update_checksum_never_creates_pending_transaction(tmp_path: Path) -> None:
    body = make_update_zip({"app.txt": b"new"})
    manager = UpdateManager(tmp_path / "updates", session=StreamSession(body))  # type: ignore[arg-type]
    bad = Release("0.2.0", "https://updates.example/a.zip", "0" * 64, len(body))
    with pytest.raises(EngineError):
        manager.stage(bad)
    assert manager.pending() is None


@pytest.mark.parametrize(
    "member",
    ["../escape.txt", "..\\escape.txt", "C:\\escape.txt"],
)
def test_update_archive_cannot_escape_staging_directory(
    tmp_path: Path, member: str
) -> None:
    body = make_update_zip({member: b"no"})
    manager = UpdateManager(tmp_path / "updates", session=StreamSession(body))  # type: ignore[arg-type]
    with pytest.raises(EngineError):
        manager.stage(release_for(body))
    assert not (tmp_path / "escape.txt").exists()


def test_manifest_check_requires_https_and_newer_version(tmp_path: Path) -> None:
    raw = release_for(b"archive", "0.3.0").to_dict()
    session = StreamSession(b"", raw)
    manager = UpdateManager(tmp_path, session=session)  # type: ignore[arg-type]
    assert manager.check("https://updates.example/update.json", "0.2.0") is not None
    assert manager.check("https://updates.example/update.json", "0.3.0") is None
    with pytest.raises(ValueError):
        manager.check("http://updates.example/update.json", "0.2.0")


def test_missing_release_manifest_means_no_update(tmp_path: Path) -> None:
    class MissingSession:
        def get(self, _url: str, **_kwargs: Any) -> StreamResponse:
            return StreamResponse(b"", status_code=404)

    manager = UpdateManager(tmp_path, session=MissingSession())  # type: ignore[arg-type]
    assert manager.check("https://updates.example/update.json", "1.0.0") is None


class LocalDownloader:
    def __init__(self, root: Path, bodies: dict[str, bytes]) -> None:
        self.root = root
        self.bodies = bodies

    def check_space_for(self, _specs: list[ModelSpec]) -> None:
        return None

    def ensure_file(self, spec, file_spec, on_progress=None):  # type: ignore[no-untyped-def]
        target = file_spec.path_in(self.root)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(self.bodies[spec.id])
        if on_progress is not None:
            from svc_engine.resources import DownloadProgress

            on_progress(
                DownloadProgress(
                    spec.id, file_spec.name, len(self.bodies[spec.id]), len(self.bodies[spec.id]), 1
                )
            )
        return target


def test_first_run_only_accepts_pinned_redistributable_core_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = {model_id: model_id.encode() for model_id in CORE_MODEL_IDS}
    models = {}
    for model_id, body in bodies.items():
        models[model_id] = ModelSpec(
            id=model_id,
            kind=ModelKind.F0,
            backend="test",
            engine_model=f"{model_id}.bin",
            display_name_he=model_id,
            files=(
                FileSpec(
                    name=f"{model_id}.bin",
                    urls=("https://models.example/file",),
                    sha256=hashlib.sha256(body).hexdigest(),
                    size_bytes=len(body),
                ),
            ),
            license=LicenseInfo(spdx="MIT", verified_at="now", source="test"),
        )
    app_paths = paths(tmp_path)
    provisioner = Provisioner(
        app_paths,
        downloader=LocalDownloader(app_paths.models, bodies),  # type: ignore[arg-type]
    )
    provisioner.registry = ModelRegistry(models, version=11)
    monkeypatch.setattr(provisioner, "_package_version", lambda _name: "2.13.0+xpu")
    monkeypatch.setattr(provisioner, "_ffmpeg_is_lgpl", lambda: True)
    progress: list[float] = []

    result = provisioner.run(lambda fraction, _message: progress.append(fraction))

    assert result.complete
    assert progress[-1] == 1.0
    marker = json.loads(provisioner.state_file.read_text(encoding="utf-8"))
    assert marker["models_version"] == 11


def test_packaging_manifests_enforce_lgpl_checksums_and_bench_isolation() -> None:
    repo = Path(__file__).resolve().parents[2]
    dependencies = json.loads((repo / "packaging" / "dependencies.json").read_text())
    ffmpeg = dependencies["ffmpeg"]
    assert ffmpeg["variant"] == "win64-lgpl-shared"
    assert len(ffmpeg["sha256"]) == 64
    assert {"alimiter", "acompressor", "loudnorm"} <= set(ffmpeg["required_filters"])
    spec = (repo / "packaging" / "songvoice.spec").read_text(encoding="utf-8")
    assert 'excludes=["env-bench", "benchmark", "seed_vc", "DDSP_SVC"]' in spec
    assert 'hookspath=[str(repo / "packaging" / "hooks")]' in spec
    assert (repo / "packaging" / "bundle-third-party-licenses.py").is_file()
    workflow = (repo / ".github" / "workflows" / "windows-release.yml").read_text()
    assert 'tags: ["v*"]' in workflow
    assert "build-offline-models.py" in workflow
