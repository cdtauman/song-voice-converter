"""The model catalogue and the downloader.

The downloader is tested against a fake transport rather than the network: what
matters is that it resumes, verifies, retries and falls through to the next
mirror, and none of that needs a real server to prove.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import requests

from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources import (
    DownloadManager,
    ModelKind,
    load_registry,
    sha256_of,
)
from svc_engine.resources import (
    registry as registry_mod,
)

BODY = b"weights" * 5000


class FakeResponse:
    def __init__(self, body: bytes, status: int = 200, headers: dict | None = None) -> None:
        self._body = body
        self.status_code = status
        self.headers = headers or {"content-length": str(len(body))}

    def iter_content(self, chunk_size: int = 1):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self) -> None:
        pass


class FakeSession:
    """Serves `BODY`, honouring Range, and can be told to fail the first N calls."""

    def __init__(self, fail_first: int = 0, ignore_range: bool = False,
                 body: bytes = BODY) -> None:
        self.fail_first = fail_first
        self.ignore_range = ignore_range
        self.body = body
        self.calls: list[tuple[str, dict]] = []

    def get(self, url, headers=None, stream=False, timeout=None):  # noqa: ANN001
        self.calls.append((url, headers or {}))
        if self.fail_first > 0:
            self.fail_first -= 1
            raise requests.ConnectionError("simulated network drop")
        if "unreachable" in url:
            return FakeResponse(b"", status=404, headers={})

        start = 0
        if headers and "Range" in headers and not self.ignore_range:
            start = int(headers["Range"].split("=")[1].split("-")[0])
            part = self.body[start:]
            return FakeResponse(part, status=206, headers={"content-length": str(len(part))})
        return FakeResponse(self.body)


def make_spec(urls: list[str], sha256: str | None = None, size: int | None = None):
    return registry_mod.ModelSpec(
        id="test_model",
        kind=ModelKind.SEPARATION,
        backend="audio_separator",
        engine_model="w.ckpt",
        display_name_he="בדיקה",
        files=(
            registry_mod.FileSpec(
                name="w.ckpt", urls=tuple(urls), sha256=sha256, size_bytes=size
            ),
        ),
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

def test_shipped_catalogue_parses_and_is_not_empty() -> None:
    registry = load_registry()
    assert registry.models
    assert registry.of_kind(ModelKind.SEPARATION), "there must be a separation model"


def test_every_catalogue_entry_has_a_reachable_looking_url_and_a_licence_verdict() -> None:
    """A model with no URL cannot be fetched; one with no audit trail is unreviewed."""
    for spec in load_registry().models.values():
        assert spec.files, f"{spec.id} declares no files"
        for file_spec in spec.files:
            assert file_spec.urls, f"{spec.id}/{file_spec.name} has no URL"
            assert all(u.startswith("https://") for u in file_spec.urls)
        assert spec.license.verified_at, f"{spec.id} has no licence verification date"
        assert spec.license.source, f"{spec.id} does not say where the licence came from"


def test_licence_policy_splits_the_catalogue() -> None:
    registry = load_registry()
    private = {m.id for m in registry.unlicensed()}
    assert "sep_melband_kim" not in private, "the MIT default must be redistributable"
    assert "dereverb_anvuew" in private, "a GPL checkpoint must not be shippable"


def test_unknown_model_id_raises_a_clear_key_error() -> None:
    with pytest.raises(KeyError, match="unknown model id"):
        load_registry().get("no_such_model")


def test_a_malformed_entry_is_skipped_not_fatal(tmp_path: Path) -> None:
    path = tmp_path / "models.json"
    path.write_text(
        json.dumps(
            {
                "version": 1,
                "models": {
                    "broken": {"kind": "not_a_real_kind"},
                    "good": {
                        "kind": "separation",
                        "files": [{"name": "a.ckpt", "urls": ["https://example/a"]}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    registry = load_registry(path)
    assert "good" in registry
    assert "broken" not in registry


# --------------------------------------------------------------------------- #
# downloader
# --------------------------------------------------------------------------- #

def test_download_verifies_the_checksum(tmp_path: Path) -> None:
    digest = hashlib.sha256(BODY).hexdigest()
    manager = DownloadManager(tmp_path, session=FakeSession())
    path = manager.ensure_file(
        make_spec(["https://mirror/w.ckpt"], sha256=digest),
        make_spec(["https://mirror/w.ckpt"], sha256=digest).files[0],
    )
    assert sha256_of(path) == digest


def test_a_wrong_checksum_is_rejected_rather_than_accepted(tmp_path: Path) -> None:
    spec = make_spec(["https://mirror/w.ckpt"], sha256="0" * 64)
    manager = DownloadManager(tmp_path, session=FakeSession())
    with pytest.raises(EngineError) as caught:
        manager.ensure_file(spec, spec.files[0])
    assert caught.value.code is ErrorCode.DOWNLOAD_FAILED
    assert not (tmp_path / "w.ckpt").exists(), "a bad file must not be left behind"


def test_an_existing_but_corrupt_file_is_replaced(tmp_path: Path) -> None:
    digest = hashlib.sha256(BODY).hexdigest()
    (tmp_path / "w.ckpt").write_bytes(b"garbage")
    spec = make_spec(["https://mirror/w.ckpt"], sha256=digest)
    manager = DownloadManager(tmp_path, session=FakeSession())
    assert sha256_of(manager.ensure_file(spec, spec.files[0])) == digest


def test_a_valid_existing_file_is_not_downloaded_again(tmp_path: Path) -> None:
    digest = hashlib.sha256(BODY).hexdigest()
    (tmp_path / "w.ckpt").write_bytes(BODY)
    session = FakeSession()
    spec = make_spec(["https://mirror/w.ckpt"], sha256=digest)
    DownloadManager(tmp_path, session=session).ensure_file(spec, spec.files[0])
    assert session.calls == []


def test_a_partial_file_resumes_instead_of_restarting(tmp_path: Path) -> None:
    digest = hashlib.sha256(BODY).hexdigest()
    part = tmp_path / "w.ckpt.part"
    part.write_bytes(BODY[:1000])

    session = FakeSession()
    spec = make_spec(["https://mirror/w.ckpt"], sha256=digest)
    path = DownloadManager(tmp_path, session=session).ensure_file(spec, spec.files[0])

    assert sha256_of(path) == digest
    assert session.calls[0][1].get("Range") == "bytes=1000-"


def test_a_server_ignoring_range_restarts_rather_than_doubling_the_file(
    tmp_path: Path,
) -> None:
    """Appending a full body onto a partial one silently produces a corrupt file."""
    digest = hashlib.sha256(BODY).hexdigest()
    (tmp_path / "w.ckpt.part").write_bytes(BODY[:1000])

    spec = make_spec(["https://mirror/w.ckpt"], sha256=digest)
    manager = DownloadManager(tmp_path, session=FakeSession(ignore_range=True))
    assert sha256_of(manager.ensure_file(spec, spec.files[0])) == digest


def test_transient_failures_are_retried(tmp_path: Path) -> None:
    digest = hashlib.sha256(BODY).hexdigest()
    spec = make_spec(["https://mirror/w.ckpt"], sha256=digest)
    manager = DownloadManager(
        tmp_path, session=FakeSession(fail_first=2), backoff_seconds=0.0
    )
    assert sha256_of(manager.ensure_file(spec, spec.files[0])) == digest


def test_a_dead_mirror_falls_through_to_the_next_one(tmp_path: Path) -> None:
    digest = hashlib.sha256(BODY).hexdigest()
    spec = make_spec(
        ["https://unreachable/w.ckpt", "https://mirror/w.ckpt"], sha256=digest
    )
    manager = DownloadManager(
        tmp_path, session=FakeSession(), backoff_seconds=0.0, attempts_per_url=1
    )
    assert sha256_of(manager.ensure_file(spec, spec.files[0])) == digest


def test_downloads_can_be_switched_off(tmp_path: Path) -> None:
    spec = make_spec(["https://mirror/w.ckpt"])
    manager = DownloadManager(tmp_path, session=FakeSession(), allow_downloads=False)
    with pytest.raises(EngineError) as caught:
        manager.ensure_file(spec, spec.files[0])
    assert caught.value.code is ErrorCode.MODEL_MISSING


def test_an_unpinned_file_falls_back_to_the_declared_size(tmp_path: Path) -> None:
    spec = make_spec(["https://mirror/w.ckpt"], size=len(BODY))
    manager = DownloadManager(tmp_path, session=FakeSession())
    assert manager.ensure_file(spec, spec.files[0]).stat().st_size == len(BODY)


def test_a_size_mismatch_on_an_unpinned_file_is_still_caught(tmp_path: Path) -> None:
    spec = make_spec(["https://mirror/w.ckpt"], size=len(BODY) + 999)
    manager = DownloadManager(
        tmp_path, session=FakeSession(), attempts_per_url=1, backoff_seconds=0.0
    )
    with pytest.raises(EngineError):
        manager.ensure_file(spec, spec.files[0])


def test_disk_space_is_checked_before_downloading_not_during(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = make_spec(["https://mirror/w.ckpt"], size=50 * 1024 ** 3)
    manager = DownloadManager(tmp_path, session=FakeSession())
    monkeypatch.setattr(manager, "free_space_bytes", lambda: 1024 ** 3)
    with pytest.raises(EngineError) as caught:
        manager.check_space_for([spec])
    assert caught.value.code is ErrorCode.DISK_FULL
