"""Checksum-verified application updates with transactional rollback."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import requests

from svc_engine.errors import EngineError, ErrorCode

DEFAULT_MANIFEST_URL = (
    "https://github.com/cdtauman/song-voice-converter/releases/latest/download/update.json"
)

__all__ = ["DEFAULT_MANIFEST_URL", "Release", "UpdateManager", "compare_versions"]

_CHUNK = 1 << 20


def compare_versions(left: str, right: str) -> int:
    def parts(value: str) -> tuple[int, ...]:
        value = value.strip().lstrip("v").split("+", 1)[0].split("-", 1)[0]
        try:
            return tuple(int(item) for item in value.split("."))
        except ValueError:
            return (0,)

    a, b = parts(left), parts(right)
    length = max(len(a), len(b))
    a += (0,) * (length - len(a))
    b += (0,) * (length - len(b))
    return (a > b) - (a < b)


@dataclass(frozen=True)
class Release:
    version: str
    url: str
    sha256: str
    size_bytes: int
    notes_he: str = ""

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Release:
        release = cls(
            version=str(raw["version"]),
            url=str(raw["url"]),
            sha256=str(raw["sha256"]).lower(),
            size_bytes=int(raw["size_bytes"]),
            notes_he=str(raw.get("notes_he") or ""),
        )
        if not release.url.startswith("https://"):
            raise ValueError("update URL must use HTTPS")
        if len(release.sha256) != 64 or any(c not in "0123456789abcdef" for c in release.sha256):
            raise ValueError("invalid update SHA-256")
        if release.size_bytes <= 0:
            raise ValueError("invalid update size")
        return release

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class UpdateManager:
    def __init__(self, state_dir: Path, session: requests.Session | None = None) -> None:
        self.state_dir = Path(state_dir)
        self.session = session or requests.Session()
        self.downloads = self.state_dir / "downloads"
        self.staging = self.state_dir / "staging"
        self.rollback = self.state_dir / "rollback"
        self.pending_file = self.state_dir / "pending.json"

    def check(self, manifest_url: str, current_version: str) -> Release | None:
        if not manifest_url.startswith("https://"):
            raise ValueError("manifest URL must use HTTPS")
        response = self.session.get(manifest_url, timeout=(15, 30))
        # A repository may legitimately have no published release yet.  Treat
        # GitHub's missing latest-release asset as "no update", not as an
        # application error shown to the user.
        if getattr(response, "status_code", None) == 404:
            return None
        response.raise_for_status()
        raw = response.json()
        release = Release.from_dict(raw)
        return release if compare_versions(release.version, current_version) > 0 else None

    def stage(self, release: Release, on_progress=None) -> Path:  # type: ignore[no-untyped-def]
        self.downloads.mkdir(parents=True, exist_ok=True)
        archive = self.downloads / f"songvoice-{release.version}.zip"
        part = archive.with_suffix(".zip.part")
        digest = hashlib.sha256()
        done = 0
        with self.session.get(release.url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with part.open("wb") as handle:
                for chunk in response.iter_content(_CHUNK):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    digest.update(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(min(1.0, done / release.size_bytes))
        if done != release.size_bytes or digest.hexdigest() != release.sha256:
            part.unlink(missing_ok=True)
            raise EngineError(ErrorCode.DOWNLOAD_FAILED, "update checksum or size mismatch")
        os.replace(part, archive)

        target = self.staging / release.version
        if target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True)
        self._safe_extract(archive, target)
        files = [path for path in target.rglob("*") if path.is_file()]
        if not files:
            shutil.rmtree(target)
            raise EngineError(ErrorCode.DOWNLOAD_FAILED, "update archive is empty")
        self._atomic_json(
            self.pending_file,
            {"schema": 1, "release": release.to_dict(), "staged": str(target)},
        )
        return target

    def pending(self) -> tuple[Release, Path] | None:
        try:
            raw = json.loads(self.pending_file.read_text(encoding="utf-8"))
            release = Release.from_dict(raw["release"])
            staged = Path(raw["staged"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        try:
            staged.resolve().relative_to(self.staging.resolve())
        except ValueError:
            return None
        return (release, staged) if staged.is_dir() else None

    def apply_pending(self, install_dir: Path) -> bool:
        pending = self.pending()
        if pending is None:
            return False
        release, staged = pending
        install_dir = Path(install_dir).resolve()
        backup = self.rollback / release.version
        if backup.exists():
            shutil.rmtree(backup)
        backup.mkdir(parents=True)
        replaced: list[Path] = []
        created: list[Path] = []
        try:
            for source in sorted(path for path in staged.rglob("*") if path.is_file()):
                relative = source.relative_to(staged)
                destination = install_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    saved = backup / relative
                    saved.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(destination, saved)
                    replaced.append(relative)
                else:
                    created.append(relative)
                temporary = destination.with_name(destination.name + ".update-new")
                shutil.copy2(source, temporary)
                os.replace(temporary, destination)
            self._atomic_json(
                backup / "transaction.json",
                {
                    "schema": 1,
                    "version": release.version,
                    "replaced": [path.as_posix() for path in replaced],
                    "created": [path.as_posix() for path in created],
                },
            )
        except Exception:
            self._restore(install_dir, backup, replaced, created)
            raise
        self.pending_file.unlink(missing_ok=True)
        shutil.rmtree(staged, ignore_errors=True)
        return True

    def rollback_version(self, install_dir: Path, version: str) -> None:
        backup = self.rollback / version
        raw = json.loads((backup / "transaction.json").read_text(encoding="utf-8"))
        replaced = [Path(item) for item in raw["replaced"]]
        created = [Path(item) for item in raw["created"]]
        self._restore(Path(install_dir).resolve(), backup, replaced, created)

    @staticmethod
    def _restore(
        install_dir: Path, backup: Path, replaced: list[Path], created: list[Path]
    ) -> None:
        for relative in created:
            (install_dir / relative).unlink(missing_ok=True)
        for relative in replaced:
            source = backup / relative
            if source.is_file():
                destination = install_dir / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

    @staticmethod
    def _safe_extract(archive: Path, target: Path) -> None:
        resolved_target = target.resolve()
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                # ZIP uses POSIX separators, but Windows also treats backslashes
                # and drive-qualified names as paths.  Normalize both forms and
                # verify the final filesystem path before creating anything.
                windows_name = PureWindowsPath(info.filename)
                name = PurePosixPath(info.filename.replace("\\", "/"))
                if (
                    name.is_absolute()
                    or windows_name.is_absolute()
                    or bool(windows_name.drive)
                    or ".." in name.parts
                ):
                    raise EngineError(
                        ErrorCode.DOWNLOAD_FAILED, "unsafe path in update archive"
                    )
                destination = target.joinpath(*name.parts)
                try:
                    destination.resolve().relative_to(resolved_target)
                except ValueError as exc:
                    raise EngineError(
                        ErrorCode.DOWNLOAD_FAILED, "unsafe path in update archive"
                    ) from exc
                if info.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue
                destination.parent.mkdir(parents=True, exist_ok=True)
                with zipped.open(info) as source, destination.open("wb") as output:
                    shutil.copyfileobj(source, output)

    @staticmethod
    def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
