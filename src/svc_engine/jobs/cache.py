"""Content-addressed, integrity-checked cache for completed job steps."""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from svc_engine.jobs._io import atomic_write_json, read_json_object

_MANIFEST = "manifest.json"
_CACHE_SCHEMA = 1

__all__ = ["CacheEntry", "CacheStats", "StepCache", "cache_key", "hash_file"]


@dataclass(frozen=True)
class CacheEntry:
    key: str
    outputs: dict[str, Path]
    size_bytes: int
    created_at: float
    last_accessed_at: float


@dataclass(frozen=True)
class CacheStats:
    entries: int
    size_bytes: int


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("cache parameters cannot contain NaN or infinity")
        return value
    if isinstance(value, Path):
        return {"path": str(value)}
    if isinstance(value, Enum):
        return _normalise(value.value)
    if is_dataclass(value) and not isinstance(value, type):
        return _normalise(asdict(value))
    if isinstance(value, Mapping):
        ordered = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(key): _normalise(item) for key, item in ordered}
    if isinstance(value, (Sequence, set, frozenset)) and not isinstance(value, (str, bytes)):
        items = [_normalise(item) for item in value]
        return sorted(items, key=repr) if isinstance(value, (set, frozenset)) else items
    raise TypeError(f"unsupported cache parameter type: {type(value).__name__}")


def cache_key(
    *,
    step_id: str,
    version: str,
    parameters: Mapping[str, Any],
    input_files: Sequence[Path] = (),
    dependency_keys: Sequence[str] = (),
) -> str:
    """Hash the step contract, file contents, parameters and dependency results."""
    import json

    files = [
        {"sha256": hash_file(Path(path)), "size": Path(path).stat().st_size}
        for path in input_files
    ]
    payload = {
        "schema": 1,
        "step": step_id,
        "version": version,
        "parameters": _normalise(parameters),
        "inputs": files,
        "dependencies": list(dependency_keys),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class StepCache:
    """Immutable entry directories published only after every output is durable."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def lookup(self, key: str, *, touch: bool = True) -> CacheEntry | None:
        directory = self._entry_dir(key)
        manifest_path = directory / _MANIFEST
        if not manifest_path.is_file():
            return None
        try:
            manifest = read_json_object(manifest_path)
            entry = self._entry_from_manifest(directory, manifest, verify=True)
        except (OSError, ValueError, TypeError, KeyError):
            # Never consume a partial/corrupt cache entry. Keep it for diagnosis;
            # quota cleanup will account for and eventually remove it.
            return None
        if touch:
            manifest["last_accessed_at"] = time.time()
            try:
                atomic_write_json(manifest_path, manifest)
                entry = self._entry_from_manifest(directory, manifest, verify=False)
            except OSError:
                pass  # a read-only cache hit is still a valid hit
        return entry

    def publish(self, key: str, outputs: Mapping[str, Path]) -> CacheEntry:
        if not outputs:
            raise ValueError("a cached step must publish at least one output")
        existing = self.lookup(key)
        if existing is not None:
            return existing

        final = self._entry_dir(key)
        temporary = self.root / f".{key}.{uuid.uuid4().hex}.tmp"
        files_dir = temporary / "files"
        files_dir.mkdir(parents=True)
        records: dict[str, dict[str, Any]] = {}
        total = 0
        try:
            for index, (name, source_value) in enumerate(sorted(outputs.items())):
                if not name or "\x00" in name:
                    raise ValueError("output names must be non-empty text")
                source = Path(source_value)
                if not source.is_file():
                    raise FileNotFoundError(source)
                filename = f"{index:03d}-{source.name}"
                target = files_dir / filename
                shutil.copyfile(source, target)
                # Windows' CRT rejects fsync on a read-only descriptor.
                with target.open("r+b") as stream:
                    os.fsync(stream.fileno())
                size = target.stat().st_size
                digest = hash_file(target)
                total += size
                records[name] = {"file": f"files/{filename}", "size": size, "sha256": digest}
            now = time.time()
            manifest = {
                "schema": _CACHE_SCHEMA,
                "key": key,
                "created_at": now,
                "last_accessed_at": now,
                "size_bytes": total,
                "outputs": records,
            }
            atomic_write_json(temporary / _MANIFEST, manifest)
            try:
                os.replace(temporary, final)
            except OSError:
                # Another worker may have won publication. Windows can report
                # this as PermissionError instead of FileExistsError for dirs.
                concurrent = self.lookup(key)
                if concurrent is None:
                    raise
                shutil.rmtree(temporary, ignore_errors=True)
                return concurrent
            entry = self.lookup(key)
            if entry is None:
                raise OSError(f"cache entry {key} was not published correctly")
            return entry
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def stats(self) -> CacheStats:
        entries = 0
        size = 0
        for directory in self._entry_directories():
            entries += 1
            try:
                manifest = read_json_object(directory / _MANIFEST)
                size += int(manifest.get("size_bytes", 0))
            except (OSError, ValueError, TypeError):
                size += self._directory_size(directory)
        return CacheStats(entries=entries, size_bytes=size)

    def enforce_limit(
        self, max_bytes: int, *, protected_keys: set[str] | None = None
    ) -> CacheStats:
        if max_bytes < 0:
            raise ValueError("max_bytes must be non-negative")
        protected = protected_keys or set()
        candidates: list[tuple[float, str, Path, int]] = []
        total = 0
        for directory in self._entry_directories():
            key = directory.name
            try:
                manifest = read_json_object(directory / _MANIFEST)
                size = int(manifest.get("size_bytes", 0))
                accessed = float(manifest.get("last_accessed_at", 0.0))
            except (OSError, ValueError, TypeError):
                size = self._directory_size(directory)
                accessed = directory.stat().st_mtime
            total += size
            candidates.append((accessed, key, directory, size))

        count = len(candidates)
        for _, key, directory, size in sorted(candidates):
            if total <= max_bytes:
                break
            if key in protected:
                continue
            self._remove_entry(directory)
            total -= size
            count -= 1
        return CacheStats(entries=count, size_bytes=max(total, 0))

    def clear(self, *, protected_keys: set[str] | None = None) -> CacheStats:
        return self.enforce_limit(0, protected_keys=protected_keys)

    def remove_abandoned_temporaries(self) -> int:
        removed = 0
        for path in self.root.glob(".*.tmp"):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)
            removed += 1
        return removed

    def _entry_dir(self, key: str) -> Path:
        if len(key) != 64 or any(char not in "0123456789abcdef" for char in key):
            raise ValueError("cache key must be a lowercase SHA-256 digest")
        return self.root / key

    def _entry_directories(self) -> list[Path]:
        return [
            path
            for path in self.root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        ]

    def _entry_from_manifest(
        self, directory: Path, manifest: Mapping[str, Any], *, verify: bool
    ) -> CacheEntry:
        if int(manifest["schema"]) != _CACHE_SCHEMA or manifest["key"] != directory.name:
            raise ValueError("cache manifest identity mismatch")
        outputs: dict[str, Path] = {}
        for name, raw in dict(manifest["outputs"]).items():
            record = dict(raw)
            path = (directory / str(record["file"])).resolve()
            if directory.resolve() not in path.parents or not path.is_file():
                raise ValueError("cache manifest points outside its entry")
            size_mismatch = path.stat().st_size != int(record["size"])
            digest_mismatch = verify and hash_file(path) != record["sha256"]
            if verify and (size_mismatch or digest_mismatch):
                raise ValueError("cache output failed integrity verification")
            outputs[str(name)] = path
        return CacheEntry(
            key=directory.name,
            outputs=outputs,
            size_bytes=int(manifest["size_bytes"]),
            created_at=float(manifest["created_at"]),
            last_accessed_at=float(manifest["last_accessed_at"]),
        )

    @staticmethod
    def _directory_size(directory: Path) -> int:
        return sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())

    def _remove_entry(self, directory: Path) -> None:
        resolved = directory.resolve()
        if resolved.parent != self.root.resolve():
            raise ValueError("refusing to remove a path outside the cache root")
        shutil.rmtree(resolved)
