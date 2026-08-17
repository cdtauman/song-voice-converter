"""The model catalogue.

`data/models.json` ships with the application and is the single place that says
which weights SongVoice is willing to use, where they come from, what they
hash to, and under what licence. Nothing downloads a file that is not listed
here -- that is what makes the licence audit in docs/models.md meaningful.

A `sha256` of null means "not pinned yet". Such a file still downloads, but
`verify()` reports it as unpinned so it is visible rather than silently trusted.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

__all__ = [
    "DATA_FILE",
    "ModelKind",
    "FileSpec",
    "LicenseInfo",
    "ModelSpec",
    "ModelRegistry",
    "load_registry",
    "sha256_of",
]

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "models.json"

_HASH_CHUNK = 1 << 20


class ModelKind(StrEnum):
    SEPARATION = "separation"
    KARAOKE = "karaoke"
    DEREVERB = "dereverb"
    DEECHO = "deecho"
    DENOISE = "denoise"
    STEMS = "stems"


@dataclass(frozen=True)
class LicenseInfo:
    """What we actually confirmed, and where. `spdx=None` means no declaration."""

    spdx: str | None = None
    verified_at: str = ""
    source: str = ""
    note_he: str = ""

    @property
    def is_redistributable(self) -> bool:
        """Only a declared permissive licence may ship in a distributed build."""
        return self.spdx in {"MIT", "Apache-2.0", "BSD-3-Clause", "CC-BY-4.0"}


@dataclass(frozen=True)
class FileSpec:
    name: str
    urls: tuple[str, ...]
    sha256: str | None = None
    size_bytes: int | None = None

    def path_in(self, models_dir: Path) -> Path:
        return models_dir / self.name


@dataclass(frozen=True)
class ModelSpec:
    id: str
    kind: ModelKind
    backend: str
    engine_model: str
    display_name_he: str
    files: tuple[FileSpec, ...]
    stems: tuple[str, ...] = ()
    sdr: float | None = None
    license: LicenseInfo = field(default_factory=LicenseInfo)
    note_he: str = ""

    @property
    def size_bytes(self) -> int:
        return sum(f.size_bytes or 0 for f in self.files)

    @property
    def size_mb(self) -> float:
        return self.size_bytes / (1024 ** 2)

    def is_present(self, models_dir: Path) -> bool:
        return all(f.path_in(models_dir).exists() for f in self.files)

    def missing_files(self, models_dir: Path) -> list[FileSpec]:
        return [f for f in self.files if not f.path_in(models_dir).exists()]


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelRegistry:
    models: dict[str, ModelSpec]
    version: int = 1
    source: str = ""

    def __contains__(self, model_id: object) -> bool:
        return model_id in self.models

    def get(self, model_id: str) -> ModelSpec:
        try:
            return self.models[model_id]
        except KeyError:
            raise KeyError(f"unknown model id: {model_id}") from None

    def of_kind(self, kind: ModelKind) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.kind is kind]

    def for_backend(self, backend: str) -> list[ModelSpec]:
        return [m for m in self.models.values() if m.backend == backend]

    def by_engine_model(self, backend: str, engine_model: str) -> ModelSpec | None:
        for spec in self.models.values():
            if spec.backend == backend and spec.engine_model == engine_model:
                return spec
        return None

    def unpinned(self) -> list[tuple[ModelSpec, FileSpec]]:
        """Files with no recorded sha256 -- reported, never silently accepted."""
        return [(m, f) for m in self.models.values() for f in m.files if not f.sha256]

    def unlicensed(self) -> list[ModelSpec]:
        """Models we may use privately but must not put in a distributed build."""
        return [m for m in self.models.values() if not m.license.is_redistributable]


def _parse_model(model_id: str, raw: dict) -> ModelSpec:
    lic_raw = raw.get("license") or {}
    files = tuple(
        FileSpec(
            name=str(f["name"]),
            urls=tuple(str(u) for u in (f.get("urls") or [])),
            sha256=(str(f["sha256"]).lower() if f.get("sha256") else None),
            size_bytes=(int(f["size_bytes"]) if f.get("size_bytes") else None),
        )
        for f in (raw.get("files") or [])
    )
    return ModelSpec(
        id=model_id,
        kind=ModelKind(raw["kind"]),
        backend=str(raw.get("backend") or "audio_separator"),
        engine_model=str(raw.get("engine_model") or (files[0].name if files else "")),
        display_name_he=str(raw.get("display_name_he") or model_id),
        files=files,
        stems=tuple(str(s) for s in (raw.get("stems") or [])),
        sdr=(float(raw["sdr"]) if raw.get("sdr") is not None else None),
        license=LicenseInfo(
            spdx=(str(lic_raw["spdx"]) if lic_raw.get("spdx") else None),
            verified_at=str(lic_raw.get("verified_at") or ""),
            source=str(lic_raw.get("source") or ""),
            note_he=str(lic_raw.get("note_he") or ""),
        ),
        note_he=str(raw.get("note_he") or ""),
    )


def load_registry(path: Path | None = None) -> ModelRegistry:
    """Read the catalogue. A malformed entry is skipped, not fatal."""
    path = path or DATA_FILE
    raw = json.loads(path.read_text(encoding="utf-8"))
    models: dict[str, ModelSpec] = {}
    for model_id, entry in (raw.get("models") or {}).items():
        try:
            models[model_id] = _parse_model(model_id, entry)
        except (KeyError, ValueError, TypeError):
            continue
    return ModelRegistry(
        models=models, version=int(raw.get("version") or 1), source=str(path)
    )
