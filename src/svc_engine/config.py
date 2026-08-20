"""Application paths and settings.

All user data lives under %LOCALAPPDATA%\\SongVoice (see docs/architecture.md).
Nothing here imports torch or any AI library.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from platformdirs import user_data_dir, user_log_dir

APP_NAME = "SongVoice"
APP_AUTHOR = "SongVoice"

__all__ = ["Paths", "Settings", "load_settings", "save_settings", "paths"]


@dataclass(frozen=True)
class Paths:
    root: Path
    models: Path
    voices: Path
    projects: Path
    cache: Path
    logs: Path
    work: Path
    db: Path

    def ensure(self) -> None:
        for p in (self.root, self.models, self.voices, self.projects, self.cache,
                  self.logs, self.work):
            p.mkdir(parents=True, exist_ok=True)


def paths(override_root: Path | None = None) -> Paths:
    """Resolve application paths.

    SONGVOICE_HOME overrides the location -- used by tests and by the spike so
    they never touch real user data.
    """
    if override_root is not None:
        root = Path(override_root)
    elif os.environ.get("SONGVOICE_HOME"):
        root = Path(os.environ["SONGVOICE_HOME"])
    else:
        root = Path(user_data_dir(APP_NAME, APP_AUTHOR, roaming=False))

    logs = (
        Path(user_log_dir(APP_NAME, APP_AUTHOR))
        if override_root is None and not os.environ.get("SONGVOICE_HOME")
        else root / "logs"
    )

    return Paths(
        root=root,
        models=root / "models",
        voices=root / "voices",
        projects=root / "projects",
        cache=root / "cache",
        logs=logs,
        # ASCII-only scratch dir: some AI/C++ tooling breaks on non-ASCII paths.
        work=root / "work",
        db=root / "db.sqlite",
    )


@dataclass
class Settings:
    """User-visible settings. Defaults are safe for a first run."""

    quality: str = "balanced"          # fast | balanced | max
    prefer_gpu: bool = True
    target_lufs: float = -14.0
    output_dir: str = ""
    keep_cache_gb: float = 20.0
    check_updates: bool = True
    allow_model_downloads: bool = True
    language: str = "he"
    theme: str = "system"  # system | light | dark
    advanced_mode: bool = False
    _version: int = field(default=1)

    @staticmethod
    def _file(p: Paths) -> Path:
        return p.root / "settings.json"


def load_settings(p: Paths | None = None) -> Settings:
    p = p or paths()
    f = Settings._file(p)
    if not f.exists():
        return Settings()
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        # A corrupt settings file must never stop the app from starting.
        return Settings()
    known = {k: v for k, v in raw.items() if k in Settings.__dataclass_fields__}
    return Settings(**known)


def save_settings(s: Settings, p: Paths | None = None) -> None:
    p = p or paths()
    p.ensure()
    Settings._file(p).write_text(
        json.dumps(asdict(s), ensure_ascii=False, indent=2), encoding="utf-8"
    )
