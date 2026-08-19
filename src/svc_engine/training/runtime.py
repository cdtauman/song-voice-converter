"""Pinned Applio runtime bootstrap and command construction.

SongVoice wraps Applio's maintained training implementation instead of copying
its neural network.  The source archive is pinned and verified; all model files
are provided by SongVoice's checksum-audited model catalogue.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests

from svc_engine.config import Paths, load_settings
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources import DownloadManager, load_registry

APPLIO_COMMIT = "085197e738ce9dd4c0bae1e0a74df5de25b89444"
APPLIO_ARCHIVE = f"https://github.com/IAHispano/Applio/archive/{APPLIO_COMMIT}.zip"
APPLIO_ARCHIVE_SHA256 = "648c6322fe951401f7647a3aa4d007f6f1af253988ab865a6c010e257a798807"

__all__ = ["APPLIO_COMMIT", "ApplioRuntime"]


class ApplioRuntime:
    def __init__(self, app_paths: Paths, python: str | None = None) -> None:
        self.paths = app_paths
        self.root = app_paths.models / "training" / "applio"
        self.python = python or sys.executable

    @property
    def titan_g(self) -> Path:
        return self.root / "rvc/models/pretraineds/titan/G-f048k-TITAN-Medium.pth"

    @property
    def titan_d(self) -> Path:
        return self.root / "rvc/models/pretraineds/titan/D-f048k-TITAN-Medium.pth"

    def ensure(self, on_progress=None) -> Path:  # type: ignore[no-untyped-def]
        self._check_dependencies()
        if not (self.root / "rvc/train/train.py").is_file():
            self._install_source()
        registry = load_registry()
        manager = DownloadManager(
            self.paths.models,
            allow_downloads=load_settings(self.paths).allow_model_downloads,
        )
        specs = [registry.get("rvc_training_runtime"), registry.get("titan_medium_48k")]
        manager.check_space_for(specs)
        for spec in specs:
            manager.ensure_model(spec, on_progress=on_progress)
        template = self.root / "assets/config_template.json"
        config = self.root / "assets/config.json"
        if not config.exists() and template.exists():
            shutil.copy2(template, config)
        return self.root

    def _check_dependencies(self) -> None:
        missing = [
            name
            for name in ("torch", "tensorboard", "faiss", "sklearn", "transformers")
            if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise EngineError(
                ErrorCode.MODEL_MISSING,
                "training runtime dependencies missing: " + ", ".join(missing),
            )

    def _install_source(self) -> None:
        self.root.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="svc_applio_", dir=self.root.parent) as tmp_raw:
            tmp = Path(tmp_raw)
            archive = tmp / "source.zip"
            digest = hashlib.sha256()
            with requests.get(APPLIO_ARCHIVE, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                with archive.open("wb") as stream:
                    for chunk in response.iter_content(1 << 20):
                        if chunk:
                            stream.write(chunk)
                            digest.update(chunk)
            if digest.hexdigest() != APPLIO_ARCHIVE_SHA256:
                raise EngineError(ErrorCode.DOWNLOAD_FAILED, "Applio source checksum mismatch")
            extracted = tmp / "extracted"
            with zipfile.ZipFile(archive) as package:
                for member in package.infolist():
                    parts = Path(member.filename).parts
                    if not parts or ".." in parts:
                        raise EngineError(ErrorCode.DOWNLOAD_FAILED, "unsafe Applio archive path")
                package.extractall(extracted)
            children = [item for item in extracted.iterdir() if item.is_dir()]
            if len(children) != 1:
                raise EngineError(ErrorCode.DOWNLOAD_FAILED, "unexpected Applio archive layout")
            staged = self.root.with_name(".applio-installing")
            shutil.rmtree(staged, ignore_errors=True)
            children[0].replace(staged)
            shutil.rmtree(self.root, ignore_errors=True)
            staged.replace(self.root)

    def preprocess_command(self, session) -> list[str]:  # type: ignore[no-untyped-def]
        experiment = self.root / "logs" / session.applio_experiment
        return [
            self.python,
            "rvc/train/preprocess/preprocess.py",
            str(experiment),
            str(session.dataset_dir),
            "48000",
            str(max(1, min(8, os.cpu_count() or 1))),
            "Skip",
            "False",
            "False",
            "0.0",
            "10.0",
            "0.0",
            "none",
        ]

    def extract_command(self, session) -> list[str]:  # type: ignore[no-untyped-def]
        experiment = self.root / "logs" / session.applio_experiment
        return [
            self.python,
            "rvc/train/extract/extract.py",
            str(experiment),
            "rmvpe",
            str(max(1, min(8, os.cpu_count() or 1))),
            "-",
            "48000",
            "contentvec",
            "None",
            "2",
        ]

    def train_command(self, session) -> list[str]:  # type: ignore[no-untyped-def]
        return [
            self.python,
            "rvc/train/train.py",
            session.applio_experiment,
            "10",
            str(session.total_epochs),
            str(self.titan_g),
            str(self.titan_d),
            "-",
            "4",
            "48000",
            "True",
            "True",
            "False",
            "False",
            "HiFi-GAN",
            "True",
        ]

    def index_command(self, session) -> list[str]:  # type: ignore[no-untyped-def]
        experiment = self.root / "logs" / session.applio_experiment
        return [self.python, "rvc/train/process/extract_index.py", str(experiment), "Auto"]
