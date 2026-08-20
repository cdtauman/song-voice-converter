"""First-run provisioning for the packaged Windows application.

The packaged runtime already contains the locked Python/PyTorch matrix.  First
run selects the best usable backend, verifies that runtime, downloads only the
redistributable SHA-256-pinned production models, and records completion
atomically.  Keeping this in the engine process preserves the GUI/torch
boundary.
"""

from __future__ import annotations

import importlib.metadata
import json
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from svc_engine.compute import DeviceManager
from svc_engine.config import Paths
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources import DownloadManager, DownloadProgress, load_registry, sha256_of

__all__ = ["CORE_MODEL_IDS", "ProvisioningStatus", "Provisioner"]

CORE_MODEL_IDS = ("sep_melband_kim", "f0_rmvpe", "content_hubert")
_STATE_VERSION = 1


@dataclass(frozen=True)
class ProvisioningStatus:
    complete: bool
    backend: str
    device_name: str
    torch_version: str
    ffmpeg_ok: bool
    missing_models: tuple[str, ...]
    detail_he: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["missing_models"] = list(self.missing_models)
        return data


Progress = Callable[[float, str], None]


class Provisioner:
    def __init__(
        self,
        app_paths: Paths,
        *,
        downloader: DownloadManager | None = None,
    ) -> None:
        self.paths = app_paths
        self.registry = load_registry()
        self.downloader = downloader or DownloadManager(app_paths.models)

    @property
    def state_file(self) -> Path:
        return self.paths.root / "setup-complete.json"

    def status(self) -> ProvisioningStatus:
        device = DeviceManager().preferred()
        missing = tuple(
            model_id
            for model_id in CORE_MODEL_IDS
            if not self._model_verified(model_id)
        )
        torch_version = self._package_version("torch")
        ffmpeg_ok = self._ffmpeg_is_lgpl()
        marker_ok = self._marker_is_current()
        complete = marker_ok and not missing and bool(torch_version) and ffmpeg_ok
        if complete:
            detail = "המערכת מוכנה ליצירת קאבר."
        elif not ffmpeg_ok:
            detail = "רכיב עיבוד השמע המופץ חסר או שאינו build מסוג LGPL."
        elif not torch_version:
            detail = "סביבת PyTorch המשובצת חסרה. יש להתקין מחדש את SongVoice."
        else:
            detail = "נדרשת השלמת ההתקנה הראשונית."
        return ProvisioningStatus(
            complete=complete,
            backend=device.backend.value,
            device_name=device.name,
            torch_version=torch_version,
            ffmpeg_ok=ffmpeg_ok,
            missing_models=missing,
            detail_he=detail,
        )

    def run(self, on_progress: Progress | None = None) -> ProvisioningStatus:
        self.paths.ensure()
        report = on_progress or (lambda _fraction, _message: None)
        report(0.03, "מזהים את החומרה המתאימה…")
        before = self.status()
        if not before.torch_version:
            raise EngineError(
                ErrorCode.BACKEND_UNAVAILABLE, "packaged torch runtime missing"
            )
        if not before.ffmpeg_ok:
            raise EngineError(ErrorCode.FFMPEG_MISSING, "LGPL ffmpeg is missing or GPL-enabled")

        specs = [self.registry.get(model_id) for model_id in CORE_MODEL_IDS]
        for spec in specs:
            if not spec.license.is_redistributable:
                raise EngineError(ErrorCode.MODEL_CORRUPT, f"{spec.id} is not redistributable")
            for file_spec in spec.files:
                if not file_spec.sha256:
                    raise EngineError(ErrorCode.MODEL_CORRUPT, f"{spec.id} is not SHA-256 pinned")
        self.downloader.check_space_for(specs)

        total = sum(file.size_bytes or 0 for spec in specs for file in spec.files) or 1
        completed = 0
        for spec in specs:
            for file_spec in spec.files:
                size = file_spec.size_bytes or 0

                def progress(
                    item: DownloadProgress, base: int = completed, span: int = size
                ) -> None:
                    done = min(span, item.done_bytes) if span else 0
                    report(0.08 + 0.82 * ((base + done) / total), item.message_he)

                self.downloader.ensure_file(spec, file_spec, progress)
                completed += size

        report(0.94, "בודקים את תקינות הקבצים…")
        after = self.status()
        if after.missing_models:
            raise EngineError(ErrorCode.MODEL_CORRUPT, ", ".join(after.missing_models))
        self._write_marker(after)
        report(1.0, "הכול מוכן. אפשר ליצור קאבר ראשון.")
        return self.status()

    def _model_verified(self, model_id: str) -> bool:
        spec = self.registry.get(model_id)
        for file_spec in spec.files:
            path = file_spec.path_in(self.paths.models)
            if not path.is_file() or not file_spec.sha256:
                return False
            try:
                if sha256_of(path) != file_spec.sha256:
                    return False
            except OSError:
                return False
        return True

    def _marker_is_current(self) -> bool:
        try:
            raw = json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return (
            raw.get("schema") == _STATE_VERSION
            and raw.get("models_version") == self.registry.version
        )

    def _write_marker(self, status: ProvisioningStatus) -> None:
        payload = {
            "schema": _STATE_VERSION,
            "models_version": self.registry.version,
            "backend": status.backend,
            "device_name": status.device_name,
            "torch_version": status.torch_version,
        }
        temporary = self.state_file.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, self.state_file)

    @staticmethod
    def _package_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return ""

    @staticmethod
    def _ffmpeg_is_lgpl() -> bool:
        exe = shutil.which("ffmpeg")
        if not exe:
            return False
        import subprocess  # noqa: PLC0415

        try:
            result = subprocess.run(
                [exe, "-version"], capture_output=True, text=True, timeout=15, check=False
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        text = f"{result.stdout}\n{result.stderr}".lower()
        return (
            result.returncode == 0
            and "--enable-gpl" not in text
            and "--enable-nonfree" not in text
        )
