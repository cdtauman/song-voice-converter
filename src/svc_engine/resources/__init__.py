"""Model catalogue and downloads."""

from svc_engine.resources.download import DownloadManager, DownloadProgress, ProgressCallback
from svc_engine.resources.registry import (
    DATA_FILE,
    FileSpec,
    LicenseInfo,
    ModelKind,
    ModelRegistry,
    ModelSpec,
    load_registry,
    sha256_of,
)

__all__ = [
    "DATA_FILE",
    "DownloadManager",
    "DownloadProgress",
    "FileSpec",
    "LicenseInfo",
    "ModelKind",
    "ModelRegistry",
    "ModelSpec",
    "ProgressCallback",
    "load_registry",
    "sha256_of",
]
