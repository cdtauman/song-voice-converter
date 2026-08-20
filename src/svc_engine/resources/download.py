"""Model downloads: resumable, verified, mirrored.

audio-separator and pymss both download their own weights, and both do it the
simple way -- one GET, no checksum, no resume, no retry. On a 900MB file over a
home connection that is the difference between "it works" and "it silently
wrote half a checkpoint and the model loads as noise".

So downloading is ours: HTTP Range resume into a `.part` file, SHA-256 on
completion, retry with growing backoff, then the next mirror. The engines are
then pointed at a directory where the files already exist, and skip their own
download path entirely.
"""

from __future__ import annotations

import errno
import logging
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import requests

from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources.registry import FileSpec, ModelSpec, sha256_of

__all__ = ["DownloadProgress", "DownloadManager", "ProgressCallback"]

log = logging.getLogger(__name__)

_CHUNK = 1 << 20
_CONNECT_TIMEOUT = 30
_READ_TIMEOUT = 120


@dataclass(frozen=True)
class DownloadProgress:
    """One progress tick. `total_bytes` is None when the server hides the size."""

    model_id: str
    file_name: str
    done_bytes: int
    total_bytes: int | None
    attempt: int

    @property
    def fraction(self) -> float | None:
        if not self.total_bytes:
            return None
        return min(1.0, self.done_bytes / self.total_bytes)

    @property
    def message_he(self) -> str:
        pct = self.fraction
        if pct is None:
            return f"מורידים קובץ נדרש… ({self.done_bytes / 1024 ** 2:.0f}MB)"
        return f"מורידים קובץ נדרש… {pct * 100:.0f}%"


ProgressCallback = Callable[[DownloadProgress], None]


class DownloadManager:
    """Fetches model files into `models_dir`, verifying every one of them."""

    def __init__(
        self,
        models_dir: Path,
        attempts_per_url: int = 3,
        backoff_seconds: float = 2.0,
        session: requests.Session | None = None,
        allow_downloads: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.attempts_per_url = max(1, attempts_per_url)
        self.backoff_seconds = backoff_seconds
        self.allow_downloads = allow_downloads
        self._session = session or requests.Session()

    # -- public API -------------------------------------------------------- #

    def ensure_model(
        self, spec: ModelSpec, on_progress: ProgressCallback | None = None
    ) -> Path:
        """Make every file of `spec` present and verified. Returns models_dir."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        for file_spec in spec.files:
            self.ensure_file(spec, file_spec, on_progress)
        return self.models_dir

    def ensure_file(
        self,
        spec: ModelSpec,
        file_spec: FileSpec,
        on_progress: ProgressCallback | None = None,
    ) -> Path:
        target = file_spec.path_in(self.models_dir)
        if target.exists():
            if self._verify(target, file_spec):
                return target
            log.warning("checksum mismatch, re-downloading: %s", target.name)
            target.unlink(missing_ok=True)

        if not self.allow_downloads:
            raise EngineError(
                ErrorCode.MODEL_MISSING,
                f"{file_spec.name} is missing and downloads are disabled",
            )
        if not file_spec.urls:
            raise EngineError(
                ErrorCode.MODEL_MISSING, f"no download URL for {file_spec.name}"
            )

        last_error = ""
        for url in file_spec.urls:
            try:
                self._fetch_with_retries(spec, file_spec, url, target, on_progress)
            except EngineError as exc:
                if exc.code is ErrorCode.DISK_FULL:
                    raise
                last_error = exc.detail
                log.warning("mirror failed (%s): %s", url, exc.detail)
                continue
            if self._verify(target, file_spec):
                return target
            last_error = "checksum mismatch after download"
            target.unlink(missing_ok=True)

        raise EngineError(ErrorCode.DOWNLOAD_FAILED, f"{file_spec.name}: {last_error}")

    def free_space_bytes(self) -> int:
        probe = self.models_dir
        while not probe.exists() and probe.parent != probe:
            probe = probe.parent
        try:
            return shutil.disk_usage(probe).free
        except OSError:
            return 0

    def check_space_for(self, specs: list[ModelSpec]) -> None:
        """Fail before downloading rather than halfway through it."""
        needed = sum(
            f.size_bytes or 0
            for spec in specs
            for f in spec.missing_files(self.models_dir)
        )
        if not needed:
            return
        # Headroom for the .part file living beside the finished one.
        if self.free_space_bytes() < needed * 2:
            raise EngineError(
                ErrorCode.DISK_FULL,
                f"need about {needed * 2 / 1024 ** 3:.1f}GB free for model downloads",
            )

    # -- internals --------------------------------------------------------- #

    def _verify(self, path: Path, file_spec: FileSpec) -> bool:
        """No recorded hash means we cannot verify -- size is the weak fallback."""
        if file_spec.sha256:
            return sha256_of(path) == file_spec.sha256
        if file_spec.size_bytes:
            return path.stat().st_size == file_spec.size_bytes
        return path.stat().st_size > 0

    def _fetch_with_retries(
        self,
        spec: ModelSpec,
        file_spec: FileSpec,
        url: str,
        target: Path,
        on_progress: ProgressCallback | None,
    ) -> None:
        part = target.with_suffix(target.suffix + ".part")
        last_error = ""
        for attempt in range(1, self.attempts_per_url + 1):
            try:
                self._fetch_once(spec, file_spec, url, part, attempt, on_progress)
                part.replace(target)
                return
            except (requests.RequestException, OSError) as exc:
                if isinstance(exc, OSError) and _is_disk_full(exc):
                    part.unlink(missing_ok=True)
                    raise EngineError(ErrorCode.DISK_FULL, str(exc)) from exc
                last_error = str(exc)
                log.warning(
                    "download attempt %d/%d failed for %s: %s",
                    attempt, self.attempts_per_url, file_spec.name, exc,
                )
                if attempt < self.attempts_per_url:
                    time.sleep(self.backoff_seconds * (2 ** (attempt - 1)))
        raise EngineError(ErrorCode.DOWNLOAD_FAILED, last_error)

    def _fetch_once(
        self,
        spec: ModelSpec,
        file_spec: FileSpec,
        url: str,
        part: Path,
        attempt: int,
        on_progress: ProgressCallback | None,
    ) -> None:
        part.parent.mkdir(parents=True, exist_ok=True)
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}

        response = self._session.get(
            url, headers=headers, stream=True,
            timeout=(_CONNECT_TIMEOUT, _READ_TIMEOUT),
        )
        if response.status_code == 416:
            # Already have the whole body; the hash check decides if it is good.
            return
        if response.status_code not in (200, 206):
            response.close()
            raise requests.RequestException(f"HTTP {response.status_code} from {url}")

        # A server that ignores Range answers 200 with the whole file: start over
        # rather than appending a second copy onto the first.
        resuming = response.status_code == 206 and have > 0
        if have and not resuming:
            have = 0

        declared = response.headers.get("content-length")
        total = (have + int(declared)) if declared and declared.isdigit() else file_spec.size_bytes

        done = have
        try:
            with part.open("ab" if resuming else "wb") as fh:
                for chunk in response.iter_content(chunk_size=_CHUNK):
                    if not chunk:
                        continue
                    fh.write(chunk)
                    done += len(chunk)
                    if on_progress is not None:
                        on_progress(
                            DownloadProgress(
                                model_id=spec.id,
                                file_name=file_spec.name,
                                done_bytes=done,
                                total_bytes=total,
                                attempt=attempt,
                            )
                        )
        finally:
            response.close()


def _is_disk_full(exc: OSError) -> bool:
    """Recognise POSIX and Windows disk-full errors from an active write."""
    return exc.errno == errno.ENOSPC or getattr(exc, "winerror", None) == 112
