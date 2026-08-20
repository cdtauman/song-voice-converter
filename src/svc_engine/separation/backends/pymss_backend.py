"""`SeparationBackend` on top of `pymss` (MIT) -- the second implementation.

Having two backends behind one interface is not symmetry for its own sake. The
Phase 2 URL audit found that five of seven checkpoints hosted in the UVR release
repo now return 404. pymss carries an independent registry with an independent
mirror, so a model that has evaporated from one source is often still reachable
from the other. That is worth more here than any quality difference between the
two libraries.

It is not the default: audio-separator has the wider model coverage and far more
real-world use. This backend is what the ensemble and the benchmark compare
against, and what the pipeline falls back to when a model is unavailable.
"""

from __future__ import annotations

import importlib.util
import logging
from contextlib import suppress
from pathlib import Path

import numpy as np

from svc_engine.audio.buffers import fit_length
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint
from svc_engine.backends.separation import SeparationRequest, StemKind, Stems
from svc_engine.compute.devices import ComputeBackend
from svc_engine.errors import EngineError, ErrorCode

__all__ = ["PymssBackend", "BACKEND_ID"]

log = logging.getLogger(__name__)

BACKEND_ID = "pymss"

_STEM_ALIASES: dict[str, StemKind] = {
    "vocals": StemKind.VOCALS,
    "vocal": StemKind.VOCALS,
    "instrumental": StemKind.INSTRUMENTAL,
    "other": StemKind.OTHER,
    "bass": StemKind.BASS,
    "drums": StemKind.DRUMS,
    "lead": StemKind.LEAD,
    "backing": StemKind.BACKING,
    "reverb": StemKind.AMBIENCE,
    "echo": StemKind.AMBIENCE,
    "noreverb": StemKind.VOCALS,
    "dry": StemKind.VOCALS,
}


class PymssBackend:
    """Wraps `pymss.MSSeparator`, which already returns arrays rather than files."""

    def __init__(self, models_dir: Path, work_dir: Path | None = None) -> None:
        self.models_dir = Path(models_dir)
        self.work_dir = Path(work_dir) if work_dir else None
        self._separator: object | None = None
        self._loaded_model: str | None = None

    # -- SeparationBackend -------------------------------------------------- #

    def info(self) -> BackendInfo:
        available = importlib.util.find_spec("pymss") is not None
        return BackendInfo(
            backend_id=BACKEND_ID,
            display_name_he="מנוע הפרדה חלופי",
            available=available,
            unavailable_reason=None if available else "החבילה pymss אינה מותקנת.",
            supports_gpu=True,
            capabilities=frozenset({"roformer", "mdx23c", "scnet", "independent_mirror"}),
            model_root=self.models_dir,
        )

    def list_models(self) -> list[str]:
        if importlib.util.find_spec("pymss") is None:
            return []
        import pymss

        return sorted(m.name for m in pymss.list_models(supported=True))

    def separate(
        self,
        audio: AudioBuffer,
        request: SeparationRequest,
        device: DeviceHint,
    ) -> Stems:
        separator = self._ensure_loaded(request, device)

        # pymss speaks (samples, channels); ours is (channels, samples).
        mix = np.ascontiguousarray(audio.samples.T.astype(np.float32))
        try:
            result = separator.separate(mix, pbar=False)
        except Exception as exc:
            if "not found" in str(exc).lower():
                raise EngineError(ErrorCode.MODEL_MISSING, str(exc)) from exc
            raise

        raw = result[0] if isinstance(result, tuple) else result
        if not isinstance(raw, dict):
            raise EngineError(ErrorCode.INTERNAL, "pymss returned an unexpected result type")

        parts: dict[StemKind, AudioBuffer] = {}
        for name, array in raw.items():
            kind = _STEM_ALIASES.get(str(name).strip().lower())
            if kind is None:
                log.warning("unmapped pymss stem name %r", name)
                continue
            data = np.asarray(array, dtype=np.float32)
            if data.ndim == 1:
                data = data[None, :]
            elif data.shape[0] > data.shape[1]:
                data = data.T
            parts[kind] = fit_length(
                AudioBuffer(samples=np.ascontiguousarray(data), sample_rate=audio.sample_rate),
                audio.frames,
            )

        return Stems(parts=parts, model_id=request.model_id)

    def unload(self) -> None:
        separator, self._separator, self._loaded_model = self._separator, None, None
        if separator is None:
            return
        close = getattr(separator, "close", None)
        if callable(close):
            with suppress(Exception):
                close()

    # -- internals ---------------------------------------------------------- #

    def _ensure_loaded(self, request: SeparationRequest, device: DeviceHint):  # type: ignore[no-untyped-def]
        if self._separator is not None and self._loaded_model == request.model_id:
            self._update_params(request)
            return self._separator

        self.unload()
        try:
            import pymss
        except ImportError as exc:
            raise EngineError(ErrorCode.BACKEND_UNAVAILABLE, str(exc)) from exc

        self.models_dir.mkdir(parents=True, exist_ok=True)
        try:
            separator = pymss.create_separator(
                request.model_id,
                model_dir=str(self.models_dir),
                device=self._device_string(device),
                device_ids=[device.device_index],
                store_dirs=str(self.work_dir or self.models_dir),
                inference_params=self._inference_params(request),
            )
        except FileNotFoundError as exc:
            raise EngineError(ErrorCode.MODEL_MISSING, str(exc)) from exc

        self._separator = separator
        self._loaded_model = request.model_id
        return separator

    @staticmethod
    def _device_string(device: DeviceHint) -> str:
        if device.backend is ComputeBackend.CUDA:
            return "cuda"
        if device.backend is ComputeBackend.XPU:
            return "xpu"
        return "cpu"

    @staticmethod
    def _inference_params(request: SeparationRequest) -> dict[str, object]:
        return {
            "batch_size": max(1, request.batch_size),
            "overlap_size": max(2, request.overlap),
            "chunk_size": request.segment_size,
            "normalize": False,
        }

    def _update_params(self, request: SeparationRequest) -> None:
        update = getattr(self._separator, "update_inference_params", None)
        if update is None:
            return
        with suppress(Exception):
            update(**self._inference_params(request))
