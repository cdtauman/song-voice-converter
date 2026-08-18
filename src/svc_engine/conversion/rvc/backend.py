"""`ConversionBackend` on top of the vendored RVC v2 engine.

This is the adapter the rest of SongVoice talks to: `info` / `load` / `convert`
/ `unload`, nothing RVC-specific leaking upward. The heavy work -- torch,
HuBERT, faiss, the vendored network -- is imported only inside `load` and
`convert`, so constructing the backend and asking `info()` whether it is
available never touches the AI stack.

What is real and tested here: availability detection, parameter mapping from
`ConversionParams` to RVC's knobs, the resample-in/resample-out length contract,
and that `unload` releases the model. What needs the runtime deps + HuBERT
weights + a real voice model to exercise end to end is documented in
docs/phase-reports/phase-5.md.
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import Any

import numpy as np

from svc_engine.audio.buffers import fit_length
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionParams, VoiceHandle
from svc_engine.resources import DownloadManager, ModelRegistry, load_registry
from svc_engine.voices.manifest import INDEX_FILE, MODEL_FILE

__all__ = ["RVCv2Backend", "BACKEND_ID", "HUBERT_MODEL_ID"]

log = logging.getLogger(__name__)

BACKEND_ID = "rvc_v2"
#: Catalogue id of the HuBERT content encoder (docs/models.md). Downloaded once,
#: shared by every voice.
HUBERT_MODEL_ID = "content_hubert"

_HUBERT_DIRNAME = "content_hubert"


def _stack_available() -> bool:
    """True if the packages RVC inference needs are importable."""
    return all(
        importlib.util.find_spec(pkg) is not None
        for pkg in ("torch", "transformers")
    )


class RVCv2Backend:
    """Runs RVC v2 inference for one loaded voice at a time."""

    def __init__(
        self,
        models_dir: Path,
        registry: ModelRegistry | None = None,
        downloader: DownloadManager | None = None,
        allow_downloads: bool = True,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.registry = registry or load_registry()
        self.downloader = downloader or DownloadManager(
            self.models_dir, allow_downloads=allow_downloads
        )
        self._inferencer: Any | None = None
        self._voice_id: str | None = None
        self._device: str = "cpu"

    # -- ConversionBackend -------------------------------------------------- #

    def info(self) -> BackendInfo:
        stack = _stack_available()
        hubert_ready = self._hubert_present()
        available = stack and hubert_ready
        if not stack:
            reason: str | None = "חבילות ההמרה (torch/transformers) אינן מותקנות."
        elif not hubert_ready:
            reason = "מודל תוכן הדיבור (HuBERT) עדיין לא הורד."
        else:
            reason = None
        return BackendInfo(
            backend_id=BACKEND_ID,
            display_name_he="מנוע המרת קול (RVC v2)",
            available=available,
            unavailable_reason=reason,
            supports_gpu=True,
            capabilities=frozenset({"rvc_v2", "external_f0", "retrieval_index"}),
            model_root=self.models_dir,
        )

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:
        """Bring a voice into memory. Idempotent for the same voice + device."""
        target = device.torch_device
        if self._inferencer is not None and self._voice_id == voice.voice_id \
                and self._device == target:
            return
        self.unload()
        try:
            from svc_engine.conversion.rvc.hubert import load_hubert
            from svc_engine.conversion.rvc.infer import RvcInferencer
            from svc_engine.conversion.rvc.model import load_rvc_model

            hubert_dir = self._ensure_hubert()
            model_path = voice.root / MODEL_FILE
            model = load_rvc_model(model_path, device=target)
            hubert = load_hubert(hubert_dir, device=target)

            index_vectors, index_search = self._load_index(voice)

            self._inferencer = RvcInferencer(
                model=model,
                hubert=hubert,
                index_vectors=index_vectors,
                index_search=index_search,
            )
            self._voice_id = voice.voice_id
            self._device = target
        except Exception:
            # A model may already occupy accelerator memory even when HuBERT,
            # the index, or inferencer construction fails.  No partial load is
            # observable after this point and the allocator cache is released.
            self._inferencer = None
            self._voice_id = None
            self._device = "cpu"
            self._empty_cache()
            raise

    def convert(
        self,
        audio: AudioBuffer,
        f0: F0Curve,
        params: ConversionParams,
    ) -> AudioBuffer:
        """Convert a segment. Output length matches the input exactly."""
        if self._inferencer is None:
            from svc_engine.errors import EngineError, ErrorCode

            raise EngineError(
                ErrorCode.BACKEND_UNAVAILABLE, "convert() called before load()"
            )

        from svc_engine.conversion.rvc.infer import HUBERT_SR, RvcInferenceConfig

        mono = audio.samples.mean(axis=0) if audio.channels > 1 else audio.samples[0]
        audio_16k = _resample(mono, audio.sample_rate, HUBERT_SR)
        f0_hz = np.asarray(f0.hz, dtype=np.float64).ravel() if f0.frames else None

        cfg = RvcInferenceConfig(
            semitones=float(params.semitones),
            index_rate=float(params.index_rate),
            protect=float(params.protect),
            rms_mix_rate=float(params.rms_mix_rate),
        )
        out_np = self._inferencer.convert_segment(audio_16k, f0_hz, cfg)

        out_at_input = _resample(out_np, self._inferencer.model.target_sr, audio.sample_rate)
        buffer = AudioBuffer(
            samples=out_at_input.reshape(1, -1).astype(np.float32),
            sample_rate=audio.sample_rate,
        )
        return fit_length(buffer, audio.frames)

    def unload(self) -> None:
        """Release the model and any accelerator memory it holds."""
        had = self._inferencer is not None
        self._inferencer = None
        self._voice_id = None
        if had:
            self._empty_cache()

    # -- internals ---------------------------------------------------------- #

    def _hubert_present(self) -> bool:
        """True if the HuBERT weights are already on disk (catalogue-driven)."""
        if HUBERT_MODEL_ID not in self.registry:
            return False
        return self.registry.get(HUBERT_MODEL_ID).is_present(self.models_dir)

    def _ensure_hubert(self) -> Path:
        """Download the HuBERT content encoder if missing; return its folder."""
        if HUBERT_MODEL_ID in self.registry:
            spec = self.registry.get(HUBERT_MODEL_ID)
            self.downloader.check_space_for([spec])
            self.downloader.ensure_model(spec)
        return self.models_dir / _HUBERT_DIRNAME

    def _load_index(self, voice: VoiceHandle) -> tuple[np.ndarray | None, Any]:
        """Load the voice's faiss index if present and faiss is installed."""
        index_path = voice.root / INDEX_FILE
        if not index_path.exists():
            return None, None
        if importlib.util.find_spec("faiss") is None:
            log.info("faiss not installed; running voice %s without its index", voice.voice_id)
            return None, None
        try:
            from svc_engine.conversion.rvc.index import load_index

            index, vectors = load_index(index_path)
            return vectors, index.search
        except Exception as exc:  # noqa: BLE001 -- a bad index must not be fatal
            log.warning("could not load index for %s: %s", voice.voice_id, exc)
            return None, None

    def _empty_cache(self) -> None:
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                xpu.empty_cache()
        except Exception:  # noqa: BLE001 -- best-effort cleanup
            pass


def _resample(signal: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a 1-D signal. Uses librosa when rates differ; else a copy."""
    signal = np.asarray(signal, dtype=np.float32)
    if orig_sr == target_sr:
        return signal.copy()
    import librosa  # noqa: PLC0415

    return librosa.resample(signal, orig_sr=orig_sr, target_sr=target_sr).astype(np.float32)
