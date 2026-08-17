"""F0 extraction adapters: FCPE (fast) and RMVPE (accurate).

Both satisfy the `F0Extractor` protocol defined back in Phase 1
(`backends/f0.py`) and both return the same `F0Curve` on whatever hop the caller
asks for, so the range/segment/preview stages never learn which model ran.

* **FCPE** is `torchfcpe` with bundled weights -- nothing to download, and it is
  the one F0 path already proven end-to-end on the Intel accelerator (Phase 1).
  It is the analysis default and the Preview path.
* **RMVPE** is the accuracy model research section 5 names for final conversion.
  Its weights come from the catalogue; it runs on CPU because no RMVPE
  accelerator proof exists yet (the support matrix says so, and this respects
  it rather than borrowing FCPE's proof).

torch, torchfcpe, librosa and the vendored network are all imported lazily,
inside the methods -- this module is imported by the CLI, which must load in the
light CI environment where none of them exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve
from svc_engine.backends.f0 import F0Extractor

__all__ = [
    "RMVPE_MODEL_ID",
    "FcpeExtractor",
    "RmvpeExtractor",
    "resample_f0",
    "make_extractor",
    "default_method",
]

#: Catalogue id of the RMVPE weights (see data/models.json).
RMVPE_MODEL_ID = "f0_rmvpe"

_MODEL_SR = 16000


def resample_f0(f0_hz: np.ndarray, duration: float, hop_seconds: float) -> np.ndarray:
    """Put an F0 track produced at the model's own frame rate onto `hop_seconds`.

    Nearest-frame, never interpolated: a 0.0 means "unvoiced", and averaging it
    with a neighbouring pitch across a voicing boundary would invent a note that
    was never sung.
    """
    src = np.asarray(f0_hz, dtype=np.float64).ravel()
    n_src = src.size
    if n_src == 0 or duration <= 0.0:
        return np.zeros(0, dtype=np.float64)
    n_dst = max(1, int(round(duration / hop_seconds)))
    # Centre of each source and destination frame, in seconds.
    src_t = (np.arange(n_src) + 0.5) * (duration / n_src)
    dst_t = (np.arange(n_dst) + 0.5) * hop_seconds
    idx = np.searchsorted(src_t, dst_t)
    idx = np.clip(idx, 0, n_src - 1)
    left = np.clip(idx - 1, 0, n_src - 1)
    choose_left = np.abs(dst_t - src_t[left]) <= np.abs(dst_t - src_t[idx])
    idx = np.where(choose_left, left, idx)
    return src[idx]


def _to_model_mono(audio: AudioBuffer) -> np.ndarray:
    """Mono float32 at 16 kHz, the rate both F0 models were trained at."""
    import librosa

    samples = audio.samples
    mono = samples.mean(axis=0) if samples.ndim == 2 else samples
    mono = np.ascontiguousarray(mono, dtype=np.float32)
    if audio.sample_rate != _MODEL_SR:
        mono = librosa.resample(
            mono, orig_sr=audio.sample_rate, target_sr=_MODEL_SR, res_type="soxr_hq"
        )
    return mono.astype(np.float32, copy=False)


class FcpeExtractor:
    """torchfcpe with its bundled weights. Fast; proven on the accelerator."""

    def __init__(self, threshold: float = 0.006) -> None:
        self.threshold = threshold
        self._model: Any = None
        self._device = "cpu"

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend_id="fcpe",
            display_name_he="זיהוי גובה מהיר (FCPE)",
            available=True,
            supports_gpu=True,
            capabilities=frozenset({"f0", "preview"}),
        )

    def _ensure(self, device: str) -> Any:
        import torchfcpe

        if self._model is None or self._device != device:
            self._model = torchfcpe.spawn_bundled_infer_model(device=device)
            self._device = device
        return self._model

    def extract(
        self, audio: AudioBuffer, device: DeviceHint, hop_seconds: float = 0.01
    ) -> F0Curve:
        import torch

        mono = _to_model_mono(audio)
        model = self._ensure(device.torch_device)
        tensor = torch.from_numpy(mono).to(device.torch_device).unsqueeze(0).unsqueeze(-1)
        with torch.no_grad():
            f0 = model.infer(
                tensor,
                sr=_MODEL_SR,
                decoder_mode="local_argmax",
                threshold=self.threshold,
            )
        f0_np = f0.squeeze().detach().cpu().numpy().astype(np.float64)
        duration = mono.size / _MODEL_SR
        return F0Curve(hz=resample_f0(f0_np, duration, hop_seconds), hop_seconds=hop_seconds)

    def unload(self) -> None:
        self._model = None


class RmvpeExtractor:
    """The vendored RMVPE network. Accurate; CPU-only until it earns a proof."""

    def __init__(self, weights_path: Path | str, threshold: float = 0.03) -> None:
        self.weights_path = Path(weights_path)
        self.threshold = threshold
        self._model: Any = None
        self._device = "cpu"

    def info(self) -> BackendInfo:
        return BackendInfo(
            backend_id="rmvpe",
            display_name_he="זיהוי גובה מדויק (RMVPE)",
            available=self.weights_path.exists(),
            unavailable_reason=(
                None if self.weights_path.exists() else "משקולות RMVPE לא נמצאו"
            ),
            supports_gpu=False,  # no end-to-end accelerator proof yet
            capabilities=frozenset({"f0"}),
            model_root=self.weights_path.parent,
        )

    def _ensure(self, device: str) -> Any:
        from svc_engine.analysis.rmvpe_model import RmvpeModel

        if self._model is None or self._device != device:
            self._model = RmvpeModel(str(self.weights_path), device=device)
            self._device = device
        return self._model

    def extract(
        self, audio: AudioBuffer, device: DeviceHint, hop_seconds: float = 0.01
    ) -> F0Curve:
        mono = _to_model_mono(audio)
        # RMVPE's torch.stft path is unproven on XPU; keep it on CPU per the
        # support matrix. FCPE is the accelerated F0 path.
        model = self._ensure("cpu")
        f0_np = model.infer_from_audio(mono, thred=self.threshold).astype(np.float64)
        duration = mono.size / _MODEL_SR
        return F0Curve(hz=resample_f0(f0_np, duration, hop_seconds), hop_seconds=hop_seconds)

    def unload(self) -> None:
        self._model = None


def default_method() -> str:
    """FCPE: the only F0 model that needs no download and is accelerator-proven."""
    return "fcpe"


def make_extractor(
    method: str,
    weights_path: Path | str | None = None,
) -> F0Extractor:
    """Construct an extractor by name. `rmvpe` needs its weights path."""
    method = method.lower()
    if method == "fcpe":
        return FcpeExtractor()
    if method == "rmvpe":
        if weights_path is None:
            raise ValueError("rmvpe requires weights_path")
        return RmvpeExtractor(weights_path)
    raise ValueError(f"unknown F0 method: {method!r}")
