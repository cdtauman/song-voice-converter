"""Run one RVC conversion pass, wiring the tested DSP to the torch network.

This is RVC-Project's `Pipeline.vc` (infer/vc/pipeline.py, MIT) restated so the
pure-numpy parts are our unit-tested modules (`f0`, `index`, `rms`) and only the
genuinely neural steps -- HuBERT feature extraction and the synthesizer's
`infer` -- stay in torch. The windowing RVC does by hand is instead handled a
level up by `conversion.chunking`, so this converts a single already-bounded
segment: 16 kHz in, target-rate out, length set by the model.

torch, librosa and the HuBERT/network modules are imported lazily; importing
this module never pulls the AI stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from svc_engine.conversion.rvc.f0 import apply_up_key, f0_to_coarse
from svc_engine.conversion.rvc.index import SearchFn, blend_with_index
from svc_engine.conversion.rvc.model import RvcModel
from svc_engine.conversion.rvc.rms import blend_rms

__all__ = ["RvcInferenceConfig", "RvcInferencer", "HUBERT_SR", "WINDOW"]

#: HuBERT input rate and RVC's per-frame hop, both fixed by the models.
HUBERT_SR = 16000
WINDOW = 160  # samples per frame at 16 kHz -> 100 fps, i.e. a 10 ms hop


@dataclass(frozen=True)
class RvcInferenceConfig:
    """The knobs one conversion honours. Mirrors `ConversionParams` after the
    backend has mapped names across."""

    semitones: float = 0.0
    index_rate: float = 0.70
    protect: float = 0.33
    rms_mix_rate: float = 0.25


class RvcInferencer:
    """Holds a loaded voice + HuBERT and converts 16 kHz segments."""

    def __init__(
        self,
        model: RvcModel,
        hubert: Any,
        index_vectors: np.ndarray | None = None,
        index_search: SearchFn | None = None,
    ) -> None:
        self.model = model
        self.hubert = hubert
        self.index_vectors = index_vectors
        self.index_search = index_search

    def convert_segment(
        self,
        audio_16k: np.ndarray,
        f0_hz: np.ndarray | None,
        cfg: RvcInferenceConfig,
        speaker_id: int = 0,
    ) -> np.ndarray:
        """Convert one mono 16 kHz segment. Returns mono float32 at the model's
        target rate. `f0_hz` is the per-frame source pitch (before the octave
        shift); pass None only for a pitchless (`if_f0 == 0`) model."""
        import torch  # noqa: PLC0415

        device = self.model.device
        net_g = self.model.net_g
        audio_16k = np.asarray(audio_16k, dtype=np.float32)

        # -- content features (HuBERT), then the retrieval blend --------------
        from svc_engine.conversion.rvc.hubert import extract_features

        feats_t = extract_features(
            self.hubert, torch.from_numpy(audio_16k), self.model.version, device
        )
        feats = feats_t[0].detach().cpu().numpy().astype(np.float32)
        feats0 = feats.copy()  # pre-blend copy, for the `protect` mix

        if (
            self.index_vectors is not None
            and self.index_search is not None
            and cfg.index_rate > 0.0
        ):
            feats = blend_with_index(
                feats, self.index_vectors, self.index_search, cfg.index_rate
            )

        # RVC upsamples the 50 fps features to 100 fps to line up with pitch.
        feats = _upsample2(feats)
        feats0 = _upsample2(feats0)
        p_len = min(audio_16k.shape[0] // WINDOW, feats.shape[0])

        pitch_t = pitchf_t = None
        if self.model.uses_f0 and f0_hz is not None:
            coarse, continuous = self._prepare_pitch(f0_hz, p_len, cfg.semitones)
            # `protect`: keep the original (unblended) features on unvoiced
            # frames, so consonants and breaths are not smeared by retrieval.
            if cfg.protect < 0.5:
                feats = _protect_blend(feats, feats0, continuous, cfg.protect)
            pitch_t = torch.from_numpy(coarse[:p_len]).unsqueeze(0).long().to(device)
            pitchf_t = (
                torch.from_numpy(continuous[:p_len]).unsqueeze(0).float().to(device)
            )

        feats_in = torch.from_numpy(feats[:p_len]).unsqueeze(0).float().to(device)
        length = torch.tensor([p_len], device=device).long()
        sid = torch.tensor([speaker_id], device=device).long()

        with torch.no_grad():
            if pitch_t is not None and pitchf_t is not None:
                out = net_g.infer(feats_in, length, pitch_t, pitchf_t, sid)[0]
            else:
                out = net_g.infer(feats_in, length, sid)[0]
            audio_out = out[0, 0].detach().cpu().float().numpy()

        if cfg.rms_mix_rate < 1.0:
            audio_out = blend_rms(
                audio_16k, HUBERT_SR, audio_out, self.model.target_sr, cfg.rms_mix_rate
            )
        return audio_out.astype(np.float32)

    def _prepare_pitch(
        self, f0_hz: np.ndarray, p_len: int, semitones: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Octave-shift and coarse-quantise the pitch, aligned to `p_len`."""
        shifted = apply_up_key(_resize(np.asarray(f0_hz, dtype=np.float64), p_len), semitones)
        coarse = f0_to_coarse(shifted)
        return coarse, shifted.astype(np.float32)


def _upsample2(feats: np.ndarray) -> np.ndarray:
    """Nearest-neighbour x2 along time -- RVC's `interpolate(scale_factor=2)`."""
    return np.repeat(feats, 2, axis=0)


def _resize(values: np.ndarray, length: int) -> np.ndarray:
    """Linearly resample a 1-D curve to `length`, holding zeros (unvoiced)."""
    if length <= 0:
        return np.zeros(0, dtype=np.float64)
    if values.size == length:
        return values
    if values.size == 0:
        return np.zeros(length, dtype=np.float64)
    src = np.linspace(0.0, 1.0, values.size)
    dst = np.linspace(0.0, 1.0, length)
    return np.interp(dst, src, values)


def _protect_blend(
    feats: np.ndarray, feats0: np.ndarray, pitchf: np.ndarray, protect: float
) -> np.ndarray:
    """Blend blended/original features by voicing, per RVC's `protect`."""
    n = min(feats.shape[0], feats0.shape[0], pitchf.shape[0])
    mask = np.where(pitchf[:n] > 0, 1.0, protect).astype(np.float32)[:, None]
    out = feats.copy()
    out[:n] = feats[:n] * mask + feats0[:n] * (1.0 - mask)
    return out
