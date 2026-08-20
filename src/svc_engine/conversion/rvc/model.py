"""Load an RVC voice checkpoint into its synthesizer network.

A `.pth` RVC model is a dict: `config` (the synthesizer constructor args, whose
last entry is the target sample rate), `weight` (the state dict), plus `f0`
(1 = pitch-conditioned) and `version` (`v1` 256-dim / `v2` 768-dim). The recipe
here is RVC-Project's own (infer/vc/modules.py, MIT): pick the synthesizer class
from `(version, f0)`, build it from `config`, drop the training-only posterior
encoder, and load the weights non-strictly.

torch and the vendored network are imported lazily, so this module is safe to
import without the AI stack; only `load_rvc_model` needs it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["RvcModel", "load_rvc_model"]


@dataclass
class RvcModel:
    """A loaded RVC voice: the network plus what the pipeline needs to know."""

    net_g: Any            # the torch synthesizer, on `device`, in eval mode
    target_sr: int        # output sample rate baked into the checkpoint
    version: str          # "v1" | "v2" -> HuBERT 256-dim vs 768-dim features
    if_f0: int            # 1 = pitch-conditioned, 0 = pitchless
    n_spk: int            # speaker count (emb_g rows)
    device: str

    @property
    def uses_f0(self) -> bool:
        return self.if_f0 == 1


def load_rvc_model(model_path: Path | str, device: str = "cpu") -> RvcModel:
    """Build the synthesizer for a `.pth` voice and put it on `device`.

    Follows RVC's loader exactly so real community checkpoints load. Raises
    `EngineError(MODEL_CORRUPT)` if the file is not a recognisable RVC model.
    """
    import torch  # noqa: PLC0415 -- heavy dep, imported on demand

    from svc_engine.conversion.rvc.infer_pack import models as rvc_models
    cpt = _load_checkpoint(torch, Path(model_path))

    config, target_sr, version, if_f0, n_spk, weights = _checkpoint_parts(cpt)
    # n_spk is authoritative from the embedding table, not the stored config.
    config[-3] = n_spk

    synthesizers = {
        ("v1", 1): rvc_models.SynthesizerTrnMs256NSFsid,
        ("v1", 0): rvc_models.SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): rvc_models.SynthesizerTrnMs768NSFsid,
        ("v2", 0): rvc_models.SynthesizerTrnMs768NSFsid_nono,
    }
    synthesizer = synthesizers[(version, if_f0)]

    net_g = synthesizer(*config, is_half=False)
    # The posterior encoder is only used in training; inference checkpoints omit
    # it, so drop it before a non-strict load to avoid a spurious mismatch.
    if hasattr(net_g, "enc_q"):
        del net_g.enc_q
    net_g.load_state_dict(weights, strict=False)
    net_g = net_g.eval().to(device).float()

    return RvcModel(
        net_g=net_g,
        target_sr=target_sr,
        version=version,
        if_f0=if_f0,
        n_spk=n_spk,
        device=device,
    )


def _load_checkpoint(torch: Any, model_path: Path) -> dict[str, Any]:
    """Load only tensor and primitive checkpoint data, never pickle code.

    Voice archives are user-provided.  ``weights_only=True`` keeps PyTorch's
    restricted unpickler on the boundary; do not fall back to regular pickle
    loading for older torch releases or checkpoints requiring custom globals.
    """
    from svc_engine.errors import EngineError, ErrorCode

    try:
        cpt = torch.load(str(model_path), map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 -- malformed or unsafe checkpoint
        raise EngineError(
            ErrorCode.MODEL_CORRUPT, f"could not safely read RVC model: {exc}"
        ) from exc

    if (
        not isinstance(cpt, dict)
        or not isinstance(cpt.get("config"), (list, tuple))
        or len(cpt["config"]) < 3
        or not isinstance(cpt.get("weight"), dict)
    ):
        raise EngineError(ErrorCode.MODEL_CORRUPT, "file is not an RVC voice checkpoint")
    return cpt


def _checkpoint_parts(cpt: dict[str, Any]) -> tuple[list[Any], int, str, int, int, dict]:
    """Validate the RVC-specific shape before constructing a network."""
    from svc_engine.errors import EngineError, ErrorCode

    config = list(cpt["config"])
    raw_sample_rate = config[-1]
    raw_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    weights = cpt["weight"]

    if (
        type(raw_sample_rate) is not int
        or raw_sample_rate <= 0
        or type(raw_f0) is not int
        or raw_f0 not in {0, 1}
        or not isinstance(version, str)
        or version not in {"v1", "v2"}
        or not isinstance(weights, dict)
    ):
        raise EngineError(ErrorCode.MODEL_CORRUPT, "RVC checkpoint has invalid metadata")

    embedding = weights.get("emb_g.weight")
    if embedding is None:
        raise EngineError(
            ErrorCode.MODEL_CORRUPT, "RVC checkpoint is missing speaker embeddings"
        )
    try:
        n_spk = int(embedding.shape[0])
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise EngineError(
            ErrorCode.MODEL_CORRUPT, "RVC checkpoint is missing speaker embeddings"
        ) from exc
    if n_spk <= 0:
        raise EngineError(ErrorCode.MODEL_CORRUPT, "RVC checkpoint has no speakers")

    return config, int(raw_sample_rate), version, raw_f0, n_spk, weights
