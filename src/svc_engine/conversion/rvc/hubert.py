"""The HuBERT/ContentVec content encoder RVC conditions on.

RVC represents *what* is being sung -- phonetic content, separated from the
original singer's timbre -- with a HuBERT model. That representation is what lets
the target voice re-sing the same words and melody (docs/research.md 3). Modern
RVC ships HuBERT in Transformers format with an added `final_proj`; this wraps
loading it and pulling the right layer: v1 uses encoder layer 9 through
`final_proj` (256-dim), v2 uses the final hidden state (768-dim). Adapted from
RVC-Project `infer/hubert.py` (MIT), minus the CUDA-graph and DirectML paths.

torch and transformers are imported lazily; this module is import-safe without
them, and only `load_hubert` / `extract_features` need them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = ["load_hubert", "extract_features"]


def load_hubert(model_dir: Path | str, device: str = "cpu") -> Any:
    """Load the local Transformers HuBERT (with `final_proj`) in eval mode.

    `model_dir` is a Transformers model folder (`config.json` + weights).
    Raises `EngineError(MODEL_MISSING)` if it is not there -- the catalogue entry
    `content_hubert` fetches it (docs/models.md).
    """
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415
    from transformers import HubertModel  # noqa: PLC0415

    from svc_engine.errors import EngineError, ErrorCode

    model_dir = Path(model_dir)
    if not (model_dir / "config.json").is_file():
        raise EngineError(
            ErrorCode.MODEL_MISSING,
            f"HuBERT model not found at {model_dir}",
        )

    class HubertModelWithFinalProj(HubertModel):
        def __init__(self, config: Any) -> None:
            super().__init__(config)
            self.final_proj = nn.Linear(config.hidden_size, config.classifier_proj_size)

    model: Any = HubertModelWithFinalProj.from_pretrained(
        str(model_dir),
        local_files_only=True,
        torch_dtype=torch.float32,
    )
    return model.to(device).eval()


def extract_features(model: Any, audio_16k: Any, version: str, device: str = "cpu") -> Any:
    """Return the [1, frames, dim] content features for a 16 kHz mono waveform.

    `version` selects the RVC feature: "v1" -> 256-dim (layer 9 + final_proj),
    "v2" -> 768-dim (final hidden state). `audio_16k` is a 1-D float tensor.
    """
    import torch  # noqa: PLC0415

    if version not in {"v1", "v2"}:
        raise ValueError(f"unsupported RVC feature version: {version!r}")

    feats = audio_16k
    if feats.dim() == 2:
        feats = feats.mean(-1)
    feats = feats.view(1, -1).to(device)

    with torch.no_grad():
        if version == "v1":
            outputs = model(
                input_values=feats, output_hidden_states=True, return_dict=True
            )
            return model.final_proj(outputs.hidden_states[9])
        outputs = model(
            input_values=feats, output_hidden_states=False, return_dict=True
        )
        return outputs.last_hidden_state
