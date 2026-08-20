"""RVC v2 voice conversion -- the MVP `ConversionBackend`.

Reuse decision, per docs/reuse-policy.md: **subset insertion (degree 4).** There
is no maintained pip package for RVC inference, so the network architecture is
vendored from RVC-Project (MIT) under `infer_pack/`, the same way Phase 3
vendored RMVPE. Everything else in this package -- the F0 quantisation, the
retrieval-index blend, the RMS envelope match, the model loader and the backend
adapter -- is ours, wrapping that network behind the `ConversionBackend`
interface so nothing above it is locked to RVC. See docs/third-party.md.

The heavy pieces (torch, transformers/HuBERT, faiss) are imported lazily inside
`RVCv2Backend`, so importing this package -- and the pure-numpy DSP in `f0`,
`index` and `rms` -- never drags in the AI stack. The DSP is unit-tested; the
full inference path is exercised only where the runtime deps, the HuBERT weights
and a real voice model are present (see docs/phase-reports/phase-5.md).
"""

from __future__ import annotations

from svc_engine.conversion.rvc.f0 import apply_up_key, f0_to_coarse
from svc_engine.conversion.rvc.index import BruteForceIndex, blend_with_index
from svc_engine.conversion.rvc.rms import blend_rms

__all__ = [
    "apply_up_key",
    "f0_to_coarse",
    "BruteForceIndex",
    "blend_with_index",
    "blend_rms",
]
