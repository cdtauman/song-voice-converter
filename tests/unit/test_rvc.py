"""RVC engine: F0 quantisation, retrieval blend, RMS match, backend contract.

The neural path (HuBERT + the vendored network) needs torch + weights and is
covered by docs/phase-reports/phase-5.md's runtime plan; here we test every
pure-numpy piece of the conversion maths and the torch-free backend logic.
"""

from __future__ import annotations

import numpy as np
import pytest

from svc_engine.backends.base import AudioBuffer, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionBackend, ConversionParams
from svc_engine.conversion.rvc import (
    BruteForceIndex,
    apply_up_key,
    blend_rms,
    blend_with_index,
    f0_to_coarse,
)
from svc_engine.conversion.rvc.backend import RVCv2Backend
from svc_engine.conversion.rvc.model import _load_checkpoint
from svc_engine.errors import EngineError, ErrorCode

# --- F0 quantisation ------------------------------------------------------- #

def test_f0_to_coarse_range_and_buckets() -> None:
    f0 = np.array([0.0, 30.0, 50.0, 110.0, 440.0, 1100.0, 2000.0])
    coarse = f0_to_coarse(f0)
    assert coarse.dtype == np.int32
    assert coarse.min() >= 1 and coarse.max() <= 255
    # Unvoiced and sub-floor pitches fall in bucket 1; the top of range saturates.
    assert coarse[0] == 1
    assert coarse[-1] == 255
    # Buckets rise monotonically with pitch across the voiced range.
    voiced = coarse[3:6]
    assert np.all(np.diff(voiced) > 0)


def test_apply_up_key_shifts_by_octave() -> None:
    f0 = np.array([0.0, 220.0, 440.0])
    up = apply_up_key(f0, 12)
    assert up[0] == 0.0  # unvoiced stays unvoiced
    assert up[1] == pytest.approx(440.0)
    assert up[2] == pytest.approx(880.0)
    down = apply_up_key(f0, -12)
    assert down[1] == pytest.approx(110.0)


# --- retrieval index ------------------------------------------------------- #

def test_bruteforce_search_matches_faiss_semantics() -> None:
    vectors = np.array([[0.0, 0.0], [1.0, 0.0], [5.0, 5.0]], dtype=np.float32)
    idx = BruteForceIndex(vectors)
    dist, ix = idx.search(np.array([[0.1, 0.0]], dtype=np.float32), k=2)
    # Squared L2, ascending: nearest is [0,0] then [1,0].
    assert list(ix[0]) == [0, 1]
    assert dist[0, 0] == pytest.approx(0.01, abs=1e-5)
    assert dist[0, 0] <= dist[0, 1]  # returned nearest-first


def test_blend_rate_zero_is_identity() -> None:
    feats = np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32)
    vectors = np.random.default_rng(1).standard_normal((20, 4)).astype(np.float32)
    idx = BruteForceIndex(vectors)
    out = blend_with_index(feats, vectors, idx.search, index_rate=0.0)
    assert np.array_equal(out, feats)


def test_blend_rate_one_moves_to_neighbours() -> None:
    # A feature sitting exactly on a training vector should, at rate 1, come back
    # essentially that vector (its nearest neighbours dominate the weighting).
    vectors = np.array([[3.0, 3.0], [0.0, 0.0], [-3.0, -3.0]], dtype=np.float32)
    idx = BruteForceIndex(vectors)
    feats = np.array([[3.0, 3.0]], dtype=np.float32)
    out = blend_with_index(feats, vectors, idx.search, index_rate=1.0, k=3)
    assert np.allclose(out, [[3.0, 3.0]], atol=1e-3)
    assert out.shape == feats.shape


def test_blend_is_finite_on_exact_match() -> None:
    # Exact match -> distance 0; the 1/d weighting must not explode to NaN/inf.
    vectors = np.array([[1.0, 2.0], [9.0, 9.0]], dtype=np.float32)
    idx = BruteForceIndex(vectors)
    out = blend_with_index(
        np.array([[1.0, 2.0]], dtype=np.float32), vectors, idx.search, 0.7, k=2
    )
    assert np.all(np.isfinite(out))


# --- RMS envelope ---------------------------------------------------------- #

def test_blend_rms_rate_one_is_identity() -> None:
    conv = np.sin(np.linspace(0, 20, 8000)).astype(np.float32)
    out = blend_rms(np.ones(8000, dtype=np.float32), 16000, conv, 16000, rate=1.0)
    assert np.array_equal(out, conv.astype(np.float32))


def test_blend_rms_rate_zero_follows_source_loudness() -> None:
    # Source loud in the first half, quiet in the second; a flat converted signal
    # should inherit that shape at rate 0.
    src = np.concatenate([np.ones(16000), 0.1 * np.ones(16000)]).astype(np.float32)
    conv = 0.5 * np.ones(32000, dtype=np.float32)
    out = blend_rms(src, 16000, conv, 16000, rate=0.0)
    assert np.abs(out[:8000]).mean() > np.abs(out[-8000:]).mean() * 3


# --- backend contract (torch-free parts) ----------------------------------- #

def test_backend_satisfies_protocol(tmp_path) -> None:  # type: ignore[no-untyped-def]
    backend = RVCv2Backend(models_dir=tmp_path)
    assert isinstance(backend, ConversionBackend)


def test_backend_unavailable_without_stack(tmp_path) -> None:  # type: ignore[no-untyped-def]
    # torch/transformers are not in the test env, so the backend reports why.
    info = RVCv2Backend(models_dir=tmp_path).info()
    assert info.backend_id == "rvc_v2"
    assert info.available is False
    assert info.unavailable_reason  # a Hebrew explanation, not a crash
    assert info.supports_gpu is True


def test_convert_before_load_raises(tmp_path) -> None:  # type: ignore[no-untyped-def]
    backend = RVCv2Backend(models_dir=tmp_path)
    audio = AudioBuffer(np.zeros((1, 16000), dtype=np.float32), 44100)
    f0 = F0Curve(hz=np.full(100, 220.0), hop_seconds=0.01)
    with pytest.raises(EngineError) as exc:
        backend.convert(audio, f0, ConversionParams())
    assert exc.value.code is ErrorCode.BACKEND_UNAVAILABLE


def test_unload_is_safe_when_nothing_loaded(tmp_path) -> None:  # type: ignore[no-untyped-def]
    RVCv2Backend(models_dir=tmp_path).unload()  # must not raise


def test_load_hint_maps_to_torch_device() -> None:
    # DeviceHint -> torch device string is what the backend passes down.
    assert DeviceHint().torch_device == "cpu"


def test_checkpoint_loader_uses_torch_restricted_weights_mode(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class FakeTorch:
        def __init__(self) -> None:
            self.kwargs: dict[str, object] = {}

        def load(self, path: str, **kwargs: object) -> dict[str, object]:
            self.kwargs = kwargs
            assert path.endswith("voice.pth")
            return {"config": [1, 2, 3], "weight": {}}

    torch = FakeTorch()
    assert _load_checkpoint(torch, tmp_path / "voice.pth") == {
        "config": [1, 2, 3], "weight": {}
    }
    assert torch.kwargs == {"map_location": "cpu", "weights_only": True}


def test_checkpoint_loader_rejects_unsafe_pickle(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class FakeTorch:
        def load(self, path: str, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("unsafe global rejected")

    with pytest.raises(EngineError) as exc:
        _load_checkpoint(FakeTorch(), tmp_path / "unsafe.pth")
    assert exc.value.code is ErrorCode.MODEL_CORRUPT


def test_checkpoint_loader_rejects_non_rvc_weights(tmp_path) -> None:  # type: ignore[no-untyped-def]
    class FakeTorch:
        def load(self, path: str, **kwargs: object) -> dict[str, object]:
            return {"config": [], "weight": {}}

    with pytest.raises(EngineError) as exc:
        _load_checkpoint(FakeTorch(), tmp_path / "not-rvc.pth")
    assert exc.value.code is ErrorCode.MODEL_CORRUPT
