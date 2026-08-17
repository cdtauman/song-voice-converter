"""Multi-backend execution: detection, preference, and per-component selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from svc_engine.backends.base import DeviceHint
from svc_engine.compute import (
    PREFERENCE,
    Component,
    ComponentSupport,
    ComputeBackend,
    DeviceInfo,
    DeviceManager,
    ProofLevel,
    SupportMatrix,
    detect_display_adapters,
    load_matrix,
)


def _device(backend: ComputeBackend, name: str = "x") -> DeviceInfo:
    return DeviceInfo(backend=backend, index=0, name=name, available=True)


class _FakeManager(DeviceManager):
    def __init__(self, backends: list[ComputeBackend]) -> None:
        super().__init__()
        self._fake = [_device(b, b.value) for b in backends]

    def detect(self) -> list[DeviceInfo]:
        return self._fake


# --- preference order ------------------------------------------------------ #

def test_preference_is_cuda_then_xpu_then_cpu() -> None:
    assert PREFERENCE == (ComputeBackend.CUDA, ComputeBackend.XPU, ComputeBackend.CPU)


def test_cuda_wins_when_present() -> None:
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU, ComputeBackend.CUDA])
    assert m.preferred().backend is ComputeBackend.CUDA


def test_xpu_wins_over_cpu() -> None:
    """Intel is a real accelerator, not a fallback."""
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU])
    assert m.preferred().backend is ComputeBackend.XPU


def test_cpu_only_machine() -> None:
    m = _FakeManager([ComputeBackend.CPU])
    assert m.preferred().backend is ComputeBackend.CPU


def test_select_falls_back_to_cpu_when_allowed_set_is_unavailable() -> None:
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU])
    chosen = m.select({ComputeBackend.CUDA})
    assert chosen.backend is ComputeBackend.CPU


def test_available_backends_are_ordered_by_preference() -> None:
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU, ComputeBackend.CUDA])
    assert m.available_backends() == [
        ComputeBackend.CUDA, ComputeBackend.XPU, ComputeBackend.CPU
    ]


# --- torch device strings -------------------------------------------------- #

@pytest.mark.parametrize(
    ("backend", "expected"),
    [(ComputeBackend.CUDA, "cuda:0"), (ComputeBackend.XPU, "xpu:0"), (ComputeBackend.CPU, "cpu")],
)
def test_torch_device_string(backend: ComputeBackend, expected: str) -> None:
    assert _device(backend).torch_device == expected
    assert DeviceHint(backend=backend).torch_device == expected


def test_device_hint_from_device() -> None:
    hint = DeviceHint.from_device(_device(ComputeBackend.XPU), max_vram_mb=2048)
    assert hint.backend is ComputeBackend.XPU
    assert hint.max_vram_mb == 2048


# --- proof policy ---------------------------------------------------------- #

def test_unproven_backend_is_not_allowed() -> None:
    """Detection alone must never authorise a backend."""
    s = ComponentSupport(Component.SEPARATION)
    assert not s.allows(ComputeBackend.XPU)
    assert not s.allows(ComputeBackend.CUDA)
    assert s.allows(ComputeBackend.CPU)


def test_ops_proof_is_not_enough_for_production() -> None:
    """Operator compatibility is evidence, not permission to route real models."""
    s = ComponentSupport(Component.F0, {ComputeBackend.XPU: ProofLevel.OPS})
    assert not s.allows(ComputeBackend.XPU)


def test_end_to_end_proof_is_enough() -> None:
    s = ComponentSupport(Component.F0, {ComputeBackend.XPU: ProofLevel.END_TO_END})
    assert s.allows(ComputeBackend.XPU)


def test_explicit_none_is_rejected() -> None:
    s = ComponentSupport(Component.F0, {ComputeBackend.XPU: ProofLevel.NONE})
    assert not s.allows(ComputeBackend.XPU)


# --- per-component selection ----------------------------------------------- #

def test_components_may_use_different_backends() -> None:
    """Only the component with real end-to-end proof may use the accelerator."""
    matrix = SupportMatrix(components={
        Component.SEPARATION: ComponentSupport(
            Component.SEPARATION, {ComputeBackend.XPU: ProofLevel.OPS}
        ),
        Component.F0: ComponentSupport(
            Component.F0, {ComputeBackend.XPU: ProofLevel.END_TO_END}
        ),
        Component.CONVERSION: ComponentSupport(
            Component.CONVERSION, {ComputeBackend.XPU: ProofLevel.OPS}
        ),
        Component.PITCH_SHIFT: ComponentSupport(Component.PITCH_SHIFT),
    })
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU])

    assert matrix.device_for(Component.SEPARATION, m).backend is ComputeBackend.CPU
    assert matrix.device_for(Component.F0, m).backend is ComputeBackend.XPU
    assert matrix.device_for(Component.CONVERSION, m).backend is ComputeBackend.CPU
    assert matrix.device_for(Component.PITCH_SHIFT, m).backend is ComputeBackend.CPU


def test_proven_backend_is_ignored_when_hardware_absent() -> None:
    matrix = SupportMatrix(components={
        Component.F0: ComponentSupport(
            Component.F0, {ComputeBackend.CUDA: ProofLevel.END_TO_END}
        ),
    })
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU])
    assert matrix.device_for(Component.F0, m).backend is ComputeBackend.CPU


# --- matrix loading -------------------------------------------------------- #

def test_defaults_are_cpu_only() -> None:
    m = _FakeManager([ComputeBackend.CPU, ComputeBackend.XPU, ComputeBackend.CUDA])
    from svc_engine.compute import DEFAULTS

    for component in Component:
        assert DEFAULTS.device_for(component, m).backend is ComputeBackend.CPU


def test_load_matrix_missing_file_returns_defaults(tmp_path: Path) -> None:
    assert load_matrix(tmp_path / "nope.json").source == "defaults"


def test_load_matrix_corrupt_file_returns_defaults(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert load_matrix(bad).source == "defaults"


def test_load_matrix_reads_recorded_proof(tmp_path: Path) -> None:
    f = tmp_path / "support.json"
    f.write_text(json.dumps({
        "source": "spike test",
        "components": {
            "f0": {
                "proofs": {"xpu": "end_to_end"},
                "implementation_proofs": {"rmvpe": {"xpu": "end_to_end"}},
                "note_he": "אומת",
            },
            "separation": {"proofs": {"xpu": "ops"}, "note_he": ""},
        },
    }), encoding="utf-8")

    matrix = load_matrix(f)
    assert matrix.source == "spike test"
    assert matrix.get(Component.F0).allows(ComputeBackend.XPU)
    assert matrix.get(Component.F0).implementation_proofs["rmvpe"][ComputeBackend.XPU] \
        is ProofLevel.END_TO_END
    assert not matrix.get(Component.SEPARATION).allows(ComputeBackend.XPU)
    # components missing from the file keep conservative defaults
    assert not matrix.get(Component.CONVERSION).allows(ComputeBackend.XPU)


def test_load_matrix_ignores_unknown_names(tmp_path: Path) -> None:
    f = tmp_path / "support.json"
    f.write_text(json.dumps({
        "components": {
            "not_a_component": {"proofs": {"xpu": "ops"}},
            "f0": {"proofs": {"not_a_backend": "ops", "xpu": "not_a_level"}},
        },
    }), encoding="utf-8")
    matrix = load_matrix(f)
    assert not matrix.get(Component.F0).allows(ComputeBackend.XPU)


def test_roundtrip_to_dict_preserves_implementation_evidence() -> None:
    matrix = SupportMatrix(components={
        Component.F0: ComponentSupport(
            Component.F0,
            {ComputeBackend.XPU: ProofLevel.OPS},
            implementation_proofs={
                "torchfcpe": {ComputeBackend.XPU: ProofLevel.END_TO_END}
            },
        ),
    })
    payload = matrix.to_dict()
    assert payload["components"]["f0"]["proofs"]["xpu"] == "ops"
    assert payload["components"]["f0"]["implementation_proofs"]["torchfcpe"]["xpu"] \
        == "end_to_end"


# --- hardware detection ---------------------------------------------------- #

def test_display_adapter_detection_returns_a_list() -> None:
    assert isinstance(detect_display_adapters(), list)


def test_real_manager_always_offers_cpu() -> None:
    assert ComputeBackend.CPU in DeviceManager().available_backends()
