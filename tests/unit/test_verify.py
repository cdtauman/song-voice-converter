"""Backend proof: real workloads, and how their results become the support matrix."""

from __future__ import annotations

import importlib.util

import pytest

from svc_engine.compute import Component, ComputeBackend, load_matrix
from svc_engine.compute.verify import (
    WORKLOAD_COMPONENTS,
    WORKLOADS,
    available_devices,
    build_support_payload,
    verify_all,
    verify_component,
)

HAS_TORCH = importlib.util.find_spec("torch") is not None
needs_torch = pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")


# --- shape of the API ------------------------------------------------------ #

def test_pitch_shift_has_no_gpu_workload() -> None:
    """CPU is where that work belongs, so there is nothing to prove on a GPU."""
    assert "pitch_shift" not in WORKLOADS
    assert "pitch_shift" not in WORKLOAD_COMPONENTS


def test_every_workload_component_has_a_callable() -> None:
    for component in WORKLOAD_COMPONENTS:
        assert callable(WORKLOADS[component])


def test_cpu_is_always_available() -> None:
    assert available_devices()[0] == "cpu"


def test_unknown_component_fails_cleanly() -> None:
    result = verify_component("not_a_component", "cpu")
    assert not result.ok
    assert "no workload" in result.error


def test_failure_on_a_bogus_device_is_a_result_not_a_crash() -> None:
    result = verify_component("conversion", "definitely_not_a_device")
    assert not result.ok
    assert result.error


# --- turning results into the matrix --------------------------------------- #

def _entry(
    ok: bool,
    end_to_end: bool = False,
    implementation: str = "probe",
    production_eligible: bool = False,
) -> dict:
    return {
        "ok": ok,
        "seconds": 0.1,
        "ops": "x",
        "end_to_end": end_to_end,
        "implementation": implementation,
        "production_eligible": production_eligible,
        "error": "",
    }


def test_cpu_success_is_never_recorded_as_a_proof() -> None:
    """CPU is the baseline. Recording it would make every component look proven."""
    payload = build_support_payload({"f0": {"cpu": _entry(True)}}, source="test")
    assert payload["components"]["f0"]["proofs"] == {}
    assert payload["components"]["f0"]["implementation_proofs"] == {}


def test_successful_accelerator_operator_run_is_recorded_as_ops() -> None:
    payload = build_support_payload(
        {"f0": {"cpu": _entry(True), "xpu": _entry(True)}}, source="test"
    )
    assert payload["components"]["f0"]["proofs"] == {"xpu": "ops"}


def test_production_eligible_end_to_end_can_authorize_generic_component() -> None:
    payload = build_support_payload(
        {
            "f0": {
                "xpu": _entry(
                    True,
                    end_to_end=True,
                    implementation="rmvpe",
                    production_eligible=True,
                )
            }
        },
        source="test",
    )
    assert payload["components"]["f0"]["proofs"]["xpu"] == "end_to_end"
    assert payload["components"]["f0"]["implementation_proofs"]["rmvpe"]["xpu"] \
        == "end_to_end"


def test_torchfcpe_end_to_end_does_not_authorize_rmvpe_route() -> None:
    """A real FCPE run is not proof that SongVoice's primary RMVPE path works."""
    payload = build_support_payload(
        {
            "f0": {
                "xpu": _entry(
                    True,
                    end_to_end=True,
                    implementation="torchfcpe",
                    production_eligible=False,
                )
            }
        },
        source="test",
    )
    component = payload["components"]["f0"]
    assert component["proofs"]["xpu"] == "ops"
    assert component["implementation_proofs"]["torchfcpe"]["xpu"] == "end_to_end"
    assert "RMVPE" in component["note_he"]


def test_failed_accelerator_run_is_not_recorded() -> None:
    payload = build_support_payload(
        {"separation": {"cpu": _entry(True), "xpu": _entry(False)}}, source="test"
    )
    assert payload["components"]["separation"]["proofs"] == {}


def test_missing_components_are_filled_in_as_unverified() -> None:
    payload = build_support_payload({}, source="test")
    for component in (*WORKLOAD_COMPONENTS, "pitch_shift"):
        assert component in payload["components"]
        assert payload["components"][component]["proofs"] == {}
        assert payload["components"][component]["implementation_proofs"] == {}


def test_notes_are_hebrew() -> None:
    import re

    payload = build_support_payload({"f0": {"xpu": _entry(True)}}, source="test")
    for data in payload["components"].values():
        assert re.search(r"[֐-׿]", data["note_he"])


def test_payload_feeds_straight_into_the_support_matrix(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Only production-eligible end-to-end evidence may authorize the runtime route."""
    import json

    payload = build_support_payload(
        {
            "f0": {
                "xpu": _entry(
                    True,
                    end_to_end=True,
                    implementation="rmvpe",
                    production_eligible=True,
                )
            },
            "separation": {"xpu": _entry(True, implementation="roformer-operator-set")},
        },
        source="test",
    )
    f = tmp_path / "support.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    matrix = load_matrix(f)
    assert matrix.get(Component.F0).allows(ComputeBackend.XPU)
    assert not matrix.get(Component.SEPARATION).allows(ComputeBackend.XPU)
    assert not matrix.get(Component.PITCH_SHIFT).allows(ComputeBackend.XPU)


def test_implementation_evidence_survives_loading_without_authorizing_route(tmp_path) -> None:  # type: ignore[no-untyped-def]
    import json

    payload = build_support_payload(
        {
            "f0": {
                "xpu": _entry(
                    True,
                    end_to_end=True,
                    implementation="torchfcpe",
                    production_eligible=False,
                )
            }
        },
        source="test",
    )
    f = tmp_path / "support.json"
    f.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    matrix = load_matrix(f)

    assert not matrix.get(Component.F0).allows(ComputeBackend.XPU)
    assert matrix.get(Component.F0).implementation_proofs["torchfcpe"][ComputeBackend.XPU].value \
        == "end_to_end"


# --- the workloads actually run -------------------------------------------- #

@needs_torch
@pytest.mark.parametrize("component", WORKLOAD_COMPONENTS)
def test_workload_runs_on_cpu(component: str) -> None:
    """CPU must always work; if it does not, the workload itself is broken."""
    result = verify_component(component, "cpu")
    assert result.ok, result.error
    assert result.ops
    assert result.implementation


@needs_torch
def test_separation_workload_preserves_length() -> None:
    result = verify_component("separation", "cpu")
    assert result.ok
    assert result.detail is not None
    assert result.detail["output_frames"] == 44100 * 2
    assert not result.production_eligible


@needs_torch
def test_verify_all_covers_every_component_on_every_device() -> None:
    matrix = verify_all(["cpu"])
    for component in (*WORKLOAD_COMPONENTS, "pitch_shift"):
        assert component in matrix
    for component in WORKLOAD_COMPONENTS:
        assert matrix[component]["cpu"]["ok"], matrix[component]["cpu"].get("error")
