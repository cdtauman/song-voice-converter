"""Per-component backend support matrix.

Rule: a component is treated as production-supported on an accelerator only when
the real implementation has been executed end to end on that backend and recorded.
Operator-set workloads are useful compatibility evidence, but they never authorize
production routing by themselves. Installing cleanly, importing successfully, or
the device merely being present proves nothing.

Anything not proven end to end falls back to CPU. That is deliberately conservative
-- a wrong "supported" costs a mysterious runtime failure much later.

The matrix ships with conservative defaults and is overwritten by the
Compatibility Spike (`spike/run_spike.py`) or `svc verify-backends`, which write
the evidence gathered on the actual machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from svc_engine.compute.devices import ComputeBackend, DeviceInfo, DeviceManager

__all__ = ["Component", "ProofLevel", "ComponentSupport", "SupportMatrix", "load_matrix"]

DATA_FILE = Path(__file__).resolve().parent.parent / "data" / "compute-support.json"


class Component(StrEnum):
    SEPARATION = "separation"
    F0 = "f0"
    CONVERSION = "conversion"
    PITCH_SHIFT = "pitch_shift"


class ProofLevel(StrEnum):
    #: Never executed on this backend. Treated as unsupported.
    NONE = "none"
    #: The operator set the component relies on ran on this backend.
    #: Compatibility evidence only; never enough for production routing.
    OPS = "ops"
    #: The real production implementation ran end to end on this backend.
    END_TO_END = "end_to_end"


#: Production routing is deliberately stricter than compatibility probing.
MIN_PROOF = ProofLevel.END_TO_END


_PROOF_ORDER = (ProofLevel.NONE, ProofLevel.OPS, ProofLevel.END_TO_END)


@dataclass(frozen=True)
class ComponentSupport:
    component: Component
    proofs: dict[ComputeBackend, ProofLevel] = field(default_factory=dict)
    note_he: str = ""
    implementation_proofs: dict[str, dict[ComputeBackend, ProofLevel]] = field(
        default_factory=dict
    )

    def proof(self, backend: ComputeBackend) -> ProofLevel:
        if backend is ComputeBackend.CPU:
            return ProofLevel.END_TO_END  # CPU is the baseline, always available
        return self.proofs.get(backend, ProofLevel.NONE)

    def implementation_proof(
        self, implementation: str, backend: ComputeBackend
    ) -> ProofLevel:
        """Proof for one exact implementation, never inherited from a sibling model."""
        if backend is ComputeBackend.CPU:
            return ProofLevel.END_TO_END
        per_backend = self.implementation_proofs.get(implementation, {})
        return per_backend.get(backend, ProofLevel.NONE)

    def allows(self, backend: ComputeBackend) -> bool:
        return _PROOF_ORDER.index(self.proof(backend)) >= _PROOF_ORDER.index(MIN_PROOF)

    def allows_implementation(self, implementation: str, backend: ComputeBackend) -> bool:
        return _PROOF_ORDER.index(
            self.implementation_proof(implementation, backend)
        ) >= _PROOF_ORDER.index(MIN_PROOF)

    def allowed_backends(self) -> set[ComputeBackend]:
        return {b for b in ComputeBackend if self.allows(b)}

    def allowed_backends_for_implementation(self, implementation: str) -> set[ComputeBackend]:
        return {
            b for b in ComputeBackend if self.allows_implementation(implementation, b)
        }


@dataclass(frozen=True)
class SupportMatrix:
    components: dict[Component, ComponentSupport]
    source: str = "defaults"

    def get(self, component: Component) -> ComponentSupport:
        return self.components.get(component, ComponentSupport(component))

    def device_for(
        self,
        component: Component,
        manager: DeviceManager,
    ) -> DeviceInfo:
        """Fastest backend this component is generically production-proven on."""
        return manager.select(self.get(component).allowed_backends())

    def device_for_implementation(
        self,
        component: Component,
        implementation: str,
        manager: DeviceManager,
    ) -> DeviceInfo:
        """Fastest backend proven for this exact model/implementation.

        A successful Mel-Band RoFormer run must not authorize an unrelated
        RoFormer checkpoint, cleanup model, RMVPE, or RVC model merely because
        their operator families overlap.
        """
        support = self.get(component)
        return manager.select(support.allowed_backends_for_implementation(implementation))

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "components": {
                c.value: {
                    "proofs": {b.value: p.value for b, p in s.proofs.items()},
                    "implementation_proofs": {
                        implementation: {
                            b.value: p.value for b, p in per_backend.items()
                        }
                        for implementation, per_backend in s.implementation_proofs.items()
                    },
                    "note_he": s.note_he,
                }
                for c, s in self.components.items()
            },
        }


#: Conservative starting point. `pitch_shift` is CPU by nature -- python-stretch
#: is a C++ DSP library with no GPU path at all, so it is not a fallback but the
#: correct place for that work.
DEFAULTS = SupportMatrix(
    components={
        Component.SEPARATION: ComponentSupport(
            Component.SEPARATION,
            note_he="טרם אומת מודל הפרדה מלא על מאיץ — מסלול הייצור נשאר על המעבד.",
        ),
        Component.F0: ComponentSupport(
            Component.F0,
            note_he="טרם אומת מנוע F0 ראשי מלא על מאיץ — מסלול הייצור נשאר על המעבד.",
        ),
        Component.CONVERSION: ComponentSupport(
            Component.CONVERSION,
            note_he="טרם אומת מודל RVC מלא על מאיץ — מסלול הייצור נשאר על המעבד.",
        ),
        Component.PITCH_SHIFT: ComponentSupport(
            Component.PITCH_SHIFT,
            note_he="רץ על המעבד מעצם טבעו — ספריית DSP ב-C++ בלי מסלול GPU.",
        ),
    },
)


def _parse_proofs(raw: object) -> dict[ComputeBackend, ProofLevel]:
    proofs: dict[ComputeBackend, ProofLevel] = {}
    if not isinstance(raw, dict):
        return proofs
    for backend_name, level_name in raw.items():
        try:
            proofs[ComputeBackend(str(backend_name))] = ProofLevel(str(level_name))
        except ValueError:
            continue
    return proofs


def load_matrix(path: Path | None = None) -> SupportMatrix:
    """Load the recorded matrix, or fall back to conservative defaults."""
    path = path or DATA_FILE
    if not path.exists():
        return DEFAULTS
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return DEFAULTS

    components: dict[Component, ComponentSupport] = {}
    for name, entry in (raw.get("components") or {}).items():
        try:
            component = Component(name)
        except ValueError:
            continue
        if not isinstance(entry, dict):
            continue
        implementation_proofs: dict[str, dict[ComputeBackend, ProofLevel]] = {}
        for implementation, per_backend in (entry.get("implementation_proofs") or {}).items():
            parsed = _parse_proofs(per_backend)
            if parsed:
                implementation_proofs[str(implementation)] = parsed
        components[component] = ComponentSupport(
            component=component,
            proofs=_parse_proofs(entry.get("proofs")),
            note_he=str(entry.get("note_he") or ""),
            implementation_proofs=implementation_proofs,
        )

    for component, fallback in DEFAULTS.components.items():
        components.setdefault(component, fallback)

    return SupportMatrix(components=components, source=str(raw.get("source") or str(path)))
