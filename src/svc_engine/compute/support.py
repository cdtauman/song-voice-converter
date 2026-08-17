"""Per-component backend support matrix.

Rule: a component is treated as supported on a backend only when a real workload
has been executed on that backend and recorded. Installing cleanly, importing
successfully, or the device merely being present proves nothing.

Anything unproven falls back to CPU. That is deliberately conservative -- a wrong
"supported" costs a mysterious runtime failure much later.

The matrix ships with conservative defaults and is overwritten by the
Compatibility Spike (`spike/run_spike.py`), which writes the evidence it gathered
on the actual machine.
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
    OPS = "ops"
    #: The real model ran end to end on this backend.
    END_TO_END = "end_to_end"


#: Minimum proof required before a backend is used in production.
MIN_PROOF = ProofLevel.OPS


@dataclass(frozen=True)
class ComponentSupport:
    component: Component
    proofs: dict[ComputeBackend, ProofLevel] = field(default_factory=dict)
    note_he: str = ""

    def proof(self, backend: ComputeBackend) -> ProofLevel:
        if backend is ComputeBackend.CPU:
            return ProofLevel.END_TO_END  # CPU is the baseline, always available
        return self.proofs.get(backend, ProofLevel.NONE)

    def allows(self, backend: ComputeBackend) -> bool:
        order = (ProofLevel.NONE, ProofLevel.OPS, ProofLevel.END_TO_END)
        return order.index(self.proof(backend)) >= order.index(MIN_PROOF)

    def allowed_backends(self) -> set[ComputeBackend]:
        return {b for b in ComputeBackend if self.allows(b)}


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
        """Fastest backend this component is actually proven on."""
        return manager.select(self.get(component).allowed_backends())

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "components": {
                c.value: {
                    "proofs": {b.value: p.value for b, p in s.proofs.items()},
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
            note_he="טרם אומת על מאיץ חומרה — רץ על המעבד עד שיוכח אחרת.",
        ),
        Component.F0: ComponentSupport(
            Component.F0,
            note_he="טרם אומת על מאיץ חומרה — רץ על המעבד עד שיוכח אחרת.",
        ),
        Component.CONVERSION: ComponentSupport(
            Component.CONVERSION,
            note_he="טרם אומת על מאיץ חומרה — רץ על המעבד עד שיוכח אחרת.",
        ),
        Component.PITCH_SHIFT: ComponentSupport(
            Component.PITCH_SHIFT,
            note_he="רץ על המעבד מעצם טבעו — ספריית DSP ב-C++ בלי מסלול GPU.",
        ),
    },
)


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
        proofs: dict[ComputeBackend, ProofLevel] = {}
        for backend_name, level_name in (entry.get("proofs") or {}).items():
            try:
                proofs[ComputeBackend(backend_name)] = ProofLevel(level_name)
            except ValueError:
                continue
        components[component] = ComponentSupport(
            component=component,
            proofs=proofs,
            note_he=str(entry.get("note_he") or ""),
        )

    for component, fallback in DEFAULTS.components.items():
        components.setdefault(component, fallback)

    return SupportMatrix(components=components, source=str(raw.get("source") or str(path)))
