"""Hardware execution layer.

Three first-class backends: CUDA (NVIDIA), XPU (Intel Arc / Core Ultra), CPU.
Selection is per pipeline component, based on recorded proof of real execution.
"""

from svc_engine.compute.devices import (
    PREFERENCE,
    ComputeBackend,
    DeviceInfo,
    DeviceManager,
    detect_display_adapters,
    has_intel_level_zero,
    has_nvidia_driver,
    looks_like_intel_gpu,
)
from svc_engine.compute.oom import (
    MIN_SEGMENT_SIZE,
    OomStep,
    ResourcePlan,
    is_oom_error,
    oom_ladder,
    run_with_oom_ladder,
)
from svc_engine.compute.support import (
    DEFAULTS,
    Component,
    ComponentSupport,
    ProofLevel,
    SupportMatrix,
    load_matrix,
)

__all__ = [
    "DEFAULTS",
    "MIN_SEGMENT_SIZE",
    "PREFERENCE",
    "Component",
    "ComponentSupport",
    "ComputeBackend",
    "DeviceInfo",
    "DeviceManager",
    "OomStep",
    "ProofLevel",
    "ResourcePlan",
    "SupportMatrix",
    "detect_display_adapters",
    "has_intel_level_zero",
    "has_nvidia_driver",
    "is_oom_error",
    "load_matrix",
    "looks_like_intel_gpu",
    "oom_ladder",
    "run_with_oom_ladder",
]
