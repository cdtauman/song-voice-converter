"""How much memory the chosen device actually has, and what to start with.

The OOM ladder in `compute.oom` recovers *after* a failure. This module tries to
avoid the first failure: an 8GB integrated Arc and a 24GB RTX should not begin
at the same window size, and on CPU the limit is system RAM, not VRAM.

Being wrong here is cheap -- it costs one rung of the ladder. Being absent is
not: every first run on a small card would start by crashing.
"""

from __future__ import annotations

import ctypes
import logging
import sys
from dataclasses import dataclass

from svc_engine.backends.base import DeviceHint
from svc_engine.compute.devices import ComputeBackend

__all__ = [
    "PeakMemory",
    "SEGMENT_BY_MEMORY",
    "available_memory_gb",
    "measure_peak",
    "suggested_segment_size",
]

log = logging.getLogger(__name__)

#: (minimum free GB, segment size). Checked top down; the last row always wins.
SEGMENT_BY_MEMORY: tuple[tuple[float, int], ...] = (
    (16.0, 512),
    (10.0, 384),
    (6.0, 256),
    (4.0, 128),
    (0.0, 64),
)


def _host_available_gb() -> float | None:
    if sys.platform == "win32":
        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        st = _MemStatus()
        st.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
            return st.ullAvailPhys / (1024 ** 3)
        return None
    try:
        import os

        return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 ** 3)
    except (ValueError, AttributeError, OSError):
        return None


def available_memory_gb(device: DeviceHint) -> float | None:
    """Free memory on `device`, or None when it cannot be measured."""
    if device.backend is ComputeBackend.CPU:
        return _host_available_gb()

    try:
        import torch
    except ImportError:
        return None

    try:
        if device.backend is ComputeBackend.CUDA and torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info(device.device_index)
            return free / (1024 ** 3)

        xpu = getattr(torch, "xpu", None)
        if device.backend is ComputeBackend.XPU and xpu is not None and xpu.is_available():
            # Integrated Arc shares system RAM, and torch exposes only the
            # total. Reserve the part already in use rather than assuming free.
            props = xpu.get_device_properties(device.device_index)
            total = props.total_memory / (1024 ** 3)
            reserved = xpu.memory_reserved(device.device_index) / (1024 ** 3)
            return max(0.0, total - reserved)
    except Exception as exc:  # noqa: BLE001  measurement must never be fatal
        log.debug("could not read device memory: %s", exc)
    return None


@dataclass(frozen=True)
class PeakMemory:
    """High-water marks for one run. `None` where the platform cannot report it."""

    device_mb: float | None = None
    host_mb: float | None = None
    backend: str = "cpu"

    def summary(self) -> str:
        parts = []
        if self.device_mb is not None:
            parts.append(f"{self.backend} {self.device_mb:.0f}MB")
        if self.host_mb is not None:
            parts.append(f"RAM {self.host_mb:.0f}MB")
        return " · ".join(parts) or "not measurable"


def _host_peak_mb() -> float | None:
    """Peak working set of this process."""
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        handle = kernel32.GetCurrentProcess()

        # argtypes are required, not optional: the pseudo-handle is -1, and
        # without a declared pointer type ctypes passes it as a 32-bit int,
        # which the 64-bit API rejects.
        try:
            get_info = ctypes.WinDLL("psapi").GetProcessMemoryInfo
        except (OSError, AttributeError):  # pragma: no cover - very old Windows
            return None
        get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong]
        get_info.restype = ctypes.c_int

        if get_info(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize / (1024 ** 2)
        return None
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except (ImportError, OSError):  # pragma: no cover - non-POSIX
        return None


def reset_peak(device: DeviceHint) -> None:
    """Zero the accelerator's high-water mark so the next run measures itself."""
    try:
        import torch

        if device.backend is ComputeBackend.CUDA and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device.device_index)
        xpu = getattr(torch, "xpu", None)
        if device.backend is ComputeBackend.XPU and xpu is not None and xpu.is_available():
            xpu.reset_peak_memory_stats(device.device_index)
    except Exception as exc:  # noqa: BLE001  instrumentation is never fatal
        log.debug("could not reset peak memory stats: %s", exc)


def measure_peak(device: DeviceHint) -> PeakMemory:
    """Read the high-water marks accumulated since the last `reset_peak`."""
    device_mb: float | None = None
    try:
        import torch

        if device.backend is ComputeBackend.CUDA and torch.cuda.is_available():
            device_mb = torch.cuda.max_memory_reserved(device.device_index) / (1024 ** 2)
        xpu = getattr(torch, "xpu", None)
        if device.backend is ComputeBackend.XPU and xpu is not None and xpu.is_available():
            device_mb = xpu.max_memory_reserved(device.device_index) / (1024 ** 2)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not read peak memory: %s", exc)

    return PeakMemory(
        device_mb=device_mb, host_mb=_host_peak_mb(), backend=device.backend.value
    )


def suggested_segment_size(device: DeviceHint, default: int = 256) -> int:
    """Starting window size for `device`, capped by an explicit VRAM budget.

    Only accelerators are sized down. On CPU the allocator is backed by system
    RAM and the page file, so a momentarily low "available physical memory"
    reading is not a reason to shrink the window -- and doing so silently
    changed the result depending on what else the machine happened to be
    running, which made two runs of the same job incomparable.
    """
    if device.backend is ComputeBackend.CPU:
        return default

    free = available_memory_gb(device)
    if free is None:
        return default

    budget = free
    if device.max_vram_mb:
        budget = min(budget, device.max_vram_mb / 1024.0)

    for threshold, segment in SEGMENT_BY_MEMORY:
        if budget >= threshold:
            return min(default, segment)
    return default  # pragma: no cover - the table's last row is 0.0
