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

from svc_engine.backends.base import DeviceHint
from svc_engine.compute.devices import ComputeBackend

__all__ = ["available_memory_gb", "suggested_segment_size", "SEGMENT_BY_MEMORY"]

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
