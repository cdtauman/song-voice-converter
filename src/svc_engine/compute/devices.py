"""Compute backend detection and selection.

SongVoice targets three execution backends as first-class citizens:

    cuda  -- NVIDIA GPUs
    xpu   -- Intel Arc / Core Ultra integrated Arc GPUs, via PyTorch XPU
    cpu   -- universal fallback

Preference order is CUDA, then XPU, then CPU. CPU is chosen only when no
accelerator is present, or when a specific pipeline component has not been
proven on the available accelerator (see svc_engine.compute.support).

Nothing here imports torch at module level -- detection must work before the AI
stack is installed.
"""

from __future__ import annotations

import ctypes
import importlib.util
import sys
from dataclasses import dataclass
from enum import StrEnum

__all__ = [
    "ComputeBackend",
    "DeviceInfo",
    "DeviceManager",
    "detect_display_adapters",
]


class ComputeBackend(StrEnum):
    CUDA = "cuda"
    XPU = "xpu"
    CPU = "cpu"


#: Highest preference first. CPU always exists and always comes last.
PREFERENCE: tuple[ComputeBackend, ...] = (
    ComputeBackend.CUDA,
    ComputeBackend.XPU,
    ComputeBackend.CPU,
)

_BACKEND_LABEL_HE: dict[ComputeBackend, str] = {
    ComputeBackend.CUDA: "כרטיס NVIDIA",
    ComputeBackend.XPU: "כרטיס Intel",
    ComputeBackend.CPU: "מעבד",
}


@dataclass(frozen=True)
class DeviceInfo:
    backend: ComputeBackend
    index: int
    name: str
    available: bool
    total_memory_gb: float | None = None
    detail: str = ""

    @property
    def torch_device(self) -> str:
        if self.backend is ComputeBackend.CPU:
            return "cpu"
        return f"{self.backend.value}:{self.index}"

    @property
    def label_he(self) -> str:
        return _BACKEND_LABEL_HE[self.backend]


# --------------------------------------------------------------------------- #
# hardware detection that does not need torch
# --------------------------------------------------------------------------- #

def detect_display_adapters() -> list[str]:
    """Names of installed display adapters, read from the Windows registry.

    Dependency-free on purpose: `svc doctor` must be able to explain the machine
    before any AI package is installed.
    """
    if sys.platform != "win32":
        return []

    import winreg  # noqa: PLC0415  (Windows-only import, guarded above)

    key_path = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    names: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path) as root:
            index = 0
            while True:
                try:
                    sub = winreg.EnumKey(root, index)
                except OSError:
                    break
                index += 1
                if not sub.isdigit():
                    continue
                try:
                    with winreg.OpenKey(root, sub) as adapter:
                        desc, _ = winreg.QueryValueEx(adapter, "DriverDesc")
                except OSError:
                    continue
                if isinstance(desc, str) and desc not in names:
                    names.append(desc)
    except OSError:
        return []
    return names


def has_nvidia_driver() -> bool:
    if sys.platform != "win32":
        from pathlib import Path  # noqa: PLC0415

        return Path("/proc/driver/nvidia/version").exists()
    try:
        ctypes.WinDLL("nvcuda.dll")
        return True
    except OSError:
        return False


def has_intel_level_zero() -> bool:
    """Level Zero is the runtime PyTorch XPU talks to on Intel GPUs."""
    lib = "ze_loader.dll" if sys.platform == "win32" else "libze_loader.so.1"
    try:
        ctypes.CDLL(lib) if sys.platform != "win32" else ctypes.WinDLL(lib)
        return True
    except OSError:
        return False


def looks_like_intel_gpu(adapters: list[str] | None = None) -> bool:
    names = adapters if adapters is not None else detect_display_adapters()
    return any("intel" in n.lower() for n in names)


# --------------------------------------------------------------------------- #
# torch-backed detection
# --------------------------------------------------------------------------- #

class DeviceManager:
    """Detects usable compute devices and picks one.

    Detection is cached per instance; call `refresh()` after installing torch.
    """

    def __init__(self) -> None:
        self._devices: list[DeviceInfo] | None = None

    def refresh(self) -> None:
        self._devices = None

    # -- detection --------------------------------------------------------- #

    def detect(self) -> list[DeviceInfo]:
        if self._devices is None:
            self._devices = self._detect_uncached()
        return self._devices

    def _detect_uncached(self) -> list[DeviceInfo]:
        devices: list[DeviceInfo] = []

        if importlib.util.find_spec("torch") is not None:
            devices.extend(self._detect_cuda())
            devices.extend(self._detect_xpu())

        devices.append(
            DeviceInfo(
                backend=ComputeBackend.CPU,
                index=0,
                name="CPU",
                available=True,
                detail=f"{sys.platform}",
            )
        )
        return devices

    @staticmethod
    def _detect_cuda() -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        try:
            import torch  # noqa: PLC0415

            if not torch.cuda.is_available():
                return out
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                out.append(
                    DeviceInfo(
                        backend=ComputeBackend.CUDA,
                        index=i,
                        name=torch.cuda.get_device_name(i),
                        available=True,
                        total_memory_gb=round(props.total_memory / 1024**3, 1),
                        detail=f"compute capability {props.major}.{props.minor}",
                    )
                )
        except Exception:  # noqa: BLE001  a broken install must not crash detection
            return out
        return out

    @staticmethod
    def _detect_xpu() -> list[DeviceInfo]:
        out: list[DeviceInfo] = []
        try:
            import torch  # noqa: PLC0415

            xpu = getattr(torch, "xpu", None)
            if xpu is None or not xpu.is_available():
                return out
            for i in range(xpu.device_count()):
                name = xpu.get_device_name(i)
                total: float | None = None
                detail = ""
                try:
                    props = xpu.get_device_properties(i)
                    total = round(props.total_memory / 1024**3, 1)
                    detail = getattr(props, "driver_version", "") or ""
                except Exception:  # noqa: BLE001  properties are best-effort
                    pass
                out.append(
                    DeviceInfo(
                        backend=ComputeBackend.XPU,
                        index=i,
                        name=name,
                        available=True,
                        total_memory_gb=total,
                        detail=detail,
                    )
                )
        except Exception:  # noqa: BLE001
            return out
        return out

    # -- selection --------------------------------------------------------- #

    def available_backends(self) -> list[ComputeBackend]:
        seen = {d.backend for d in self.detect() if d.available}
        return [b for b in PREFERENCE if b in seen]

    def has(self, backend: ComputeBackend) -> bool:
        return backend in self.available_backends()

    def preferred(self) -> DeviceInfo:
        """Best available device, ignoring per-component limitations."""
        for backend in PREFERENCE:
            for device in self.detect():
                if device.backend is backend and device.available:
                    return device
        raise RuntimeError("no compute device at all, not even CPU")  # pragma: no cover

    def select(self, allowed: set[ComputeBackend]) -> DeviceInfo:
        """Best device restricted to `allowed`. Falls back to CPU."""
        for backend in PREFERENCE:
            if backend not in allowed:
                continue
            for device in self.detect():
                if device.backend is backend and device.available:
                    return device
        return next(d for d in self.detect() if d.backend is ComputeBackend.CPU)
