"""System checks -- the engine side of `svc doctor`.

Every check is independent, never raises, and returns a Hebrew message.
Nothing here imports torch at module level: torch is optional in Phase 1.
"""

from __future__ import annotations

import contextlib
import ctypes
import importlib.util
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

__all__ = ["Status", "CheckResult", "run_all_checks", "REQUIRED_FFMPEG_FILTERS"]

#: ffmpeg filters the mixing stage depends on (decision: ffmpeg instead of custom DSP).
REQUIRED_FFMPEG_FILTERS = ("loudnorm", "alimiter", "acompressor", "aresample")

#: Python packages the AI pipeline needs, probed by import name.
PIPELINE_PACKAGES: tuple[tuple[str, str, bool], ...] = (
    # (import name, Hebrew label, required for MVP)
    ("torch", "מנוע ה-AI", True),
    ("soundfile", "קריאת קבצי שמע", True),
    ("librosa", "ניתוח שמע", True),
    ("pyloudnorm", "מדידת עוצמה", True),
    ("audio_separator", "הפרדת שירה ממוזיקה", True),
    ("python_stretch", "הזזת גובה", True),
    ("torchfcpe", "זיהוי גובה מהיר", False),
)

_MIN_FREE_GB = 20.0
_MIN_RAM_GB = 12.0


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class CheckResult:
    key: str
    label_he: str
    status: Status
    message_he: str
    detail: str = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _total_ram_gb() -> float | None:
    """Total physical RAM in GB, Windows-first with a POSIX fallback."""
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
            return st.ullTotalPhys / (1024 ** 3)
        return None
    try:
        return (os_sysconf := __import__("os")).sysconf("SC_PAGE_SIZE") * os_sysconf.sysconf(
            "SC_PHYS_PAGES"
        ) / (1024 ** 3)
    except (ValueError, AttributeError, OSError):
        return None


def _run(cmd: list[str], timeout: int = 20) -> tuple[int, str]:
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.SubprocessError):
        return -1, ""


def _has_nvidia_driver() -> bool:
    """True if the NVIDIA CUDA driver library is present on this machine."""
    if sys.platform != "win32":
        return Path("/proc/driver/nvidia/version").exists()
    try:
        ctypes.WinDLL("nvcuda.dll")
        return True
    except OSError:
        return False


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #

def check_python() -> CheckResult:
    """3.11 only -- verified by the Phase 1 spike, see docs/compat-matrix.md."""
    v = sys.version_info
    txt = f"{v.major}.{v.minor}.{v.micro}"
    if (v.major, v.minor) == (3, 11):
        return CheckResult("python", "גרסת Python", Status.OK, f"Python {txt}")
    return CheckResult(
        "python", "גרסת Python", Status.FAIL,
        f"Python {txt} — נדרשת גרסה 3.11.",
        detail=sys.executable,
    )


def _windows_name() -> str:
    """platform.release() still reports '10' on Windows 11; use the build number."""
    build = 0
    with contextlib.suppress(ValueError, IndexError):
        build = int(platform.version().split(".")[-1])
    major = "11" if build >= 22000 else platform.release()
    return f"Windows {major}" + (f" (build {build})" if build else "")


def check_os() -> CheckResult:
    label = "מערכת ההפעלה"
    if sys.platform == "win32":
        return CheckResult("os", label, Status.OK, _windows_name())
    return CheckResult(
        "os", label, Status.WARN,
        f"{platform.system()} — התוכנה מיועדת ל-Windows. ההרצה כאן לפיתוח בלבד.",
    )


def check_ram() -> CheckResult:
    gb = _total_ram_gb()
    label = "זיכרון המחשב"
    if gb is None:
        return CheckResult("ram", label, Status.WARN, "לא הצלחנו למדוד את הזיכרון.")
    txt = f"{gb:.0f}GB"
    if gb >= 24:
        return CheckResult("ram", label, Status.OK, txt)
    if gb >= _MIN_RAM_GB:
        return CheckResult(
            "ram", label, Status.WARN,
            f"{txt} — יעבוד, אבל שירים ארוכים ואימון קול עלולים להיות איטיים.",
        )
    return CheckResult(
        "ram", label, Status.FAIL,
        f"{txt} — מעט מדי. נדרשים לפחות {_MIN_RAM_GB:.0f}GB.",
    )


def check_disk(target: Path) -> CheckResult:
    label = "מקום בדיסק"
    probe = target
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free_gb = shutil.disk_usage(probe).free / (1024 ** 3)
    except OSError as exc:
        return CheckResult("disk", label, Status.WARN, "לא הצלחנו לבדוק את הדיסק.", str(exc))
    txt = f"{free_gb:.0f}GB פנויים"
    if free_gb >= _MIN_FREE_GB:
        return CheckResult("disk", label, Status.OK, txt)
    return CheckResult(
        "disk", label, Status.FAIL,
        f"{txt} — צריך לפחות {_MIN_FREE_GB:.0f}GB פנויים.",
    )


def check_gpu() -> CheckResult:
    label = "כרטיס מסך"

    if importlib.util.find_spec("torch") is not None:
        try:
            import torch  # noqa: PLC0415  (optional dependency, probed on purpose)

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                free = torch.cuda.mem_get_info(0)[0] / (1024 ** 3)
                return CheckResult(
                    "gpu", label, Status.OK,
                    f"{name} ({total:.0f}GB, {free:.1f}GB פנויים)",
                    detail=f"torch {torch.__version__} cuda {torch.version.cuda}",
                )
        except Exception as exc:  # noqa: BLE001  a broken torch must not kill doctor
            return CheckResult(
                "gpu", label, Status.WARN,
                "לא הצלחנו לתשאל את כרטיס המסך.", detail=str(exc),
            )

    if _has_nvidia_driver():
        return CheckResult(
            "gpu", label, Status.WARN,
            "נמצא כרטיס NVIDIA, אבל רכיבי ה-AI עדיין לא הותקנו.",
        )

    return CheckResult(
        "gpu", label, Status.FAIL,
        "לא נמצא כרטיס מסך של NVIDIA. אפשר לעבוד על המעבד, אבל זה יהיה איטי מאוד.",
    )


def check_torch_cuda_match() -> CheckResult:
    label = "התאמת AI לכרטיס המסך"
    if importlib.util.find_spec("torch") is None:
        return CheckResult("torch", label, Status.FAIL, "מנוע ה-AI עדיין לא הותקן.")
    try:
        import torch  # noqa: PLC0415

        built = torch.version.cuda
        if built is None:
            return CheckResult(
                "torch", label, Status.WARN,
                f"מנוע ה-AI מותקן בגרסת מעבד בלבד (torch {torch.__version__}).",
            )
        if torch.cuda.is_available():
            return CheckResult(
                "torch", label, Status.OK,
                f"תואמים (torch {torch.__version__}, CUDA {built})",
            )
        return CheckResult(
            "torch", label, Status.WARN,
            f"מנוע ה-AI נבנה ל-CUDA {built} אבל לא מצליח לגשת לכרטיס המסך.",
        )
    except Exception as exc:  # noqa: BLE001
        return CheckResult("torch", label, Status.FAIL, "מנוע ה-AI לא נטען.", str(exc))


def check_ffmpeg() -> CheckResult:
    label = "עיבוד שמע (ffmpeg)"
    exe = shutil.which("ffmpeg")
    if not exe:
        return CheckResult("ffmpeg", label, Status.FAIL, "ffmpeg לא נמצא.")

    rc, out = _run([exe, "-version"])
    if rc != 0:
        return CheckResult("ffmpeg", label, Status.FAIL, "ffmpeg נמצא אבל לא רץ.", exe)
    version = out.splitlines()[0] if out else "ffmpeg"

    rc, filters = _run([exe, "-hide_banner", "-filters"], timeout=30)
    missing = [f for f in REQUIRED_FFMPEG_FILTERS if f not in filters]
    if missing:
        return CheckResult(
            "ffmpeg", label, Status.FAIL,
            "לגרסת ffmpeg המותקנת חסרים רכיבי מיקס: " + ", ".join(missing),
            detail=exe,
        )
    return CheckResult(
        "ffmpeg", label, Status.OK,
        "זמין, כולל רכיבי המיקס הנדרשים",
        detail=f"{version} | {exe}",
    )


def check_pipeline_packages() -> list[CheckResult]:
    results: list[CheckResult] = []
    for mod, label, required in PIPELINE_PACKAGES:
        present = importlib.util.find_spec(mod) is not None
        if present:
            results.append(CheckResult(f"pkg.{mod}", label, Status.OK, "מותקן"))
        else:
            results.append(
                CheckResult(
                    f"pkg.{mod}", label,
                    Status.FAIL if required else Status.WARN,
                    "לא מותקן",
                )
            )
    return results


def check_backend_interfaces() -> CheckResult:
    """The four adapters from the approved architecture must be importable."""
    label = "ממשקי המנוע"
    try:
        from svc_engine import backends  # noqa: PLC0415

        missing = [
            n for n in backends.REQUIRED_PROTOCOLS if not hasattr(backends, n)
        ]
    except Exception as exc:  # noqa: BLE001
        return CheckResult("backends", label, Status.FAIL, "לא נטענו.", str(exc))
    if missing:
        return CheckResult(
            "backends", label, Status.FAIL, "חסרים: " + ", ".join(missing)
        )
    return CheckResult(
        "backends", label, Status.OK,
        f"{len(backends.REQUIRED_PROTOCOLS)} ממשקים מוגדרים",
    )


def run_all_checks(work_dir: Path) -> list[CheckResult]:
    results = [
        check_os(),
        check_python(),
        check_ram(),
        check_disk(work_dir),
        check_gpu(),
        check_torch_cuda_match(),
        check_ffmpeg(),
        check_backend_interfaces(),
    ]
    results.extend(check_pipeline_packages())
    return results
