"""Configure paths to dependencies shipped beside the Windows executable."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def configure_bundled_runtime() -> Path:
    root = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path.cwd()
    bundle = Path(getattr(sys, "_MEIPASS", root))
    candidates = (
        bundle / "runtime" / "ffmpeg" / "bin",
        root / "runtime" / "ffmpeg" / "bin",
        root / "ffmpeg" / "bin",
    )
    for candidate in candidates:
        if (candidate / "ffmpeg.exe").is_file():
            current = os.environ.get("PATH", "")
            os.environ["PATH"] = str(candidate) + (os.pathsep + current if current else "")
            break
    return root
