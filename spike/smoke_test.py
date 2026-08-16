"""Runs INSIDE a candidate virtual environment and reports what actually works.

Prints a single JSON object to stdout. Never raises: every probe reports its own
failure so one broken component does not hide the others.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

RESULT: dict[str, Any] = {}


def probe(name: str):  # type: ignore[no-untyped-def]
    def wrap(fn):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        try:
            data = fn()
            RESULT[name] = {"ok": True, "seconds": round(time.perf_counter() - started, 2), **data}
        except Exception as exc:  # noqa: BLE001  probes must never abort the run
            RESULT[name] = {
                "ok": False,
                "seconds": round(time.perf_counter() - started, 2),
                "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc(limit=3),
            }
        return fn

    return wrap


def _first_import(*names: str):  # type: ignore[no-untyped-def]
    errors = []
    for n in names:
        try:
            return n, importlib.import_module(n)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{n}: {type(exc).__name__}: {exc}")
    raise ImportError("; ".join(errors))


# --------------------------------------------------------------------------- #

@probe("python")
def _python() -> dict[str, Any]:
    return {"version": sys.version.split()[0], "executable": sys.executable}


@probe("torch")
def _torch() -> dict[str, Any]:
    import torch

    info: dict[str, Any] = {
        "version": torch.__version__,
        "built_cuda": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        info["device_name"] = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        info["vram_gb"] = round(props.total_memory / 1024**3, 1)
        info["compute_capability"] = f"{props.major}.{props.minor}"

    # A real op, not just an import: catches broken CUDA installs.
    t0 = time.perf_counter()
    a = torch.randn(512, 512, device=device)
    b = torch.randn(512, 512, device=device)
    c = a @ b
    if device == "cuda":
        torch.cuda.synchronize()
    info["matmul_device"] = device
    info["matmul_ok"] = bool(c.shape == (512, 512))
    info["matmul_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return info


@probe("numpy")
def _numpy() -> dict[str, Any]:
    import numpy as np

    return {"version": np.__version__}


@probe("soundfile")
def _soundfile() -> dict[str, Any]:
    import tempfile
    from pathlib import Path

    import numpy as np
    import soundfile as sf

    sr = 44100
    tone = (0.2 * np.sin(2 * np.pi * 220 * np.arange(sr) / sr)).astype("float32")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "probe.wav"
        sf.write(p, tone, sr)
        back, sr_back = sf.read(p, dtype="float32")
    return {
        "version": sf.__version__,
        "roundtrip_ok": bool(sr_back == sr and back.shape[0] == sr),
    }


@probe("librosa")
def _librosa() -> dict[str, Any]:
    import librosa
    import numpy as np

    sr = 22050
    y = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype("float32")
    f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)
    return {"version": librosa.__version__, "yin_median_hz": round(float(np.median(f0)), 1)}


@probe("pyloudnorm")
def _pyloudnorm() -> dict[str, Any]:
    import numpy as np
    import pyloudnorm as pyln

    sr = 44100
    y = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(sr * 3) / sr)).astype("float64")
    lufs = pyln.Meter(sr).integrated_loudness(y)
    return {"version": getattr(pyln, "__version__", "?"), "lufs": round(float(lufs), 2)}


@probe("pitch_shift")
def _pitch_shift() -> dict[str, Any]:
    """python-stretch (MIT) -- the approved default pitch shifter.

    The hard contract is length preservation, so that is what we verify.
    """
    import numpy as np

    modname, mod = _first_import("python_stretch", "stretch", "pystretch")

    sr = 44100
    n = sr  # one second
    audio = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype("float32")
    audio = audio.reshape(1, -1)  # (channels, samples)

    stretch_cls = getattr(mod, "Signalsmith", None)
    stretch = stretch_cls.Stretch() if stretch_cls is not None else mod.Stretch()
    stretch.preset(1, sr)
    stretch.setTransposeSemitones(-12.0)
    out = stretch.process(audio)

    out_n = int(np.asarray(out).shape[-1])
    return {
        "module": modname,
        "input_frames": n,
        "output_frames": out_n,
        "length_preserved": bool(out_n == n),
        "finite": bool(np.isfinite(np.asarray(out)).all()),
    }


@probe("audio_separator")
def _audio_separator() -> dict[str, Any]:
    """Import + API surface only. Downloading models is Phase 2 work."""
    from audio_separator.separator import Separator

    return {
        "importable": True,
        "has_load_model": hasattr(Separator, "load_model"),
        "has_separate": hasattr(Separator, "separate"),
    }


@probe("torchfcpe")
def _torchfcpe() -> dict[str, Any]:
    import torchfcpe

    return {"importable": True, "version": getattr(torchfcpe, "__version__", "?")}


@probe("onnxruntime")
def _onnxruntime() -> dict[str, Any]:
    import onnxruntime as ort

    return {"version": ort.__version__, "providers": ort.get_available_providers()}


@probe("ffmpeg")
def _ffmpeg() -> dict[str, Any]:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise FileNotFoundError("ffmpeg is not on PATH")
    ver = subprocess.run([exe, "-version"], capture_output=True, text=True, timeout=20)
    flt = subprocess.run(
        [exe, "-hide_banner", "-filters"], capture_output=True, text=True, timeout=30
    )
    wanted = ("loudnorm", "alimiter", "acompressor", "aresample")
    return {
        "path": exe,
        "version": (ver.stdout or "").splitlines()[0] if ver.stdout else "?",
        "filters": {f: (f in flt.stdout) for f in wanted},
        "all_filters_present": all(f in flt.stdout for f in wanted),
    }


if __name__ == "__main__":
    ok = all(v.get("ok") for k, v in RESULT.items() if k != "onnxruntime")
    print(json.dumps({"all_required_ok": ok, "probes": RESULT}, ensure_ascii=False, indent=2))
