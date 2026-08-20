"""Runs INSIDE a candidate virtual environment and reports what actually works.

Prints a single JSON object to stdout. Never raises: every probe reports its own
failure so one broken component does not hide the others.

Accelerator policy: a component counts as supported on a backend only when a real
workload runs on that backend and the numbers come back correct. Import success,
device detection and clean installation prove nothing.
"""

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
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


def _accelerators() -> list[str]:
    """Accelerator device strings torch reports as usable right now."""
    out: list[str] = []
    try:
        import torch
    except Exception:  # noqa: BLE001
        return out
    try:
        if torch.cuda.is_available():
            out.append("cuda")
    except Exception:  # noqa: BLE001
        pass
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            out.append("xpu")
    except Exception:  # noqa: BLE001
        pass
    return out


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


# --------------------------------------------------------------------------- #
# environment probes
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
        "built_xpu": getattr(torch.version, "xpu", None),
        "cuda_available": bool(torch.cuda.is_available()),
        "xpu_available": bool(
            getattr(torch, "xpu", None) is not None and torch.xpu.is_available()
        ),
        "accelerators": _accelerators(),
    }

    t0 = time.perf_counter()
    a = torch.randn(512, 512)
    c = a @ a
    info["matmul_device"] = "cpu"
    info["matmul_ok"] = bool(c.shape == (512, 512))
    info["matmul_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return info


@probe("xpu")
def _xpu() -> dict[str, Any]:
    """Intel XPU: detection AND a real tensor operation with verified numerics."""
    import torch

    xpu = getattr(torch, "xpu", None)
    if xpu is None:
        raise RuntimeError("this torch build has no torch.xpu module")
    if not xpu.is_available():
        raise RuntimeError("torch.xpu is present but reports no available device")

    info: dict[str, Any] = {
        "device_count": xpu.device_count(),
        "device_name": xpu.get_device_name(0),
    }
    try:
        props = xpu.get_device_properties(0)
        info["total_memory_gb"] = round(props.total_memory / 1024**3, 1)
        for attr in ("driver_version", "max_compute_units", "gpu_eu_count", "type"):
            value = getattr(props, attr, None)
            if value is not None:
                info[attr] = value
    except Exception as exc:  # noqa: BLE001  properties are best-effort
        info["properties_error"] = str(exc)

    # Real work, checked against CPU: a device that reports available but computes
    # garbage is worse than one that reports unavailable.
    torch.manual_seed(0)
    a_cpu = torch.randn(256, 256, dtype=torch.float32)
    b_cpu = torch.randn(256, 256, dtype=torch.float32)
    expected = a_cpu @ b_cpu

    t0 = time.perf_counter()
    got = (a_cpu.to("xpu") @ b_cpu.to("xpu")).to("cpu")
    torch.xpu.synchronize()
    info["matmul_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    info["max_abs_error"] = float((got - expected).abs().max())
    info["numerics_ok"] = bool(torch.allclose(got, expected, atol=1e-3, rtol=1e-3))

    if not info["numerics_ok"]:
        raise RuntimeError(f"xpu matmul disagrees with cpu: {info['max_abs_error']}")
    return info


# --------------------------------------------------------------------------- #
# per-component accelerator workloads
# --------------------------------------------------------------------------- #

@probe("component_backends")
def _component_backends() -> dict[str, Any]:
    """Record, per component and per accelerator, whether real work succeeded.

    The workloads live in `svc_engine/compute/verify.py` so that the spike and
    the shipped `svc verify-backends` command prove support exactly the same
    way. The module is loaded straight off disk: candidate environments do not
    have SongVoice installed, and some of them run a Python version the project
    does not even support.
    """
    import torch  # noqa: F401  (fail fast and clearly if torch is missing)

    src = str(Path(__file__).resolve().parent.parent / "src")
    if src not in sys.path:
        sys.path.insert(0, src)

    from svc_engine.compute.verify import available_devices, verify_all

    devices = available_devices()
    return {"devices": devices, "matrix": verify_all(devices)}


# --------------------------------------------------------------------------- #
# library probes
# --------------------------------------------------------------------------- #

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
    """python-stretch (MIT). The hard contract is exact length preservation."""
    import numpy as np

    modname, mod = _first_import("python_stretch", "stretch", "pystretch")

    sr = 44100
    n = sr
    audio = (0.2 * np.sin(2 * np.pi * 440 * np.arange(n) / sr)).astype("float32")
    audio = audio.reshape(1, -1)

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


@probe("transformers")
def _transformers() -> dict[str, Any]:
    import transformers
    from transformers import HubertModel

    return {"version": transformers.__version__, "has_hubert": HubertModel is not None}


@probe("faiss")
def _faiss() -> dict[str, Any]:
    import faiss
    import numpy as np

    index = faiss.IndexFlatL2(2)
    index.add(np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32))
    distances, indices = index.search(np.array([[0.1, 0.1]], dtype=np.float32), 1)
    return {
        "version": getattr(faiss, "__version__", "?"),
        "search_ok": bool(indices[0, 0] == 0 and distances[0, 0] >= 0),
    }


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


#: `xpu` and `onnxruntime` are informational -- absence is not a candidate failure.
OPTIONAL_PROBES = {"xpu", "onnxruntime"}

if __name__ == "__main__":
    ok = all(v.get("ok") for k, v in RESULT.items() if k not in OPTIONAL_PROBES)
    payload = json.dumps(
        {"all_required_ok": ok, "probes": RESULT}, ensure_ascii=False, indent=2
    )

    # Some libraries (torchfcpe, for one) print banners straight to stdout, which
    # would corrupt the report. Writing to a file keeps the result parseable no
    # matter how noisy the dependencies are.
    out_path = None
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]
    if out_path:
        from pathlib import Path as _Path

        _Path(out_path).write_text(payload, encoding="utf-8")
    else:
        print(payload)
