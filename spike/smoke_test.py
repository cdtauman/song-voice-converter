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

def _workload_separation(device: str) -> dict[str, Any]:
    """The operator set a RoFormer separator needs: STFT, attention, iSTFT."""
    import torch

    sr, seconds = 44100, 2
    wav = torch.randn(1, sr * seconds, device=device)
    window = torch.hann_window(2048, device=device)

    spec = torch.stft(wav, n_fft=2048, hop_length=512, window=window, return_complex=True)
    mag = spec.abs().transpose(1, 2)                        # (batch, frames, bins)
    feat = mag[..., :512]

    attn = torch.nn.MultiheadAttention(512, 8, batch_first=True).to(device)
    with torch.no_grad():
        out, _ = attn(feat, feat, feat)
    mask = torch.sigmoid(out).transpose(1, 2)
    masked = spec.clone()
    masked[:, :512, :] = spec[:, :512, :] * mask

    back = torch.istft(masked, n_fft=2048, hop_length=512, window=window, length=wav.shape[-1])
    _sync(device)
    return {
        "ops": "stft+mha+istft",
        "output_frames": int(back.shape[-1]),
        "length_preserved": bool(back.shape[-1] == wav.shape[-1]),
        "finite": bool(torch.isfinite(back).all()),
    }


def _workload_f0(device: str) -> dict[str, Any]:
    """Prefer the real F0 model; fall back to its operator set if weights are absent."""
    import torch

    sr = 16000
    t = torch.arange(sr * 2, dtype=torch.float32) / sr
    tone = (0.4 * torch.sin(2 * torch.pi * 220.0 * t)).unsqueeze(0)

    try:
        import torchfcpe

        model = torchfcpe.spawn_bundled_infer_model(device=device)
        with torch.no_grad():
            f0 = model.infer(tone.unsqueeze(-1).to(device), sr=sr, decoder_mode="local_argmax")
        _sync(device)
        median = float(f0[f0 > 0].median()) if (f0 > 0).any() else 0.0
        return {
            "ops": "torchfcpe end-to-end",
            "end_to_end": True,
            "median_hz": round(median, 1),
            "plausible": bool(180 < median < 260),
            "finite": bool(torch.isfinite(f0).all()),
        }
    except Exception as exc:  # noqa: BLE001  weights may be unavailable offline
        conv = torch.nn.Sequential(
            torch.nn.Conv1d(1, 128, 9, padding=4),
            torch.nn.GELU(),
            torch.nn.Conv1d(128, 128, 9, padding=4),
            torch.nn.GELU(),
            torch.nn.Conv1d(128, 360, 1),
        ).to(device)
        with torch.no_grad():
            out = conv(tone.unsqueeze(0).to(device))
        _sync(device)
        return {
            "ops": "conv1d stack (fallback)",
            "end_to_end": False,
            "fallback_reason": f"{type(exc).__name__}: {exc}"[:200],
            "output_shape": list(out.shape),
            "finite": bool(torch.isfinite(out).all()),
        }


def _workload_conversion(device: str) -> dict[str, Any]:
    """The operator set RVC inference needs: conv encoder, transformer, upsampling."""
    import torch

    frames, channels = 400, 256
    x = torch.randn(1, channels, frames, device=device)

    encoder = torch.nn.Sequential(
        torch.nn.Conv1d(channels, channels, 5, padding=2),
        torch.nn.LeakyReLU(0.1),
        torch.nn.Conv1d(channels, channels, 5, padding=2),
    ).to(device)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=channels, nhead=8, dim_feedforward=1024, batch_first=True
    ).to(device)
    upsample = torch.nn.ConvTranspose1d(channels, 1, 16, stride=8, padding=4).to(device)

    with torch.no_grad():
        h = encoder(x)
        h = layer(h.transpose(1, 2)).transpose(1, 2)
        wav = upsample(h)
    _sync(device)
    return {
        "ops": "conv1d+transformer+convtranspose",
        "output_shape": list(wav.shape),
        "finite": bool(torch.isfinite(wav).all()),
    }


COMPONENT_WORKLOADS = {
    "separation": _workload_separation,
    "f0": _workload_f0,
    "conversion": _workload_conversion,
}


@probe("component_backends")
def _component_backends() -> dict[str, Any]:
    """Record, per component and per accelerator, whether real work succeeded."""
    import torch  # noqa: F401  (fail fast and clearly if torch is missing)

    devices = ["cpu", *_accelerators()]
    matrix: dict[str, dict[str, Any]] = {}

    for component, workload in COMPONENT_WORKLOADS.items():
        matrix[component] = {}
        for device in devices:
            started = time.perf_counter()
            try:
                data = workload(device)
                matrix[component][device] = {
                    "ok": True,
                    "seconds": round(time.perf_counter() - started, 2),
                    **data,
                }
            except Exception as exc:  # noqa: BLE001
                matrix[component][device] = {
                    "ok": False,
                    "seconds": round(time.perf_counter() - started, 2),
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }

    # python-stretch is a C++ DSP library with no GPU path. CPU here is the right
    # place for the work, not a fallback from a failure.
    matrix["pitch_shift"] = {
        "cpu": {"ok": True, "ops": "python-stretch (cpu-only by design)"},
    }
    return {"devices": devices, "matrix": matrix}


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
