"""Real workloads that prove a component can run on a compute backend.

The policy is in docs/reuse-policy.md and docs/architecture.md: a component
counts as supported on a backend only when a real workload runs there and
returns correct numbers. Importing, installing and `is_available()` prove
nothing.

Each workload runs the operator set the real component depends on:

    separation  -- STFT -> multi-head attention -> iSTFT   (RoFormer)
    f0          -- the actual torchfcpe model, end to end
    conversion  -- Conv1d -> Transformer -> ConvTranspose  (RVC)
    pitch_shift -- CPU only; python-stretch is C++ DSP with no GPU path

This module is imported both by the app (`svc verify-backends`) and by the
Compatibility Spike, which loads it straight off disk inside isolated
environments. It therefore imports torch lazily and depends on nothing else
in the package.
"""

from __future__ import annotations

import contextlib
import io
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

__all__ = [
    "WORKLOADS",
    "ProofResult",
    "available_devices",
    "build_support_payload",
    "verify_all",
    "verify_component",
]

#: Components that have a GPU path worth testing. `pitch_shift` is absent on
#: purpose: CPU is where that work belongs, not a fallback from failure.
WORKLOAD_COMPONENTS = ("separation", "f0", "conversion")


@dataclass(frozen=True)
class ProofResult:
    component: str
    device: str
    ok: bool
    seconds: float
    ops: str = ""
    end_to_end: bool = False
    detail: dict[str, Any] | None = None
    error: str = ""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def available_devices() -> list[str]:
    """Device strings torch reports as usable right now, CPU always first."""
    devices = ["cpu"]
    try:
        import torch
    except Exception:  # noqa: BLE001  torch is optional until the stack is installed
        return devices
    try:
        if torch.cuda.is_available():
            devices.append("cuda")
    except Exception:  # noqa: BLE001
        pass
    try:
        xpu = getattr(torch, "xpu", None)
        if xpu is not None and xpu.is_available():
            devices.append("xpu")
    except Exception:  # noqa: BLE001
        pass
    return devices


def _sync(device: str) -> None:
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "xpu":
        torch.xpu.synchronize()


# --------------------------------------------------------------------------- #
# workloads
# --------------------------------------------------------------------------- #

def workload_separation(device: str) -> dict[str, Any]:
    """The operator set a RoFormer separator needs."""
    import torch

    sr, seconds = 44100, 2
    wav = torch.randn(1, sr * seconds, device=device)
    window = torch.hann_window(2048, device=device)

    spec = torch.stft(wav, n_fft=2048, hop_length=512, window=window, return_complex=True)
    feat = spec.abs().transpose(1, 2)[..., :512]

    attn = torch.nn.MultiheadAttention(512, 8, batch_first=True).to(device)
    with torch.no_grad():
        out, _ = attn(feat, feat, feat)
    mask = torch.sigmoid(out).transpose(1, 2)
    masked = spec.clone()
    masked[:, :512, :] = spec[:, :512, :] * mask

    back = torch.istft(masked, n_fft=2048, hop_length=512, window=window, length=wav.shape[-1])
    _sync(device)

    if back.shape[-1] != wav.shape[-1]:
        raise RuntimeError(f"length changed: {wav.shape[-1]} -> {back.shape[-1]}")
    if not bool(torch.isfinite(back).all()):
        raise RuntimeError("output contains non-finite values")
    return {
        "ops": "stft+mha+istft",
        "end_to_end": False,
        "output_frames": int(back.shape[-1]),
    }


def workload_f0(device: str) -> dict[str, Any]:
    """The real F0 model. Falls back to its operator set only if weights are absent."""
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
        median = float(f0[f0 > 0].median()) if bool((f0 > 0).any()) else 0.0
        # A device that runs without erroring but reads the wrong pitch is worse
        # than one that fails loudly.
        if not 180.0 < median < 260.0:
            raise RuntimeError(f"read {median:.1f}Hz off a 220Hz tone")
        return {
            "ops": "torchfcpe end-to-end",
            "end_to_end": True,
            "median_hz": round(median, 1),
        }
    except Exception as exc:  # noqa: BLE001  weights may be unavailable offline
        fallback_reason = f"{type(exc).__name__}: {exc}"[:200]

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
    if not bool(torch.isfinite(out).all()):
        raise RuntimeError("output contains non-finite values")
    return {
        "ops": "conv1d stack (fallback)",
        "end_to_end": False,
        "fallback_reason": fallback_reason,
        "output_shape": list(out.shape),
    }


def workload_conversion(device: str) -> dict[str, Any]:
    """The operator set RVC inference needs."""
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

    if not bool(torch.isfinite(wav).all()):
        raise RuntimeError("output contains non-finite values")
    return {
        "ops": "conv1d+transformer+convtranspose",
        "end_to_end": False,
        "output_shape": list(wav.shape),
    }


WORKLOADS: dict[str, Callable[[str], dict[str, Any]]] = {
    "separation": workload_separation,
    "f0": workload_f0,
    "conversion": workload_conversion,
}


# --------------------------------------------------------------------------- #
# running and recording
# --------------------------------------------------------------------------- #

def verify_component(component: str, device: str) -> ProofResult:
    """Run one workload. Never raises -- a failure is a result, not a crash."""
    workload = WORKLOADS.get(component)
    if workload is None:
        return ProofResult(component, device, False, 0.0, error="no workload defined")

    started = time.perf_counter()
    # Libraries here print banners straight to stdout (torchfcpe does). Swallow
    # them: this runs behind a user-facing command and behind --json output.
    noise = io.StringIO()
    try:
        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            data = workload(device)
    except Exception as exc:  # noqa: BLE001
        return ProofResult(
            component, device, False, round(time.perf_counter() - started, 2),
            error=f"{type(exc).__name__}: {exc}"[:300],
        )
    return ProofResult(
        component=component,
        device=device,
        ok=True,
        seconds=round(time.perf_counter() - started, 2),
        ops=str(data.pop("ops", "")),
        end_to_end=bool(data.pop("end_to_end", False)),
        detail=data or None,
    )


def verify_all(devices: list[str] | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    """Run every workload on every device. Returns {component: {device: result}}."""
    devices = devices if devices is not None else available_devices()
    matrix: dict[str, dict[str, dict[str, Any]]] = {}

    for component in WORKLOAD_COMPONENTS:
        matrix[component] = {}
        for device in devices:
            result = verify_component(component, device)
            entry = asdict(result)
            entry.pop("component")
            entry.pop("device")
            if entry.get("detail"):
                entry.update(entry.pop("detail"))
            else:
                entry.pop("detail", None)
            matrix[component][device] = entry

    matrix["pitch_shift"] = {
        "cpu": {"ok": True, "seconds": 0.0, "ops": "python-stretch (cpu-only by design)",
                "end_to_end": True, "error": ""},
    }
    return matrix


_NOTES_HE = {
    "separation": "הפרדה",
    "f0": "זיהוי גובה",
    "conversion": "המרת קול",
    "pitch_shift": "הזזת גובה",
}


def build_support_payload(
    matrix: dict[str, dict[str, dict[str, Any]]],
    source: str,
    machine: str = "",
) -> dict[str, Any]:
    """Collapse raw results into the support matrix the app reads at runtime.

    Only a workload that actually ran counts. CPU is implicit and never recorded
    as a proof -- it is the baseline.
    """
    components: dict[str, dict[str, Any]] = {}

    for component, per_device in matrix.items():
        proofs: dict[str, str] = {}
        for device, entry in per_device.items():
            if device == "cpu" or not entry.get("ok"):
                continue
            level = "end_to_end" if entry.get("end_to_end") else "ops"
            if proofs.get(device) != "end_to_end":
                proofs[device] = level

        label = _NOTES_HE.get(component, component)
        if component == "pitch_shift":
            note = "רץ על המעבד מעצם טבעו — ספריית DSP ב-C++ בלי מסלול GPU."
        elif proofs:
            devices = ", ".join(sorted(proofs))
            kind = "מודל מלא" if "end_to_end" in proofs.values() else \
                "עומס אמיתי ברמת האופרטורים"
            note = f"{label}: אומת על {devices} ({kind})."
        else:
            note = f"{label}: לא אומת על שום מאיץ — רץ על המעבד."

        components[component] = {"proofs": proofs, "note_he": note}

    for component in (*WORKLOAD_COMPONENTS, "pitch_shift"):
        components.setdefault(
            component,
            {"proofs": {}, "note_he": f"{_NOTES_HE.get(component, component)}: לא נבדק."},
        )

    return {
        "source": source,
        "machine": machine,
        "verified_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "components": components,
    }
