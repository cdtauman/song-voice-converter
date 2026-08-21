"""Runtime imports required for a distributable engine smoke test."""

from __future__ import annotations

from importlib import import_module


def verify_packaged_runtime() -> None:
    """Fail fast when a frozen build omitted the heavy engine runtime."""
    torch = import_module("torch")
    separator_module = import_module("audio_separator.separator")

    if not torch.__version__:
        raise RuntimeError("Torch runtime has no version")
    if getattr(separator_module, "Separator", None) is None:
        raise RuntimeError("audio-separator runtime is unavailable")
