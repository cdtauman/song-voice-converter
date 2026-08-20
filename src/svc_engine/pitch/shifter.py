"""``PythonStretchShifter`` -- length-preserving pitch shift over python-stretch.

python-stretch 0.3.1 (MIT) wraps Signalsmith Stretch and was proven on this
machine in the Phase 1 spike: it loads, runs, and returns exactly as many frames
as it was given (``docs/compat-matrix.md``; ``docs/third-party.md`` section 1.1).
The API that spike exercised is the one used here:

    stretch = python_stretch.Stretch()
    stretch.preset(channels, sample_rate)
    stretch.setTransposeSemitones(semitones)
    out = stretch.process(samples)          # (channels, n) float32 -> (channels, n)

This is the default and only shipped shifter. Rubber Band (GPL) may appear only
as a benchmark-time comparison behind the same ``PitchShifter`` protocol and is
never packaged (research 4.5, ``docs/decisions.md``).

The heavy import is lazy and the CPU-only routing is deliberate: the spike
recorded ``pitch_shift`` as proven on CPU only, and the engine honours that
proof exactly the way the RMVPE path does rather than borrowing another
component's accelerator claim.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

from svc_engine.audio.buffers import fit_length
from svc_engine.backends.base import AudioBuffer, BackendInfo
from svc_engine.errors import EngineError, ErrorCode

__all__ = ["PythonStretchShifter", "BACKEND_ID"]

BACKEND_ID = "python_stretch"

#: Signalsmith's "cheaper" preset id, as used in the Phase 1 spike. Preset 1 is
#: the balanced default; the number is the library's own preset enum.
_PRESET = 1


class PythonStretchShifter:
    """Wraps ``python_stretch.Stretch``. Guarantees frame-exact output length."""

    def __init__(self) -> None:
        self._stretch: Any = None
        self._configured: tuple[int, int] | None = None  # (channels, sample_rate)

    # -- PitchShifter ------------------------------------------------------- #

    def info(self) -> BackendInfo:
        available = importlib.util.find_spec("python_stretch") is not None
        return BackendInfo(
            backend_id=BACKEND_ID,
            display_name_he="הזזת גובה (Signalsmith)",
            available=available,
            unavailable_reason=None if available else "החבילה python-stretch אינה מותקנת.",
            supports_gpu=False,  # CPU-only by design; see module docstring.
            capabilities=frozenset({"pitch_shift", "length_preserving"}),
        )

    def shift(self, audio: AudioBuffer, semitones: float) -> AudioBuffer:
        """Shift by ``semitones`` without changing duration.

        A shift of zero is returned untouched -- there is no reason to send audio
        through the resampler to get itself back, and it keeps ``r == 0`` exact.
        """
        if semitones == 0:
            return audio

        stretch = self._ensure_ready(audio.channels, audio.sample_rate)
        samples = np.ascontiguousarray(audio.samples, dtype=np.float32)

        stretch.setTransposeSemitones(float(semitones))
        out = np.asarray(stretch.process(samples), dtype=np.float32)
        if out.ndim == 1:
            out = out[None, :]

        shifted = AudioBuffer(samples=np.ascontiguousarray(out), sample_rate=audio.sample_rate)
        # The library preserves length, but the contract is non-negotiable and
        # cheap to enforce: never let a rounding difference leak downstream where
        # every stem must stay sample-aligned.
        return fit_length(shifted, audio.frames)

    # -- internals ---------------------------------------------------------- #

    def _ensure_ready(self, channels: int, sample_rate: int) -> Any:
        if self._stretch is not None and self._configured == (channels, sample_rate):
            return self._stretch
        try:
            import python_stretch
        except ImportError as exc:  # pragma: no cover - exercised only without the wheel
            raise EngineError(ErrorCode.BACKEND_UNAVAILABLE, str(exc)) from exc

        # The Stretch class sits under a Signalsmith namespace in some builds and
        # at the top level in others; getattr keeps both paths untyped-Any so no
        # attribute is resolved against the compiled module at check time.
        container = getattr(python_stretch, "Signalsmith", python_stretch)
        stretch = container.Stretch()
        stretch.preset(channels, sample_rate)
        self._stretch = stretch
        self._configured = (channels, sample_rate)
        return stretch
