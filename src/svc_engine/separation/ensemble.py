"""Combining the output of several separation models.

Different checkpoints fail in different places -- one leaks a guitar into the
vocal at the chorus, another chops a breath. Combining them keeps what they
agree on and suppresses what only one of them produced.

Which combiner wins is a listening question, not an arithmetic one, so all four
are implemented and the choice is left to the benchmark. What is *not* a
judgement call: every input must be the same length before combining, or the
result smears. `align_all` enforces that.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np

from svc_engine.audio.buffers import align_all
from svc_engine.backends.base import AudioBuffer
from svc_engine.backends.separation import StemKind, Stems

__all__ = ["EnsembleMode", "combine_buffers", "combine_stems"]


class EnsembleMode(StrEnum):
    #: Single model; the combiner is never called.
    NONE = "none"
    #: Plain mean. Safest, slightly softer than any single input.
    AVERAGE = "avg"
    #: Per-sample median. Rejects a single model's artifact outright, which
    #: makes it the natural default for an odd number of inputs.
    MEDIAN = "median"
    #: Per-sample smallest magnitude -- the most conservative, least bleed,
    #: at the cost of thinning quiet detail.
    MIN_MAG = "min_mag"
    #: Per-sample largest magnitude -- keeps every breath, keeps every artifact.
    MAX_MAG = "max_mag"


def _stack(buffers: list[AudioBuffer]) -> np.ndarray:
    aligned = align_all(buffers)
    channels = max(b.channels for b in aligned)
    rows = []
    for buf in aligned:
        data = buf.samples
        if buf.channels < channels:
            data = np.repeat(data, channels // buf.channels, axis=0)[:channels]
        rows.append(data[:channels])
    return np.stack(rows, axis=0)


def combine_buffers(
    buffers: list[AudioBuffer], mode: EnsembleMode = EnsembleMode.MEDIAN
) -> AudioBuffer:
    """Combine N buffers of the same stem into one."""
    if not buffers:
        raise ValueError("nothing to combine")
    if len(buffers) == 1 or mode is EnsembleMode.NONE:
        return buffers[0]

    stack = _stack(buffers)
    sample_rate = buffers[0].sample_rate

    if mode is EnsembleMode.AVERAGE:
        out = stack.mean(axis=0)
    elif mode is EnsembleMode.MEDIAN:
        out = np.median(stack, axis=0)
    elif mode in (EnsembleMode.MIN_MAG, EnsembleMode.MAX_MAG):
        magnitudes = np.abs(stack)
        picker = magnitudes.argmin(axis=0) if mode is EnsembleMode.MIN_MAG else (
            magnitudes.argmax(axis=0)
        )
        out = np.take_along_axis(stack, picker[None, ...], axis=0)[0]
    else:  # pragma: no cover - StrEnum is exhaustive above
        raise ValueError(f"unknown ensemble mode: {mode}")

    return AudioBuffer(samples=out.astype(np.float32), sample_rate=sample_rate)


def combine_stems(
    results: list[Stems], mode: EnsembleMode = EnsembleMode.MEDIAN
) -> Stems:
    """Combine several full separation results, stem by stem.

    A stem only one model produced is passed through unchanged rather than
    dropped: losing the ambience layer because the second model has no concept
    of it would be a worse outcome than not combining it.
    """
    if not results:
        raise ValueError("nothing to combine")
    if len(results) == 1 or mode is EnsembleMode.NONE:
        return results[0]

    per_kind: dict[StemKind, list[AudioBuffer]] = {}
    for result in results:
        for kind, buffer in result.parts.items():
            per_kind.setdefault(kind, []).append(buffer)

    combined = {
        kind: combine_buffers(buffers, mode) for kind, buffers in per_kind.items()
    }
    model_id = "+".join(r.model_id for r in results)
    return Stems(parts=combined, model_id=f"{model_id}({mode.value})")
