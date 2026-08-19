"""Small, bounded preview search using objective artifact proxies.

The grid deliberately stays at four variants.  It includes the user's/manual
starting point, so auto-tuning can never lose that candidate through search.
Human blind preference is still the Phase-10 acceptance authority; the score is
only the deterministic selector used before that listening gate.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from svc_engine.backends.base import AudioBuffer
from svc_engine.tuning.config import AdvancedConfig

__all__ = ["CandidateResult", "TuningResult", "auto_tune", "candidate_grid", "score_audio"]


@dataclass(frozen=True)
class CandidateResult:
    candidate_id: str
    config: AdvancedConfig
    score: float
    metrics: dict[str, float]
    audio: AudioBuffer


@dataclass(frozen=True)
class TuningResult:
    winner: CandidateResult
    candidates: tuple[CandidateResult, ...]


def candidate_grid(base: AdvancedConfig) -> tuple[AdvancedConfig, ...]:
    """Return the manual baseline and three conservative nearby variants."""
    return (
        base,
        replace(
            base,
            index_rate=min(1.0, base.index_rate + 0.15),
            protect=min(0.5, base.protect + 0.12),
        ),
        replace(
            base,
            index_rate=max(0.0, base.index_rate - 0.20),
            rms_mix_rate=min(1.0, base.rms_mix_rate + 0.20),
        ),
        replace(
            base,
            protect=max(0.0, base.protect - 0.13),
            filter_radius=min(7, base.filter_radius + 2),
        ),
    )


def score_audio(audio: AudioBuffer) -> tuple[float, dict[str, float]]:
    """Score finite, unclipped, continuous audio while preserving dynamics."""
    samples = np.asarray(audio.samples, dtype=np.float64)
    if samples.size == 0 or not np.isfinite(samples).all():
        return float("-inf"), {"finite": 0.0}
    mono = samples.mean(axis=0)
    peak = float(np.max(np.abs(mono), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(mono)) + 1e-12))
    clipped = float(np.mean(np.abs(mono) >= 0.999))
    discontinuity = (
        float(np.percentile(np.abs(np.diff(mono)), 99.9)) if mono.size > 1 else 0.0
    )
    crest = peak / max(rms, 1e-9)
    silence = float(np.mean(np.abs(mono) < 1e-4))
    # Values are deliberately transparent, bounded proxies rather than a claim
    # of perceptual MOS. Human blind votes are retained separately.
    score = (
        100.0
        - 5000.0 * clipped
        - 35.0 * max(0.0, peak - 0.98)
        - 12.0 * max(0.0, discontinuity - 0.35)
        - 8.0 * max(0.0, 1.5 - crest)
        - 10.0 * max(0.0, silence - 0.45)
    )
    return score, {
        "peak": peak,
        "rms": rms,
        "clipped_fraction": clipped,
        "discontinuity_p999": discontinuity,
        "crest_factor": crest,
        "silence_fraction": silence,
    }


def auto_tune(
    base: AdvancedConfig,
    render: Callable[[AdvancedConfig], AudioBuffer],
) -> TuningResult:
    results: list[CandidateResult] = []
    for index, config in enumerate(candidate_grid(base)):
        audio = render(config)
        score, metrics = score_audio(audio)
        results.append(CandidateResult(f"candidate-{index + 1}", config, score, metrics, audio))
    winner = max(results, key=lambda item: (item.score, -int(item.candidate_id.rsplit("-", 1)[1])))
    return TuningResult(winner=winner, candidates=tuple(results))
