"""F0 preservation check plus a bounded, global fine correction."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from svc_engine.backends.base import AudioBuffer, F0Curve
from svc_engine.backends.pitch import PitchShifter

__all__ = ["MelodyReport", "check_melody", "correct_global_pitch"]


@dataclass(frozen=True)
class MelodyReport:
    correlation: float | None
    median_error_cents: float | None
    p95_absolute_error_cents: float | None
    voiced_frames: int
    passed: bool | None
    correction_semitones: float = 0.0
    note: str = ""


def _align(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).ravel()
    if values.size == size:
        return values
    if not values.size or size <= 0:
        return np.zeros(max(0, size), dtype=np.float64)
    return np.interp(
        np.linspace(0.0, 1.0, size), np.linspace(0.0, 1.0, values.size), values
    )


def check_melody(
    reference: F0Curve,
    converted: F0Curve | None,
    semitones: float,
    *,
    correlation_floor: float = 0.97,
) -> MelodyReport:
    """Compare converted F0 with the source after the requested transposition.

    The 0.97 floor is informational until the project's human calibration set
    exists (testing.md 3.3); callers must not turn it into a merge gate yet.
    """
    if converted is None:
        return MelodyReport(None, None, None, 0, None, note="output F0 was not measured")
    n = max(reference.frames, converted.frames)
    expected = _align(reference.hz, n) * 2.0 ** (float(semitones) / 12.0)
    actual = _align(converted.hz, n)
    voiced = (expected > 20.0) & (actual > 20.0)
    count = int(np.count_nonzero(voiced))
    if count < 3:
        return MelodyReport(None, None, None, count, None, note="too few jointly voiced frames")

    exp_log = np.log2(expected[voiced])
    act_log = np.log2(actual[voiced])
    if np.std(exp_log) < 1e-9 or np.std(act_log) < 1e-9:
        correlation = 1.0 if np.std(act_log - exp_log) < 0.01 else 0.0
    else:
        correlation = float(np.corrcoef(exp_log, act_log)[0, 1])
    error = 1200.0 * (act_log - exp_log)
    median = float(np.median(error))
    p95 = float(np.percentile(np.abs(error), 95))
    return MelodyReport(
        correlation=correlation,
        median_error_cents=median,
        p95_absolute_error_cents=p95,
        voiced_frames=count,
        passed=bool(correlation >= correlation_floor),
        note="provisional threshold; not a calibrated quality gate",
    )


def correct_global_pitch(
    audio: AudioBuffer,
    report: MelodyReport,
    shifter: PitchShifter,
    *,
    trigger_cents: float = 15.0,
    max_correction_semitones: float = 0.5,
) -> tuple[AudioBuffer, MelodyReport]:
    """Correct only a stable global offset, capped at half a semitone.

    Local note rewriting would change the performance and is intentionally not
    attempted. The cap keeps this a safety correction rather than a new melody.
    """
    error = report.median_error_cents
    if error is None or abs(error) < trigger_cents:
        return audio, report
    correction = float(np.clip(-error / 100.0, -max_correction_semitones,
                               max_correction_semitones))
    return shifter.shift(audio, correction), replace(report, correction_semitones=correction)
