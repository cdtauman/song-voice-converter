"""Optional pitch-jump softening -- off by default.

Research section 4.7: gently compressing the F0 curve toward its median pulls the
most extreme notes inward, which can rescue a phrase that clips the target's
range. It is optional and off by default for one honest reason: **it changes the
melody.** A note pulled 10% toward the median is no longer quite the note the
singer sang. So this is a labelled switch in advanced mode, never automatic, and
never part of the default recommendation.

The compression runs in the MIDI domain -- an octave is 12 semitones anywhere,
so scaling there is musically even, unlike scaling hertz. Unvoiced frames (0 Hz)
are left as they are. Pure numpy.
"""

from __future__ import annotations

import numpy as np

from svc_engine.analysis.midi import hz_to_midi, midi_to_hz
from svc_engine.analysis.range import weighted_percentile
from svc_engine.backends.base import F0Curve

__all__ = ["soften_f0"]


def soften_f0(
    curve: F0Curve,
    amount: float = 0.0,
    energy: np.ndarray | None = None,
) -> F0Curve:
    """Compress voiced pitches toward the median by ``amount`` in [0, 1].

    ``amount == 0`` is an exact no-op (the default): the curve is returned
    unchanged. ``amount == 1`` would flatten everything onto the median, which is
    never wanted; sensible use is a few percent. The centre is the
    energy-weighted median when ``energy`` is given, so loud, held notes anchor
    the compression rather than passing grace notes.
    """
    if not 0.0 <= amount <= 1.0:
        raise ValueError(f"amount must be in [0, 1], got {amount}")
    if amount == 0.0:
        return curve

    hz = np.asarray(curve.hz, dtype=np.float64).ravel()
    midi = hz_to_midi(hz)
    voiced = ~np.isnan(midi)
    if not voiced.any():
        return curve

    if energy is None:
        centre = float(np.median(midi[voiced]))
    else:
        w = np.clip(np.asarray(energy, dtype=np.float64).ravel(), 0.0, None)
        vw = w[voiced]
        centre = (
            float(weighted_percentile(midi[voiced], vw, np.array([50.0]))[0])
            if vw.sum() > 0.0
            else float(np.median(midi[voiced]))
        )

    factor = 1.0 - amount
    new_midi = midi.copy()
    new_midi[voiced] = centre + (midi[voiced] - centre) * factor

    new_hz = hz.copy()
    new_hz[voiced] = midi_to_hz(new_midi[voiced])
    return F0Curve(hz=new_hz.astype(np.float64), hop_seconds=curve.hop_seconds)
