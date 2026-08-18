"""F0 preparation for RVC: octave shift and mel-scale coarse quantisation.

Pure numpy, matched sample-for-sample to RVC-Project's `Pipeline.get_f0`
(infer/vc/pipeline.py, MIT). RVC feeds the network two things per frame: the
continuous pitch in Hz (`pitchf`) and a coarse integer bucket in 1..255
(`pitch`) produced by quantising the mel-scaled pitch. Getting the buckets wrong
detunes the whole conversion, so this is worth having on its own and testing.

Kept here rather than vendored because it is short, self-contained maths -- and
because the same octave decomposition the pitch engine reasons about
(`s = 12k + r`) enters RVC exactly through the `f0_up_key` applied here.
"""

from __future__ import annotations

import numpy as np

__all__ = ["F0_MIN", "F0_MAX", "apply_up_key", "f0_to_coarse"]

#: RVC's fixed voiced-pitch bounds, in Hz. The mel range derived from these sets
#: where the 255 buckets fall; they are constants of the trained models.
F0_MIN = 50.0
F0_MAX = 1100.0

_F0_MEL_MIN = 1127.0 * np.log(1.0 + F0_MIN / 700.0)
_F0_MEL_MAX = 1127.0 * np.log(1.0 + F0_MAX / 700.0)


def apply_up_key(f0_hz: np.ndarray, semitones: float) -> np.ndarray:
    """Shift a pitch curve by `semitones` half-steps: `f0 * 2**(n/12)`.

    This is where the pitch engine's decision reaches the vocoder. Unvoiced
    frames are zero and scaling leaves them zero, so they stay unvoiced.
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    return f0 * float(2.0 ** (semitones / 12.0))


def f0_to_coarse(f0_hz: np.ndarray) -> np.ndarray:
    """Quantise a pitch curve (Hz) to RVC's integer buckets in 1..255.

    Voiced frames map onto a mel scale stretched across buckets 1..255; unvoiced
    (0 Hz) frames and anything below the floor collapse to bucket 1. Verbatim
    arithmetic from RVC so a checkpoint sees exactly the buckets it trained on.
    """
    f0 = np.asarray(f0_hz, dtype=np.float64)
    f0_mel = 1127.0 * np.log(1.0 + f0 / 700.0)
    voiced = f0_mel > 0
    f0_mel[voiced] = (f0_mel[voiced] - _F0_MEL_MIN) * 254.0 / (
        _F0_MEL_MAX - _F0_MEL_MIN
    ) + 1.0
    f0_mel[f0_mel <= 1] = 1.0
    f0_mel[f0_mel > 255] = 255.0
    return np.rint(f0_mel).astype(np.int32)
