"""Pitch <-> MIDI helpers.

The whole range/pitch story is told in semitones, not hertz: an octave is
exactly 12 whatever the register, which is what makes the shift decomposition
`s = 12k + r` in Phase 4 arithmetic instead of guesswork. So the analysis stage
speaks MIDI everywhere and only touches hertz at the edges (the F0 model in,
the plot out).

Pure numpy, no torch -- this runs in CI where the heavy stack is absent.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "A4_HZ",
    "A4_MIDI",
    "NOTE_NAMES",
    "hz_to_midi",
    "midi_to_hz",
    "midi_to_note_name",
    "note_name",
]

A4_HZ = 440.0
A4_MIDI = 69

#: Sharps only; enough to name a note for a report. No enharmonic spelling.
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def hz_to_midi(hz: np.ndarray | float) -> np.ndarray:
    """Convert frequency to fractional MIDI. Non-positive values become NaN.

    F0 models write 0.0 for unvoiced frames; mapping those to NaN keeps them out
    of every statistic instead of poisoning it with a huge negative number.
    """
    arr = np.asarray(hz, dtype=np.float64)
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    voiced = arr > 0.0
    out[voiced] = A4_MIDI + 12.0 * np.log2(arr[voiced] / A4_HZ)
    return out


def midi_to_hz(midi: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(midi, dtype=np.float64)
    return A4_HZ * np.power(2.0, (arr - A4_MIDI) / 12.0)


def midi_to_note_name(midi: float) -> str:
    """Nearest named note, e.g. 69.0 -> 'A4', 60.4 -> 'C4'."""
    rounded = int(round(midi))
    return f"{NOTE_NAMES[rounded % 12]}{rounded // 12 - 1}"


def note_name(midi: float) -> str:
    """Alias kept for readability at call sites that read a single pitch."""
    return midi_to_note_name(midi)
