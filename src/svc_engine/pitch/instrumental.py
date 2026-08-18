"""Two ways to move the backing track with the singing -- A and B.

Research section 4.4: when the shift's remainder ``r`` is non-zero the playback
must move by ``r`` too, or the song goes out of key. *How* to move it is an
empirical question, not a rule, because the two obvious answers each pay a
different price:

    A -- whole shift    move the instrumental as one buffer.
                        Cost: drums, which have no defined pitch, smear.
    B -- split          bass / drums / other; shift everything but the drums.
                        Cost: the extra separation adds its own artifacts and
                        the re-summed stems can mismatch at the seams.

The old plan hard-coded "never shift the drums". That was too absolute -- it
ignored B's separation cost -- so both strategies live here behind one call and
the Phase 4 benchmark (``docs/testing.md`` 5a) decides which wins at each ``|r|``.
The result is a *table of defaults by |r|*, filled by measurement.

Every output is trimmed back to the source length to the sample: timing is never
allowed to drift (research 4.6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum

from svc_engine.audio.buffers import add, fit_length
from svc_engine.backends.base import AudioBuffer
from svc_engine.backends.pitch import PitchShifter
from svc_engine.backends.separation import StemKind

__all__ = ["PlaybackStrategy", "PitchedStems", "shift_playback"]


class PlaybackStrategy(StrEnum):
    WHOLE = "A"   # shift the instrumental as a single buffer
    SPLIT = "B"   # split into bass/drums/other, shift all but the drums


#: In strategy B these carry pitch and must follow the singing; the drums do not.
_PITCHED = (StemKind.BASS, StemKind.OTHER)


@dataclass(frozen=True)
class PitchedStems:
    """The split-strategy inputs: the pitched stems plus the untouched drums.

    Any stem may be absent -- a two-stem split with no separate drums still works
    (it just behaves like strategy A on whatever it was given). Lengths are
    reconciled to the drums, or to the first pitched stem, at shift time.
    """

    bass: AudioBuffer | None = None
    other: AudioBuffer | None = None
    drums: AudioBuffer | None = None

    @classmethod
    def from_mapping(cls, stems: Mapping[StemKind, AudioBuffer]) -> PitchedStems:
        return cls(
            bass=stems.get(StemKind.BASS),
            other=stems.get(StemKind.OTHER),
            drums=stems.get(StemKind.DRUMS),
        )

    def present(self) -> list[AudioBuffer]:
        return [b for b in (self.bass, self.other, self.drums) if b is not None]


def shift_playback(
    playback: AudioBuffer | PitchedStems | Mapping[StemKind, AudioBuffer],
    remainder: int,
    shifter: PitchShifter,
    strategy: PlaybackStrategy = PlaybackStrategy.WHOLE,
) -> AudioBuffer:
    """Move the playback by ``remainder`` semitones using the chosen strategy.

    ``remainder`` is the ``r`` of ``s = 12k + r`` -- the octaves never reach the
    playback. ``r == 0`` returns the playback unchanged (an octave-only shift),
    which is also the cheapest and most common recommendation.
    """
    if strategy is PlaybackStrategy.WHOLE:
        if not isinstance(playback, AudioBuffer):
            raise TypeError("strategy A (WHOLE) takes a single instrumental AudioBuffer")
        return _shift_whole(playback, remainder, shifter)

    stems = (
        playback
        if isinstance(playback, PitchedStems)
        else PitchedStems.from_mapping(playback)
        if isinstance(playback, Mapping)
        else None
    )
    if stems is None:
        raise TypeError("strategy B (SPLIT) takes PitchedStems or a StemKind mapping")
    return _shift_split(stems, remainder, shifter)


def _shift_whole(instrumental: AudioBuffer, r: int, shifter: PitchShifter) -> AudioBuffer:
    if r == 0:
        return instrumental
    return fit_length(shifter.shift(instrumental, r), instrumental.frames)


def _shift_split(stems: PitchedStems, r: int, shifter: PitchShifter) -> AudioBuffer:
    present = stems.present()
    if not present:
        raise ValueError("no stems to shift")
    length = present[0].frames

    parts: list[AudioBuffer] = []
    for stem in (stems.bass, stems.other):
        if stem is None:
            continue
        moved = stem if r == 0 else shifter.shift(stem, r)
        parts.append(fit_length(moved, length))
    if stems.drums is not None:  # drums are never shifted -- keep their attack
        parts.append(fit_length(stems.drums, length))

    mix = parts[0]
    for part in parts[1:]:
        mix = add(mix, part)
    return fit_length(mix, length)
