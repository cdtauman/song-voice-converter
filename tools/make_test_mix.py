"""Generate a synthetic song with known stems.

The real test set is five songs the user has rights to (docs/testing.md section
1). This is not a substitute for it. It exists so that the separation plumbing
can be exercised on any machine, in CI, with no audio in the repository and no
licensing question at all: because the mix is built from stems we generated, the
ground truth is exact and reconstruction error is measurable.

What it does *not* tell you is whether separation sounds good. A sawtooth with
vibrato is not a singer, and a separation model may do anything with it.

    python tools/make_test_mix.py --out tests/fixtures --seconds 20
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.audio import io as audio_io  # noqa: E402
from svc_engine.backends.base import AudioBuffer  # noqa: E402

SAMPLE_RATE = 44100

#: A short melodic phrase, as MIDI note numbers. Roughly a female alto range,
#: so the pitch analysis in Phase 3 has something plausible to chew on.
MELODY = (67, 69, 71, 72, 71, 69, 67, 64)
#: A simple chord progression under it, one chord per two melody notes.
CHORDS = ((48, 52, 55), (45, 48, 52), (43, 47, 50), (48, 52, 55))


def _midi_hz(note: float) -> float:
    return 440.0 * (2.0 ** ((note - 69) / 12.0))


def _adsr(length: int, attack: float = 0.02, release: float = 0.15) -> np.ndarray:
    env = np.ones(length, dtype=np.float32)
    a = max(1, int(length * attack))
    r = max(1, int(length * release))
    env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
    env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
    return env


def make_vocal(seconds: float) -> AudioBuffer:
    """A vibrato-ed sawtooth phrase -- vaguely voice-shaped, definitely not a voice."""
    note_frames = int(SAMPLE_RATE * seconds / len(MELODY))
    parts = []
    for note in MELODY:
        t = np.arange(note_frames, dtype=np.float32) / SAMPLE_RATE
        vibrato = 1.0 + 0.012 * np.sin(2 * np.pi * 5.2 * t)
        phase = 2 * np.pi * _midi_hz(note) * np.cumsum(vibrato) / SAMPLE_RATE
        # A few harmonics with falling amplitude reads more like a voice than a
        # pure sine, and gives the separator actual spectral structure to find.
        wave = sum(np.sin(phase * h) / h for h in (1, 2, 3, 4, 5))
        parts.append((wave * _adsr(note_frames) * 0.28).astype(np.float32))
    mono = np.concatenate(parts)[None, :]
    return AudioBuffer(samples=np.repeat(mono, 2, axis=0), sample_rate=SAMPLE_RATE)


def make_instrumental(seconds: float, frames: int) -> AudioBuffer:
    """Sustained chords plus a metronomic noise burst standing in for drums."""
    chord_frames = frames // len(CHORDS)
    parts = []
    for chord in CHORDS:
        t = np.arange(chord_frames, dtype=np.float32) / SAMPLE_RATE
        wave = sum(np.sin(2 * np.pi * _midi_hz(n) * t) for n in chord) / len(chord)
        parts.append((wave * 0.22).astype(np.float32))
    pad = np.concatenate(parts)

    rng = np.random.default_rng(20260817)
    drums = np.zeros(len(pad), dtype=np.float32)
    beat = int(SAMPLE_RATE * 0.5)
    burst = int(SAMPLE_RATE * 0.04)
    for start in range(0, len(pad) - burst, beat):
        noise = rng.standard_normal(burst).astype(np.float32)
        drums[start : start + burst] += noise * np.linspace(0.35, 0.0, burst) ** 2

    mono = (pad + drums)[None, :]
    stereo = np.concatenate([mono * 0.95, mono * 1.05], axis=0).astype(np.float32)
    return AudioBuffer(samples=stereo, sample_rate=SAMPLE_RATE)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="tests/fixtures", help="output directory")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    vocal = make_vocal(args.seconds)
    instrumental = make_instrumental(args.seconds, vocal.frames)
    frames = min(vocal.frames, instrumental.frames)
    mix = AudioBuffer(
        samples=(vocal.samples[:, :frames] + instrumental.samples[:, :frames]).astype(
            np.float32
        ),
        sample_rate=SAMPLE_RATE,
    )

    for name, buffer in (
        ("dry_vocal.wav", vocal),
        ("dry_instrumental.wav", instrumental),
        ("mix.wav", mix),
    ):
        path = audio_io.save_wav(buffer, out / name, bit_depth=24)
        print(f"{path}  {buffer.seconds:.1f}s  {buffer.channels}ch")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
