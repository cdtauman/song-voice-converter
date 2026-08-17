"""Musical key of a song: chroma + Krumhansl-Schmuckler.

Two independent halves so the decision logic stays testable without a decoder:

* `chroma_from_audio` folds a magnitude spectrum onto the twelve pitch classes,
  using nothing but `numpy.fft`. It runs on the *instrumental* by preference --
  a bare vocal spends most of its time on a handful of scale degrees and is a
  poor witness to the harmony.
* `estimate_key` correlates that 12-vector against the twenty-four
  Krumhansl-Kessler tonal profiles and reports the best fit plus the runners-up.

Key is context Phase 4 uses when it decides whether a non-octave shift is worth
its cost; it is not a hard input, so a confident-but-wrong estimate on an
ambiguous song is a documented limitation, not a failure.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from svc_engine.analysis.midi import A4_HZ, A4_MIDI, NOTE_NAMES

__all__ = ["KeyEstimate", "chroma_from_audio", "estimate_key", "estimate_key_from_audio"]

# Krumhansl & Kessler (1982) probe-tone profiles, tonic at index 0.
_MAJOR_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
)
_MINOR_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
)

_MODE_HE = {"major": "מז'ור", "minor": "מינור"}


@dataclass(frozen=True)
class KeyEstimate:
    tonic: int          # 0..11, C=0
    mode: str           # "major" | "minor"
    correlation: float  # fit of the winning profile, -1..1
    confidence: float   # winner minus runner-up, 0..~2
    alternatives: tuple[tuple[str, float], ...] = ()

    @property
    def tonic_name(self) -> str:
        return NOTE_NAMES[self.tonic % 12]

    @property
    def name(self) -> str:
        return f"{self.tonic_name} {self.mode}"

    @property
    def name_he(self) -> str:
        return f"{self.tonic_name} {_MODE_HE.get(self.mode, self.mode)}"

    def to_dict(self) -> dict[str, object]:
        return {
            "tonic": self.tonic,
            "tonic_name": self.tonic_name,
            "mode": self.mode,
            "name": self.name,
            "name_he": self.name_he,
            "correlation": round(float(self.correlation), 4),
            "confidence": round(float(self.confidence), 4),
            "alternatives": [
                {"name": name, "correlation": round(float(corr), 4)}
                for name, corr in self.alternatives
            ],
        }


def chroma_from_audio(
    mono: np.ndarray,
    sample_rate: int,
    n_fft: int = 8192,
    hop: int | None = None,
    fmin: float = 55.0,
    fmax: float = 2093.0,
) -> np.ndarray:
    """A 12-vector of pitch-class energy, L1-normalised.

    Magnitude STFT, then every bin between `fmin` and `fmax` is dropped onto its
    nearest pitch class weighted by magnitude. `fmax` defaults to C7: above that
    the bins are too coarse per semitone to place cleanly.
    """
    x = np.asarray(mono, dtype=np.float64).ravel()
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size))
    hop = hop or n_fft // 2

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate)
    band = (freqs >= fmin) & (freqs <= fmax)
    # Pitch class of each in-band bin, computed once.
    pitch = A4_MIDI + 12.0 * np.log2(np.where(band, freqs, A4_HZ) / A4_HZ)
    pitch_class = np.mod(np.rint(pitch).astype(int), 12)

    window = np.hanning(n_fft)
    chroma = np.zeros(12, dtype=np.float64)
    frames = 0
    for start in range(0, max(1, x.size - n_fft + 1), hop):
        frame = x[start : start + n_fft]
        if frame.size < n_fft:
            break
        mag = np.abs(np.fft.rfft(frame * window))
        contrib = mag * band
        np.add.at(chroma, pitch_class, contrib)
        frames += 1

    if frames == 0 or chroma.sum() <= 0.0:
        return np.zeros(12, dtype=np.float64)
    return chroma / chroma.sum()


def _correlate(chroma: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Pearson correlation of `chroma` against `profile` for all 12 rotations."""
    c = chroma - chroma.mean()
    p = profile - profile.mean()
    denom_c = np.sqrt((c**2).sum())
    denom_p = np.sqrt((p**2).sum())
    if denom_c == 0.0 or denom_p == 0.0:
        return np.zeros(12, dtype=np.float64)
    out = np.empty(12, dtype=np.float64)
    for tonic in range(12):
        rotated = np.roll(chroma, -tonic)
        rc = rotated - rotated.mean()
        out[tonic] = float((rc * p).sum() / (denom_c * denom_p))
    return out


def estimate_key(chroma: np.ndarray) -> KeyEstimate:
    """Best-fitting major/minor key for a 12-vector of pitch-class energy."""
    chroma = np.asarray(chroma, dtype=np.float64).ravel()
    if chroma.shape != (12,) or chroma.sum() <= 0.0:
        return KeyEstimate(tonic=0, mode="major", correlation=0.0, confidence=0.0)

    major = _correlate(chroma, _MAJOR_PROFILE)
    minor = _correlate(chroma, _MINOR_PROFILE)

    candidates: list[tuple[float, int, str]] = []
    for tonic in range(12):
        candidates.append((float(major[tonic]), tonic, "major"))
        candidates.append((float(minor[tonic]), tonic, "minor"))
    candidates.sort(key=lambda c: c[0], reverse=True)

    best_corr, best_tonic, best_mode = candidates[0]
    runner_up = candidates[1][0]
    alternatives = tuple(
        (f"{NOTE_NAMES[t]} {m}", corr) for corr, t, m in candidates[1:4]
    )
    return KeyEstimate(
        tonic=best_tonic,
        mode=best_mode,
        correlation=best_corr,
        confidence=best_corr - runner_up,
        alternatives=alternatives,
    )


def estimate_key_from_audio(mono: np.ndarray, sample_rate: int) -> KeyEstimate:
    return estimate_key(chroma_from_audio(mono, sample_rate))
