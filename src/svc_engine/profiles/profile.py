"""The range profile of a target voice.

Research section 4.3, step 2: every target voice carries a small profile the
pitch engine reads when it decides how far to shift a song. Phase 3 computes the
range half of it from a sample of the voice -- the comfortable band (p05..p95)
and the absolute limits -- in MIDI, the same units the song's range comes in, so
Phase 4 can compare the two directly.

The *quality-vs-shift* curve that research section 4.3 step 3 also puts in the
profile is deliberately not here: it is a measurement over a running conversion
model, which is Phase 4/5 work. `quality_vs_shift` is left absent, not faked.

The dataclass and its (de)serialisation are pure numpy and unit-tested; only
`compute_profile` runs an F0 model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svc_engine.analysis.midi import midi_to_note_name
from svc_engine.analysis.range import RangeStats, compute_range
from svc_engine.backends.base import AudioBuffer, DeviceHint
from svc_engine.backends.f0 import F0Extractor

__all__ = ["VoiceProfile", "compute_profile"]

PROFILE_VERSION = 1


@dataclass(frozen=True)
class VoiceProfile:
    """A target voice's usable range, in MIDI semitones."""

    name: str
    comfort_low: float   # p05
    comfort_high: float  # p95
    abs_low: float       # lowest voiced pitch measured
    abs_high: float      # highest voiced pitch measured
    median: float        # p50
    f0_method: str
    sample_seconds: float
    voiced_frames: int

    @property
    def comfortable_span(self) -> float:
        return self.comfort_high - self.comfort_low

    def to_dict(self) -> dict[str, object]:
        return {
            "version": PROFILE_VERSION,
            "name": self.name,
            "unit": "midi",
            "comfort_low": round(self.comfort_low, 3),
            "comfort_high": round(self.comfort_high, 3),
            "abs_low": round(self.abs_low, 3),
            "abs_high": round(self.abs_high, 3),
            "median": round(self.median, 3),
            "comfort_low_note": midi_to_note_name(self.comfort_low),
            "comfort_high_note": midi_to_note_name(self.comfort_high),
            "median_note": midi_to_note_name(self.median),
            "comfortable_span_semitones": round(self.comfortable_span, 3),
            "f0_method": self.f0_method,
            "sample_seconds": round(self.sample_seconds, 3),
            "voiced_frames": self.voiced_frames,
            # Reserved for Phase 4's measured penalty curve; absent, not guessed.
            "quality_vs_shift": None,
        }

    def save(self, path: Path | str) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return path

    @classmethod
    def from_dict(cls, data: dict) -> VoiceProfile:
        return cls(
            name=str(data.get("name") or ""),
            comfort_low=float(data["comfort_low"]),
            comfort_high=float(data["comfort_high"]),
            abs_low=float(data["abs_low"]),
            abs_high=float(data["abs_high"]),
            median=float(data["median"]),
            f0_method=str(data.get("f0_method") or ""),
            sample_seconds=float(data.get("sample_seconds") or 0.0),
            voiced_frames=int(data.get("voiced_frames") or 0),
        )

    @classmethod
    def load(cls, path: Path | str) -> VoiceProfile:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_range(
        cls, stats: RangeStats, name: str, f0_method: str, sample_seconds: float
    ) -> VoiceProfile:
        return cls(
            name=name,
            comfort_low=stats.p05,
            comfort_high=stats.p95,
            abs_low=stats.min_midi,
            abs_high=stats.max_midi,
            median=stats.p50,
            f0_method=f0_method,
            sample_seconds=sample_seconds,
            voiced_frames=stats.voiced_frames,
        )


def compute_profile(
    audio: AudioBuffer,
    extractor: F0Extractor,
    name: str,
    device: DeviceHint | None = None,
    hop_seconds: float = 0.01,
) -> VoiceProfile:
    """Measure a voice's range from an audio sample of it."""
    device = device or DeviceHint()
    curve = extractor.extract(audio, device, hop_seconds)
    f0_hz = np.asarray(curve.hz, dtype=np.float64).ravel()

    mono = audio.samples.mean(axis=0) if audio.samples.ndim == 2 else audio.samples
    from svc_engine.analysis.features import align_to, frame_rms

    energy = align_to(frame_rms(mono, audio.sample_rate, curve.hop_seconds), f0_hz.size)
    stats = compute_range(f0_hz, energy)
    if stats.voiced_frames == 0:
        raise ValueError("no voiced frames in the voice sample; cannot build a profile")
    return VoiceProfile.from_range(
        stats, name=name, f0_method=extractor.info().backend_id, sample_seconds=audio.seconds
    )
