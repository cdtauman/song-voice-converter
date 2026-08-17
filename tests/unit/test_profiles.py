"""Target-voice range profile: the math and the file format, without an F0 model."""

from __future__ import annotations

import numpy as np
import pytest

from svc_engine.analysis.range import compute_range
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve
from svc_engine.profiles import VoiceProfile, compute_profile


def test_profile_from_range_carries_the_percentiles() -> None:
    hz = np.concatenate([np.full(50, 220.0), np.full(50, 440.0)])  # A3..A4
    stats = compute_range(hz)
    profile = VoiceProfile.from_range(stats, name="yossi", f0_method="fcpe", sample_seconds=1.0)
    assert profile.comfort_low == pytest.approx(stats.p05)
    assert profile.comfort_high == pytest.approx(stats.p95)
    assert profile.abs_low == pytest.approx(57.0, abs=0.1)   # A3
    assert profile.abs_high == pytest.approx(69.0, abs=0.1)  # A4
    assert profile.comfortable_span > 0.0


def test_profile_roundtrips_through_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    stats = compute_range(np.full(200, 330.0))
    profile = VoiceProfile.from_range(stats, name="dana", f0_method="rmvpe", sample_seconds=2.0)
    path = profile.save(tmp_path / "dana.json")

    loaded = VoiceProfile.load(path)
    assert loaded.name == "dana"
    assert loaded.f0_method == "rmvpe"
    # The JSON stores 3-decimal MIDI on purpose; the roundtrip is exact to that.
    assert loaded.median == pytest.approx(profile.median, abs=1e-3)
    assert loaded.comfort_low == pytest.approx(profile.comfort_low, abs=1e-3)


def test_profile_dict_has_note_names_and_reserved_quality_curve() -> None:
    stats = compute_range(np.full(100, 440.0))
    d = VoiceProfile.from_range(stats, "x", "fcpe", 1.0).to_dict()
    assert d["median_note"] == "A4"
    assert d["unit"] == "midi"
    # Phase 4 fills this in; Phase 3 leaves it explicitly empty rather than faked.
    assert d["quality_vs_shift"] is None


class _FakeExtractor:
    def __init__(self, hz: float) -> None:
        self.hz = hz

    def info(self) -> BackendInfo:
        return BackendInfo(backend_id="fake", display_name_he="בדיקה", available=True)

    def extract(
        self, audio: AudioBuffer, device: DeviceHint, hop_seconds: float = 0.01
    ) -> F0Curve:
        n = int(round(audio.seconds / hop_seconds))
        return F0Curve(hz=np.full(n, self.hz), hop_seconds=hop_seconds)

    def unload(self) -> None:
        pass


def _tone(hz: float, seconds: float, sr: int = 44100) -> AudioBuffer:
    t = np.arange(int(sr * seconds)) / sr
    wave = (0.4 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    return AudioBuffer(samples=wave[None, :], sample_rate=sr)


def test_compute_profile_measures_the_sample() -> None:
    profile = compute_profile(_tone(440.0, 3.0), _FakeExtractor(440.0), name="a4")
    assert profile.name == "a4"
    assert profile.median == pytest.approx(69.0, abs=0.2)
    assert profile.f0_method == "fake"


def test_compute_profile_rejects_a_silent_sample() -> None:
    silence = AudioBuffer(samples=np.zeros((1, 44100), dtype=np.float32), sample_rate=44100)
    with pytest.raises(ValueError):
        compute_profile(silence, _FakeExtractor(0.0), name="silent")
