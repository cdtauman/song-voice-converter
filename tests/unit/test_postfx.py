"""Phase 6 DSP, mastering and orchestration tests."""

from __future__ import annotations

import math
import shutil
from pathlib import Path

import numpy as np
import pytest

from svc_engine.backends.base import AudioBuffer, BackendInfo, F0Curve
from svc_engine.postfx import (
    AmbienceStrategy,
    MixConfig,
    PostFxConfig,
    PostFxPipeline,
    apply_ambience,
    check_melody,
    clean_output_path,
    deess,
    envelope_correlation,
    export_audio,
    match_envelope,
    mix_and_master,
    repair_artifacts,
)

SR = 44100


def tone(frequency: float, seconds: float = 1.0, gain: float = 0.2,
         channels: int = 1) -> AudioBuffer:
    t = np.arange(int(SR * seconds)) / SR
    wave = gain * np.sin(2.0 * np.pi * frequency * t)
    return AudioBuffer(
        samples=np.repeat(wave[None, :], channels, axis=0).astype(np.float32),
        sample_rate=SR,
    )


class FakeShifter:
    def __init__(self) -> None:
        self.shifts: list[float] = []

    def info(self) -> BackendInfo:
        return BackendInfo("fake", "בדיקה", True)

    def shift(self, audio: AudioBuffer, semitones: float) -> AudioBuffer:
        self.shifts.append(semitones)
        return audio


def test_repair_removes_an_isolated_click_and_nonfinite_sample() -> None:
    audio = tone(220.0)
    broken = audio.samples.copy()
    broken[0, 1000] = 1.0
    broken[0, 2000] = np.nan
    repaired, report = repair_artifacts(AudioBuffer(broken, SR))
    assert report.repaired_samples >= 1
    assert report.nonfinite_samples == 1
    assert np.isfinite(repaired.samples).all()
    assert abs(float(repaired.samples[0, 1000])) < 0.3
    assert repaired.frames == audio.frames


def test_melody_check_measures_the_requested_transposition() -> None:
    source = F0Curve(np.array([220.0, 246.94, 261.63, 293.66]), 0.01)
    exact = F0Curve(source.hz * 2.0, 0.01)
    report = check_melody(source, exact, 12)
    assert report.passed
    assert report.correlation == pytest.approx(1.0)
    assert report.median_error_cents == pytest.approx(0.0, abs=0.01)


def test_melody_correction_is_fine_and_bounded() -> None:
    source = F0Curve(np.array([220.0, 246.94, 261.63, 293.66]), 0.01)
    sharp = F0Curve(source.hz * 2.0 ** (0.8 / 12.0), 0.01)
    report = check_melody(source, sharp, 0)
    shifter = FakeShifter()
    from svc_engine.postfx.melody_check import correct_global_pitch

    _, corrected = correct_global_pitch(tone(220), report, shifter)
    assert shifter.shifts == [pytest.approx(-0.5)]
    assert corrected.correction_semitones == pytest.approx(-0.5)


def test_envelope_matching_improves_dynamic_correlation() -> None:
    reference = tone(220, seconds=2.0)
    shape = np.linspace(0.15, 1.0, reference.frames, dtype=np.float32)
    reference = AudioBuffer(reference.samples * shape, SR)
    flat = tone(220, seconds=2.0, gain=0.15)
    matched, report = match_envelope(reference, flat, rms_mix_rate=0.0)
    assert report.correlation_after is not None
    assert report.correlation_after > 0.95
    assert envelope_correlation(reference, matched) == pytest.approx(
        report.correlation_after
    )


def test_deesser_reduces_a_sibilant_high_band_without_changing_length() -> None:
    low = tone(300, gain=0.08)
    high = tone(9000, gain=0.35)
    audio = AudioBuffer(low.samples + high.samples, SR)
    result, report = deess(audio)
    assert result.frames == audio.frames
    assert report.maximum_reduction_db > 1.0
    # The alternating 9 kHz component loses energy.
    assert float(np.std(result.samples - low.samples)) < float(
        np.std(audio.samples - low.samples)
    )


def test_all_ambience_strategies_are_length_exact() -> None:
    dry = tone(220, seconds=1.2)
    ambience = AudioBuffer((tone(330, seconds=1.2, gain=0.03).samples), SR)
    levels = {}
    for strategy in AmbienceStrategy:
        result = apply_ambience(dry, dry, ambience, strategy)
        assert result.vocal.frames == dry.frames
        assert result.measurement.available
        levels[strategy] = float(np.sqrt(np.mean(result.vocal.samples ** 2)))
    assert levels[AmbienceStrategy.ORIGINAL] != levels[AmbienceStrategy.PARAMETRIC]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_ffmpeg_master_hits_lufs_and_does_not_clip(tmp_path: Path) -> None:
    vocal = tone(220, seconds=3.0, gain=0.18)
    playback = tone(110, seconds=3.0, gain=0.12, channels=2)
    mastered, report = mix_and_master(
        vocal, playback, MixConfig(target_lufs=-14.0), work_dir=tmp_path
    )
    assert mastered.frames == playback.frames
    assert mastered.channels == playback.channels
    assert abs(report.integrated_lufs - report.target_lufs) <= 0.5
    assert not report.clipped
    assert report.peak_dbfs <= 0.0
    assert report.used_filters == ("amix", "acompressor", "loudnorm", "alimiter")


def test_output_names_are_windows_safe() -> None:
    clean = clean_output_path(Path("out") / "bad:name?.WAV")
    assert clean.name == "bad-name-.wav"
    assert clean_output_path("CON.mp3").name == "CON-cover.mp3"
    assert clean_output_path("cover.unknown").suffix == ".wav"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_export_contract_is_wav24_or_mp3_320(tmp_path: Path) -> None:
    wav = export_audio(tone(220), tmp_path / "voice:cover.wav")
    mp3 = export_audio(tone(220), tmp_path / "voice:cover.mp3")
    assert wav.path.exists() and wav.quality == "24-bit PCM"
    assert mp3.path.exists() and mp3.quality == "320 kbps"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is required")
def test_complete_postfx_chain_reports_every_stage(tmp_path: Path) -> None:
    source = tone(220, seconds=2.0)
    converted = tone(110, seconds=2.0)
    playback = tone(55, seconds=2.0, gain=0.08, channels=2)
    f0_source = F0Curve(np.linspace(210, 230, 200), 0.01)
    f0_out = F0Curve(np.linspace(105, 115, 200), 0.01)
    outcome = PostFxPipeline(
        PostFxConfig(ambience_strategy=AmbienceStrategy.PARAMETRIC), tmp_path
    ).run(
        original_vocal=source,
        converted_vocal=converted,
        playback=playback,
        original_ambience=tone(440, seconds=2.0, gain=0.02),
        reference_f0=f0_source,
        converted_f0=f0_out,
        semitones=-12,
        shifter=FakeShifter(),
    )
    assert outcome.cover.frames == playback.frames
    assert abs(outcome.report.mix.integrated_lufs + 14.0) <= 0.5
    payload = outcome.report.to_dict()
    assert set(payload) == {"repair", "melody", "envelope", "deess", "ambience", "mix"}
    assert math.isfinite(outcome.report.mix.integrated_lufs)
