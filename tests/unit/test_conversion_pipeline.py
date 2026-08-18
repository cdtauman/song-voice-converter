"""The full song-to-cover orchestration, driven with fakes (torch-free)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionParams, VoiceHandle
from svc_engine.backends.separation import StemKind
from svc_engine.config import paths
from svc_engine.conversion import ConversionPipeline, mix_cover, params_for_voice
from svc_engine.pitch import PlaybackStrategy
from svc_engine.profiles import VoiceProfile

SR = 44100


class _FakeBackend:
    def __init__(self) -> None:
        self.loaded = self.unloaded = False
        self.seen_params: ConversionParams | None = None

    def info(self) -> BackendInfo:
        return BackendInfo("fake", "fake", available=True)

    def load(self, voice: VoiceHandle, device: DeviceHint) -> None:
        self.loaded = True

    def convert(self, audio: AudioBuffer, f0: F0Curve, params: ConversionParams) -> AudioBuffer:
        self.seen_params = params
        return AudioBuffer(samples=audio.samples.copy(), sample_rate=audio.sample_rate)

    def unload(self) -> None:
        self.unloaded = True


class _FakeShifter:
    def __init__(self) -> None:
        self.shift_calls = 0

    def info(self) -> BackendInfo:
        return BackendInfo("fakeshift", "fakeshift", available=True)

    def shift(self, audio: AudioBuffer, semitones: float) -> AudioBuffer:
        self.shift_calls += 1
        return AudioBuffer(samples=audio.samples.copy(), sample_rate=audio.sample_rate)


class _FakeF0:
    def info(self) -> BackendInfo:
        return BackendInfo("f0", "f0", available=True)

    def extract(self, audio: AudioBuffer, device: DeviceHint, hop: float = 0.01) -> F0Curve:
        n = max(1, audio.frames // int(hop * audio.sample_rate))
        return F0Curve(hz=np.full(n, 330.0), hop_seconds=hop)

    def unload(self) -> None:
        pass


def _vocals(seconds: float = 2.0) -> AudioBuffer:
    n = int(seconds * SR)
    t = np.arange(n)
    wave = 0.3 * np.sin(2 * np.pi * 330.0 * t / SR)
    return AudioBuffer(samples=wave.reshape(1, -1).astype(np.float32), sample_rate=SR)


def _instrumental(frames: int) -> AudioBuffer:
    rng = np.random.default_rng(3)
    return AudioBuffer(
        samples=(0.1 * rng.standard_normal((2, frames))).astype(np.float32),
        sample_rate=SR,
    )


def _profile(median: float = 45.0) -> VoiceProfile:
    return VoiceProfile(
        name="baritone",
        comfort_low=median - 7,
        comfort_high=median + 7,
        abs_low=median - 12,
        abs_high=median + 12,
        median=median,
        f0_method="fake",
        sample_seconds=10.0,
        voiced_frames=500,
    )


# --- pure helpers ---------------------------------------------------------- #

def test_mix_cover_is_song_length_and_sums() -> None:
    vocal = _vocals(1.0)
    inst = _instrumental(int(1.5 * SR))  # different length on purpose
    cover = mix_cover(vocal, inst)
    assert cover.frames == inst.frames
    assert cover.channels == inst.channels


def test_params_for_voice_fills_in_shift() -> None:
    base = ConversionParams(index_rate=0.6, protect=0.4, rms_mix_rate=0.2)
    out = params_for_voice(base, semitones=-12)
    assert out.semitones == -12
    assert out.index_rate == pytest.approx(0.6)
    assert out.protect == pytest.approx(0.4)


# --- render_cover orchestration -------------------------------------------- #

def test_render_cover_wires_decision_convert_shift_mix(tmp_path: Path) -> None:
    p = paths(override_root=tmp_path)
    backend, shifter = _FakeBackend(), _FakeShifter()
    pipe = ConversionPipeline(paths=p, conversion_backend=backend, shifter=shifter)

    vocals = _vocals(2.0)
    inst = _instrumental(vocals.frames)
    f0 = F0Curve(hz=np.full(200, 660.0), hop_seconds=0.01)  # a high song
    profile = _profile(median=45.0)  # a low target voice -> expect a downward shift

    cover, decision = pipe.render_cover(
        vocals, f0, inst, profile,
        VoiceHandle("v", tmp_path), ConversionParams(),
        strategy=PlaybackStrategy.WHOLE,
    )

    assert backend.loaded and backend.unloaded  # VRAM released after the run
    assert cover.frames == inst.frames
    # The pitch engine's shift reached the conversion params.
    assert backend.seen_params is not None
    assert backend.seen_params.semitones == decision.best.semitones
    # A high song into a low voice shifts down.
    assert decision.best.semitones < 0


def test_render_cover_reports_progress_to_completion(tmp_path: Path) -> None:
    p = paths(override_root=tmp_path)
    pipe = ConversionPipeline(
        paths=p, conversion_backend=_FakeBackend(), shifter=_FakeShifter()
    )
    seen: list[float] = []
    pipe.render_cover(
        _vocals(1.5), F0Curve(hz=np.full(150, 330.0), hop_seconds=0.01),
        _instrumental(int(1.5 * SR)), _profile(),
        VoiceHandle("v", tmp_path), ConversionParams(),
        on_progress=lambda frac, msg: seen.append(frac),
    )
    assert seen and seen[-1] == pytest.approx(1.0)
    assert seen == sorted(seen)  # monotonic progress


# --- run end to end (fakes for every heavy stage) -------------------------- #

def _library_with_voice(tmp_path: Path):  # type: ignore[no-untyped-def]
    from svc_engine.voices import VoiceLibrary
    from svc_engine.voices.manifest import VoiceManifest, VoiceSource

    p = paths(override_root=tmp_path)
    p.ensure()
    lib = VoiceLibrary(p)
    vdir = lib.voice_dir("yossi")
    vdir.mkdir(parents=True)
    (vdir / "model.pth").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    _profile(45.0).save(vdir / "profile.json")
    VoiceManifest(
        "yossi", "יוסי", VoiceSource.IMPORTED, consent_confirmed=True
    ).save(vdir)
    lib.refresh_health("yossi")
    return p, lib


def test_run_end_to_end_with_fakes(tmp_path: Path) -> None:
    p, lib = _library_with_voice(tmp_path)
    backend, shifter = _FakeBackend(), _FakeShifter()
    pipe = ConversionPipeline(
        paths=p, library=lib, conversion_backend=backend, shifter=shifter
    )

    vocals = _vocals(2.0)
    stems = {
        StemKind.VOCALS: vocals,
        StemKind.INSTRUMENTAL: _instrumental(vocals.frames),
    }

    outcome = pipe.run(
        tmp_path / "song.mp3",  # never opened -- separation is injected
        "yossi",
        f0_extractor=_FakeF0(),
        device=DeviceHint(),
        separate=lambda song: (vocals, stems),
        strategy=PlaybackStrategy.WHOLE,
    )

    assert outcome.voice_id == "yossi"
    assert outcome.cover.frames == vocals.frames
    assert backend.loaded and backend.unloaded
    assert "separate" in outcome.timings and "convert" in outcome.timings
    assert outcome.summary_he()


def test_run_refuses_voice_without_profile(tmp_path: Path) -> None:
    from svc_engine.errors import EngineError, ErrorCode
    from svc_engine.voices import VoiceLibrary
    from svc_engine.voices.manifest import VoiceManifest, VoiceSource

    p = paths(override_root=tmp_path)
    p.ensure()
    lib = VoiceLibrary(p)
    vdir = lib.voice_dir("noprof")
    vdir.mkdir(parents=True)
    (vdir / "model.pth").write_bytes(b"PK\x03\x04" + b"\x00" * 64)
    VoiceManifest("noprof", "N", VoiceSource.IMPORTED, consent_confirmed=True).save(vdir)
    lib.refresh_health("noprof")

    pipe = ConversionPipeline(
        paths=p, library=lib, conversion_backend=_FakeBackend(), shifter=_FakeShifter()
    )
    with pytest.raises(EngineError) as exc:
        pipe.run(
            tmp_path / "s.mp3", "noprof", f0_extractor=_FakeF0(), device=DeviceHint(),
            separate=lambda song: (_vocals(1.0), {StemKind.INSTRUMENTAL: _instrumental(SR)}),
        )
    assert exc.value.code is ErrorCode.VOICE_CORRUPT
