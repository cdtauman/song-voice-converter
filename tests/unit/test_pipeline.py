"""The separation pipeline's orchestration, against a fake backend.

No weights, no GPU: what is under test is the sequencing and the decisions --
which models get chosen, what happens when one is unavailable, what the user is
told, and what the pipeline refuses to hand back.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

from svc_engine.audio import io as audio_io
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint
from svc_engine.backends.separation import SeparationRequest, StemKind, Stems
from svc_engine.config import Paths
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.separation import Progress, QualityLevel, SeparationPipeline

pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"
)


def make_paths(tmp_path: Path) -> Paths:
    paths = Paths(
        root=tmp_path,
        models=tmp_path / "models",
        voices=tmp_path / "voices",
        projects=tmp_path / "projects",
        cache=tmp_path / "cache",
        logs=tmp_path / "logs",
        work=tmp_path / "work",
        db=tmp_path / "db.sqlite",
    )
    paths.ensure()
    return paths


def write_mix(tmp_path: Path, seconds: float = 0.5) -> Path:
    rate = 44100
    t = np.arange(int(rate * seconds), dtype=np.float32) / rate
    mono = (0.3 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)[None, :]
    buffer = AudioBuffer(samples=np.repeat(mono, 2, axis=0), sample_rate=rate)
    return audio_io.save_wav(buffer, tmp_path / "mix.wav")


class SplittingBackend:
    """Splits any input into two halves by gain. Deterministic and instant."""

    def __init__(self, vocal_gain: float = 0.4) -> None:
        self.vocal_gain = vocal_gain
        self.requests: list[SeparationRequest] = []
        self.unloaded = 0

    def info(self) -> BackendInfo:
        return BackendInfo(backend_id="fake", display_name_he="בדיקה", available=True)

    def list_models(self) -> list[str]:
        return []

    def separate(
        self, audio: AudioBuffer, request: SeparationRequest, device: DeviceHint
    ) -> Stems:
        self.requests.append(request)
        vocals = AudioBuffer(
            samples=audio.samples * self.vocal_gain, sample_rate=audio.sample_rate
        )
        rest = AudioBuffer(
            samples=audio.samples * (1.0 - self.vocal_gain),
            sample_rate=audio.sample_rate,
        )
        return Stems(
            parts={StemKind.VOCALS: vocals, StemKind.INSTRUMENTAL: rest},
            model_id=request.model_id,
        )

    def unload(self) -> None:
        self.unloaded += 1


def build(tmp_path: Path, backend: object, **kwargs: object) -> SeparationPipeline:
    return SeparationPipeline(
        paths=make_paths(tmp_path), backend=backend, **kwargs  # type: ignore[arg-type]
    )


def test_a_balanced_run_produces_both_stems_at_the_input_length(tmp_path: Path) -> None:
    source = write_mix(tmp_path)
    outcome = build(tmp_path, SplittingBackend()).run(source, QualityLevel.BALANCED)

    assert set(outcome.stems) == {StemKind.VOCALS, StemKind.INSTRUMENTAL}
    frames = {b.frames for b in outcome.stems.values()}
    assert len(frames) == 1, "every stem must come back the same length"


def test_maximum_quality_runs_every_model_in_the_profile(tmp_path: Path) -> None:
    backend = SplittingBackend()
    outcome = build(tmp_path, backend).run(write_mix(tmp_path), QualityLevel.MAX)

    assert len(backend.requests) == 2
    assert len(outcome.model_ids) == 2
    assert "ensemble" in outcome.timings


def test_the_quality_level_reaches_the_backend_as_real_parameters(
    tmp_path: Path,
) -> None:
    backend = SplittingBackend()
    build(tmp_path, backend).run(write_mix(tmp_path), QualityLevel.FAST)
    fast = backend.requests[0].overlap

    backend = SplittingBackend()
    build(tmp_path, backend).run(write_mix(tmp_path), QualityLevel.MAX)
    assert backend.requests[0].overlap > fast


def test_progress_is_reported_from_zero_to_one_in_hebrew(tmp_path: Path) -> None:
    seen: list[Progress] = []
    build(tmp_path, SplittingBackend()).run(
        write_mix(tmp_path), QualityLevel.FAST, on_progress=seen.append
    )

    assert seen[0].fraction == 0.0
    assert seen[-1].fraction == 1.0
    assert all(p.message_he for p in seen)
    assert all(
        not any(ch.isascii() and ch.isalpha() for ch in p.message_he) for p in seen
    ), "progress text is user-facing and must not leak technical English"


def test_an_instrumental_track_is_reported_not_silently_converted(
    tmp_path: Path,
) -> None:
    """Otherwise the user waits through the whole pipeline to be handed silence."""
    with pytest.raises(EngineError) as caught:
        build(tmp_path, SplittingBackend(vocal_gain=0.0)).run(write_mix(tmp_path))
    assert caught.value.code is ErrorCode.NO_VOCALS
    assert caught.value.user_message.action


def test_licence_policy_can_reduce_an_ensemble_and_says_so(tmp_path: Path) -> None:
    backend = SplittingBackend()
    outcome = build(tmp_path, backend, allow_private_models=False).run(
        write_mix(tmp_path), QualityLevel.MAX
    )

    assert len(backend.requests) == 1, "the unlicensed partner model must be dropped"
    assert outcome.notes_he, "dropping a model silently would be worse than dropping it"


def test_a_level_left_with_no_usable_model_fails_clearly(tmp_path: Path) -> None:
    pipeline = build(tmp_path, SplittingBackend(), allow_private_models=False)
    # Pretend even the MIT model became non-redistributable.
    for spec in pipeline.registry.models.values():
        object.__setattr__(spec.license, "spdx", None)

    with pytest.raises(EngineError) as caught:
        pipeline.run(write_mix(tmp_path), QualityLevel.BALANCED)
    assert caught.value.code is ErrorCode.MODEL_MISSING


def test_write_produces_one_named_file_per_stem(tmp_path: Path) -> None:
    pipeline = build(tmp_path, SplittingBackend())
    outcome = pipeline.run(write_mix(tmp_path), QualityLevel.FAST)
    written = pipeline.write(outcome, tmp_path / "out")

    assert {p.name for p in written.values()} == {"vocals.wav", "instrumental.wav"}
    for path in written.values():
        assert audio_io.probe(path).channels == 2


def test_the_device_is_released_after_a_run(tmp_path: Path) -> None:
    backend = SplittingBackend()
    build(tmp_path, backend).run(write_mix(tmp_path), QualityLevel.FAST)
    assert backend.unloaded >= 1


def test_timings_are_recorded_for_every_step(tmp_path: Path) -> None:
    outcome = build(tmp_path, SplittingBackend()).run(
        write_mix(tmp_path), QualityLevel.BALANCED
    )
    assert "load" in outcome.timings
    assert any(k.startswith("separate:") for k in outcome.timings)
    assert outcome.total_seconds > 0
    assert outcome.realtime_factor is not None
    assert outcome.summary_he()


def test_only_the_requested_window_is_processed(tmp_path: Path) -> None:
    source = write_mix(tmp_path, seconds=2.0)
    outcome = build(tmp_path, SplittingBackend()).run(
        source, QualityLevel.FAST, duration_seconds=0.5
    )
    assert outcome.seconds_of_audio == pytest.approx(0.5, abs=0.05)
