"""Separation logic that does not need a model: ensemble, cleanup, profiles, OOM.

The backends themselves are exercised by `tools/bench_separation.py`, which
needs 900MB of weights and a few minutes. Everything here runs in milliseconds
against a fake backend, so the decision logic is covered on every commit.
"""

from __future__ import annotations

import numpy as np
import pytest

from svc_engine.audio.buffers import rms
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint
from svc_engine.backends.separation import (
    SeparationBackend,
    SeparationRequest,
    StemKind,
    Stems,
)
from svc_engine.compute import ComputeBackend, ResourcePlan
from svc_engine.compute.memory import measure_peak, reset_peak, suggested_segment_size
from svc_engine.compute.oom import is_oom_error, oom_ladder, run_with_oom_ladder
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources import load_registry
from svc_engine.separation import (
    PROFILES,
    CleanupStep,
    EnsembleMode,
    QualityLevel,
    VocalCleanup,
    combine_buffers,
    profile_for,
)


def buf(value: float, frames: int = 1000, channels: int = 2) -> AudioBuffer:
    return AudioBuffer(
        samples=np.full((channels, frames), value, dtype=np.float32), sample_rate=44100
    )


def noise(frames: int = 1000, seed: int = 0, gain: float = 0.3) -> AudioBuffer:
    rng = np.random.default_rng(seed)
    return AudioBuffer(
        samples=(rng.standard_normal((2, frames)) * gain).astype(np.float32),
        sample_rate=44100,
    )


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #

def test_median_rejects_a_single_model_outlier() -> None:
    """The reason median is the default: one bad model must not reach the output."""
    good, outlier = buf(0.2), buf(0.9)
    combined = combine_buffers([good, good, outlier], EnsembleMode.MEDIAN)
    assert np.allclose(combined.samples, 0.2)


def test_average_is_pulled_by_the_outlier_median_is_not() -> None:
    inputs = [buf(0.2), buf(0.2), buf(0.9)]
    assert combine_buffers(inputs, EnsembleMode.AVERAGE).samples[0, 0] > 0.4
    assert combine_buffers(inputs, EnsembleMode.MEDIAN).samples[0, 0] == pytest.approx(0.2)


def test_min_and_max_magnitude_pick_per_sample() -> None:
    a = AudioBuffer(samples=np.array([[0.1, -0.9]], dtype=np.float32), sample_rate=44100)
    b = AudioBuffer(samples=np.array([[-0.8, 0.2]], dtype=np.float32), sample_rate=44100)
    assert np.allclose(
        combine_buffers([a, b], EnsembleMode.MIN_MAG).samples, [[0.1, 0.2]]
    )
    assert np.allclose(
        combine_buffers([a, b], EnsembleMode.MAX_MAG).samples, [[-0.8, -0.9]]
    )


def test_a_single_input_is_returned_untouched() -> None:
    only = buf(0.3)
    assert combine_buffers([only], EnsembleMode.MEDIAN) is only


def test_combining_different_lengths_uses_the_shortest() -> None:
    combined = combine_buffers([buf(0.2, 1000), buf(0.2, 600)], EnsembleMode.AVERAGE)
    assert combined.frames == 600


def test_combine_stems_keeps_a_stem_only_one_model_produced() -> None:
    """Losing the ambience layer to an ensemble would be worse than not combining."""
    from svc_engine.separation import combine_stems

    a = Stems(parts={StemKind.VOCALS: buf(0.2), StemKind.AMBIENCE: buf(0.1)}, model_id="a")
    b = Stems(parts={StemKind.VOCALS: buf(0.2)}, model_id="b")
    combined = combine_stems([a, b], EnsembleMode.MEDIAN)
    assert StemKind.AMBIENCE in combined.parts


def test_combining_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="nothing to combine"):
        combine_buffers([], EnsembleMode.MEDIAN)


# --------------------------------------------------------------------------- #
# quality profiles
# --------------------------------------------------------------------------- #

def test_every_quality_level_has_a_profile() -> None:
    for level in QualityLevel:
        assert profile_for(level).level is level


def test_the_levels_are_ordered_by_the_work_they_do() -> None:
    """`overlap` is a coverage factor, so more of it is strictly more work."""
    fast, balanced, maximum = (PROFILES[level] for level in QualityLevel)
    assert fast.overlap < balanced.overlap < maximum.overlap
    assert len(maximum.models) > len(fast.models)
    assert not fast.is_ensemble and maximum.is_ensemble


def test_every_profile_names_models_that_exist_in_the_catalogue() -> None:
    registry = load_registry()
    for profile in PROFILES.values():
        for model_id in profile.models:
            assert model_id in registry, f"{profile.level.value} wants missing {model_id}"


def test_the_default_model_is_the_redistributable_one() -> None:
    registry = load_registry()
    primary = PROFILES[QualityLevel.BALANCED].models[0]
    assert registry.get(primary).license.is_redistributable


# --------------------------------------------------------------------------- #
# OOM ladder
# --------------------------------------------------------------------------- #

def test_the_ladder_only_ever_gets_cheaper() -> None:
    plan = ResourcePlan(segment_size=512, batch_size=4, overlap=4,
                        backend=ComputeBackend.CUDA)
    rungs = list(oom_ladder(plan))
    costs = [(r.segment_size, r.batch_size, r.overlap) for r in rungs]
    assert costs == sorted(costs, reverse=True)
    assert rungs[-1].backend is ComputeBackend.CPU, "the last rung must always work"


def test_a_cpu_plan_does_not_get_a_pointless_cpu_rung() -> None:
    rungs = list(ResourcePlan(segment_size=256, backend=ComputeBackend.CPU) and
                 oom_ladder(ResourcePlan(segment_size=256, backend=ComputeBackend.CPU)))
    assert all(r.backend is ComputeBackend.CPU for r in rungs)
    assert len(rungs) < 5


def test_success_on_the_first_rung_does_not_retry() -> None:
    seen: list[ResourcePlan] = []
    result = run_with_oom_ladder(
        ResourcePlan(segment_size=256), lambda plan: seen.append(plan) or "ok"
    )
    assert result == "ok"
    assert len(seen) == 1


def test_an_oom_backs_off_and_then_succeeds() -> None:
    seen: list[ResourcePlan] = []

    def flaky(plan: ResourcePlan) -> str:
        seen.append(plan)
        if len(seen) < 3:
            raise RuntimeError("CUDA error: out of memory")
        return "ok"

    assert run_with_oom_ladder(ResourcePlan(segment_size=512, batch_size=4), flaky) == "ok"
    assert seen[-1].segment_size < seen[0].segment_size


def test_a_non_memory_error_is_raised_immediately_not_retried() -> None:
    """Retrying a real bug four times just makes it take four times as long."""
    attempts = []

    def broken(plan: ResourcePlan) -> str:
        attempts.append(plan)
        raise ValueError("this is a real bug")

    with pytest.raises(ValueError, match="real bug"):
        run_with_oom_ladder(ResourcePlan(), broken)
    assert len(attempts) == 1


def test_exhausting_the_ladder_reports_a_hebrew_oom_error() -> None:
    def always_oom(plan: ResourcePlan) -> str:
        raise MemoryError("nope")

    with pytest.raises(EngineError) as caught:
        run_with_oom_ladder(ResourcePlan(segment_size=512), always_oom)
    assert caught.value.code is ErrorCode.GPU_OOM
    assert caught.value.user_message.what


@pytest.mark.parametrize(
    "exc",
    [
        MemoryError("x"),
        RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB"),
        RuntimeError("Native API failed: UR_RESULT_ERROR_OUT_OF_DEVICE_MEMORY"),
        RuntimeError("std::bad_alloc"),
    ],
)
def test_out_of_memory_is_recognised_across_backends(exc: BaseException) -> None:
    assert is_oom_error(exc)


def test_an_ordinary_error_is_not_mistaken_for_out_of_memory() -> None:
    assert not is_oom_error(FileNotFoundError("model.ckpt"))
    assert not is_oom_error(ValueError("bad shape"))


def test_cpu_window_size_does_not_follow_free_ram() -> None:
    """Otherwise the same job gives different output depending on machine load."""
    assert suggested_segment_size(DeviceHint(backend=ComputeBackend.CPU), 256) == 256


def test_suggested_window_never_exceeds_the_profile_it_was_given() -> None:
    """The profile is the ceiling; memory can only lower it."""
    for backend in ComputeBackend:
        assert suggested_segment_size(DeviceHint(backend=backend), 128) <= 128


def test_a_tight_vram_budget_lowers_the_window() -> None:
    hint = DeviceHint(backend=ComputeBackend.XPU, max_vram_mb=2048)
    assert suggested_segment_size(hint, 512) < 512


# --------------------------------------------------------------------------- #
# memory instrumentation
# --------------------------------------------------------------------------- #

def test_host_peak_memory_is_actually_readable() -> None:
    """This silently returned None until the Win32 handle was declared properly."""
    peak = measure_peak(DeviceHint(backend=ComputeBackend.CPU))
    assert peak.host_mb is not None
    assert peak.host_mb > 0
    assert peak.summary() != "not measurable"


def test_reset_peak_is_safe_on_a_machine_without_that_accelerator() -> None:
    for backend in ComputeBackend:
        reset_peak(DeviceHint(backend=backend))  # must never raise
        assert measure_peak(DeviceHint(backend=backend)).backend == backend.value


# --------------------------------------------------------------------------- #
# cleanup
# --------------------------------------------------------------------------- #

class FakeBackend:
    """Returns whatever it is told to, so cleanup decisions can be tested alone."""

    def __init__(self, responses: dict[str, dict[StemKind, AudioBuffer]]) -> None:
        self.responses = responses
        self.calls: list[str] = []
        self.unloaded = 0

    def info(self) -> BackendInfo:
        return BackendInfo(backend_id="fake", display_name_he="בדיקה", available=True)

    def list_models(self) -> list[str]:
        return list(self.responses)

    def separate(
        self, audio: AudioBuffer, request: SeparationRequest, device: DeviceHint
    ) -> Stems:
        self.calls.append(request.model_id)
        if request.model_id not in self.responses:
            raise EngineError(ErrorCode.MODEL_MISSING, request.model_id)
        return Stems(parts=self.responses[request.model_id], model_id=request.model_id)

    def unload(self) -> None:
        self.unloaded += 1


def test_the_fake_backend_satisfies_the_protocol() -> None:
    assert isinstance(FakeBackend({}), SeparationBackend)


def test_dereverb_keeps_the_removed_tail_because_phase_6_needs_it() -> None:
    registry = load_registry()
    vocals, ambience = noise(seed=1), noise(seed=2, gain=0.05)
    backend = FakeBackend(
        {"dereverb_anvuew": {StemKind.VOCALS: vocals, StemKind.AMBIENCE: ambience}}
    )
    result = VocalCleanup(backend, registry).run(
        noise(seed=3), (CleanupStep.DEREVERB,), DeviceHint()
    )
    assert result.applied == (CleanupStep.DEREVERB,)
    assert result.ambience is not None
    assert rms(result.ambience) > 0


def test_a_removed_layer_is_derived_when_the_model_does_not_return_it() -> None:
    registry = load_registry()
    original = noise(seed=4)
    kept = AudioBuffer(samples=original.samples * 0.6, sample_rate=original.sample_rate)
    backend = FakeBackend({"dereverb_anvuew": {StemKind.VOCALS: kept}})
    result = VocalCleanup(backend, registry).run(
        original, (CleanupStep.DEREVERB,), DeviceHint()
    )
    assert result.ambience is not None
    assert np.allclose(
        result.ambience.samples, original.samples - kept.samples, atol=1e-5
    )


def test_a_pass_that_would_erase_the_vocal_is_refused() -> None:
    """Shipping a silent vocal is worse than shipping an untreated one."""
    registry = load_registry()
    original = noise(seed=5)
    backend = FakeBackend(
        {"denoise_aufr33": {StemKind.VOCALS: buf(0.0, original.frames)}}
    )
    result = VocalCleanup(backend, registry).run(
        original, (CleanupStep.DENOISE,), DeviceHint()
    )
    assert result.applied == ()
    assert CleanupStep.DENOISE in result.skipped
    assert np.array_equal(result.vocals.samples, original.samples)


def test_a_failing_pass_does_not_lose_the_separation_that_worked() -> None:
    registry = load_registry()
    original = noise(seed=6)
    backend = FakeBackend({})  # every model missing
    result = VocalCleanup(backend, registry).run(
        original, (CleanupStep.DENOISE, CleanupStep.DEREVERB), DeviceHint()
    )
    assert result.applied == ()
    assert set(result.skipped) == {CleanupStep.DENOISE, CleanupStep.DEREVERB}
    assert np.array_equal(result.vocals.samples, original.samples)
    assert all(text for text in result.skipped.values()), "every skip needs a reason"


def test_passes_run_in_the_fixed_order_not_the_order_asked_for() -> None:
    registry = load_registry()
    original = noise(seed=7)
    kept = AudioBuffer(samples=original.samples * 0.8, sample_rate=original.sample_rate)
    backend = FakeBackend(
        {
            "denoise_aufr33": {StemKind.VOCALS: kept},
            "dereverb_anvuew": {StemKind.VOCALS: kept},
        }
    )
    result = VocalCleanup(backend, registry).run(
        original, (CleanupStep.DEREVERB, CleanupStep.DENOISE), DeviceHint()
    )
    assert result.applied == (CleanupStep.DENOISE, CleanupStep.DEREVERB)
    assert backend.calls == ["denoise_aufr33", "dereverb_anvuew"]


def test_licence_policy_can_switch_off_the_non_redistributable_passes() -> None:
    registry = load_registry()
    original = noise(seed=8)
    kept = AudioBuffer(samples=original.samples * 0.8, sample_rate=original.sample_rate)
    backend = FakeBackend({"dereverb_anvuew": {StemKind.VOCALS: kept}})
    result = VocalCleanup(backend, registry, allow_private_models=False).run(
        original, (CleanupStep.DEREVERB,), DeviceHint()
    )
    assert result.applied == ()
    assert backend.calls == [], "a blocked model must not even be loaded"
    assert CleanupStep.DEREVERB in result.skipped


def test_the_karaoke_split_produces_a_lead_and_a_backing_stem() -> None:
    registry = load_registry()
    original = noise(seed=9)
    lead = AudioBuffer(samples=original.samples * 0.7, sample_rate=original.sample_rate)
    backing = AudioBuffer(samples=original.samples * 0.3, sample_rate=original.sample_rate)
    backend = FakeBackend(
        {"karaoke_aufr33_viperx": {StemKind.LEAD: lead, StemKind.BACKING: backing}}
    )
    result = VocalCleanup(backend, registry).run(
        original, (CleanupStep.KARAOKE,), DeviceHint()
    )
    assert result.applied == (CleanupStep.KARAOKE,)
    assert StemKind.LEAD in result.parts
    assert StemKind.BACKING in result.parts
    assert np.allclose(result.vocals.samples, lead.samples)


def test_cleanup_releases_the_device_when_it_finishes() -> None:
    backend = FakeBackend({})
    VocalCleanup(backend, load_registry()).run(noise(), (), DeviceHint())
    assert backend.unloaded == 1
