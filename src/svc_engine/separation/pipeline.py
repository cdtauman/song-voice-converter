"""The separation stage, end to end.

Takes a file the user dropped on the window and returns labelled stems in
memory, having picked a device, downloaded whatever was missing, run one or
more models, combined them, optionally cleaned the vocal, and survived running
out of GPU memory.

Everything it reports is measured on the run that just happened -- device,
per-step seconds, which fallback rungs were used, which cleanup passes were
skipped and why. Phase 7 turns those steps into cached job nodes; the shape of
this module is what it will cache.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from svc_engine.audio import io as audio_io
from svc_engine.audio.buffers import fit_length, is_silent
from svc_engine.backends.base import AudioBuffer, DeviceHint
from svc_engine.backends.separation import (
    SeparationBackend,
    SeparationRequest,
    StemKind,
    Stems,
)
from svc_engine.compute import (
    Component,
    ComputeBackend,
    DeviceInfo,
    DeviceManager,
    OomStep,
    ResourcePlan,
    SupportMatrix,
    load_matrix,
    run_with_oom_ladder,
)
from svc_engine.compute.memory import (
    PeakMemory,
    measure_peak,
    reset_peak,
    suggested_segment_size,
)
from svc_engine.config import Paths
from svc_engine.config import paths as default_paths
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources import DownloadManager, ModelRegistry, load_registry
from svc_engine.separation.backends import AudioSeparatorBackend, PymssBackend
from svc_engine.separation.cleanup import CleanupResult, VocalCleanup
from svc_engine.separation.ensemble import combine_stems
from svc_engine.separation.quality import (
    CleanupStep,
    QualityLevel,
    SeparationProfile,
    profile_for,
)

__all__ = ["SeparationOutcome", "SeparationPipeline", "Progress", "ProgressHook"]

log = logging.getLogger(__name__)

#: What the user reads while each part runs. No model names, no percentages of
#: VRAM -- docs/architecture.md section 7.1.
_MESSAGE_LOAD = "קוראים את השיר…"
_MESSAGE_SEPARATE = "מפרידים את השירה מהמוזיקה…"
_MESSAGE_COMBINE = "משלבים את התוצאות…"
_MESSAGE_DONE = "ההפרדה הושלמה."


@dataclass(frozen=True)
class Progress:
    fraction: float
    message_he: str


ProgressHook = Callable[[Progress], None]


@dataclass(frozen=True)
class SeparationOutcome:
    stems: dict[StemKind, AudioBuffer]
    profile: SeparationProfile
    model_ids: tuple[str, ...]
    device: DeviceInfo
    source: Path
    seconds_of_audio: float
    timings: dict[str, float] = field(default_factory=dict)
    cleanup: CleanupResult | None = None
    fallback_steps: tuple[str, ...] = ()
    notes_he: tuple[str, ...] = ()
    peak_memory: PeakMemory | None = None

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    @property
    def realtime_factor(self) -> float | None:
        """Seconds of processing per second of audio. Lower is better."""
        if not self.seconds_of_audio:
            return None
        return self.total_seconds / self.seconds_of_audio

    def summary_he(self) -> str:
        rtf = self.realtime_factor
        speed = f" (פי {rtf:.1f} מאורך השיר)" if rtf else ""
        return (
            f"{self.profile.display_name_he} · {self.device.label_he} · "
            f"{self.total_seconds:.0f} שניות{speed}"
        )


class SeparationPipeline:
    """Owns the separation stage: device choice, models, retries, cleanup."""

    def __init__(
        self,
        paths: Paths | None = None,
        registry: ModelRegistry | None = None,
        backend: SeparationBackend | None = None,
        matrix: SupportMatrix | None = None,
        manager: DeviceManager | None = None,
        allow_downloads: bool = True,
        allow_private_models: bool = True,
    ) -> None:
        self.paths = paths or default_paths()
        self.paths.ensure()
        self.registry = registry or load_registry()
        self.matrix = matrix or load_matrix()
        self.manager = manager or DeviceManager()
        self.allow_private_models = allow_private_models
        self.downloader = DownloadManager(
            self.paths.models, allow_downloads=allow_downloads
        )
        self.backend = backend or AudioSeparatorBackend(
            models_dir=self.paths.models,
            registry=self.registry,
            downloader=self.downloader,
            work_dir=self.paths.work,
        )

    # -- public API --------------------------------------------------------- #

    def alternative_backend(self) -> SeparationBackend:
        """The independent second source, for when a model cannot be fetched."""
        return PymssBackend(models_dir=self.paths.models, work_dir=self.paths.work)

    def _implementation_id(self, model_id: str) -> str:
        """Stable proof key for one backend/checkpoint pair."""
        return f"{self.backend.info().backend_id}:{model_id}"

    def _device_for_model(self, model_id: str) -> DeviceInfo:
        """Choose acceleration only when this exact model was proven end to end.

        Compatibility evidence for a sibling model must never authorize this
        checkpoint. Old injected matrices used by tests may contain generic proof
        only; they retain their old behavior until implementation-specific proof
        exists in that matrix.
        """
        support = self.matrix.get(Component.SEPARATION)
        if support.implementation_proofs:
            return self.matrix.device_for_implementation(
                Component.SEPARATION,
                self._implementation_id(model_id),
                self.manager,
            )
        return self.matrix.device_for(Component.SEPARATION, self.manager)

    def device(self) -> DeviceInfo:
        """Device for the primary production separation model."""
        if "sep_melband_kim" in self.registry:
            return self._device_for_model("sep_melband_kim")
        return self.matrix.device_for(Component.SEPARATION, self.manager)

    def run(
        self,
        source: Path | str,
        level: QualityLevel | str = QualityLevel.BALANCED,
        cleanup: tuple[CleanupStep, ...] = (),
        on_progress: ProgressHook | None = None,
        start_seconds: float | None = None,
        duration_seconds: float | None = None,
    ) -> SeparationOutcome:
        profile = profile_for(level)
        source = Path(source)
        timings: dict[str, float] = {}
        notes: list[str] = []
        fallbacks: list[str] = []

        def report(fraction: float, message: str) -> None:
            if on_progress is not None:
                on_progress(Progress(fraction=fraction, message_he=message))

        report(0.0, _MESSAGE_LOAD)
        started = time.perf_counter()
        mix = audio_io.load_audio(
            source,
            start_seconds=start_seconds,
            duration_seconds=duration_seconds,
        )
        timings["load"] = time.perf_counter() - started

        models = self._resolve_models(profile, notes)
        self.downloader.check_space_for(models)

        # The outcome's device describes the primary production model. Ensemble
        # members are routed independently below and may legitimately use CPU.
        device = self._device_for_model(models[0].id)
        primary_hint = DeviceHint.from_device(device)
        reset_peak(primary_hint)

        steps = tuple(dict.fromkeys(cleanup)) + profile.cleanup
        # Separation is the bulk of the work; cleanup passes are each about as
        # expensive as one separation pass, so weight the bar accordingly.
        total_units = len(models) + len(steps)
        done_units = 0

        results: list[Stems] = []
        model_backends: list[tuple[str, str]] = []
        for spec in models:
            report(done_units / total_units * 0.95, _MESSAGE_SEPARATE)
            model_device = self._device_for_model(spec.id)
            model_hint = DeviceHint.from_device(model_device)
            model_backends.append((spec.id, model_device.backend.value))
            started = time.perf_counter()
            stems, used = self._separate_one(mix, spec.id, profile, model_hint)
            timings[f"separate:{spec.id}"] = time.perf_counter() - started
            fallbacks.extend(used)
            results.append(stems)
            done_units += 1

        distinct_backends = {backend for _model, backend in model_backends}
        if len(distinct_backends) > 1:
            detail = ", ".join(f"{model}: {backend}" for model, backend in model_backends)
            notes.append(f"מודלי ההפרדה רצו במסלולים שונים לפי האימות שלהם: {detail}.")

        if len(results) > 1:
            report(done_units / total_units * 0.95, _MESSAGE_COMBINE)
            started = time.perf_counter()
            combined = combine_stems(results, profile.ensemble_mode)
            timings["ensemble"] = time.perf_counter() - started
        else:
            combined = results[0]

        parts = {k: fit_length(v, mix.frames) for k, v in combined.parts.items()}
        self.backend.unload()

        # An instrumental track separates "successfully" into a silent vocal and
        # a copy of itself. Saying so here is far kinder than letting the user
        # wait through conversion for silence.
        separated_vocal = parts.get(StemKind.VOCALS)
        if separated_vocal is not None and is_silent(separated_vocal):
            raise EngineError(
                ErrorCode.NO_VOCALS,
                f"the separated vocal from {source.name} is below the silence threshold",
            )

        cleanup_result: CleanupResult | None = None
        if steps:
            vocals = parts.get(StemKind.VOCALS)
            if vocals is None:
                notes.append("לא התקבלה שכבת שירה, אז שלבי הניקוי לא רצו.")
            else:
                # None of the cleanup checkpoints has its own XPU end-to-end
                # proof yet. Keep optional cleanup on CPU until each exact model
                # earns one rather than inheriting the main separator's proof.
                cleanup_hint = DeviceHint(backend=ComputeBackend.CPU)
                if primary_hint.backend is not ComputeBackend.CPU:
                    notes.append(
                        "שלבי הניקוי רצו על המעבד כי מודלי הניקוי טרם אומתו "
                        "מקצה לקצה על המאיץ."
                    )
                started = time.perf_counter()
                cleanup_result = self._run_cleanup(
                    vocals, steps, cleanup_hint, profile,
                    lambda _step, message: report(
                        (done_units + 0.5) / total_units * 0.95, message
                    ),
                )
                timings["cleanup"] = time.perf_counter() - started
                parts[StemKind.VOCALS] = cleanup_result.vocals
                parts.update(cleanup_result.parts)
                notes.extend(cleanup_result.skipped.values())

        report(1.0, _MESSAGE_DONE)
        return SeparationOutcome(
            stems=parts,
            profile=profile,
            model_ids=tuple(m.id for m in models),
            device=device,
            source=source,
            seconds_of_audio=mix.seconds,
            timings=timings,
            cleanup=cleanup_result,
            fallback_steps=tuple(fallbacks),
            notes_he=tuple(dict.fromkeys(notes)),
            peak_memory=measure_peak(primary_hint),
        )

    def write(self, outcome: SeparationOutcome, out_dir: Path | str) -> dict[StemKind, Path]:
        """Write every stem as 24-bit WAV, named the way the roadmap specifies."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: dict[StemKind, Path] = {}
        for kind, buffer in outcome.stems.items():
            path = out_dir / f"{kind.value}.wav"
            audio_io.save_wav(buffer, path, bit_depth=24)
            written[kind] = path
        return written

    # -- internals ---------------------------------------------------------- #

    def _resolve_models(self, profile: SeparationProfile, notes: list[str]):  # type: ignore[no-untyped-def]
        chosen = []
        for model_id in profile.models:
            if model_id not in self.registry:
                notes.append(f"המודל {model_id} אינו בקטלוג ולכן דילגנו עליו.")
                continue
            spec = self.registry.get(model_id)
            if not self.allow_private_models and not spec.license.is_redistributable:
                notes.append(spec.license.note_he or f"{spec.id}: אינו מאושר להפצה.")
                continue
            chosen.append(spec)

        if not chosen:
            raise EngineError(
                ErrorCode.MODEL_MISSING,
                f"no usable separation model for quality level {profile.level.value}",
            )
        if len(chosen) < len(profile.models):
            notes.append("רמת האיכות רצה עם פחות מודלים מהמתוכנן.")
        return chosen

    def _separate_one(
        self,
        mix: AudioBuffer,
        model_id: str,
        profile: SeparationProfile,
        hint: DeviceHint,
    ) -> tuple[Stems, list[str]]:
        """One model, with the OOM ladder wrapped around it."""
        plan = ResourcePlan(
            segment_size=min(
                profile.segment_size, suggested_segment_size(hint, profile.segment_size)
            ),
            batch_size=profile.batch_size,
            overlap=profile.overlap,
            backend=hint.backend,
        )
        used: list[str] = []

        def note(step: OomStep) -> None:
            if step.index > 1:
                used.append(f"{model_id}:{step.reason_he}")

        def attempt(rung: ResourcePlan) -> Stems:
            attempt_device = (
                hint
                if rung.backend is hint.backend
                else DeviceHint(backend=ComputeBackend.CPU)
            )
            request = SeparationRequest(
                model_id=model_id,
                wanted=profile.wanted,
                overlap=rung.overlap,
                batch_size=rung.batch_size,
                segment_size=rung.segment_size,
            )
            try:
                return self.backend.separate(mix, request, attempt_device)
            except Exception:
                # Free the card before the next, smaller attempt -- otherwise
                # the retry meets the same exhausted allocator.
                self.backend.unload()
                raise

        return run_with_oom_ladder(plan, attempt, on_step=note), used

    def _run_cleanup(
        self,
        vocals: AudioBuffer,
        steps: tuple[CleanupStep, ...],
        hint: DeviceHint,
        profile: SeparationProfile,
        on_step: Callable[[CleanupStep, str], None],
    ) -> CleanupResult:
        cleaner = VocalCleanup(
            backend=self.backend,
            registry=self.registry,
            allow_private_models=self.allow_private_models,
            work_dir=self.paths.work,
        )
        return cleaner.run(
            vocals,
            steps,
            hint,
            segment_size=profile.segment_size,
            overlap=profile.overlap,
            on_progress=on_step,
        )
