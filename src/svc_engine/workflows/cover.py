"""Durable GUI-facing song-to-cover production workflow.

Preview and full covers are real Phase-7 jobs.  Their separation, analysis,
conversion/mastering and delivery nodes publish immutable outputs to StepCache;
the request needed to reconstruct the graph is stored beside RecoveryStore.
"""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from svc_engine.config import Paths, load_settings
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.jobs import Job, JobProgress, JobRunner, Step, StepContext
from svc_engine.jobs._io import atomic_write_json, read_json_object, validate_identifier

ProgressHook = Callable[[float, str], None]
JobHook = Callable[[str], None]

_SCHEMA = 1
_AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}


def run_cover(
    app_paths: Paths,
    *,
    song: str,
    voice_id: str,
    quality: str = "balanced",
    output: str | None = None,
    preview: bool = False,
    preview_seconds: float = 30.0,
    on_progress: ProgressHook | None = None,
    on_job: JobHook | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Create and execute a new durable preview or cover job."""
    source = _validate_source(song)
    _validate_options(quality, preview_seconds)
    app_paths.ensure()
    identifier = validate_identifier(job_id or f"cover-{uuid.uuid4().hex}", label="job id")
    destination = _destination(app_paths, source, output=output, preview=preview)
    request_path = _request_path(app_paths, identifier)
    if request_path.exists():
        raise EngineError(ErrorCode.INTERNAL, f"cover job already exists: {identifier}")
    request: dict[str, Any] = {
        "schema": _SCHEMA,
        "job_id": identifier,
        "song": str(source.resolve()),
        "voice_id": voice_id,
        "quality": quality,
        "output": str(destination.resolve()),
        "preview": preview,
        "preview_seconds": preview_seconds,
    }
    atomic_write_json(request_path, request)
    try:
        return _execute(app_paths, request, on_progress=on_progress, on_job=on_job)
    except BaseException:
        from svc_engine.jobs import RecoveryStore

        if RecoveryStore(app_paths.root / "jobs").load(identifier) is None:
            request_path.unlink(missing_ok=True)
        raise


def resume_cover(
    app_paths: Paths,
    job_id: str,
    *,
    on_progress: ProgressHook | None = None,
    on_job: JobHook | None = None,
) -> dict[str, Any]:
    """Reconstruct a cover graph from its durable request and resume it."""
    request = load_cover_request(app_paths, job_id)
    if request is None:
        raise EngineError(ErrorCode.INTERNAL, f"missing recoverable cover request: {job_id}")
    return _execute(app_paths, request, on_progress=on_progress, on_job=on_job)


def load_cover_request(app_paths: Paths, job_id: str) -> dict[str, Any] | None:
    """Read user-safe recovery metadata, or return None for a non-cover job."""
    path = _request_path(app_paths, validate_identifier(job_id, label="job id"))
    if not path.is_file():
        return None
    raw = read_json_object(path)
    if int(raw.get("schema", -1)) != _SCHEMA or str(raw.get("job_id")) != job_id:
        raise ValueError("unsupported or mismatched cover request")
    return raw


def _execute(
    app_paths: Paths,
    request: dict[str, Any],
    *,
    on_progress: ProgressHook | None,
    on_job: JobHook | None,
) -> dict[str, Any]:
    identifier = validate_identifier(str(request["job_id"]), label="job id")
    if on_job is not None:
        on_job(identifier)
    settings = load_settings(app_paths)
    steps = _production_steps(app_paths, request)
    runner = JobRunner(app_paths, settings=settings)

    def progress(item: JobProgress) -> None:
        if on_progress is not None:
            on_progress(item.overall_fraction, item.message_he)

    result = runner.run(
        Job(
            "תצוגה מקדימה" if bool(request["preview"]) else "יצירת קאבר",
            steps,
            job_id=identifier,
        ),
        on_progress=progress,
    )
    metadata_path = result.steps["render"].outputs["metadata"]
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise EngineError(ErrorCode.INTERNAL, "cover metadata is not an object")
    payload["output"] = str(Path(str(request["output"])).resolve())
    payload["job_id"] = identifier
    payload["resumed"] = result.resumed
    payload["cache_hits"] = {
        step_id: step.cache_hit for step_id, step in result.steps.items()
    }
    payload["timings"] = {
        step_id: step.seconds for step_id, step in result.steps.items()
    }
    _request_path(app_paths, identifier).unlink(missing_ok=True)
    return payload


def _production_steps(app_paths: Paths, request: dict[str, Any]) -> tuple[Step, ...]:
    """Build the actual production DAG. Tests replace only this factory."""
    source = Path(str(request["song"]))
    voice_id = str(request["voice_id"])
    quality = str(request["quality"])
    preview = bool(request["preview"])
    duration = float(request["preview_seconds"]) if preview else None
    destination = Path(str(request["output"]))
    settings = load_settings(app_paths)

    # Validate the voice before spending time on separation and include every
    # file that can affect conversion in the render node's content key.
    from svc_engine.voices import VoiceLibrary

    entry = VoiceLibrary(app_paths).get(voice_id)
    if not entry.manifest.consent_confirmed:
        raise EngineError(ErrorCode.CONSENT_REQUIRED, f"voice {voice_id} has no consent")
    if entry.profile_path is None:
        raise EngineError(ErrorCode.VOICE_CORRUPT, f"voice {voice_id} has no range profile")
    voice_inputs = [entry.model_path, entry.profile_path, entry.root / "voice.json"]
    if entry.index_path is not None:
        voice_inputs.append(entry.index_path)

    def separate(context: StepContext) -> dict[str, Path]:
        from svc_engine.audio import io as audio_io
        from svc_engine.separation import CleanupStep, QualityLevel, SeparationPipeline

        separator = SeparationPipeline(
            paths=app_paths,
            allow_downloads=settings.allow_model_downloads,
            allow_private_models=False,
        )
        outcome = separator.run(
            source,
            level=QualityLevel(quality),
            cleanup=(CleanupStep.DEREVERB,),
            duration_seconds=duration,
            on_progress=lambda item: context.progress(item.fraction, item.message_he),
        )
        outputs: dict[str, Path] = {}
        for kind, audio in outcome.stems.items():
            target = context.output_dir / f"{kind.value}.wav"
            audio_io.save_wav(audio, target)
            outputs[kind.value] = target
        if "vocals" not in outputs or "instrumental" not in outputs:
            raise EngineError(ErrorCode.NO_VOCALS, "separation did not produce required stems")
        return outputs

    method, device = _analysis_route(preview)

    def analyze(context: StepContext) -> dict[str, Path]:
        import numpy as np

        from svc_engine.audio import io as audio_io

        vocals = audio_io.load_audio(context.dependencies["separate"]["vocals"])
        extractor = _new_extractor(
            app_paths,
            method,
            allow_downloads=settings.allow_model_downloads,
            progress=context.progress,
        )
        context.progress(0.05, "מנתחים את גובה הקול…")
        try:
            curve = extractor.extract(vocals, device, 0.01)
        finally:
            extractor.unload()
        target = context.output_dir / "f0.npz"
        with target.open("wb") as stream:
            np.savez(stream, hz=curve.hz, hop_seconds=curve.hop_seconds)
        context.progress(1.0, "ניתוח השירה הושלם.")
        return {"f0": target}

    def render(context: StepContext) -> dict[str, Path]:
        import numpy as np

        from svc_engine.audio import io as audio_io
        from svc_engine.backends.base import F0Curve
        from svc_engine.backends.conversion import ConversionParams
        from svc_engine.backends.separation import StemKind
        from svc_engine.conversion import ConversionPipeline
        from svc_engine.pitch import PlaybackStrategy
        from svc_engine.postfx import AmbienceStrategy, PostFxConfig

        separated = context.dependencies["separate"]
        vocals = audio_io.load_audio(separated["vocals"])
        stems = {
            StemKind(key): audio_io.load_audio(path)
            for key, path in separated.items()
            if key in {kind.value for kind in StemKind}
        }
        with np.load(context.dependencies["analyze"]["f0"], allow_pickle=False) as saved:
            curve = F0Curve(
                hz=np.asarray(saved["hz"], dtype=np.float32),
                hop_seconds=float(saved["hop_seconds"]),
            )
        profile = entry.profile()
        if profile is None:  # guarded above, but keep the action self-contained
            raise EngineError(ErrorCode.VOICE_CORRUPT, "voice profile disappeared")
        recommended = ConversionParams(
            index_rate=entry.manifest.recommended.index_rate,
            protect=entry.manifest.recommended.protect,
            rms_mix_rate=entry.manifest.recommended.rms_mix_rate,
        )
        output_extractor = _new_extractor(
            app_paths,
            method,
            allow_downloads=settings.allow_model_downloads,
            progress=context.progress,
        )
        pipeline = ConversionPipeline(
            paths=app_paths,
            postfx_config=PostFxConfig(
                ambience_strategy=AmbienceStrategy.PARAMETRIC,
                target_lufs=settings.target_lufs,
            ),
        )
        cover, decision, postfx, _seconds = pipeline._render_cover_detailed(
            vocals,
            curve,
            stems[StemKind.INSTRUMENTAL],
            profile,
            entry.handle(),
            recommended,
            device=device,
            strategy=PlaybackStrategy.WHOLE,
            ambience=stems.get(StemKind.AMBIENCE),
            output_f0_extractor=output_extractor,
            on_progress=context.progress,
        )
        cover_path = context.output_dir / "cover.wav"
        audio_io.save_wav(cover, cover_path)
        best = decision.best
        playback = (
            "המוזיקה נשארה כמו שהיא"
            if best.remainder == 0
            else f"המוזיקה הוזזה ב-{best.remainder} חצאי-טונים"
        )
        summary = (
            f"קאבר בקול '{voice_id}' · הזזת שירה {best.semitones:+d} חצאי-טונים · "
            f"{playback} · {postfx.mix.integrated_lufs:.1f} LUFS"
        )
        metadata = {
            "source": str(source.resolve()),
            "voice_id": voice_id,
            "preview": preview,
            "summary_he": summary,
            "recommendation": {
                "semitones": best.semitones,
                "octaves": best.octaves,
                "playback_semitones": best.remainder,
                "needs_playback_shift": best.needs_playback_shift,
            },
            "audio_seconds": vocals.seconds,
            "master": {
                "lufs": postfx.mix.integrated_lufs,
                "peak_dbfs": postfx.mix.peak_dbfs,
                "clipped": postfx.mix.clipped,
            },
        }
        metadata_path = context.output_dir / "result.json"
        metadata_path.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return {"cover": cover_path, "metadata": metadata_path}

    def deliver(context: StepContext) -> dict[str, Path]:
        from svc_engine.audio import io as audio_io

        cached_cover = context.dependencies["render"]["cover"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        context.progress(0.1, "שומרים את התוצאה…")
        if destination.suffix.lower() == ".wav":
            shutil.copy2(cached_cover, destination)
        else:
            audio_io.save_audio(audio_io.load_audio(cached_cover), destination)
        receipt = context.output_dir / "delivered.json"
        receipt.write_text(
            json.dumps({"output": str(destination.resolve())}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        context.progress(1.0, "השיר מוכן.")
        return {"receipt": receipt}

    return (
        Step(
            "separate",
            separate,
            parameters={"quality": quality, "seconds": duration, "private_models": False},
            input_files=(source,),
            version="cover-separate-v1",
            weight=3.0,
            vram_hint_mb=6000,
        ),
        Step(
            "analyze",
            analyze,
            parameters={"method": method, "device": device.torch_device},
            needs=("separate",),
            version="cover-analyze-v1",
            weight=1.0,
            vram_hint_mb=1500,
        ),
        Step(
            "render",
            render,
            parameters={
                "voice_id": voice_id,
                "method": method,
                "device": device.torch_device,
                "target_lufs": settings.target_lufs,
                "ambience": "B",
            },
            input_files=tuple(voice_inputs),
            needs=("separate", "analyze"),
            version="cover-render-v1",
            weight=5.0,
            vram_hint_mb=6000,
        ),
        Step(
            "deliver",
            deliver,
            parameters={"job_id": str(request["job_id"]), "output": str(destination)},
            needs=("render",),
            version="cover-deliver-v1",
            weight=0.2,
            vram_hint_mb=0,
        ),
    )


def _analysis_route(preview: bool) -> tuple[str, Any]:
    from svc_engine.backends.base import DeviceHint

    if not preview:
        return "rmvpe", DeviceHint()
    from svc_engine.compute import Component, DeviceManager, load_matrix

    selected = load_matrix().device_for_implementation(
        Component.F0, "torchfcpe", DeviceManager()
    )
    return "fcpe", DeviceHint.from_device(selected)


def _new_extractor(
    app_paths: Paths,
    method: str,
    *,
    allow_downloads: bool,
    progress: Callable[[float, str], None],
) -> Any:
    if method == "fcpe":
        from svc_engine.analysis.f0 import FcpeExtractor

        return FcpeExtractor()
    from svc_engine.analysis.f0 import RMVPE_MODEL_ID, RmvpeExtractor
    from svc_engine.resources import DownloadManager, load_registry

    spec = load_registry().get(RMVPE_MODEL_ID)
    downloader = DownloadManager(app_paths.models, allow_downloads=allow_downloads)
    downloader.check_space_for([spec])
    downloader.ensure_model(
        spec,
        on_progress=lambda item: progress(
            min(0.15, 0.15 * float(item.fraction or 0.0)), item.message_he
        ),
    )
    return RmvpeExtractor(spec.files[0].path_in(app_paths.models))


def _request_path(app_paths: Paths, job_id: str) -> Path:
    return app_paths.root / "jobs" / "requests" / f"{job_id}.json"


def _validate_source(song: str) -> Path:
    source = Path(song)
    if not source.is_file() or source.suffix.lower() not in _AUDIO_SUFFIXES:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, f"unsupported input: {source}")
    return source


def _validate_options(quality: str, preview_seconds: float) -> None:
    if quality not in {"fast", "balanced", "max"}:
        raise EngineError(ErrorCode.INTERNAL, f"invalid quality: {quality}")
    if preview_seconds <= 0 or preview_seconds > 90:
        raise EngineError(ErrorCode.INTERNAL, "preview duration must be between 0 and 90")


def _destination(app_paths: Paths, source: Path, *, output: str | None, preview: bool) -> Path:
    if preview:
        return app_paths.work / "previews" / f"preview-{uuid.uuid4().hex}.wav"
    if not output:
        raise EngineError(ErrorCode.INTERNAL, "full conversion requires an output path")
    destination = Path(output)
    if destination.suffix.lower() not in {".wav", ".mp3"}:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, "output must be WAV or MP3")
    if destination.resolve() == source.resolve():
        raise EngineError(ErrorCode.INTERNAL, "output must not overwrite the source")
    return destination
