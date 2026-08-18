"""The GUI-facing song-to-cover workflow.

This module deliberately owns the engine wiring that used to live only in the
CLI command.  The app sends plain JSON parameters and receives plain progress
events; every heavy import remains inside the engine process.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from svc_engine.config import Paths, load_settings
from svc_engine.errors import EngineError, ErrorCode

ProgressHook = Callable[[float, str], None]

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
) -> dict[str, Any]:
    """Run either a short preview or the complete production cover pipeline."""
    source = Path(song)
    if not source.is_file() or source.suffix.lower() not in _AUDIO_SUFFIXES:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, f"unsupported input: {source}")
    if quality not in {"fast", "balanced", "max"}:
        raise EngineError(ErrorCode.INTERNAL, f"invalid quality: {quality}")
    if preview_seconds <= 0 or preview_seconds > 90:
        raise EngineError(ErrorCode.INTERNAL, "preview duration must be between 0 and 90")

    app_paths.ensure()
    destination = _destination(app_paths, source, output=output, preview=preview)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # All imports below this point belong only to the engine process.
    from svc_engine.analysis.f0 import RMVPE_MODEL_ID, FcpeExtractor, RmvpeExtractor
    from svc_engine.backends.base import DeviceHint
    from svc_engine.backends.f0 import F0Extractor
    from svc_engine.backends.separation import StemKind
    from svc_engine.compute import Component, DeviceManager, load_matrix
    from svc_engine.conversion import ConversionPipeline
    from svc_engine.pitch import PlaybackStrategy
    from svc_engine.postfx import AmbienceStrategy, PostFxConfig
    from svc_engine.resources import DownloadManager, load_registry
    from svc_engine.separation import CleanupStep, QualityLevel, SeparationPipeline

    settings = load_settings(app_paths)
    reporter = _MonotonicReporter(on_progress)
    separator = SeparationPipeline(
        paths=app_paths,
        allow_downloads=settings.allow_model_downloads,
        allow_private_models=False,
    )

    duration = preview_seconds if preview else None

    def separate(path: Path):  # type: ignore[no-untyped-def]
        outcome = separator.run(
            path,
            level=QualityLevel(quality),
            cleanup=(CleanupStep.DEREVERB,),
            duration_seconds=duration,
            on_progress=lambda progress: reporter(
                min(0.14, 0.14 * progress.fraction), progress.message_he
            ),
        )
        return outcome.stems[StemKind.VOCALS], outcome.stems

    extractor: F0Extractor
    if preview:
        manager = DeviceManager()
        selected_device = load_matrix().device_for_implementation(
            Component.F0, "torchfcpe", manager
        )
        device = DeviceHint.from_device(selected_device)
        extractor = FcpeExtractor()
    else:
        spec = load_registry().get(RMVPE_MODEL_ID)
        downloader = DownloadManager(
            app_paths.models, allow_downloads=settings.allow_model_downloads
        )
        downloader.check_space_for([spec])
        downloader.ensure_model(
            spec,
            on_progress=lambda progress: reporter(
                min(0.04, 0.04 * float(progress.fraction or 0.0)), progress.message_he
            ),
        )
        extractor = RmvpeExtractor(spec.files[0].path_in(app_paths.models))
        device = DeviceHint()
    pipeline = ConversionPipeline(
        paths=app_paths,
        postfx_config=PostFxConfig(
            ambience_strategy=AmbienceStrategy.PARAMETRIC,
            target_lufs=settings.target_lufs,
        ),
    )
    outcome = pipeline.run(
        source,
        voice_id,
        f0_extractor=extractor,
        device=device,
        separate=separate,
        strategy=PlaybackStrategy.WHOLE,
        on_progress=reporter.pipeline,
    )
    written = pipeline.write(outcome, destination)
    best = outcome.decision.best
    postfx = outcome.postfx
    return {
        "output": str(written.resolve()),
        "source": str(source.resolve()),
        "voice_id": voice_id,
        "preview": preview,
        "summary_he": outcome.summary_he(),
        "recommendation": {
            "semitones": best.semitones,
            "octaves": best.octaves,
            "playback_semitones": best.remainder,
            "needs_playback_shift": best.needs_playback_shift,
        },
        "audio_seconds": outcome.seconds_of_audio,
        "timings": outcome.timings,
        "master": None
        if postfx is None
        else {
            "lufs": postfx.mix.integrated_lufs,
            "peak_dbfs": postfx.mix.peak_dbfs,
            "clipped": postfx.mix.clipped,
        },
    }


def _destination(app_paths: Paths, source: Path, *, output: str | None, preview: bool) -> Path:
    if preview:
        root = app_paths.work / "previews"
        return root / f"preview-{uuid.uuid4().hex}.wav"
    if not output:
        raise EngineError(ErrorCode.INTERNAL, "full conversion requires an output path")
    destination = Path(output)
    if destination.suffix.lower() not in {".wav", ".mp3"}:
        raise EngineError(ErrorCode.AUDIO_UNSUPPORTED, "output must be WAV or MP3")
    if destination.resolve() == source.resolve():
        raise EngineError(ErrorCode.INTERNAL, "output must not overwrite the source")
    return destination


class _MonotonicReporter:
    """Translate pipeline-local fractions into stable GUI progress."""

    def __init__(self, hook: ProgressHook | None) -> None:
        self._hook = hook
        self._last = 0.0

    def __call__(self, fraction: float, message: str) -> None:
        fraction = max(self._last, min(1.0, float(fraction)))
        self._last = fraction
        if self._hook is not None:
            self._hook(fraction, message)

    def pipeline(self, fraction: float, message: str) -> None:
        # ConversionPipeline emits fractions local to its stages. Keep the
        # visible bar useful and monotonic without leaking technical stage ids.
        if "מפרידים" in message:
            mapped = 0.0
        elif "מנתחים" in message:
            mapped = 0.18
        elif "ממירים" in message:
            mapped = 0.20 + 0.60 * min(1.0, max(0.0, fraction))
        elif "מחברים" in message:
            mapped = 0.86
        elif "מוכן" in message:
            mapped = 1.0
        else:
            mapped = fraction
        self(mapped, message)
