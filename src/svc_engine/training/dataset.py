"""Prepare an RVC dataset: isolate vocals, clean, trim silence and slice."""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svc_engine.audio import load_audio, save_wav, to_mono
from svc_engine.backends.base import AudioBuffer
from svc_engine.backends.separation import StemKind
from svc_engine.config import Paths
from svc_engine.separation import SeparationPipeline
from svc_engine.separation.quality import CleanupStep, QualityLevel

__all__ = ["DatasetBuilder", "DatasetResult", "PreparationOptions"]

Progress = Callable[[float, str], None]
Cleaner = Callable[[Path], AudioBuffer]


@dataclass(frozen=True)
class PreparationOptions:
    sample_rate: int = 48000
    min_segment_seconds: float = 3.0
    max_segment_seconds: float = 10.0
    silence_db: float = -45.0
    separate_mix: bool = True


_DEFAULT_OPTIONS = PreparationOptions()


@dataclass(frozen=True)
class DatasetResult:
    root: Path
    segments: tuple[Path, ...]
    sample: Path
    seconds: float

    def to_dict(self) -> dict[str, object]:
        return {
            "root": str(self.root),
            "segments": [str(p) for p in self.segments],
            "sample": str(self.sample),
            "seconds": round(self.seconds, 3),
        }


def _resample(audio: AudioBuffer, target: int) -> AudioBuffer:
    mono = to_mono(audio)
    if mono.sample_rate == target:
        return mono
    samples = mono.samples[0]
    positions = np.linspace(
        0, samples.size - 1, int(round(samples.size * target / mono.sample_rate))
    )
    converted = np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
    return AudioBuffer(converted[None, :], target)


def _active_regions(
    samples: np.ndarray, sample_rate: int, silence_db: float
) -> list[tuple[int, int]]:
    frame = max(1, int(0.02 * sample_rate))
    count = samples.size // frame
    if not count:
        return []
    blocks = samples[: count * frame].reshape(count, frame).astype(np.float64, copy=False)
    levels = np.sqrt(np.mean(np.square(blocks), axis=1) + 1e-12)
    peak = max(float(np.max(levels)), 1e-9)
    active = levels >= max(10 ** (silence_db / 20.0), peak * 10 ** (-40.0 / 20.0))
    # Bridge pauses shorter than 300 ms, then keep 100 ms of context.
    bridge = max(1, int(0.3 / 0.02))
    for start in np.flatnonzero(active[:-1] & ~active[1:]):
        later = np.flatnonzero(active[start + 2 : start + 2 + bridge])
        if later.size:
            active[start + 1 : start + 2 + int(later[0])] = True
    indices = np.flatnonzero(active)
    if not indices.size:
        return []
    splits = np.flatnonzero(np.diff(indices) > 1)
    groups = np.split(indices, splits + 1)
    pad = int(0.1 * sample_rate)
    return [
        (max(0, int(g[0] * frame) - pad), min(samples.size, int((g[-1] + 1) * frame) + pad))
        for g in groups
    ]


def _segments(audio: AudioBuffer, options: PreparationOptions) -> list[AudioBuffer]:
    samples = audio.samples[0]
    minimum = int(options.min_segment_seconds * audio.sample_rate)
    maximum = int(options.max_segment_seconds * audio.sample_rate)
    result: list[AudioBuffer] = []
    pending = np.zeros(0, dtype=np.float32)
    for start, end in _active_regions(samples, audio.sample_rate, options.silence_db):
        pending = np.concatenate([pending, samples[start:end]])
        while pending.size >= maximum:
            result.append(AudioBuffer(pending[:maximum][None, :].copy(), audio.sample_rate))
            pending = pending[maximum:]
        if pending.size >= minimum:
            result.append(AudioBuffer(pending[None, :].copy(), audio.sample_rate))
            pending = np.zeros(0, dtype=np.float32)
    if pending.size >= minimum:
        result.append(AudioBuffer(pending[None, :].copy(), audio.sample_rate))
    return result


class DatasetBuilder:
    def __init__(self, cleaner: Cleaner | None = None, paths: Paths | None = None) -> None:
        self.cleaner = cleaner
        self.paths = paths

    def _production_cleaner(self, path: Path) -> AudioBuffer:
        outcome = SeparationPipeline(self.paths, allow_private_models=True).run(
            path,
            level=QualityLevel.FAST,
            cleanup=(CleanupStep.DENOISE, CleanupStep.DEREVERB),
        )
        return outcome.stems[StemKind.VOCALS]

    def build(
        self,
        recordings: list[Path | str],
        output_dir: Path | str,
        options: PreparationOptions = _DEFAULT_OPTIONS,
        on_progress: Progress | None = None,
    ) -> DatasetResult:
        source_paths = [Path(path) for path in recordings]
        if not source_paths:
            raise ValueError("at least one recording is required")
        root = Path(output_dir)
        staged = root.with_name(f".{root.name}.preparing")
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True)
        written: list[Path] = []
        sample_audio: AudioBuffer | None = None
        seconds = 0.0
        try:
            for source_index, source in enumerate(source_paths):
                if on_progress:
                    on_progress(source_index / len(source_paths), "מפרידים ומנקים את ההקלטות…")
                if options.separate_mix:
                    cleaner = self.cleaner or self._production_cleaner
                    audio = cleaner(source)
                else:
                    audio = load_audio(source)
                audio = _resample(audio, options.sample_rate)
                for part in _segments(audio, options):
                    peak = float(np.max(np.abs(part.samples)))
                    if peak > 0.0:
                        part = AudioBuffer(
                            (part.samples * min(1.0, 0.95 / peak)).astype(np.float32),
                            part.sample_rate,
                        )
                    destination = staged / f"segment-{len(written):05d}.wav"
                    save_wav(part, destination, bit_depth=24)
                    written.append(destination)
                    seconds += part.seconds
                    if sample_audio is None or part.seconds > sample_audio.seconds:
                        sample_audio = part
            if not written or sample_audio is None:
                raise ValueError("no voiced segments remained after silence trimming")
            sample_path = staged / "sample.wav"
            save_wav(sample_audio, sample_path, bit_depth=24)
            shutil.rmtree(root, ignore_errors=True)
            staged.replace(root)
        except BaseException:
            shutil.rmtree(staged, ignore_errors=True)
            raise
        segments = tuple(root / path.name for path in written)
        if on_progress:
            on_progress(1.0, "חומר האימון נקי ומוכן.")
        return DatasetResult(
            root=root, segments=segments, sample=root / "sample.wav", seconds=seconds
        )
