"""The full song-to-cover pipeline: separation -> analysis -> pitch -> convert.

Phase 5 wires the earlier engines together for the first time. Given a song and
a target voice this separates the stems (Phase 2), measures the singing's pitch
(Phase 3), decides how far to shift it (Phase 4), converts the vocal into the
voice (this phase), shifts the playback by the remainder `r`, and mixes the
cover back together.

The heavy stages are injected -- the conversion backend, the F0 extractor and
the pitch shifter -- so the orchestration (`render_cover`) is tested end to end
with fakes, torch-free, while `run` wires the real engines. `mix_cover` is pure
numpy and length-exact: the converted vocal and the shifted playback come back
at the song's length, sample for sample.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from svc_engine.audio.buffers import add, fit_length
from svc_engine.backends.base import AudioBuffer, DeviceHint, F0Curve
from svc_engine.backends.conversion import ConversionBackend, ConversionParams, VoiceHandle
from svc_engine.backends.f0 import F0Extractor
from svc_engine.backends.pitch import PitchShifter
from svc_engine.backends.separation import StemKind
from svc_engine.config import Paths
from svc_engine.config import paths as default_paths
from svc_engine.conversion.chunking import convert_in_chunks
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.pitch import (
    PitchDistribution,
    PlaybackStrategy,
    ShiftDecision,
    decide_shift,
    shift_playback,
)
from svc_engine.profiles import VoiceProfile
from svc_engine.voices import VoiceLibrary

__all__ = ["ConversionOutcome", "ConversionPipeline", "mix_cover", "params_for_voice"]

log = logging.getLogger(__name__)

ProgressHook = Callable[[float, str], None]

_MSG_SEPARATE = "מפרידים את השירה מהמוזיקה…"
_MSG_ANALYZE = "מנתחים את גובה הקול…"
_MSG_CONVERT = "ממירים את הקול…"
_MSG_MIX = "מחברים הכול יחד…"
_MSG_DONE = "השיר מוכן."


def mix_cover(vocal: AudioBuffer, playback: AudioBuffer) -> AudioBuffer:
    """Sum the converted vocal onto the playback at the playback's length.

    The playback carries the song's full length and channel count; the vocal is
    fit to it (mono is widened by `add`). Length is exact -- the cover is exactly
    as long as the song.
    """
    vocal_fit = fit_length(vocal, playback.frames)
    return add(playback, vocal_fit)


def params_for_voice(
    profile_recommended: ConversionParams, semitones: int
) -> ConversionParams:
    """The conversion knobs for this run: the voice's recommended params, with
    the pitch engine's shift filled in."""
    return ConversionParams(
        semitones=semitones,
        index_rate=profile_recommended.index_rate,
        protect=profile_recommended.protect,
        rms_mix_rate=profile_recommended.rms_mix_rate,
        filter_radius=profile_recommended.filter_radius,
        formant_shift=profile_recommended.formant_shift,
    )


@dataclass(frozen=True)
class ConversionOutcome:
    """The finished cover plus how it was made."""

    cover: AudioBuffer
    decision: ShiftDecision
    voice_id: str
    source: Path
    seconds_of_audio: float
    playback_strategy: PlaybackStrategy
    timings: dict[str, float] = field(default_factory=dict)
    notes_he: tuple[str, ...] = ()

    @property
    def total_seconds(self) -> float:
        return sum(self.timings.values())

    def summary_he(self) -> str:
        r = self.decision.best.remainder
        playback = "המוזיקה נשארה כמו שהיא" if r == 0 else f"המוזיקה הוזזה ב-{r} חצאי-טונים"
        return (
            f"קאבר בקול '{self.voice_id}' · הזזת שירה {self.decision.best.semitones:+d} "
            f"חצאי-טונים · {playback} · {self.total_seconds:.0f} שניות עיבוד"
        )


class ConversionPipeline:
    """Owns the song-to-cover flow. Heavy stages are injected for testability."""

    def __init__(
        self,
        paths: Paths | None = None,
        library: VoiceLibrary | None = None,
        conversion_backend: ConversionBackend | None = None,
        shifter: PitchShifter | None = None,
    ) -> None:
        self.paths = paths or default_paths()
        self.paths.ensure()
        self.library = library or VoiceLibrary(self.paths)
        self._backend = conversion_backend
        self._shifter = shifter

    # -- public API --------------------------------------------------------- #

    def render_cover(
        self,
        vocals: AudioBuffer,
        f0: F0Curve,
        playback: AudioBuffer | dict[StemKind, AudioBuffer],
        profile: VoiceProfile,
        voice: VoiceHandle,
        recommended: ConversionParams,
        device: DeviceHint | None = None,
        strategy: PlaybackStrategy = PlaybackStrategy.WHOLE,
        on_progress: ProgressHook | None = None,
    ) -> tuple[AudioBuffer, ShiftDecision]:
        """The orchestration heart: decide the shift, convert, shift, mix.

        Torch-free given fakes for the injected backend/shifter, so this is
        tested end to end without the AI stack. Returns the cover and the
        decision that produced it.
        """
        device = device or DeviceHint()
        backend = self._require_backend()
        shifter = self._require_shifter()

        f0_hz = np.asarray(f0.hz, dtype=np.float64).ravel()
        mono = vocals.samples.mean(axis=0) if vocals.channels > 1 else vocals.samples[0]
        energy = _frame_energy(mono, vocals.sample_rate, f0.hop_seconds, f0_hz.size)
        dist = PitchDistribution.from_f0(f0_hz, energy)
        decision = decide_shift(dist, profile)

        params = params_for_voice(recommended, decision.best.semitones)

        if on_progress is not None:
            on_progress(0.05, _MSG_CONVERT)
        backend.load(voice, device)
        try:
            converted = convert_in_chunks(
                vocals, f0, backend, params,
                on_progress=(
                    (lambda done, total: on_progress(0.05 + 0.75 * done / total, _MSG_CONVERT))
                    if on_progress is not None
                    else None
                ),
            )
        finally:
            backend.unload()  # release VRAM after every run (Phase 5 DoD)

        if on_progress is not None:
            on_progress(0.85, _MSG_MIX)
        shifted = shift_playback(playback, decision.best.remainder, shifter, strategy)
        cover = mix_cover(converted, shifted)
        if on_progress is not None:
            on_progress(1.0, _MSG_DONE)
        return cover, decision

    def run(
        self,
        song: Path | str,
        voice_id: str,
        f0_extractor: F0Extractor,
        device: DeviceHint,
        separate: Callable[[Path], tuple[AudioBuffer, dict[StemKind, AudioBuffer]]],
        strategy: PlaybackStrategy = PlaybackStrategy.WHOLE,
        on_progress: ProgressHook | None = None,
    ) -> ConversionOutcome:
        """End to end: separate, analyse, decide, convert, mix.

        `separate` is injected (the default CLI wiring passes a closure over the
        Phase 2 `SeparationPipeline`) so this stays testable and does not hard-
        wire the heavy separation stack.
        """
        song = Path(song)
        if voice_id not in self.library:
            raise EngineError(
                ErrorCode.VOICE_CORRUPT, f"no voice '{voice_id}' in the library"
            )
        entry = self.library.get(voice_id)
        if not entry.manifest.consent_confirmed:
            raise EngineError(
                ErrorCode.CONSENT_REQUIRED, f"voice {voice_id} has no confirmed consent"
            )
        profile = entry.profile()
        if profile is None:
            raise EngineError(
                ErrorCode.VOICE_CORRUPT,
                f"voice {voice_id} has no range profile; cannot decide the shift",
            )

        timings: dict[str, float] = {}
        if on_progress is not None:
            on_progress(0.0, _MSG_SEPARATE)
        t = time.perf_counter()
        vocals, stems = separate(song)
        timings["separate"] = time.perf_counter() - t

        instrumental = stems.get(StemKind.INSTRUMENTAL)
        if instrumental is None:
            raise EngineError(
                ErrorCode.NO_VOCALS, "separation returned no instrumental to mix under"
            )
        playback = _playback_input(stems, strategy)

        if on_progress is not None:
            on_progress(0.15, _MSG_ANALYZE)
        t = time.perf_counter()
        f0 = f0_extractor.extract(vocals, device, 0.01)
        f0_extractor.unload()
        timings["analyze"] = time.perf_counter() - t

        recommended = _params_from_manifest(entry.manifest.recommended)
        t = time.perf_counter()
        cover, decision = self.render_cover(
            vocals, f0, playback, profile, entry.handle(), recommended,
            device=device, strategy=strategy, on_progress=on_progress,
        )
        timings["convert"] = time.perf_counter() - t

        return ConversionOutcome(
            cover=cover,
            decision=decision,
            voice_id=voice_id,
            source=song,
            seconds_of_audio=vocals.seconds,
            playback_strategy=strategy,
            timings=timings,
        )

    def write(self, outcome: ConversionOutcome, out_path: Path | str) -> Path:
        from svc_engine.audio import io as audio_io

        return audio_io.save_audio(outcome.cover, Path(out_path))

    # -- internals ---------------------------------------------------------- #

    def _require_backend(self) -> ConversionBackend:
        if self._backend is None:
            from svc_engine.conversion.rvc.backend import RVCv2Backend

            self._backend = RVCv2Backend(models_dir=self.paths.models)
        return self._backend

    def _require_shifter(self) -> PitchShifter:
        if self._shifter is None:
            from svc_engine.pitch import PythonStretchShifter

            self._shifter = PythonStretchShifter()
        return self._shifter


def _playback_input(
    stems: dict[StemKind, AudioBuffer], strategy: PlaybackStrategy
) -> AudioBuffer | dict[StemKind, AudioBuffer]:
    """Strategy A shifts the whole instrumental; B needs the split stems."""
    if strategy is PlaybackStrategy.SPLIT and (
        StemKind.BASS in stems or StemKind.DRUMS in stems or StemKind.OTHER in stems
    ):
        return stems
    instrumental = stems.get(StemKind.INSTRUMENTAL)
    if instrumental is None:
        raise EngineError(ErrorCode.NO_VOCALS, "no instrumental stem for playback")
    return instrumental


def _params_from_manifest(recommended: object) -> ConversionParams:
    """Voice manifest RecommendedParams -> ConversionParams (defaults elsewhere)."""
    return ConversionParams(
        index_rate=getattr(recommended, "index_rate", 0.70),
        protect=getattr(recommended, "protect", 0.33),
        rms_mix_rate=getattr(recommended, "rms_mix_rate", 0.25),
    )


def _frame_energy(
    mono: np.ndarray, sample_rate: int, hop_seconds: float, size: int
) -> np.ndarray:
    from svc_engine.analysis.features import align_to, frame_rms

    return align_to(frame_rms(mono, sample_rate, hop_seconds), size)
