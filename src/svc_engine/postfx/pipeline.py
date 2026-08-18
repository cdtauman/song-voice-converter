"""One deterministic Phase-6 chain over the raw Phase-5 conversion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from svc_engine.backends.base import AudioBuffer, F0Curve
from svc_engine.backends.pitch import PitchShifter
from svc_engine.postfx.ambience import AmbienceResult, AmbienceStrategy, apply_ambience
from svc_engine.postfx.deess import DeEssReport, deess
from svc_engine.postfx.envelope import EnvelopeReport, match_envelope
from svc_engine.postfx.melody_check import MelodyReport, check_melody, correct_global_pitch
from svc_engine.postfx.mix import MixConfig, MixReport, mix_and_master
from svc_engine.postfx.repair import RepairReport, repair_artifacts

__all__ = ["PostFxConfig", "PostFxOutcome", "PostFxPipeline", "PostFxReport"]


@dataclass(frozen=True)
class PostFxConfig:
    ambience_strategy: AmbienceStrategy = AmbienceStrategy.PARAMETRIC
    rms_mix_rate: float = 0.25
    target_lufs: float = -14.0
    true_peak_db: float = -1.0
    melody_correction: bool = True
    deess_enabled: bool = True


@dataclass(frozen=True)
class PostFxReport:
    repair: RepairReport
    melody: MelodyReport
    envelope: EnvelopeReport
    deess: DeEssReport
    ambience: AmbienceResult
    mix: MixReport

    def to_dict(self) -> dict[str, object]:
        return {
            "repair": asdict(self.repair),
            "melody": asdict(self.melody),
            "envelope": asdict(self.envelope),
            "deess": asdict(self.deess),
            "ambience": {
                "strategy": self.ambience.strategy.value,
                "measurement": asdict(self.ambience.measurement),
                "original_layer_gain": self.ambience.original_layer_gain,
            },
            "mix": asdict(self.mix),
        }


@dataclass(frozen=True)
class PostFxOutcome:
    cover: AudioBuffer
    vocal: AudioBuffer
    report: PostFxReport


class PostFxPipeline:
    """Runs repair -> melody -> envelope -> de-ess -> ambience -> ffmpeg mix."""

    def __init__(self, config: PostFxConfig | None = None, work_dir: Path | None = None) -> None:
        self.config = config or PostFxConfig()
        self.work_dir = work_dir

    def run(
        self,
        *,
        original_vocal: AudioBuffer,
        converted_vocal: AudioBuffer,
        playback: AudioBuffer,
        original_ambience: AudioBuffer | None,
        reference_f0: F0Curve,
        converted_f0: F0Curve | None,
        semitones: float,
        shifter: PitchShifter,
    ) -> PostFxOutcome:
        vocal, repair = repair_artifacts(converted_vocal)
        melody = check_melody(reference_f0, converted_f0, semitones)
        if self.config.melody_correction:
            vocal, melody = correct_global_pitch(vocal, melody, shifter)
        vocal, envelope = match_envelope(
            original_vocal, vocal, rms_mix_rate=self.config.rms_mix_rate
        )
        if self.config.deess_enabled:
            vocal, deess_report = deess(vocal)
        else:
            deess_report = DeEssReport(0.0, 0.0, 5500.0)
        ambience = apply_ambience(
            vocal,
            original_vocal,
            original_ambience,
            self.config.ambience_strategy,
        )
        cover, mix = mix_and_master(
            ambience.vocal,
            playback,
            MixConfig(
                target_lufs=self.config.target_lufs,
                true_peak_db=self.config.true_peak_db,
            ),
            work_dir=self.work_dir,
        )
        report = PostFxReport(repair, melody, envelope, deess_report, ambience, mix)
        return PostFxOutcome(cover=cover, vocal=ambience.vocal, report=report)
