"""Optional passes over a separated vocal: denoise, de-reverb, de-echo, karaoke.

Each pass is another Roformer, run on the vocal instead of on the mix. They all
work the same way -- split the input into "what we keep" and "what we removed" --
so they share one implementation and differ only in which checkpoint runs.

**The removed half is kept, never discarded.** The reverb tail carries the room
the song was recorded in, and Phase 6 has to decide what to do with it: put the
original back, synthesise a matching one, or blend. That decision needs the
original tail as evidence, and it cannot be recovered later.

Two rules keep this honest:

* **Order matters.** Denoise, then de-reverb, then de-echo, then the karaoke
  split. Removing reverb from a noisy signal teaches the model to treat the
  noise as room; splitting lead from backing after cleaning gives it a cleaner
  decision.
* **A pass that would remove almost everything is refused.** A model applied to
  material it was not trained for can return near-silence, and shipping a
  silent vocal is worse than shipping an untreated one.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from svc_engine.audio.buffers import fit_length, is_silent, rms, subtract
from svc_engine.backends.base import AudioBuffer, DeviceHint
from svc_engine.backends.separation import SeparationBackend, SeparationRequest, StemKind
from svc_engine.errors import EngineError
from svc_engine.resources import ModelKind, ModelRegistry
from svc_engine.separation.quality import CleanupStep

__all__ = ["CleanupResult", "VocalCleanup", "STEP_ORDER", "STEP_LABELS_HE"]

log = logging.getLogger(__name__)

#: Fixed pipeline order, independent of the order the caller listed them in.
STEP_ORDER: tuple[CleanupStep, ...] = (
    CleanupStep.DENOISE,
    CleanupStep.DEREVERB,
    CleanupStep.DEECHO,
    CleanupStep.KARAOKE,
)

STEP_LABELS_HE: dict[CleanupStep, str] = {
    CleanupStep.DENOISE: "מנקים רעש מהשירה…",
    CleanupStep.DEREVERB: "מסירים הדהוד מהשירה…",
    CleanupStep.DEECHO: "מסירים הד מהשירה…",
    CleanupStep.KARAOKE: "מפרידים את השירה הראשית מהליווי…",
}

_STEP_KINDS: dict[CleanupStep, ModelKind] = {
    CleanupStep.DENOISE: ModelKind.DENOISE,
    CleanupStep.DEREVERB: ModelKind.DEREVERB,
    CleanupStep.DEECHO: ModelKind.DEECHO,
    CleanupStep.KARAOKE: ModelKind.KARAOKE,
}

#: Below this fraction of the input level, we assume the model misfired.
_COLLAPSE_RATIO = 0.05


@dataclass(frozen=True)
class CleanupResult:
    """The cleaned vocal plus everything that was taken out of it."""

    vocals: AudioBuffer
    parts: dict[StemKind, AudioBuffer] = field(default_factory=dict)
    applied: tuple[CleanupStep, ...] = ()
    skipped: dict[CleanupStep, str] = field(default_factory=dict)

    @property
    def ambience(self) -> AudioBuffer | None:
        """The removed room sound. Phase 6 needs this."""
        return self.parts.get(StemKind.AMBIENCE)


ProgressHook = Callable[[CleanupStep, str], None]


class VocalCleanup:
    """Runs cleanup passes on a vocal using the same backend as separation."""

    def __init__(
        self,
        backend: SeparationBackend,
        registry: ModelRegistry,
        allow_private_models: bool = True,
        work_dir: Path | None = None,
    ) -> None:
        self.backend = backend
        self.registry = registry
        self.allow_private_models = allow_private_models
        self.work_dir = work_dir

    def run(
        self,
        vocals: AudioBuffer,
        steps: tuple[CleanupStep, ...],
        device: DeviceHint,
        segment_size: int = 256,
        overlap: int = 4,
        on_progress: ProgressHook | None = None,
    ) -> CleanupResult:
        current = vocals
        parts: dict[StemKind, AudioBuffer] = {}
        applied: list[CleanupStep] = []
        skipped: dict[CleanupStep, str] = {}

        for step in STEP_ORDER:
            if step not in steps:
                continue

            model = self._model_for(step)
            if model is None:
                skipped[step] = "לא נמצא מודל מתאים בקטלוג."
                continue
            if not self.allow_private_models and not model.license.is_redistributable:
                skipped[step] = model.license.note_he or "המודל אינו מאושר להפצה."
                log.info("skipping %s: licence policy excludes %s", step.value, model.id)
                continue

            if on_progress is not None:
                on_progress(step, STEP_LABELS_HE[step])

            try:
                produced = self.backend.separate(
                    current,
                    SeparationRequest(
                        model_id=model.id,
                        wanted=frozenset(
                            {StemKind.VOCALS, StemKind.AMBIENCE, StemKind.LEAD, StemKind.BACKING}
                        ),
                        overlap=overlap,
                        segment_size=segment_size,
                    ),
                    device,
                )
            except EngineError as exc:
                # One optional pass failing must not lose the separation that
                # already succeeded -- record why and carry on with what we have.
                skipped[step] = exc.user_message.render()
                log.warning("cleanup step %s failed: %s", step.value, exc.detail)
                continue

            if step is CleanupStep.KARAOKE:
                current, extra, reason = self._apply_karaoke(current, produced.parts)
            else:
                current, extra, reason = self._apply_removal(current, produced.parts)

            if reason is not None:
                skipped[step] = reason
                continue

            parts.update(extra)
            applied.append(step)

        self.backend.unload()
        return CleanupResult(
            vocals=current, parts=parts, applied=tuple(applied), skipped=skipped
        )

    # -- internals ---------------------------------------------------------- #

    def _model_for(self, step: CleanupStep):  # type: ignore[no-untyped-def]
        candidates = self.registry.of_kind(_STEP_KINDS[step])
        if not candidates:
            return None
        # Prefer a redistributable model when one exists at all.
        redistributable = [m for m in candidates if m.license.is_redistributable]
        pool = redistributable or candidates
        return max(pool, key=lambda m: (m.sdr if m.sdr is not None else -1.0))

    def _apply_removal(
        self, before: AudioBuffer, produced: dict[StemKind, AudioBuffer]
    ) -> tuple[AudioBuffer, dict[StemKind, AudioBuffer], str | None]:
        kept = produced.get(StemKind.VOCALS)
        if kept is None:
            return before, {}, "המודל לא החזיר את השכבה הנקייה."

        kept = fit_length(kept, before.frames)
        if is_silent(kept) or rms(kept) < rms(before) * _COLLAPSE_RATIO:
            return before, {}, "הניקוי הסיר כמעט את כל השירה — דילגנו עליו."

        removed = produced.get(StemKind.AMBIENCE) or produced.get(StemKind.OTHER)
        extra: dict[StemKind, AudioBuffer] = {}
        if removed is not None:
            extra[StemKind.AMBIENCE] = fit_length(removed, before.frames)
        else:
            # Derive it: whatever the model kept, subtracted from what went in.
            extra[StemKind.AMBIENCE] = subtract(before, kept)
        return kept, extra, None

    def _apply_karaoke(
        self, before: AudioBuffer, produced: dict[StemKind, AudioBuffer]
    ) -> tuple[AudioBuffer, dict[StemKind, AudioBuffer], str | None]:
        lead = produced.get(StemKind.LEAD) or produced.get(StemKind.VOCALS)
        if lead is None:
            return before, {}, "המודל לא החזיר את השירה הראשית."

        lead = fit_length(lead, before.frames)
        if is_silent(lead) or rms(lead) < rms(before) * _COLLAPSE_RATIO:
            return before, {}, "לא זוהתה שירה ראשית נפרדת — השארנו את השירה כמו שהיא."

        backing = produced.get(StemKind.BACKING) or produced.get(StemKind.INSTRUMENTAL)
        extra = {
            StemKind.LEAD: lead,
            StemKind.BACKING: (
                fit_length(backing, before.frames)
                if backing is not None
                else subtract(before, lead)
            ),
        }
        return lead, extra, None
