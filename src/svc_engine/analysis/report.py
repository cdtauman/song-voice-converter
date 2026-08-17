"""Tie the analysis stages together into one report about a song.

`analyze_vocal` runs the F0 model once and hands the same curve and the same
per-frame energy to every downstream stage, so the range, the segments and the
preview all describe the same measurement. The result serialises to the JSON the
CLI writes today and the Phase 8 GUI will read later, and carries the raw F0
track (decimated) so a curve can be drawn without re-running the model.

The math stages are pure numpy; only `analyze_vocal` touches an F0 model, and
only `plot_report` touches matplotlib -- which is optional and absent in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from svc_engine.analysis.features import align_to, frame_rms
from svc_engine.analysis.key import KeyEstimate, estimate_key_from_audio
from svc_engine.analysis.midi import hz_to_midi, midi_to_note_name
from svc_engine.analysis.preview_picker import PreviewChoice, PreviewWeights, pick_preview
from svc_engine.analysis.range import RangeStats, compute_range
from svc_engine.analysis.segments import (
    Segment,
    find_voiced_segments,
    voiced_fraction,
)
from svc_engine.backends.base import AudioBuffer, DeviceHint, F0Curve
from svc_engine.backends.f0 import F0Extractor

__all__ = ["AnalysisReport", "analyze_vocal", "plot_report"]

_MAX_TRACK_POINTS = 2000
#: Frozen singleton default (avoids B008 on the analyze_vocal signature).
_DEFAULT_PREVIEW_WEIGHTS = PreviewWeights()


@dataclass(frozen=True)
class AnalysisReport:
    source: str
    seconds: float
    sample_rate: int
    f0_method: str
    hop_seconds: float
    range: RangeStats
    key: KeyEstimate
    segments: tuple[Segment, ...]
    voiced_fraction: float
    preview: PreviewChoice
    f0_hz: np.ndarray  # per-frame, 0 = unvoiced; not serialised in full

    def summary_he(self) -> str:
        r = self.range
        if r.voiced_frames == 0:
            return "לא זוהתה שירה בקובץ."
        median = midi_to_note_name(r.p50)
        low = midi_to_note_name(r.p05)
        high = midi_to_note_name(r.p95)
        pv = self.preview.best
        pv_txt = (
            f" · קטע מייצג: {pv.start_seconds:.0f}–{pv.end_seconds:.0f} שנ׳"
            if pv is not None
            else ""
        )
        return (
            f"מנעד {low}–{high} (חציון {median}) · סולם {self.key.name_he} · "
            f"שירה ב-{self.voiced_fraction * 100:.0f}% מהקובץ{pv_txt}"
        )

    def _f0_track(self) -> list[list[float | None]]:
        """Decimated [time, midi] pairs for a UI curve; unvoiced -> null midi."""
        n = self.f0_hz.size
        if n == 0:
            return []
        step = max(1, n // _MAX_TRACK_POINTS)
        idx = np.arange(0, n, step)
        midi = hz_to_midi(self.f0_hz[idx])
        times = idx * self.hop_seconds
        return [
            [round(float(t), 3), (None if np.isnan(m) else round(float(m), 3))]
            for t, m in zip(times, midi, strict=True)
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "seconds": round(self.seconds, 3),
            "sample_rate": self.sample_rate,
            "f0_method": self.f0_method,
            "hop_seconds": self.hop_seconds,
            "summary_he": self.summary_he(),
            "range": self.range.to_dict(),
            "key": self.key.to_dict(),
            "voiced_fraction": round(self.voiced_fraction, 4),
            "segments": [s.to_dict() for s in self.segments],
            "preview": self.preview.to_dict(),
            "f0_track": self._f0_track(),
        }


def analyze_vocal(
    audio: AudioBuffer,
    extractor: F0Extractor,
    device: DeviceHint | None = None,
    hop_seconds: float = 0.01,
    preview_weights: PreviewWeights = _DEFAULT_PREVIEW_WEIGHTS,
    source: str = "",
) -> AnalysisReport:
    """Full analysis of one vocal track. Runs the F0 model exactly once."""
    device = device or DeviceHint()
    curve: F0Curve = extractor.extract(audio, device, hop_seconds)
    f0_hz = np.asarray(curve.hz, dtype=np.float64).ravel()
    n_frames = f0_hz.size

    mono = audio.samples.mean(axis=0) if audio.samples.ndim == 2 else audio.samples
    energy = align_to(
        frame_rms(mono, audio.sample_rate, curve.hop_seconds), n_frames
    )

    stats = compute_range(f0_hz, energy)
    key = estimate_key_from_audio(mono, audio.sample_rate)
    segments = find_voiced_segments(f0_hz, curve.hop_seconds, energy)
    vf = voiced_fraction(segments, audio.seconds)
    preview = pick_preview(
        f0_hz, curve.hop_seconds, energy, weights=preview_weights
    )

    method = extractor.info().backend_id
    return AnalysisReport(
        source=source,
        seconds=audio.seconds,
        sample_rate=audio.sample_rate,
        f0_method=method,
        hop_seconds=curve.hop_seconds,
        range=stats,
        key=key,
        segments=tuple(segments),
        voiced_fraction=vf,
        preview=preview,
        f0_hz=f0_hz,
    )


def plot_report(report: AnalysisReport, path: Path | str) -> Path:
    """Draw the F0 curve with the range band, segments, and preview marked.

    matplotlib is an optional inspection dependency, not part of the locked
    runtime matrix. A missing install is a clear error, not a crash mid-run.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "הצגת גרף דורשת את matplotlib. התקן אותו כדי להשתמש ב---plot."
        ) from exc

    path = Path(path)
    midi = hz_to_midi(report.f0_hz)
    times = np.arange(report.f0_hz.size) * report.hop_seconds

    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(times, midi, ".", markersize=1.5, color="#1f77b4", label="F0")

    r = report.range
    if r.voiced_frames:
        ax.axhspan(r.p05, r.p95, color="#1f77b4", alpha=0.10, label="p05–p95")
        ax.axhline(r.p50, color="#ff7f0e", lw=1.0, label=f"median {midi_to_note_name(r.p50)}")

    for seg in report.segments:
        ax.axvspan(seg.start_seconds, seg.end_seconds, color="#2ca02c", alpha=0.06)

    if report.preview.best is not None:
        pv = report.preview.best
        ax.axvspan(pv.start_seconds, pv.end_seconds, color="#d62728", alpha=0.12, label="preview")

    ax.set_xlabel("time (s)")
    ax.set_ylabel("pitch (MIDI)")
    ax.set_title(f"{report.source or 'vocal'} — {report.key.name} — {report.f0_method}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
