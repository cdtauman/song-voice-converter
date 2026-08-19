"""Fast, explainable checks over recordings before expensive RVC training.

The checks intentionally run without torch.  They measure duration, clipping,
an estimated signal-to-noise ratio and an acoustic-consistency fingerprint.
The latter is not biometric identification; it catches the practical mistakes
the wizard can fix (a second speaker, a music-only file, or a very different
microphone) before the user waits hours for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.fft import dct

from svc_engine.audio import load_audio, to_mono
from svc_engine.backends.base import AudioBuffer

__all__ = ["FileQuality", "QualityIssue", "QualityReport", "inspect_recordings"]

_FRAME_SECONDS = 0.04
_CLIP_LEVEL = 10 ** (-0.1 / 20.0)
_WARN_MINUTES = 15.0
_BLOCK_MINUTES = 5.0


@dataclass(frozen=True)
class QualityIssue:
    code: str
    severity: str
    message_he: str
    action_he: str

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "severity": self.severity,
            "message_he": self.message_he,
            "action_he": self.action_he,
        }


@dataclass(frozen=True)
class FileQuality:
    path: str
    seconds: float
    active_seconds: float
    snr_db: float
    clipping_ratio: float
    peak_dbfs: float
    fingerprint: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "seconds": round(self.seconds, 3),
            "active_seconds": round(self.active_seconds, 3),
            "snr_db": round(self.snr_db, 2),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "peak_dbfs": round(self.peak_dbfs, 2),
        }


@dataclass(frozen=True)
class QualityReport:
    files: tuple[FileQuality, ...]
    total_seconds: float
    active_seconds: float
    median_snr_db: float
    clipping_ratio: float
    speaker_consistency: float
    issues: tuple[QualityIssue, ...]

    @property
    def can_train(self) -> bool:
        return not any(issue.severity == "block" for issue in self.issues)

    @property
    def score(self) -> int:
        penalty = sum(35 if i.severity == "block" else 12 for i in self.issues)
        return max(0, min(100, 100 - penalty))

    @property
    def summary_he(self) -> str:
        minutes = self.active_seconds / 60.0
        if self.can_train and not self.issues:
            return f"החומר מוכן לאימון: {minutes:.1f} דקות פעילות, ללא בעיה בולטת."
        state = "אפשר להמשיך, אבל מומלץ לשפר" if self.can_train else "צריך לתקן לפני האימון"
        return f"{state}: {minutes:.1f} דקות פעילות, {len(self.issues)} הערות."

    def to_dict(self) -> dict[str, object]:
        return {
            "files": [item.to_dict() for item in self.files],
            "total_seconds": round(self.total_seconds, 3),
            "active_seconds": round(self.active_seconds, 3),
            "median_snr_db": round(self.median_snr_db, 2),
            "clipping_ratio": round(self.clipping_ratio, 6),
            "speaker_consistency": round(self.speaker_consistency, 4),
            "score": self.score,
            "can_train": self.can_train,
            "summary_he": self.summary_he,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _frame_rms(samples: np.ndarray, sample_rate: int) -> np.ndarray:
    frame = max(1, int(round(sample_rate * _FRAME_SECONDS)))
    usable = samples[: samples.size // frame * frame]
    if not usable.size:
        return np.zeros(0, dtype=np.float64)
    framed = usable.reshape(-1, frame).astype(np.float64, copy=False)
    return np.sqrt(np.mean(np.square(framed), axis=1) + 1e-12)


def _fingerprint(samples: np.ndarray, sample_rate: int) -> tuple[float, ...]:
    """Median cepstral envelope; robust to words and pitch, sensitive to source changes."""
    if samples.size < 512:
        return tuple(0.0 for _ in range(20))
    target = 16000
    if sample_rate != target:
        positions = np.linspace(0, samples.size - 1, int(samples.size * target / sample_rate))
        samples = np.interp(positions, np.arange(samples.size), samples).astype(np.float32)
    window = 512
    hop = 256
    count = 1 + max(0, (samples.size - window) // hop)
    if count <= 0:
        return tuple(0.0 for _ in range(20))
    # At most five minutes are needed for a stable quality fingerprint.
    indices = np.linspace(0, count - 1, min(count, 18000), dtype=int)
    frames = np.stack([samples[i * hop : i * hop + window] for i in indices])
    spectrum = np.abs(np.fft.rfft(frames * np.hanning(window), axis=1))
    cepstra = dct(np.log1p(spectrum), type=2, axis=1, norm="ortho")[:, 1:21]
    vector = np.median(cepstra, axis=0)
    norm = float(np.linalg.norm(vector))
    if norm:
        vector = vector / norm
    return tuple(float(value) for value in vector)


def _inspect_one(path: Path) -> FileQuality:
    audio: AudioBuffer = to_mono(load_audio(path))
    samples = np.asarray(audio.samples[0], dtype=np.float32)
    levels = _frame_rms(samples, audio.sample_rate)
    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    peak_db = 20.0 * np.log10(max(peak, 1e-9))
    clipping = float(np.mean(np.abs(samples) >= _CLIP_LEVEL)) if samples.size else 0.0
    if levels.size:
        floor = float(np.percentile(levels, 10))
        speech = float(np.percentile(levels, 75))
        snr = 20.0 * np.log10(max(speech, 1e-9) / max(floor, 1e-9))
        threshold = max(10 ** (-55.0 / 20.0), float(np.max(levels)) * 0.01)
        active = float(np.count_nonzero(levels >= threshold)) * _FRAME_SECONDS
    else:
        snr, active = 0.0, 0.0
    return FileQuality(
        path=str(path),
        seconds=audio.seconds,
        active_seconds=min(audio.seconds, active),
        snr_db=float(np.clip(snr, 0.0, 80.0)),
        clipping_ratio=clipping,
        peak_dbfs=float(peak_db),
        fingerprint=_fingerprint(samples, audio.sample_rate),
    )


def _consistency(files: list[FileQuality]) -> float:
    if len(files) < 2:
        return 1.0
    vectors = np.asarray([item.fingerprint for item in files], dtype=np.float64)
    centre = np.median(vectors, axis=0)
    centre /= max(float(np.linalg.norm(centre)), 1e-9)
    similarities = np.clip(vectors @ centre, -1.0, 1.0)
    return float(np.median((similarities + 1.0) / 2.0))


def inspect_recordings(
    recordings: list[Path | str], *, strict_consistency: bool = False
) -> QualityReport:
    paths = [Path(path) for path in recordings]
    if not paths:
        raise ValueError("at least one recording is required")
    files = [_inspect_one(path) for path in paths]
    total = sum(item.seconds for item in files)
    active = sum(item.active_seconds for item in files)
    weighted_clip = sum(item.clipping_ratio * item.seconds for item in files) / max(total, 1e-9)
    snr = float(np.median([item.snr_db for item in files]))
    consistency = _consistency(files)
    issues: list[QualityIssue] = []
    if active < _BLOCK_MINUTES * 60:
        issues.append(
            QualityIssue(
                "too_short",
                "block",
                "יש פחות מחמש דקות של קול פעיל.",
                "הוסף הקלטות עד שיש לפחות 15 דקות.",
            )
        )
    elif active < _WARN_MINUTES * 60:
        issues.append(
            QualityIssue(
                "short",
                "warn",
                "יש פחות מ־15 דקות של קול פעיל.",
                "עוד חומר מגוון ישפר את יציבות הקול.",
            )
        )
    if snr < 12.0:
        issues.append(
            QualityIssue(
                "noise", "warn", "זוהה יחס רעש גבוה בחומר.", "העדף הקלטה קרובה למיקרופון ובחדר שקט."
            )
        )
    if weighted_clip > 0.001:
        issues.append(
            QualityIssue(
                "clipping",
                "warn",
                "בחלק מההקלטה העוצמה נחתכת.",
                "הנמך את עוצמת ההקלטה והקלט שוב קטעים שנחרכו.",
            )
        )
    if consistency < 0.72:
        issues.append(
            QualityIssue(
                "speaker_mismatch",
                "block" if strict_consistency else "warn",
                "ההקלטות אינן עקביות ונראה שיש יותר מקול אחד.",
                (
                    "השאר רק הקלטות של אותו אדם ובאותו סגנון הקלטה."
                    if strict_consistency
                    else "הבדיקה תחזור לאחר הפרדת המוזיקה והניקוי."
                ),
            )
        )
    return QualityReport(
        files=tuple(files),
        total_seconds=total,
        active_seconds=active,
        median_snr_db=snr,
        clipping_ratio=weighted_clip,
        speaker_consistency=consistency,
        issues=tuple(issues),
    )
