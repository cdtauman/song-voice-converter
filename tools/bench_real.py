"""Phase 2 real-audio benchmark: measure what can be measured, then hand the
rest to a pair of ears.

This is the harness for the remaining Phase 2 DoD item. It deliberately splits
the work in two, because the two halves have different kinds of truth:

**Part A -- the songs (no ground truth).** Five real songs, every applicable
quality mode. Objectively measurable here: runtime, peak device and host memory,
which backend actually ran, sample alignment, and whether the stems still sum
back to the mix. Bleed, fullness and artifacts are *not* objectively measurable
without a reference, so this part exports audio into a blind listening kit
instead of inventing numbers for them.

**Part B -- the dry vocals (exact ground truth).** Following
docs/testing.md section 1: mix a dry vocal with a known instrumental, separate
it, and compare against the original. Here bleed, fullness and instrumental
damage *are* real measurements, because the answer is known.

**Part C -- the blind kit.** Variants written under neutral names in a random
order, the mapping stored in a file the listener is not meant to open, plus an
empty scoring sheet. docs/testing.md section 3.4 makes blind listening the
primary metric; this makes it practical to actually do.

The synthetic fixture is not used here at all. Reconstruction accuracy on a
generated mix says the pipeline is exact; it says nothing about separation
quality, and the two must not be confused.

    python tools/bench_real.py --material D:\\svc-test --out benchmark/results/phase2
    python tools/bench_real.py --material D:\\svc-test --seconds 45   # quick pass
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.audio import io as audio_io  # noqa: E402
from svc_engine.audio.buffers import add, align_all, fit_length  # noqa: E402
from svc_engine.backends.base import AudioBuffer  # noqa: E402
from svc_engine.backends.separation import StemKind  # noqa: E402
from svc_engine.config import paths  # noqa: E402
from svc_engine.separation import (  # noqa: E402
    PROFILES,
    CleanupStep,
    QualityLevel,
    SeparationPipeline,
)

AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".wma", ".opus"}

#: Below this, a frame of the reference vocal counts as "not singing". Energy
#: the separator puts there can only have come from the instrumental.
_SILENCE_DB = -50.0
_FRAME = 2048


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _mono(buffer: AudioBuffer) -> np.ndarray:
    return buffer.samples.astype(np.float64).mean(axis=0)


def _frame_db(signal: np.ndarray, frame: int = _FRAME) -> np.ndarray:
    usable = (len(signal) // frame) * frame
    blocks = signal[:usable].reshape(-1, frame)
    power = np.sqrt(np.mean(blocks ** 2, axis=1))
    return 20.0 * np.log10(np.maximum(power, 1e-12))


def sdr_db(estimate: AudioBuffer, reference: AudioBuffer) -> float:
    est, ref = align_all([estimate, reference])
    e, r = _mono(est), _mono(ref)
    noise = float(np.sum((r - e) ** 2))
    signal = float(np.sum(r ** 2))
    if noise <= 0.0:
        return float("inf")
    if signal <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(signal / noise))


def bleed_db(estimate: AudioBuffer, reference_vocal: AudioBuffer) -> float | None:
    """Level the separator left in the vocal where the singer is silent.

    This is the honest version of "vocal bleed": in frames where the reference
    vocal is below the silence floor, anything present in the estimate came from
    the instrumental. Reported relative to the estimate's overall level, so a
    quiet mix and a loud one are comparable. Lower is better.
    """
    est, ref = align_all([estimate, reference_vocal])
    e, r = _mono(est), _mono(ref)
    ref_db = _frame_db(r)
    silent = ref_db < _SILENCE_DB
    if not silent.any():
        return None

    usable = len(ref_db) * _FRAME
    blocks = e[:usable].reshape(-1, _FRAME)
    leaked = float(np.sum(blocks[silent] ** 2))
    total = float(np.sum(e[:usable] ** 2))
    if total <= 0.0:
        return None
    return 10.0 * float(np.log10(max(leaked, 1e-20) / total))


def fullness_db(estimate: AudioBuffer, reference_vocal: AudioBuffer) -> float | None:
    """How much of the singing survived, in the frames where there was singing.

    0dB means the level was preserved. Negative means content was thinned or
    cut -- the "missing content" failure mode, which is the opposite risk to
    bleed and has to be reported alongside it.
    """
    est, ref = align_all([estimate, reference_vocal])
    e, r = _mono(est), _mono(ref)
    ref_db = _frame_db(r)
    voiced = ref_db >= _SILENCE_DB
    if not voiced.any():
        return None

    usable = len(ref_db) * _FRAME
    e_blocks = e[:usable].reshape(-1, _FRAME)
    r_blocks = r[:usable].reshape(-1, _FRAME)
    est_energy = float(np.sum(e_blocks[voiced] ** 2))
    ref_energy = float(np.sum(r_blocks[voiced] ** 2))
    if ref_energy <= 0.0:
        return None
    return 10.0 * float(np.log10(max(est_energy, 1e-20) / ref_energy))


def click_count(buffer: AudioBuffer, threshold: float = 0.25) -> int:
    """Sample-to-sample jumps large enough to be heard as a click.

    A proxy, not a verdict -- docs/testing.md section 3.3 forbids using an
    uncalibrated metric as a gate. It is here because a click count that jumps
    between two configurations is worth looking into.
    """
    signal = _mono(buffer)
    if len(signal) < 3:
        return 0
    return int(np.sum(np.abs(np.diff(signal)) > threshold))


def hf_ratio_db(buffer: AudioBuffer) -> float:
    """Energy above ~8kHz relative to the whole, in dB.

    Separation artifacts usually show up as added high-frequency fizz, so a
    large shift between two variants on the same source is a signal.
    """
    signal = _mono(buffer)
    spectrum = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
    freqs = np.fft.rfftfreq(len(signal), 1.0 / buffer.sample_rate)
    total = float(np.sum(spectrum ** 2))
    high = float(np.sum(spectrum[freqs >= 8000.0] ** 2))
    if total <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(max(high, 1e-20) / total))


# --------------------------------------------------------------------------- #
# result rows
# --------------------------------------------------------------------------- #

@dataclass
class RunRow:
    case: str
    kind: str                      # "song" | "controlled"
    quality: str
    cleanup: str
    backend_used: str
    device_name: str
    seconds_of_audio: float
    runtime_seconds: float
    realtime_factor: float | None
    peak_device_mb: float | None
    peak_host_mb: float | None
    #: Exact-length agreement with the input. Anything but 0 is a bug.
    alignment_error_samples: int
    #: vocals + instrumental vs the input mix. A pipeline check, not a quality one.
    reconstruction_db: float
    stems: list[str]
    oom_fallbacks: list[str]
    notes_he: list[str]
    # ground truth only
    sdr_vocals: float | None = None
    sdr_instrumental: float | None = None
    vocal_bleed_db: float | None = None
    vocal_fullness_db: float | None = None
    clicks_vocals: int | None = None
    hf_ratio_vocals_db: float | None = None
    hf_ratio_reference_db: float | None = None
    audio_dir: str = ""


@dataclass
class Manifest:
    created: str
    machine: str
    material_root: str
    seconds_per_case: float | None
    songs: list[str] = field(default_factory=list)
    dry_vocals: list[str] = field(default_factory=list)
    instrumentals: list[str] = field(default_factory=list)
    profiles: dict[str, dict] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# material discovery
# --------------------------------------------------------------------------- #

def _audio_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES
    )


def discover(root: Path) -> tuple[list[Path], list[Path], list[Path]]:
    """Find material under `root`.

    Convention, so no manifest is needed to get started:

        <root>/songs/          the 5 full songs
        <root>/dry/            the 2 dry vocal clips
        <root>/instrumentals/  optional: true instrumentals for the controlled mix

    Loose files directly under `root` are treated as songs, so pointing this at
    a flat folder also works.
    """
    songs = _audio_in(root / "songs") or _audio_in(root)
    dry = _audio_in(root / "dry")
    instrumentals = _audio_in(root / "instrumentals")
    return songs, dry, instrumentals


# --------------------------------------------------------------------------- #
# the runs
# --------------------------------------------------------------------------- #

def _run_once(
    pipeline: SeparationPipeline,
    source: Path,
    quality: QualityLevel,
    cleanup: tuple[CleanupStep, ...],
    seconds: float | None,
    audio_out: Path,
) -> tuple[RunRow, dict[StemKind, AudioBuffer]]:
    mix = audio_io.load_audio(source, duration_seconds=seconds)

    started = time.perf_counter()
    outcome = pipeline.run(
        source, level=quality, cleanup=cleanup, duration_seconds=seconds
    )
    runtime = time.perf_counter() - started

    stems = outcome.stems
    alignment = max(abs(b.frames - mix.frames) for b in stems.values())

    vocals = stems.get(StemKind.VOCALS)
    instrumental = stems.get(StemKind.INSTRUMENTAL)
    if vocals is not None and instrumental is not None:
        reconstruction = sdr_db(add(vocals, instrumental), mix)
    else:
        reconstruction = float("nan")

    audio_out.mkdir(parents=True, exist_ok=True)
    for kind, buffer in stems.items():
        audio_io.save_wav(buffer, audio_out / f"{kind.value}.wav", bit_depth=24)

    peak = outcome.peak_memory
    row = RunRow(
        case=source.stem,
        kind="song",
        quality=quality.value,
        cleanup="+".join(s.value for s in cleanup) or "none",
        backend_used=outcome.device.backend.value,
        device_name=outcome.device.name,
        seconds_of_audio=round(mix.seconds, 2),
        runtime_seconds=round(runtime, 1),
        realtime_factor=(
            round(runtime / mix.seconds, 2) if mix.seconds else None
        ),
        peak_device_mb=(
            round(peak.device_mb, 1) if peak and peak.device_mb is not None else None
        ),
        peak_host_mb=(
            round(peak.host_mb, 1) if peak and peak.host_mb is not None else None
        ),
        alignment_error_samples=alignment,
        reconstruction_db=round(reconstruction, 2),
        stems=sorted(k.value for k in stems),
        oom_fallbacks=list(outcome.fallback_steps),
        notes_he=list(outcome.notes_he),
        clicks_vocals=click_count(vocals) if vocals is not None else None,
        hf_ratio_vocals_db=(
            round(hf_ratio_db(vocals), 2) if vocals is not None else None
        ),
        audio_dir=str(audio_out),
    )
    return row, stems


def run_songs(
    songs: list[Path],
    qualities: list[QualityLevel],
    cleanup_sets: list[tuple[CleanupStep, ...]],
    seconds: float | None,
    out_dir: Path,
) -> list[RunRow]:
    rows: list[RunRow] = []
    pipeline = SeparationPipeline(paths=paths())

    for song in songs:
        for quality in qualities:
            # Cleanup is an extra comparison of the Balanced path only. Applying
            # it to Fast/Max would multiply expensive runs and contradict the
            # CLI contract for --with-cleanup.
            applicable_cleanup = (
                cleanup_sets if quality is QualityLevel.BALANCED else [()]
            )
            for cleanup in applicable_cleanup:
                label = f"{song.stem}__{quality.value}"
                if cleanup:
                    label += "__" + "+".join(s.value for s in cleanup)
                print(f"  {label} ...", flush=True)
                row, _ = _run_once(
                    pipeline, song, quality, cleanup, seconds,
                    out_dir / "audio" / label,
                )
                rows.append(row)
                print(
                    f"    {row.runtime_seconds:6.1f}s  x{row.realtime_factor}  "
                    f"{row.backend_used}  align={row.alignment_error_samples}  "
                    f"recon={row.reconstruction_db}dB  "
                    f"peak={row.peak_device_mb}MB",
                    flush=True,
                )
    return rows


def build_controlled_mix(
    dry_vocal: Path, instrumental: Path, seconds: float | None, work: Path
) -> tuple[Path, AudioBuffer, AudioBuffer]:
    """dry vocal + known instrumental, at a realistic relative level.

    -3dB on the vocal against the instrumental is roughly where a pop mix sits.
    Both references are returned so the estimate can be scored against them.
    """
    vocal = audio_io.load_audio(dry_vocal, duration_seconds=seconds)
    backing = audio_io.load_audio(instrumental, duration_seconds=seconds)
    frames = min(vocal.frames, backing.frames)
    vocal = fit_length(vocal, frames)
    backing = fit_length(backing, frames)

    vocal_peak = float(np.max(np.abs(vocal.samples))) or 1.0
    backing_peak = float(np.max(np.abs(backing.samples))) or 1.0
    vocal = AudioBuffer(
        samples=(vocal.samples / vocal_peak * 0.5).astype(np.float32),
        sample_rate=vocal.sample_rate,
    )
    backing = AudioBuffer(
        samples=(backing.samples / backing_peak * 0.35).astype(np.float32),
        sample_rate=backing.sample_rate,
    )

    work.mkdir(parents=True, exist_ok=True)
    mix_path = work / f"controlled_{dry_vocal.stem}_{instrumental.stem}.wav"
    audio_io.save_wav(add(vocal, backing), mix_path, bit_depth=24)
    return mix_path, vocal, backing


def run_controlled(
    dry_vocals: list[Path],
    instrumentals: list[Path],
    qualities: list[QualityLevel],
    seconds: float | None,
    out_dir: Path,
) -> list[RunRow]:
    """The only part of this benchmark with exact ground truth."""
    rows: list[RunRow] = []
    pipeline = SeparationPipeline(paths=paths())
    work = out_dir / "controlled_mixes"

    for dry in dry_vocals:
        instrumental = instrumentals[0]
        mix_path, ref_vocal, ref_backing = build_controlled_mix(
            dry, instrumental, seconds, work
        )
        for quality in qualities:
            label = f"controlled_{dry.stem}__{quality.value}"
            print(f"  {label} ...", flush=True)
            row, stems = _run_once(
                pipeline, mix_path, quality, (), seconds, out_dir / "audio" / label
            )
            row.kind = "controlled"
            row.case = f"{dry.stem}+{instrumental.stem}"

            estimate = stems.get(StemKind.VOCALS)
            estimated_backing = stems.get(StemKind.INSTRUMENTAL)
            if estimate is not None:
                row.sdr_vocals = round(sdr_db(estimate, ref_vocal), 2)
                bleed = bleed_db(estimate, ref_vocal)
                row.vocal_bleed_db = round(bleed, 2) if bleed is not None else None
                fullness = fullness_db(estimate, ref_vocal)
                row.vocal_fullness_db = (
                    round(fullness, 2) if fullness is not None else None
                )
                row.hf_ratio_reference_db = round(hf_ratio_db(ref_vocal), 2)
            if estimated_backing is not None:
                row.sdr_instrumental = round(sdr_db(estimated_backing, ref_backing), 2)

            rows.append(row)
            print(
                f"    SDR voc {row.sdr_vocals}dB  inst {row.sdr_instrumental}dB  "
                f"bleed {row.vocal_bleed_db}dB  fullness {row.vocal_fullness_db}dB",
                flush=True,
            )
    return rows


# --------------------------------------------------------------------------- #
# blind listening kit
# --------------------------------------------------------------------------- #

def build_blind_kit(rows: list[RunRow], out_dir: Path, seed: int = 20260817) -> Path:
    """Copy each variant under a neutral name, in a shuffled order.

    The mapping goes into `KEY_DO_NOT_OPEN.json`. The point is not secrecy from
    a determined reader -- it is that the listener should not know which is the
    ensemble while they are listening to it.
    """
    kit = out_dir / "blind"
    kit.mkdir(parents=True, exist_ok=True)

    by_case: dict[str, list[RunRow]] = {}
    for row in rows:
        by_case.setdefault(f"{row.kind}:{row.case}", []).append(row)

    rng = random.Random(seed)
    key: dict[str, dict[str, str]] = {}
    sheet = [
        "case,variant,better_or_worse,vocal_bleed,fullness,"
        "instrumental_damage,artifacts,notes"
    ]

    for case, variants in sorted(by_case.items()):
        if len(variants) < 2:
            continue  # nothing to compare blind
        shuffled = variants[:]
        rng.shuffle(shuffled)
        case_dir = kit / case.replace(":", "_")
        case_dir.mkdir(parents=True, exist_ok=True)
        key[case] = {}

        for index, row in enumerate(shuffled):
            name = f"variant_{chr(ord('a') + index)}"
            key[case][name] = f"{row.quality} / cleanup={row.cleanup}"
            source = Path(row.audio_dir) / "vocals.wav"
            if source.exists():
                shutil.copy2(source, case_dir / f"{name}_vocals.wav")
            source = Path(row.audio_dir) / "instrumental.wav"
            if source.exists():
                shutil.copy2(source, case_dir / f"{name}_instrumental.wav")
            sheet.append(f"{case},{name},,,,,,")

    (kit / "KEY_DO_NOT_OPEN.json").write_text(
        json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sheet_path = kit / "scoring_sheet.csv"
    sheet_path.write_text("\n".join(sheet) + "\n", encoding="utf-8")

    (kit / "README.md").write_text(
        "# האזנה עיוורת — Phase 2\n\n"
        "לכל מקרה בדיקה יש תיקייה עם כמה וריאנטים בשמות `variant_a`, `variant_b`…\n"
        "בסדר אקראי. **אל תפתח `KEY_DO_NOT_OPEN.json` לפני שסיימת לדרג.**\n\n"
        "## מה לדרג, לכל וריאנט\n\n"
        "| עמודה | מה למלא |\n"
        "|-------|----------|\n"
        "| `better_or_worse` | דירוג 1 (הגרוע) עד N (הטוב), בתוך אותו מקרה |\n"
        "| `vocal_bleed` | 0 = לא שומע כלים בשירה · 3 = מפריע |\n"
        "| `fullness` | 0 = השירה שלמה · 3 = נחתכה, חסרות נשימות/זנבות |\n"
        "| `instrumental_damage` | 0 = הפלייבק נקי · 3 = חורים או עמעום |\n"
        "| `artifacts` | 0 = נקי · 3 = רעש מטאלי / קרקושים |\n\n"
        "למלא ב-`scoring_sheet.csv` ולהחזיר. אחרי זה נקבעות ברירות המחדל של Phase 2.\n",
        encoding="utf-8",
    )
    return kit


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--material", required=True,
        help="folder holding songs/ , dry/ and optionally instrumentals/",
    )
    parser.add_argument("--out", default="benchmark/results/phase2")
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="process only the first N seconds of each case (for a quick pass)",
    )
    parser.add_argument(
        "--quality", nargs="+", default=["fast", "balanced", "max"],
    )
    parser.add_argument(
        "--with-cleanup", action="store_true",
        help="also run a dereverb+karaoke variant of the balanced level",
    )
    parser.add_argument("--skip-blind", action="store_true")
    args = parser.parse_args(argv)

    root = Path(args.material)
    if not root.is_dir():
        print(f"material folder not found: {root}")
        return 2

    songs, dry, instrumentals = discover(root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    skipped: list[str] = []
    print(f"material: {root}")
    print(f"  songs: {len(songs)}")
    print(f"  dry vocals: {len(dry)}")
    print(f"  instrumentals: {len(instrumentals)}")
    if not songs:
        print("\nno songs found -- nothing to measure.")
        return 2
    if len(songs) < 5:
        skipped.append(
            f"only {len(songs)} songs supplied; docs/testing.md asks for 5 covering "
            "different failure cases"
        )
    if dry and not instrumentals:
        skipped.append(
            "dry vocals supplied but no instrumentals/ folder, so the controlled "
            "ground-truth mixes were not built"
        )
    if not dry:
        skipped.append(
            "no dry vocals supplied; bleed, fullness and instrumental damage cannot "
            "be measured objectively without them"
        )

    qualities = [QualityLevel(q) for q in args.quality]
    cleanup_sets: list[tuple[CleanupStep, ...]] = [()]
    if args.with_cleanup:
        cleanup_sets.append((CleanupStep.DEREVERB, CleanupStep.KARAOKE))

    print("\n--- part A: songs (runtime, memory, alignment; quality by ear) ---")
    rows = run_songs(songs, qualities, cleanup_sets, args.seconds, out_dir)

    if dry and instrumentals:
        print("\n--- part B: controlled mixes (exact ground truth) ---")
        rows += run_controlled(dry, instrumentals, qualities, args.seconds, out_dir)

    results = out_dir / "results.json"
    results.write_text(
        json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    manifest = Manifest(
        created=datetime.now(UTC).isoformat(timespec="seconds"),
        machine=platform.platform(),
        material_root=str(root),
        seconds_per_case=args.seconds,
        songs=[p.name for p in songs],
        dry_vocals=[p.name for p in dry],
        instrumentals=[p.name for p in instrumentals],
        profiles={
            level.value: {
                "models": list(PROFILES[level].models),
                "overlap": PROFILES[level].overlap,
                "segment_size": PROFILES[level].segment_size,
                "ensemble": PROFILES[level].ensemble_mode.value,
            }
            for level in qualities
        },
        skipped=skipped,
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(asdict(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if not args.skip_blind:
        kit = build_blind_kit(rows, out_dir)
        print(f"\nblind listening kit: {kit}")

    print(f"results: {results}")
    if skipped:
        print("\nnot measured:")
        for note in skipped:
            print(f"  - {note}")

    bad_alignment = [r for r in rows if r.alignment_error_samples != 0]
    if bad_alignment:
        print(f"\nFAIL: {len(bad_alignment)} run(s) returned stems of the wrong length")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
