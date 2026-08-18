"""Phase 6 benchmark: dynamics calibration and acoustic-space blind matrix.

With no inputs it exercises every path on deterministic synthetic material and
records objective invariants. With four real stems it also writes anonymized
WAV variants for the human blind-listening gate:

    python tools/bench_postfx.py \
      --dry dry.wav --converted converted.wav \
      --playback instrumental.wav --ambience ambience.wav \
      --audio-out benchmark/audio/postfx-song3

Objective measurements can reject broken output; they do not select the most
natural ambience. The winner stays null until the five-song listening sheet is
filled, per docs/testing.md 3.4 and 5b.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.audio.buffers import add, fit_length, peak  # noqa: E402
from svc_engine.audio.io import ffmpeg_exe, load_audio, save_wav  # noqa: E402
from svc_engine.backends.base import AudioBuffer, BackendInfo, F0Curve  # noqa: E402
from svc_engine.postfx import (  # noqa: E402
    AmbienceStrategy,
    PostFxConfig,
    PostFxPipeline,
    envelope_correlation,
    integrated_lufs,
    match_envelope,
)

SR = 44100
RESULTS = Path(__file__).resolve().parents[1] / "benchmark" / "results" / "postfx"


class IdentityShifter:
    """Synthetic benchmark has exact F0 already; no correction is expected."""

    def info(self) -> BackendInfo:
        return BackendInfo("identity", "בדיקה", True)

    def shift(self, audio: AudioBuffer, semitones: float) -> AudioBuffer:
        return audio


def _buffer(samples: np.ndarray, channels: int = 1) -> AudioBuffer:
    samples = np.asarray(samples, dtype=np.float32).reshape(1, -1)
    return AudioBuffer(np.repeat(samples, channels, axis=0), SR)


def synthetic_material(seconds: float = 6.0) -> tuple[AudioBuffer, ...]:
    n = int(SR * seconds)
    t = np.arange(n) / SR
    phrase = 0.25 + 0.75 * np.square(np.sin(np.pi * t / seconds))
    dry = phrase * (0.20 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.sin(2 * np.pi * 440 * t))
    converted = 0.14 * np.sin(2 * np.pi * 110 * t) + 0.025 * np.sin(2 * np.pi * 7000 * t)
    delay = int(0.065 * SR)
    ambience = np.zeros(n)
    for repeats, gain in ((1, 0.12), (2, 0.07), (4, 0.035), (7, 0.018)):
        shift = repeats * delay
        ambience[shift:] += dry[:-shift] * gain
    playback = (
        0.10 * np.sin(2 * np.pi * 55 * t)
        + 0.06 * np.sin(2 * np.pi * 330 * t)
        + 0.025 * np.random.default_rng(6006).standard_normal(n)
    )
    return _buffer(dry), _buffer(converted), _buffer(playback, 2), _buffer(ambience)


def load_material(args: argparse.Namespace) -> tuple[tuple[AudioBuffer, ...], str]:
    supplied = [args.dry, args.converted, args.playback, args.ambience]
    if not any(supplied):
        return synthetic_material(), "synthetic"
    if not all(supplied):
        raise SystemExit("--dry, --converted, --playback and --ambience must be supplied together")
    material = tuple(load_audio(path) for path in supplied)
    target = min(buf.frames for buf in material)
    return tuple(fit_length(buf, target) for buf in material), "real supplied stems"


def envelope_matrix(dry: AudioBuffer, converted: AudioBuffer) -> list[dict[str, object]]:
    rows = []
    for rate in (0.0, 0.25, 0.5):
        result, report = match_envelope(dry, converted, rate)
        rows.append(
            {
                "rms_mix_rate": rate,
                "envelope_correlation": envelope_correlation(dry, result),
                "report": asdict(report),
            }
        )
    return rows


def strategy_matrix(
    material: tuple[AudioBuffer, ...], work_dir: Path
) -> tuple[list[dict[str, object]], dict[str, AudioBuffer]]:
    dry, converted, playback, ambience = material
    source_f0 = F0Curve(np.linspace(210.0, 230.0, max(3, int(dry.seconds * 100))), 0.01)
    output_f0 = F0Curve(source_f0.hz / 2.0, 0.01)
    rows: list[dict[str, object]] = []
    audio: dict[str, AudioBuffer] = {}
    raw = add(playback, fit_length(converted, playback.frames))
    audio["raw_phase5"] = raw
    raw_peak = peak(raw)
    rows.append(
        {
            "variant": "raw_phase5",
            "length_exact": raw.frames == playback.frames,
            "lufs": integrated_lufs(raw),
            "peak_dbfs": 20.0 * np.log10(raw_peak) if raw_peak > 0 else float("-inf"),
            "clipped": raw_peak > 1.0,
            "winner": None,
        }
    )
    for strategy in AmbienceStrategy:
        outcome = PostFxPipeline(
            PostFxConfig(ambience_strategy=strategy, target_lufs=-14.0), work_dir
        ).run(
            original_vocal=dry,
            converted_vocal=converted,
            playback=playback,
            original_ambience=ambience,
            reference_f0=source_f0,
            converted_f0=output_f0,
            semitones=-12,
            shifter=IdentityShifter(),
        )
        audio[strategy.value] = outcome.cover
        rows.append(
            {
                "variant": strategy.value,
                "length_exact": outcome.cover.frames == playback.frames,
                "lufs": outcome.report.mix.integrated_lufs,
                "target_error_lu": (
                    outcome.report.mix.integrated_lufs - outcome.report.mix.target_lufs
                ),
                "peak_dbfs": outcome.report.mix.peak_dbfs,
                "clipped": outcome.report.mix.clipped,
                "ambience": outcome.report.to_dict()["ambience"],
                "winner": None,
            }
        )
    return rows, audio


def write_blind_audio(audio: dict[str, AudioBuffer], out_dir: Path, seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    names = list(audio)
    random.Random(seed).shuffle(names)
    key: dict[str, str] = {}
    for index, name in enumerate(names, 1):
        label = f"variant_{index}"
        save_wav(audio[name], out_dir / f"{label}.wav", bit_depth=24)
        key[label] = name
    path = out_dir / "blind-key.json"
    path.write_text(json.dumps({"seed": seed, "key": key}, indent=2), encoding="utf-8")
    return path


def ffmpeg_build_licence() -> str:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-version"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    text = (proc.stdout or "") + (proc.stderr or "")
    return "GPL-enabled development build" if "--enable-gpl" in text.lower() else "LGPL build"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dry", "converted", "playback", "ambience"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--audio-out", type=Path)
    parser.add_argument("--out", type=Path, default=RESULTS)
    parser.add_argument("--seed", type=int, default=6006)
    args = parser.parse_args(argv)

    material, source = load_material(args)
    args.out.mkdir(parents=True, exist_ok=True)
    rows, audio = strategy_matrix(material, args.out)
    payload: dict[str, object] = {
        "phase": 6,
        "source": source,
        "objective_scope": (
            "mechanics only; objective metrics do not measure naturalness or select a winner"
        ),
        "rms_mix_rate": envelope_matrix(material[0], material[1]),
        "acoustic_space": rows,
        "selection": {
            "winner": None,
            "status": "blocked_pending_five_song_blind_listening",
            "provisional_runtime_default": "B",
            "reason": "B avoids the original singer timbre and does not require GPL at mix time",
        },
        "mix_chain": {
            "filters": ["ffmpeg amix", "ffmpeg acompressor", "ffmpeg loudnorm", "ffmpeg alimiter"],
            "python_copyleft_dependency": False,
            "filters_available_in_lgpl_ffmpeg": True,
            "tested_ffmpeg_binary": ffmpeg_build_licence(),
            "packaging_requirement": "Phase 11 must bundle an LGPL-only ffmpeg build",
        },
    }
    if args.audio_out:
        payload["blind_key"] = str(write_blind_audio(audio, args.audio_out, args.seed))
    out_path = args.out / "results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
