"""Measure separation: quality against known stems, and speed per device.

Quality here is measured against the *synthetic* fixture from
`tools/make_test_mix.py`, where the ground truth is exact. That makes the
numbers reproducible and license-free, and makes them a regression guard --
but it does not make them a judgement about how separation sounds on real
music. docs/testing.md section 1 is where that comes from, and it needs the
user's own test set.

    python tools/bench_separation.py --seconds 10 --devices xpu cpu
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.audio import io as audio_io  # noqa: E402
from svc_engine.audio.buffers import add, align_all, fit_length  # noqa: E402
from svc_engine.backends.base import AudioBuffer  # noqa: E402
from svc_engine.compute import (  # noqa: E402
    Component,
    ComputeBackend,
    DeviceManager,
    load_matrix,
)
from svc_engine.config import paths  # noqa: E402
from svc_engine.separation import QualityLevel, SeparationPipeline  # noqa: E402
from svc_engine.separation.quality import profile_for  # noqa: E402


def sdr_db(estimate: AudioBuffer, reference: AudioBuffer) -> float:
    """Signal-to-distortion ratio in dB. Higher is better; identical is +inf."""
    est, ref = align_all([estimate, reference])
    e = est.samples.astype(np.float64).ravel()
    r = ref.samples.astype(np.float64).ravel()
    noise = float(np.sum((r - e) ** 2))
    signal = float(np.sum(r ** 2))
    if noise <= 0.0:
        return float("inf")
    if signal <= 0.0:
        return float("-inf")
    return 10.0 * float(np.log10(signal / noise))


@dataclass
class Row:
    quality: str
    device: str
    seconds: float
    realtime_factor: float | None
    sdr_vocals: float
    sdr_instrumental: float
    reconstruction_db: float
    peak_vocals: float
    fallbacks: int


class _FixedDevice:
    """Pins the pipeline to one backend so the two devices are comparable."""

    def __init__(self, backend: ComputeBackend, manager: DeviceManager) -> None:
        self._backend = backend
        self._manager = manager
        self._real = load_matrix()

    def device_for(self, component: Component, manager: DeviceManager):  # noqa: ANN201
        return manager.select({self._backend})

    def device_for_implementation(
        self,
        component: Component,
        implementation: str,
        manager: DeviceManager,
    ):  # noqa: ANN201
        return manager.select({self._backend})

    def get(self, component: Component):  # noqa: ANN201
        return self._real.get(component)

    @property
    def components(self):  # noqa: ANN201
        return self._real.components

    @property
    def source(self) -> str:
        return f"pinned:{self._backend.value}"


def run(fixtures: Path, out_dir: Path, seconds: float, qualities: list[str],
        devices: list[str]) -> list[Row]:
    mix_path = fixtures / "mix.wav"
    dry_vocal = audio_io.load_audio(fixtures / "dry_vocal.wav", duration_seconds=seconds)
    dry_inst = audio_io.load_audio(
        fixtures / "dry_instrumental.wav", duration_seconds=seconds
    )
    mix = audio_io.load_audio(mix_path, duration_seconds=seconds)

    manager = DeviceManager()
    rows: list[Row] = []

    for device_name in devices:
        backend = ComputeBackend(device_name)
        if not manager.has(backend):
            print(f"skipping {device_name}: not available on this machine")
            continue

        for quality in qualities:
            profile = profile_for(quality)
            pipeline = SeparationPipeline(
                paths=paths(), matrix=_FixedDevice(backend, manager), manager=manager
            )
            started = time.perf_counter()
            outcome = pipeline.run(
                mix_path, level=QualityLevel(quality), duration_seconds=seconds
            )
            elapsed = time.perf_counter() - started

            from svc_engine.backends.separation import StemKind

            vocals = fit_length(outcome.stems[StemKind.VOCALS], mix.frames)
            inst = fit_length(outcome.stems[StemKind.INSTRUMENTAL], mix.frames)

            rows.append(
                Row(
                    quality=quality,
                    device=device_name,
                    seconds=round(elapsed, 1),
                    realtime_factor=(
                        round(outcome.realtime_factor, 2)
                        if outcome.realtime_factor
                        else None
                    ),
                    sdr_vocals=round(sdr_db(vocals, dry_vocal), 2),
                    sdr_instrumental=round(sdr_db(inst, dry_inst), 2),
                    reconstruction_db=round(sdr_db(add(vocals, inst), mix), 2),
                    peak_vocals=round(float(np.max(np.abs(vocals.samples))), 4),
                    fallbacks=len(outcome.fallback_steps),
                )
            )
            print(
                f"{quality:9s} {device_name:4s} {elapsed:6.1f}s  "
                f"vocals {rows[-1].sdr_vocals:6.2f}dB  "
                f"inst {rows[-1].sdr_instrumental:6.2f}dB  "
                f"recon {rows[-1].reconstruction_db:6.2f}dB  "
                f"({len(profile.models)} model(s))"
            )

            target = out_dir / f"{quality}_{device_name}"
            pipeline.write(outcome, target)

    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", default="tests/fixtures")
    parser.add_argument("--out", default="benchmark/results/separation")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--quality", nargs="+", default=["fast", "balanced"])
    parser.add_argument("--devices", nargs="+", default=["xpu", "cpu"])
    args = parser.parse_args(argv)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = run(
        Path(args.fixtures), out_dir, args.seconds, args.quality, args.devices
    )
    report = out_dir / "results.json"
    report.write_text(
        json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8"
    )
    print(f"\nwrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
