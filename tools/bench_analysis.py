"""Exercise the analysis engine against a real F0 model, and time it.

The unit tests cover the analysis math against a fake extractor; this runs the
actual FCPE and RMVPE models on a file so the two heavy paths are exercised on
demand. With no file it synthesises a known two-pitch tone (A3 then E4) whose
correct answer is exact, so the run doubles as a reproducible, licence-free
regression check -- the same idea as tools/bench_separation.py.

    python tools/bench_analysis.py                       # synthetic A3/E4 tone
    python tools/bench_analysis.py vocals.wav --method rmvpe --plot out.png

Judging accuracy on real singing needs the user's own test set
(docs/testing.md section 1); this proves the pipeline runs and is self-consistent.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from svc_engine.analysis import analyze_vocal, plot_report  # noqa: E402
from svc_engine.analysis.f0 import RMVPE_MODEL_ID, FcpeExtractor, RmvpeExtractor  # noqa: E402
from svc_engine.audio import io as audio_io  # noqa: E402
from svc_engine.backends.base import AudioBuffer  # noqa: E402
from svc_engine.config import paths  # noqa: E402
from svc_engine.resources import DownloadManager, load_registry  # noqa: E402

SR = 44100


def synth_tone(seconds: float = 15.0) -> AudioBuffer:
    """A3 (220 Hz) then E4 (330 Hz), harmonics + vibrato, with one silent gap."""
    t = np.arange(int(SR * seconds)) / SR
    base = np.where(t < seconds / 2, 220.0, 330.0)
    vib = base * (1 + 0.012 * np.sin(2 * np.pi * 5.5 * t))
    phase = 2 * np.pi * np.cumsum(vib) / SR
    sig = sum((1.0 / k) * np.sin(k * phase) for k in range(1, 6))
    sig[int(SR * (seconds / 2 - 0.1)) : int(SR * (seconds / 2 + 0.3))] = 0.0
    sig = 0.3 * sig / np.max(np.abs(sig))
    return AudioBuffer(samples=sig.astype(np.float32)[None, :], sample_rate=SR)


def build_extractor(method: str):  # type: ignore[no-untyped-def]
    if method == "fcpe":
        return FcpeExtractor()
    spec = load_registry().get(RMVPE_MODEL_ID)
    downloader = DownloadManager(paths().models)
    downloader.ensure_model(spec)
    return RmvpeExtractor(spec.files[0].path_in(paths().models))


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", help="vocal file; omit for a synthetic tone")
    parser.add_argument("--method", choices=["fcpe", "rmvpe", "both"], default="both")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--plot", metavar="FILE")
    args = parser.parse_args(argv)

    if args.input:
        audio = audio_io.load_audio(args.input, duration_seconds=args.seconds)
        source = Path(args.input).name
    else:
        audio = synth_tone(args.seconds or 15.0)
        source = "synthetic A3/E4"

    methods = ["fcpe", "rmvpe"] if args.method == "both" else [args.method]
    last_report = None
    for method in methods:
        extractor = build_extractor(method)
        started = time.perf_counter()
        report = analyze_vocal(audio, extractor, source=source)
        elapsed = time.perf_counter() - started
        extractor.unload()
        r = report.range
        print(f"\n[{method}]  {elapsed:.1f}s for {audio.seconds:.1f}s of audio")
        print(f"  range   {r.to_dict()['low_note']}..{r.to_dict()['high_note']} "
              f"(median {r.to_dict()['median_note']}, p50 MIDI {r.p50:.2f})")
        print(f"  key     {report.key.name}  (corr {report.key.correlation:.3f})")
        print(f"  voiced  {report.voiced_fraction:.1%}  ·  {len(report.segments)} segments")
        pv = report.preview.best
        if pv is not None:
            print(f"  preview {pv.start_seconds:.1f}-{pv.end_seconds:.1f}s (score {pv.score:.3f})")
        last_report = report

    if args.plot and last_report is not None:
        try:
            plot_report(last_report, args.plot)
            print(f"\nplot: {args.plot}")
        except RuntimeError as exc:
            print(f"\nplot skipped: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
