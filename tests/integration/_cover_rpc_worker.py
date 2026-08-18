"""Subprocess worker used by the durable cover RPC integration test."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import svc_engine.workflows.cover as cover_workflow  # noqa: E402
from svc_engine.config import paths  # noqa: E402
from svc_engine.jobs import Step  # noqa: E402
from svc_engine.rpc import Request, Server  # noqa: E402


def fake_steps(app_paths, request):  # type: ignore[no-untyped-def]
    root = app_paths.root.parent
    calls = root / "calls.log"
    crash_marker = root / "analyze-crashed"

    def record(name: str) -> None:
        with calls.open("a", encoding="utf-8") as stream:
            stream.write(f"{name}\n")
            stream.flush()
            os.fsync(stream.fileno())

    def separate(context):  # type: ignore[no-untyped-def]
        record("separate")
        artifact = context.output_dir / "vocals.wav"
        artifact.write_bytes(b"separated")
        context.progress(1.0, "ההפרדה הושלמה.")
        return {"vocals": artifact, "instrumental": artifact}

    def analyze(context):  # type: ignore[no-untyped-def]
        record("analyze")
        artifact = context.output_dir / "f0.npz"
        artifact.write_bytes(b"analysis")
        if not crash_marker.exists():
            crash_marker.write_text("once", encoding="utf-8")
            os._exit(91)
        context.progress(1.0, "הניתוח הושלם.")
        return {"f0": artifact}

    def render(context):  # type: ignore[no-untyped-def]
        record("render")
        audio = context.output_dir / "cover.wav"
        audio.write_bytes(b"durable-cover")
        metadata = context.output_dir / "result.json"
        metadata.write_text(
            json.dumps(
                {
                    "source": request["song"],
                    "voice_id": request["voice_id"],
                    "preview": False,
                    "summary_he": "קאבר בדיקה הושלם.",
                    "recommendation": {"semitones": 0, "playback_semitones": 0},
                    "master": None,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return {"cover": audio, "metadata": metadata}

    def deliver(context):  # type: ignore[no-untyped-def]
        record("deliver")
        destination = Path(request["output"])
        destination.write_bytes(context.dependencies["render"]["cover"].read_bytes())
        receipt = context.output_dir / "receipt.json"
        receipt.write_text("{}", encoding="utf-8")
        return {"receipt": receipt}

    return (
        Step(
            "separate",
            separate,
            input_files=(Path(request["song"]),),
            version="rpc-integration-v1",
        ),
        Step(
            "analyze",
            analyze,
            needs=("separate",),
            version="rpc-integration-v1",
        ),
        Step(
            "render",
            render,
            needs=("separate", "analyze"),
            version="rpc-integration-v1",
        ),
        Step(
            "deliver",
            deliver,
            needs=("render",),
            parameters={"job_id": request["job_id"]},
            version="rpc-integration-v1",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["crash", "resume"])
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    app_paths = paths(args.root / "data")
    cover_workflow._production_steps = fake_steps
    server = Server(app_paths)
    if args.mode == "crash":
        request = Request(
            id="run",
            method="covers.run",
            params={
                "song": str(args.root / "song.wav"),
                "voice_id": "test-voice",
                "quality": "balanced",
                "output": str(args.root / "cover.wav"),
                "job_id": "rpc-cover-recovery",
            },
        )
    else:
        request = Request(
            id="resume",
            method="covers.resume",
            params={"job_id": "rpc-cover-recovery"},
        )
    events = []
    response = server.handle(request, on_event=lambda item: events.append(item.to_dict()))
    print(json.dumps({"response": response.to_dict(), "events": events}, ensure_ascii=False))
    return 0 if response.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
