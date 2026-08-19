"""Detached worker that runs Applio and updates the durable session record."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from pathlib import Path

from svc_engine.config import paths
from svc_engine.training.runtime import ApplioRuntime
from svc_engine.training.session import SessionStage, TrainingSession, TrainingSessionStore
from svc_engine.training.trainer import TrainingCoordinator, training_root

_EPOCH = re.compile(r"(?:Epoch|epoch)[^0-9]{0,12}(\d+)")


def _run(
    command: list[str],
    cwd: Path,
    session: TrainingSession,
    store: TrainingSessionStore,
    *,
    start: float,
    span: float,
    training: bool = False,
) -> None:
    log = session.root / "training.log"
    begun = time.monotonic()
    with log.open("a", encoding="utf-8") as output:
        output.write("\n$ " + subprocess.list2cmdline(command) + "\n")
        output.flush()
        proc = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            output.write(line)
            output.flush()
            if training and (match := _EPOCH.search(line)):
                epoch = min(session.total_epochs, int(match.group(1)))
                if epoch >= session.current_epoch:
                    session.current_epoch = epoch
                    fraction = epoch / session.total_epochs
                    session.progress = start + span * fraction
                    elapsed = max(1.0, time.monotonic() - begun)
                    session.estimated_remaining_seconds = (
                        elapsed / max(epoch, 1) * (session.total_epochs - epoch)
                    )
                    session.message_he = (
                        f"מאמנים את הקול — epoch {epoch} מתוך {session.total_epochs}."
                    )
                    store.save(session)
        code = proc.wait()
    completed = list(
        (cwd / "logs" / session.applio_experiment).glob(
            f"{session.applio_experiment}_{session.total_epochs}e_*s.pth"
        )
    )
    if code != 0 and not (training and completed):
        raise RuntimeError(f"training command exited with {code}; see {log}")


def run(home: Path, session_id: str) -> int:
    app_paths = paths(home)
    store = TrainingSessionStore(training_root(app_paths))
    session = store.load(session_id)
    session.worker_pid = os.getpid()
    store.save(session)
    try:
        runtime = ApplioRuntime(app_paths)
        runtime.ensure()
        session.stage = SessionStage.TRAINING
        session.message_he = "מכינים את קובצי האימון…"
        store.save(session)
        _run(
            runtime.preprocess_command(session), runtime.root, session, store, start=0.0, span=0.08
        )
        session.progress = 0.08
        session.message_he = "מחלצים מאפייני קול וגובה…"
        store.save(session)
        _run(runtime.extract_command(session), runtime.root, session, store, start=0.08, span=0.12)
        session.progress = 0.20
        session.message_he = "מאמנים את הקול…"
        store.save(session)
        _run(
            runtime.train_command(session),
            runtime.root,
            session,
            store,
            start=0.20,
            span=0.72,
            training=True,
        )
        session.progress = 0.92
        session.message_he = "בונים אינדקס חיפוש לקול…"
        store.save(session)
        _run(runtime.index_command(session), runtime.root, session, store, start=0.92, span=0.04)
        experiment = runtime.root / "logs" / session.applio_experiment
        models = sorted(
            experiment.glob(f"{session.applio_experiment}_*e_*s.pth"),
            key=lambda item: item.stat().st_mtime,
        )
        indexes = sorted(experiment.glob("*.index"), key=lambda item: item.stat().st_mtime)
        if not models or not indexes:
            raise RuntimeError("Applio did not publish a model and index")
        session.stage = SessionStage.FINALIZING
        session.progress = 0.97
        session.message_he = "מחשבים את פרופיל המנעד…"
        store.save(session)
        TrainingCoordinator(app_paths).finalize(session, models[-1], indexes[-1])
        session.stage = SessionStage.READY
        session.progress = 1.0
        session.current_epoch = session.total_epochs
        session.estimated_remaining_seconds = 0.0
        session.worker_pid = None
        session.message_he = "הקול מוכן ונוסף לספרייה."
        store.save(session)
        return 0
    except Exception as exc:
        session.stage = SessionStage.FAILED
        session.worker_pid = None
        session.error_he = str(exc)
        session.message_he = "האימון נעצר בגלל שגיאה. אפשר לתקן ולהמשיך."
        store.save(session)
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home", type=Path, required=True)
    parser.add_argument("session_id")
    args = parser.parse_args(argv)
    return run(args.home, args.session_id)


if __name__ == "__main__":
    raise SystemExit(main())
