"""Coordinator for inspect -> clean -> train -> publish, with pause/resume."""

from __future__ import annotations

import contextlib
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

from svc_engine.analysis.f0 import make_extractor
from svc_engine.audio import load_audio
from svc_engine.backends.base import DeviceHint
from svc_engine.config import Paths
from svc_engine.profiles import compute_profile
from svc_engine.training.dataset import DatasetBuilder, PreparationOptions
from svc_engine.training.quality import inspect_recordings
from svc_engine.training.session import SessionStage, TrainingSession, TrainingSessionStore
from svc_engine.voices import HealthState, HealthStatus, VoiceLibrary, VoiceManifest, VoiceSource
from svc_engine.voices.manifest import INDEX_FILE, MODEL_FILE, PROFILE_FILE, SAMPLE_FILE

__all__ = ["TrainingCoordinator"]


def training_root(paths: Paths) -> Path:
    return paths.root / "training" / "sessions"


class TrainingCoordinator:
    def __init__(self, paths: Paths, *, dataset_builder: DatasetBuilder | None = None) -> None:
        self.paths = paths
        self.paths.ensure()
        self.store = TrainingSessionStore(training_root(paths))
        self.dataset_builder = dataset_builder or DatasetBuilder(paths=paths)

    def create(
        self,
        display_name: str,
        recordings: list[str],
        consent_confirmed: bool,
        consent_note: str,
        total_epochs: int = 200,
    ) -> dict[str, Any]:
        session = self.store.create(
            display_name,
            recordings,
            consent_confirmed=consent_confirmed,
            consent_note=consent_note,
            total_epochs=total_epochs,
        )
        return session.to_dict()

    def list(self) -> list[dict[str, Any]]:
        return [session.to_dict() for session in self.store.list()]

    def inspect(self, session_id: str) -> dict[str, Any]:
        session = self.store.load(session_id)
        report = inspect_recordings([Path(path) for path in session.recordings])
        session.quality = report.to_dict()
        session.stage = SessionStage.QUALITY
        session.progress = 1.0
        session.message_he = report.summary_he
        self.store.save(session)
        return session.to_dict()

    def prepare(self, session_id: str, *, separate_mix: bool = True) -> dict[str, Any]:
        session = self.store.load(session_id)
        if not session.quality or not bool(session.quality.get("can_train")):
            raise ValueError("recording quality must pass before preparation")
        session.stage = SessionStage.CLEANING
        session.progress = 0.0
        session.message_he = "מתחילים לנקות את ההקלטות…"
        self.store.save(session)

        def progress(fraction: float, message: str) -> None:
            session.progress = fraction
            session.message_he = message
            self.store.save(session)

        result = self.dataset_builder.build(
            list(session.recordings),
            session.dataset_dir,
            PreparationOptions(separate_mix=separate_mix),
            on_progress=progress,
        )
        cleaned_quality = inspect_recordings(list(result.segments), strict_consistency=True)
        if not cleaned_quality.can_train:
            session.quality = cleaned_quality.to_dict()
            session.stage = SessionStage.QUALITY
            session.progress = 1.0
            session.message_he = cleaned_quality.summary_he
            self.store.save(session)
            raise ValueError("cleaned dataset did not pass the speaker-consistency gate")
        session.dataset_seconds = result.seconds
        session.progress = 1.0
        session.message_he = "חומר האימון נקי ומוכן."
        self.store.save(session)
        return session.to_dict()

    def start(self, session_id: str) -> dict[str, Any]:
        session = self.store.load(session_id)
        if not session.dataset_dir.is_dir():
            raise ValueError("dataset has not been prepared")
        if session.worker_pid and _pid_exists(session.worker_pid):
            return session.to_dict()
        session.stage = SessionStage.TRAINING
        session.message_he = "מכינים את מנוע האימון…"
        session.error_he = ""
        session.progress = max(0.0, session.progress if session.current_epoch else 0.0)
        self.store.save(session)
        env = os.environ.copy()
        source_root = str(Path(__file__).resolve().parents[2])
        env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
        command = [
            sys.executable,
            "-m",
            "svc_engine.training.worker",
            "--home",
            str(self.paths.root),
            session_id,
        ]
        flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        proc = subprocess.Popen(
            command,
            env=env,
            creationflags=flags,
            close_fds=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        session.worker_pid = proc.pid
        self.store.save(session)
        return session.to_dict()

    def pause(self, session_id: str) -> dict[str, Any]:
        session = self.store.load(session_id)
        if session.worker_pid and _pid_exists(session.worker_pid):
            _kill_tree(session.worker_pid)
        session.worker_pid = None
        session.stage = SessionStage.PAUSED
        session.message_he = "האימון נעצר. אפשר להמשיך מאותה נקודת ביקורת."
        self.store.save(session)
        return session.to_dict()

    def status(self, session_id: str) -> dict[str, Any]:
        session = self.store.load(session_id)
        if (
            session.worker_pid
            and not _pid_exists(session.worker_pid)
            and session.stage is SessionStage.TRAINING
        ):
            session.worker_pid = None
            session.stage = SessionStage.PAUSED
            session.message_he = "האימון נעצר. אפשר להמשיך מנקודת הביקורת האחרונה."
            self.store.save(session)
        return session.to_dict()

    def finalize(self, session: TrainingSession, model_path: Path, index_path: Path) -> None:
        library = VoiceLibrary(self.paths)
        destination = library.voice_dir(session.voice_id)
        staged = destination.with_name(f".{session.voice_id}.training")
        shutil.rmtree(staged, ignore_errors=True)
        staged.mkdir(parents=True)
        shutil.copy2(model_path, staged / MODEL_FILE)
        shutil.copy2(index_path, staged / INDEX_FILE)
        shutil.copy2(session.dataset_dir / SAMPLE_FILE, staged / SAMPLE_FILE)
        profile = compute_profile(
            load_audio(staged / SAMPLE_FILE),
            make_extractor("fcpe"),
            session.display_name,
            DeviceHint(),
        )
        profile.save(staged / PROFILE_FILE)
        manifest = VoiceManifest(
            voice_id=session.voice_id,
            display_name=session.display_name,
            source=VoiceSource.TRAINED_LOCALLY,
            consent_confirmed=True,
            consent_note=session.consent_note,
            created_at=session.created_at,
            sample_rate=48000,
            has_index=True,
            has_sample=True,
            health=HealthState(HealthStatus.OK, session.updated_at),
        )
        manifest.save(staged)
        if destination.exists():
            shutil.rmtree(destination)
        staged.replace(destination)


def _pid_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True, check=False
        )
        return str(pid) in result.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _kill_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
    else:
        with contextlib.suppress(OSError, ProcessLookupError):
            os_api: Any = os
            os_api.killpg(os_api.getpgid(pid), signal.SIGTERM)
