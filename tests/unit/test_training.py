"""Phase-9 dataset, quality, recovery, runtime and wizard contracts."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from svc_engine.config import paths
from svc_engine.rpc import Request, Server
from svc_engine.training import quality as quality_module
from svc_engine.training.dataset import DatasetBuilder, PreparationOptions
from svc_engine.training.quality import inspect_recordings
from svc_engine.training.runtime import APPLIO_COMMIT, ApplioRuntime
from svc_engine.training.session import SessionStage, TrainingSessionStore
from svc_engine.training.trainer import TrainingCoordinator
from svc_engine.voices import VoiceLibrary, VoiceManifest, VoiceSource


def _tone(path: Path, seconds: float = 3.0, frequency: float = 220.0) -> Path:
    sample_rate = 16000
    time = np.arange(int(sample_rate * seconds)) / sample_rate
    samples = (0.2 * np.sin(2 * np.pi * frequency * time)).astype(np.float32)
    # Silence around the useful material exercises trimming.
    samples[: sample_rate // 4] = 0.0
    samples[-sample_rate // 4 :] = 0.0
    wavfile.write(path, sample_rate, (samples * 32767).astype(np.int16))
    return path


def test_quality_report_precedes_training_and_is_explainable(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(quality_module, "_BLOCK_MINUTES", 0.01)
    monkeypatch.setattr(quality_module, "_WARN_MINUTES", 0.02)
    report = inspect_recordings([_tone(tmp_path / "one.wav"), _tone(tmp_path / "two.wav")])

    assert report.can_train
    assert report.active_seconds > 1.0
    assert 0.0 <= report.speaker_consistency <= 1.0
    payload = report.to_dict()
    assert payload["summary_he"]
    assert all(issue["action_he"] for issue in payload["issues"])


def test_quality_blocks_too_little_material(tmp_path: Path) -> None:
    report = inspect_recordings([_tone(tmp_path / "short.wav", seconds=1.0)])
    assert not report.can_train
    assert any(issue.code == "too_short" for issue in report.issues)


def test_dataset_builder_trims_silence_and_slices_atomically(tmp_path: Path) -> None:
    source = _tone(tmp_path / "source.wav", seconds=3.0)
    builder = DatasetBuilder(
        cleaner=lambda path: __import__("svc_engine.audio", fromlist=["load_audio"]).load_audio(
            path
        )
    )
    result = builder.build(
        [source],
        tmp_path / "dataset",
        PreparationOptions(
            sample_rate=16000,
            min_segment_seconds=0.25,
            max_segment_seconds=0.75,
            separate_mix=True,
        ),
    )

    assert result.segments
    assert all(path.is_file() for path in result.segments)
    assert result.sample.is_file()
    assert not (tmp_path / ".dataset.preparing").exists()
    assert result.seconds < 3.0


def test_session_store_requires_consent_and_survives_reload(tmp_path: Path) -> None:
    recording = _tone(tmp_path / "voice.wav")
    store = TrainingSessionStore(tmp_path / "sessions")
    try:
        store.create("Demo", [recording], consent_confirmed=False, consent_note="")
    except ValueError:
        pass
    else:
        raise AssertionError("training without consent must be rejected")

    created = store.create("Demo", [recording], consent_confirmed=True, consent_note="my recording")
    created.stage = SessionStage.PAUSED
    created.current_epoch = 37
    store.save(created)
    loaded = store.load(created.session_id)
    assert loaded.stage is SessionStage.PAUSED
    assert loaded.current_epoch == 37
    assert loaded.voice_id.startswith("demo-")


def test_coordinator_inspect_then_prepare_dry_recording(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(quality_module, "_BLOCK_MINUTES", 0.01)
    monkeypatch.setattr(quality_module, "_WARN_MINUTES", 0.02)
    recording = _tone(tmp_path / "voice.wav", seconds=4.0)
    coordinator = TrainingCoordinator(paths(tmp_path))
    created = coordinator.create("Demo", [str(recording)], True, "mine")
    inspected = coordinator.inspect(str(created["session_id"]))
    assert inspected["quality"]["can_train"] is True
    prepared = coordinator.prepare(str(created["session_id"]), separate_mix=False)
    assert prepared["stage"] == "cleaning"
    assert prepared["dataset_seconds"] > 0
    assert Path(str(prepared["root"]), "dataset", "sample.wav").is_file()


def test_applio_commands_are_pinned_to_titan_and_resume_same_experiment(tmp_path: Path) -> None:
    app_paths = paths(tmp_path)
    store = TrainingSessionStore(tmp_path / "training" / "sessions")
    recording = _tone(tmp_path / "voice.wav")
    session = store.create("Demo", [recording], consent_confirmed=True, consent_note="mine")
    runtime = ApplioRuntime(app_paths, python="python")

    assert len(APPLIO_COMMIT) == 40
    train = runtime.train_command(session)
    assert session.applio_experiment in train
    assert str(runtime.titan_g) in train
    assert str(runtime.titan_d) in train
    assert train == runtime.train_command(session)


def test_training_rpc_creates_and_lists_durable_session(tmp_path: Path) -> None:
    recording = _tone(tmp_path / "voice.wav")
    server = Server(paths(tmp_path / "home"))
    created = server.handle(
        Request(
            id="create",
            method="training.create",
            params={
                "display_name": "Demo",
                "recordings": [str(recording)],
                "consent_confirmed": True,
                "consent_note": "mine",
            },
        )
    )
    listed = server.handle(Request(id="list", method="training.list"))

    assert created.ok
    assert listed.ok
    assert listed.result[0]["session_id"] == created.result["session_id"]


def test_voice_card_metadata_can_be_renamed_and_given_media(tmp_path: Path) -> None:
    app_paths = paths(tmp_path / "home")
    library = VoiceLibrary(app_paths)
    voice_dir = library.voice_dir("demo")
    voice_dir.mkdir(parents=True)
    VoiceManifest(
        voice_id="demo",
        display_name="Old",
        source=VoiceSource.TRAINED_LOCALLY,
        consent_confirmed=True,
    ).save(voice_dir)
    sample = _tone(tmp_path / "sample.wav")
    avatar = tmp_path / "avatar.png"
    avatar.write_bytes(b"\x89PNG\r\n\x1a\n")

    updated = library.update("demo", display_name="New", sample=sample, avatar=avatar)

    assert updated.manifest.display_name == "New"
    assert updated.manifest.has_sample and updated.manifest.has_avatar
    assert (voice_dir / "sample.wav").is_file()
    assert (voice_dir / "avatar.png").read_bytes().startswith(b"\x89PNG")


def test_training_wizard_has_exactly_five_rtl_steps(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from svc_app.screens.voices import TrainingWizardDialog

    class Client:
        pass

    app = QApplication.instance() or QApplication([])
    assert app is not None
    dialog = TrainingWizardDialog(
        Client(),  # type: ignore[arg-type]
        session={
            "session_id": "demo-session",
            "stage": "paused",
            "progress": 0.4,
            "message_he": "נעצר",
        },
    )
    try:
        dialog.setLayoutDirection(Qt.LayoutDirection.RightToLeft)
        assert dialog.pages.count() == 5
        assert dialog.pages.currentIndex() == 3
        assert dialog.pause_button.text() == "המשך אימון"
        for index in range(dialog.pages.count()):
            assert dialog.pages.widget(index).layoutDirection() is Qt.LayoutDirection.RightToLeft
    finally:
        dialog.close()
