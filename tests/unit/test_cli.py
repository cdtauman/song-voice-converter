"""CLI wiring regressions that must hold before any heavy backend is loaded."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint
from svc_engine.backends.separation import SeparationRequest, Stems
from svc_engine.cli import _conversion_separator, build_parser
from svc_engine.config import paths
from svc_engine.separation import CleanupStep, QualityLevel, profile_for


class RejectingBackend:
    """Fails the test if licence policy ever lets a private model reach it."""

    called = False

    def info(self) -> BackendInfo:
        return BackendInfo("rejecting", "בדיקה", True)

    def list_models(self) -> list[str]:
        return []

    def separate(
        self, audio: AudioBuffer, request: SeparationRequest, device: DeviceHint
    ) -> Stems:
        self.called = True
        raise AssertionError(f"private model reached backend: {request.model_id}")

    def unload(self) -> None:
        pass


def _convert_args(*extra: str) -> argparse.Namespace:
    return build_parser().parse_args(["convert", "song.wav", "--voice", "v", *extra])


def test_convert_blocks_private_models_by_default_and_requires_explicit_opt_in() -> None:
    assert _convert_args().allow_private_models is False
    assert _convert_args("--allow-private-models").allow_private_models is True


def test_default_convert_wiring_skips_gpl_dereverb_before_backend_or_download(
    tmp_path: Path,
) -> None:
    separator = _conversion_separator(paths(tmp_path), no_download=False)
    assert separator.allow_private_models is False
    dereverb = separator.registry.get("dereverb_anvuew")
    assert dereverb.license.spdx == "GPL-3.0"
    assert not dereverb.license.is_redistributable

    backend = RejectingBackend()
    separator.backend = backend
    audio = AudioBuffer(np.full((1, 4410), 0.1, dtype=np.float32), 44100)
    result = separator._run_cleanup(
        audio,
        (CleanupStep.DEREVERB,),
        DeviceHint(),
        profile_for(QualityLevel.FAST),
        lambda _step, _message: None,
    )

    assert not backend.called
    assert result.applied == ()
    assert CleanupStep.DEREVERB in result.skipped
    assert result.ambience is None
