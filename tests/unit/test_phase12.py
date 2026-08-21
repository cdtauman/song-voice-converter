"""Phase 12 release integration and proportional long-song acceptance."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from svc_app import __version__ as app_version
from svc_engine import __version__ as engine_version
from svc_engine.backends.base import AudioBuffer, F0Curve
from svc_engine.backends.conversion import ConversionParams
from svc_engine.conversion.chunking import convert_in_chunks
from svc_engine.resources import load_registry
from svc_engine.separation.backends.audio_separator_backend import (
    AudioSeparatorBackend,
)


class IdentityBackend:
    def convert(
        self, audio: AudioBuffer, _f0: F0Curve, _params: ConversionParams
    ) -> AudioBuffer:
        return audio


def test_ten_minute_song_is_processed_in_bounded_chunks_without_length_loss() -> None:
    # A low sample rate keeps this acceptance mechanical and fast; the chunker
    # sees the same 600-second timeline and the production 30-second windows.
    sample_rate = 100
    seconds = 10 * 60
    audio = AudioBuffer(
        samples=np.linspace(-0.1, 0.1, sample_rate * seconds, dtype=np.float32)[None, :],
        sample_rate=sample_rate,
    )
    f0 = F0Curve(np.full(seconds * 100, 220.0), hop_seconds=0.01)
    progress: list[tuple[int, int]] = []

    result = convert_in_chunks(
        audio,
        f0,
        IdentityBackend(),  # type: ignore[arg-type]
        ConversionParams(),
        on_progress=lambda done, total: progress.append((done, total)),
    )

    assert result.frames == audio.frames
    assert result.seconds == seconds
    assert progress[-1][0] == progress[-1][1]
    assert progress[-1][1] > 1


def test_release_version_and_packaging_are_synchronised() -> None:
    repo = Path(__file__).resolve().parents[2]
    pyproject = (repo / "pyproject.toml").read_text(encoding="utf-8")
    project_version = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert project_version is not None
    assert project_version.group(1) == app_version == engine_version == "1.0.1"

    installer = (repo / "packaging" / "SongVoice.iss").read_text(encoding="utf-8")
    spec = (repo / "packaging" / "songvoice.spec").read_text(encoding="utf-8")
    workflow = (repo / ".github" / "workflows" / "windows-release.yml").read_text(
        encoding="utf-8"
    )
    assert '#define MyAppVersion "1.0.1"' in installer
    assert '"user-guide-he.md"' in spec
    assert '"songvoice-quickstart-he.mp4"' in spec
    assert "SongVoice-0.1.0" not in workflow
    assert '"packaging/output/SongVoice-$version-Setup.exe"' in workflow


def test_release_documentation_and_quick_start_video_are_present() -> None:
    repo = Path(__file__).resolve().parents[2]
    guide = (repo / "docs" / "user-guide-he.md").read_text(encoding="utf-8")
    video = repo / "docs" / "media" / "songvoice-quickstart-he.mp4"
    assert all(term in guide for term in ("קאבר ראשון", "פתרון תקלות", "פרטיות"))
    assert video.read_bytes()[4:8] == b"ftyp"
    assert video.stat().st_size > 10_000


def test_offline_separator_catalog_prevents_the_library_metadata_download(
    tmp_path: Path,
) -> None:
    """audio-separator must find its mandatory metadata without reaching GitHub."""
    backend = AudioSeparatorBackend(tmp_path, load_registry())
    backend._ensure_offline_download_catalog()

    catalog = json.loads((tmp_path / "download_checks.json").read_text(encoding="utf-8"))
    assert set(catalog) == {
        "demucs_download_list",
        "vr_download_list",
        "mdx_download_list",
        "mdx_download_vip_list",
        "mdx23c_download_list",
        "mdx23c_download_vip_list",
        "roformer_download_list",
    }
