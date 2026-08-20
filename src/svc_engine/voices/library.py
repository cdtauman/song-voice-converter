"""The voice library -- discovery, health, and handing voices to the backend.

Torch-free on purpose. It knows which files a voice is made of and whether they
look right; it never loads a model. `conversion.rvc` does that, behind the
`ConversionBackend` interface, from the `VoiceHandle` this hands out.
"""

from __future__ import annotations

import datetime as _dt
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

from svc_engine.backends.conversion import VoiceHandle
from svc_engine.config import Paths
from svc_engine.config import paths as default_paths
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.profiles import VoiceProfile
from svc_engine.voices.manifest import (
    AVATAR_FILE,
    INDEX_FILE,
    MODEL_FILE,
    PROFILE_FILE,
    SAMPLE_FILE,
    VOICE_FILE,
    HealthState,
    HealthStatus,
    VoiceManifest,
    is_canonical_voice_id,
)

__all__ = ["VoiceLibrary", "VoiceEntry", "looks_like_torch_checkpoint"]

#: A PyTorch checkpoint saved the modern way is a zip archive ("PK\x03\x04");
#: a legacy one is a pickle stream, which starts with the protocol-2+ opcode
#: 0x80. Anything else is not a model file. This is a cheap, torch-free gate --
#: not a guarantee the weights load, which only the backend can prove.
_ZIP_MAGIC = b"PK\x03\x04"
_PICKLE_PROTO = 0x80


def looks_like_torch_checkpoint(path: Path) -> bool:
    """True if `path` has the byte signature of a torch checkpoint."""
    try:
        with path.open("rb") as fh:
            head = fh.read(4)
    except OSError:
        return False
    if len(head) < 2:
        return False
    return head.startswith(_ZIP_MAGIC) or head[0] == _PICKLE_PROTO


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class VoiceEntry:
    """One voice on disk: its manifest, its folder, and its file paths."""

    manifest: VoiceManifest
    root: Path

    @property
    def voice_id(self) -> str:
        return self.manifest.voice_id

    @property
    def model_path(self) -> Path:
        return self.root / MODEL_FILE

    @property
    def index_path(self) -> Path | None:
        p = self.root / INDEX_FILE
        return p if p.exists() else None

    @property
    def profile_path(self) -> Path | None:
        p = self.root / PROFILE_FILE
        return p if p.exists() else None

    def profile(self) -> VoiceProfile | None:
        p = self.profile_path
        return VoiceProfile.load(p) if p is not None else None

    def handle(self) -> VoiceHandle:
        return VoiceHandle(voice_id=self.voice_id, root=self.root)


class VoiceLibrary:
    """The user's voices under `paths.voices`. Scans, validates, removes."""

    def __init__(self, paths: Paths | None = None) -> None:
        self.paths = paths or default_paths()
        self.root = self.paths.voices

    # -- discovery ---------------------------------------------------------- #

    def list(self) -> list[VoiceEntry]:
        """Every voice folder that carries a readable manifest, id-sorted."""
        if not self.root.exists():
            return []
        entries: list[VoiceEntry] = []
        for child in sorted(self.root.iterdir()):
            if not is_canonical_voice_id(child.name):
                continue
            try:
                voice_dir = self._voice_dir(child.name)
                if not voice_dir.is_dir() or not (voice_dir / VOICE_FILE).exists():
                    continue
                manifest = VoiceManifest.load(voice_dir)
            except (EngineError, OSError, ValueError):
                continue
            if manifest.voice_id != child.name or not is_canonical_voice_id(manifest.voice_id):
                continue
            entries.append(VoiceEntry(manifest=manifest, root=voice_dir))
        return entries

    def __contains__(self, voice_id: object) -> bool:
        if not isinstance(voice_id, str) or not is_canonical_voice_id(voice_id):
            return False
        try:
            return (self._voice_dir(voice_id) / VOICE_FILE).is_file()
        except EngineError:
            return False

    def get(self, voice_id: str) -> VoiceEntry:
        voice_dir = self._voice_dir(voice_id)
        if not (voice_dir / VOICE_FILE).exists():
            raise KeyError(f"unknown voice id: {voice_id}")
        manifest = VoiceManifest.load(voice_dir)
        if manifest.voice_id != voice_id or not is_canonical_voice_id(manifest.voice_id):
            raise EngineError(
                ErrorCode.VOICE_CORRUPT, "voice manifest id does not match its folder"
            )
        return VoiceEntry(manifest=manifest, root=voice_dir)

    # -- health ------------------------------------------------------------- #

    def check_health(self, entry: VoiceEntry) -> HealthState:
        """Re-derive a voice's health from what is actually on disk."""
        if not entry.manifest.consent_confirmed:
            return HealthState(HealthStatus.NO_CONSENT, _now())
        model = entry.model_path
        if not model.exists():
            return HealthState(HealthStatus.MISSING_MODEL, _now())
        if not looks_like_torch_checkpoint(model):
            return HealthState(HealthStatus.CORRUPT_MODEL, _now())
        return HealthState(HealthStatus.OK, _now())

    def refresh_health(self, voice_id: str) -> VoiceEntry:
        """Re-check a voice and persist the result into its manifest."""
        entry = self.get(voice_id)
        health = self.check_health(entry)
        updated = entry.manifest.with_health(health)
        updated.save(entry.root)
        return VoiceEntry(manifest=updated, root=entry.root)

    # -- mutation ----------------------------------------------------------- #

    def remove(self, voice_id: str) -> None:
        """Delete a voice folder and everything in it. Idempotent."""
        voice_dir = self._voice_dir(voice_id)
        if voice_dir.exists():
            shutil.rmtree(voice_dir)

    def update(
        self,
        voice_id: str,
        *,
        display_name: str | None = None,
        sample: Path | str | None = None,
        avatar: Path | str | None = None,
    ) -> VoiceEntry:
        """Update user-facing metadata without ever replacing model weights."""
        entry = self.get(voice_id)
        manifest = entry.manifest
        if display_name is not None:
            clean_name = display_name.strip()
            if not clean_name:
                raise ValueError("voice display name may not be empty")
            manifest = replace(manifest, display_name=clean_name)
        if sample is not None:
            from svc_engine.audio import load_audio, save_wav, to_mono

            source = Path(sample)
            if not source.is_file():
                raise ValueError("voice sample does not exist")
            save_wav(to_mono(load_audio(source)), entry.root / SAMPLE_FILE, bit_depth=24)
            manifest = replace(manifest, has_sample=True)
        if avatar is not None:
            source = Path(avatar)
            if not source.is_file() or source.suffix.lower() != ".png":
                raise ValueError("voice avatar must be a PNG file")
            shutil.copy2(source, entry.root / AVATAR_FILE)
            manifest = replace(manifest, has_avatar=True)
        manifest.save(entry.root)
        return VoiceEntry(manifest=manifest, root=entry.root)

    def voice_dir(self, voice_id: str) -> Path:
        return self._voice_dir(voice_id)

    def _voice_dir(self, voice_id: str) -> Path:
        """Return a canonical child of ``root`` or reject it before filesystem I/O."""
        if not is_canonical_voice_id(voice_id):
            raise EngineError(ErrorCode.VOICE_CORRUPT, f"unsafe voice id: {voice_id!r}")
        root = self.root.resolve()
        candidate = self.root / voice_id
        try:
            candidate.resolve().relative_to(root)
        except ValueError as exc:
            raise EngineError(
                ErrorCode.VOICE_CORRUPT, "voice path escapes the voice library"
            ) from exc
        return candidate
