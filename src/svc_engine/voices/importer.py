"""Import a voice from a `.zip` the user dropped on the window.

Phase 5 DoD: "importing a voice from a zip works". A user's RVC voice ships as a
folder or zip with a `.pth` model and usually a `.index`; this turns that into a
first-class library entry -- validated, consented, health-checked.

Two things it refuses to do:

1. **Import without consent.** SongVoice uses only voices the user made or was
   explicitly allowed to use (README). `consent_confirmed` is required; there is
   no default-yes path.
2. **Trust a zip's paths.** A malicious or careless archive can name an entry
   `../../evil`; every member is flattened to its base name and written only
   inside the destination folder (zip-slip guard).
"""

from __future__ import annotations

import datetime as _dt
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import IO

from svc_engine.errors import EngineError, ErrorCode
from svc_engine.profiles import VoiceProfile
from svc_engine.voices.library import VoiceEntry, VoiceLibrary
from svc_engine.voices.manifest import (
    AVATAR_FILE,
    INDEX_FILE,
    MODEL_FILE,
    PROFILE_FILE,
    SAMPLE_FILE,
    RecommendedParams,
    VoiceManifest,
    VoiceSource,
    slugify,
)

__all__ = ["ImportResult", "import_voice_from_zip"]

#: How a file inside the archive is recognised, by suffix, and where it lands.
_MODEL_SUFFIXES = {".pth"}
_INDEX_SUFFIXES = {".index"}
_SAMPLE_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
_AVATAR_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# A normal RVC checkpoint plus its retrieval index is far below these limits.
# Keep both limits so one huge member and many smaller members are rejected.
_MAX_FILE_BYTES = 3 * 1024 ** 3
_MAX_TOTAL_BYTES = 4 * 1024 ** 3
_COPY_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ImportResult:
    """What was imported, for the caller to report."""

    voice_id: str
    root: Path
    manifest: VoiceManifest
    imported_index: bool
    imported_sample: bool
    imported_avatar: bool
    imported_profile: bool
    generated_profile: bool

    def summary_he(self) -> str:
        extras = []
        if self.imported_index:
            extras.append("קובץ חיפוש")
        if self.imported_profile:
            extras.append("פרופיל מנעד")
        elif self.generated_profile:
            extras.append("פרופיל מנעד ניטרלי")
        if self.imported_sample:
            extras.append("דוגמה")
        tail = f" (כולל {', '.join(extras)})" if extras else ""
        status = "מוכן לשימוש" if self.manifest.usable else self.manifest.health.note_he
        return f"הקול '{self.manifest.display_name}' יובא — {status}{tail}."


def _now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat()


def _safe_members(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Every real file in the archive, ignoring directories, within size limits."""
    total = 0
    members: list[zipfile.ZipInfo] = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        if info.file_size > _MAX_FILE_BYTES:
            raise EngineError(
                ErrorCode.VOICE_CORRUPT, "an archive member is too large for a voice"
            )
        total += info.file_size
        if total > _MAX_TOTAL_BYTES:
            raise EngineError(
                ErrorCode.VOICE_CORRUPT, "archive is implausibly large for a voice"
            )
        members.append(info)
    return members


def import_voice_from_zip(
    archive: Path | str,
    display_name: str,
    consent_confirmed: bool,
    consent_note: str = "",
    voice_id: str | None = None,
    rvc_version: str = "v2",
    recommended: RecommendedParams | None = None,
    library: VoiceLibrary | None = None,
    overwrite: bool = False,
) -> ImportResult:
    """Unpack a voice zip into the library and register it.

    Requires `consent_confirmed=True` -- there is no implicit consent. Raises
    `EngineError` with a Hebrew message on every failure mode.
    """
    if not consent_confirmed:
        raise EngineError(
            ErrorCode.CONSENT_REQUIRED, "voice import requires explicit consent"
        )

    archive = Path(archive)
    if not archive.exists():
        raise EngineError(ErrorCode.VOICE_CORRUPT, f"archive not found: {archive}")
    if not zipfile.is_zipfile(archive):
        raise EngineError(ErrorCode.VOICE_CORRUPT, "the file is not a valid zip archive")

    library = library or VoiceLibrary()
    library.root.mkdir(parents=True, exist_ok=True)

    base_vid = slugify(voice_id or display_name)
    vid = base_vid
    if voice_id is None and not overwrite:
        suffix = 2
        while vid in library:
            vid = f"{base_vid}-{suffix}"
            suffix += 1
    _recover_interrupted_activation(library.voice_dir(vid))
    if vid in library and not overwrite:
        raise EngineError(
            ErrorCode.VOICE_CORRUPT, f"a voice named '{vid}' already exists"
        )

    # Stage into a temp dir first so a failed/partial import never leaves a
    # half-written voice in the library that `list()` would then surface.
    with tempfile.TemporaryDirectory(prefix="svc_voice_") as tmp:
        staged = _extract_into(archive, Path(tmp))

        if staged.get("model") is None:
            raise EngineError(
                ErrorCode.VOICE_CORRUPT,
                "the archive has no model file (.pth)",
            )

        # Build a complete replacement under the library root.  It therefore
        # can be renamed into place atomically on the same filesystem.
        replacement = Path(tempfile.mkdtemp(prefix=f".{vid}.import-", dir=library.root))
        try:
            imported_profile = "profile" in staged
            placed = _place(staged, replacement)
            generated_profile = False
            if not imported_profile:
                profile_path = replacement / PROFILE_FILE
                _neutral_profile(display_name.strip() or vid).save(profile_path)
                placed["profile"] = profile_path
                generated_profile = True
            manifest = VoiceManifest(
                voice_id=vid,
                display_name=display_name.strip() or vid,
                source=VoiceSource.IMPORTED,
                consent_confirmed=True,
                consent_note=consent_note.strip(),
                created_at=_now(),
                rvc_version=rvc_version,
                has_index="index" in placed,
                has_sample="sample" in placed,
                has_avatar="avatar" in placed,
                recommended=recommended or RecommendedParams(),
            )
            manifest.save(replacement)
            health = library.check_health(VoiceEntry(manifest=manifest, root=replacement))
            manifest = manifest.with_health(health)
            manifest.save(replacement)

            voice_dir = library.voice_dir(vid)
            _activate_replacement(replacement, voice_dir)
        except Exception:
            shutil.rmtree(replacement, ignore_errors=True)
            raise

    entry = library.get(vid)
    manifest = entry.manifest
    return ImportResult(
        voice_id=vid,
        root=voice_dir,
        manifest=entry.manifest,
        imported_index="index" in placed,
        imported_sample="sample" in placed,
        imported_avatar="avatar" in placed,
        imported_profile=imported_profile,
        generated_profile=generated_profile,
    )


def _neutral_profile(display_name: str) -> VoiceProfile:
    """A conservative zero-shift profile for imported models without a sample.

    RVC archives normally contain only weights and an index.  A broad neutral
    range keeps the model usable without inventing a gender or pushing the song
    into an unmeasured octave.  Training or a measured profile can replace it.
    """
    return VoiceProfile(
        name=display_name,
        comfort_low=36.0,
        comfort_high=84.0,
        abs_low=24.0,
        abs_high=96.0,
        median=60.0,
        f0_method="unmeasured-neutral",
        sample_seconds=0.0,
        voiced_frames=0,
    )


def _extract_into(archive: Path, dest: Path) -> dict[str, Path]:
    """Extract the recognised files, flattened, into `dest`. Zip-slip safe.

    When several files of a kind are present the largest model, the largest
    index, and the first of everything else win -- the largest `.pth` is the
    real model, small `.pth` files in the wild are usually optimiser shards.
    """
    found: dict[str, Path] = {}
    extracted_total = 0
    with zipfile.ZipFile(archive) as zf:
        try:
            members = _safe_members(zf)
        except zipfile.BadZipFile as exc:
            raise EngineError(ErrorCode.VOICE_CORRUPT, str(exc)) from exc
        for position, info in enumerate(members):
            name = Path(info.filename).name  # flatten: defeats ../ traversal
            if not name:
                continue
            suffix = Path(name).suffix.lower()
            kind = _classify(name, suffix)
            if kind is None:
                continue
            # A zip can contain duplicate base names.  Give staging files a
            # unique name so an attacker cannot overwrite a selected candidate.
            out = dest / f"{position:04d}-{name}"
            try:
                with zf.open(info) as src, out.open("wb") as dst:
                    extracted_total += _stream_member(src, dst, extracted_total)
            except (OSError, zipfile.BadZipFile) as exc:
                out.unlink(missing_ok=True)
                raise EngineError(ErrorCode.VOICE_CORRUPT, str(exc)) from exc
            # For model/index, keep the larger of two candidates: the real
            # model is the big .pth, small ones are usually optimiser shards.
            if (
                kind in {"model", "index"}
                and kind in found
                and out.stat().st_size <= found[kind].stat().st_size
            ):
                out.unlink(missing_ok=True)
                continue
            found[kind] = out
    return found


def _stream_member(src: IO[bytes], dst: IO[bytes], extracted_total: int) -> int:
    """Copy one archive member in bounded chunks and enforce actual byte caps."""
    written = 0
    while True:
        chunk = src.read(_COPY_CHUNK_BYTES)
        if not chunk:
            return written
        next_member_total = written + len(chunk)
        next_archive_total = extracted_total + next_member_total
        if next_member_total > _MAX_FILE_BYTES:
            raise EngineError(ErrorCode.VOICE_CORRUPT, "an archive member is too large")
        if next_archive_total > _MAX_TOTAL_BYTES:
            raise EngineError(ErrorCode.VOICE_CORRUPT, "archive is implausibly large")
        dst.write(chunk)
        written = next_member_total


def _classify(name: str, suffix: str) -> str | None:
    lower = name.lower()
    if suffix in _MODEL_SUFFIXES:
        return "model"
    if suffix in _INDEX_SUFFIXES:
        return "index"
    if lower == PROFILE_FILE or (suffix == ".json" and "profile" in lower):
        return "profile"
    if suffix in _AVATAR_SUFFIXES:
        return "avatar"
    if suffix in _SAMPLE_SUFFIXES:
        return "sample"
    return None


def _place(staged: dict[str, Path], voice_dir: Path) -> dict[str, Path]:
    """Move staged files to their canonical names inside the voice folder."""
    targets = {
        "model": MODEL_FILE,
        "index": INDEX_FILE,
        "profile": PROFILE_FILE,
        "sample": SAMPLE_FILE,
        "avatar": AVATAR_FILE,
    }
    placed: dict[str, Path] = {}
    for kind, src in staged.items():
        target = voice_dir / targets[kind]
        src.replace(target)
        placed[kind] = target
    return placed


def _activate_replacement(replacement: Path, voice_dir: Path) -> None:
    """Atomically activate a complete voice and restore its predecessor on error."""
    _recover_interrupted_activation(voice_dir)
    backup = voice_dir.with_name(f".{voice_dir.name}.previous")
    if backup.exists():
        # Both paths mean the preceding activation completed and only its
        # post-activation cleanup was interrupted.  The active directory is
        # authoritative; the backup is stale and may be discarded.
        shutil.rmtree(backup)

    had_previous = voice_dir.exists()
    if had_previous:
        voice_dir.rename(backup)
    try:
        # ``Path.replace`` maps to MoveFileEx on Windows and can fail with
        # ERROR_ACCESS_DENIED for directory-to-directory activation, even when
        # the destination does not exist.  ``rename`` is atomic on this same
        # filesystem and works for directories on every supported platform.
        replacement.rename(voice_dir)
    except Exception:
        if had_previous and backup.exists():
            backup.rename(voice_dir)
        raise
    if had_previous:
        shutil.rmtree(backup, ignore_errors=True)


def _recover_interrupted_activation(voice_dir: Path) -> None:
    """Restore the prior voice when a crash left only its activation backup."""
    backup = voice_dir.with_name(f".{voice_dir.name}.previous")
    if backup.exists() and not voice_dir.exists():
        backup.rename(voice_dir)
