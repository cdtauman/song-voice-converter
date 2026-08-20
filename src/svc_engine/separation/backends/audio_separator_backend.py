"""`SeparationBackend` on top of `audio-separator` (MIT).

Reuse decision, per docs/reuse-policy.md: **wrap, do not reimplement.**
audio-separator already carries loaders for BS-Roformer, Mel-Band Roformer,
MDX23C, MDX and Demucs, kept working against a moving model zoo. Rewriting
Roformer inference would buy nothing and cost the rest of the phase.

Three things it does *not* do, which this adapter adds:

1. **Model files.** Its downloader has no checksum, no resume and no retry.
   We fetch everything ourselves into `models_dir` first; it then finds the
   files present and skips its own download path.
2. **Intel XPU.** It configures CUDA, MPS and DirectML only, so on an Intel Arc
   machine it silently falls back to CPU. Its `torch_device` is read when the
   model loads, so setting it before `load_model()` routes the work to the GPU.
   Gated on the proof rule -- we only do this where the support matrix says
   separation was actually executed on that backend.
3. **Levels.** It normalises every stem to a peak and writes 16-bit by default.
   Both are wrong here: the stems get mixed back together later, so their
   relative level must survive, and they get processed several more times, so
   they must stay floating point.
"""

from __future__ import annotations

import json
import logging
import tempfile
from contextlib import suppress
from pathlib import Path
from typing import Any

from svc_engine.audio import io as audio_io
from svc_engine.backends.base import AudioBuffer, BackendInfo, DeviceHint
from svc_engine.backends.separation import SeparationRequest, StemKind, Stems
from svc_engine.compute.devices import ComputeBackend
from svc_engine.errors import EngineError, ErrorCode
from svc_engine.resources import DownloadManager, ModelRegistry, ModelSpec

__all__ = ["AudioSeparatorBackend", "BACKEND_ID"]

log = logging.getLogger(__name__)

BACKEND_ID = "audio_separator"

#: How the library names a stem -> what SongVoice calls it. The library takes
#: its names from each model's own config, so this needs to be forgiving.
_STEM_ALIASES: dict[str, StemKind] = {
    "vocals": StemKind.VOCALS,
    "vocal": StemKind.VOCALS,
    "instrumental": StemKind.INSTRUMENTAL,
    "instrument": StemKind.INSTRUMENTAL,
    "no vocals": StemKind.INSTRUMENTAL,
    "music": StemKind.INSTRUMENTAL,
    "other": StemKind.OTHER,
    "bass": StemKind.BASS,
    "drums": StemKind.DRUMS,
    "lead": StemKind.LEAD,
    "backing": StemKind.BACKING,
    "bv": StemKind.BACKING,
    "reverb": StemKind.AMBIENCE,
    "echo": StemKind.AMBIENCE,
    "wet": StemKind.AMBIENCE,
    "ambience": StemKind.AMBIENCE,
    "no dry": StemKind.AMBIENCE,
    # A cleanup model's "kept" output is the vocal, minus whatever it removed.
    "noreverb": StemKind.VOCALS,
    "no reverb": StemKind.VOCALS,
    "dry": StemKind.VOCALS,
    "restored": StemKind.VOCALS,
    "noise": StemKind.OTHER,
}

# audio-separator 0.44.5 fetches this list unconditionally before it checks
# its bundled model catalogue.  The SongVoice production model is already in
# that bundled catalogue, so an empty snapshot of the remote-only sections is
# sufficient and keeps a fully provisioned offline installation offline.
_OFFLINE_DOWNLOAD_CATALOG: dict[str, dict[str, Any]] = {
    "demucs_download_list": {},
    "vr_download_list": {},
    "mdx_download_list": {},
    "mdx_download_vip_list": {},
    "mdx23c_download_list": {},
    "mdx23c_download_vip_list": {},
    "roformer_download_list": {},
}


def _canonical(name: str) -> StemKind | None:
    return _STEM_ALIASES.get(name.strip().lower())


class AudioSeparatorBackend:
    """Runs one separation model per call and hands back in-memory stems."""

    def __init__(
        self,
        models_dir: Path,
        registry: ModelRegistry,
        downloader: DownloadManager | None = None,
        work_dir: Path | None = None,
    ) -> None:
        self.models_dir = Path(models_dir)
        self.registry = registry
        self.downloader = downloader or DownloadManager(self.models_dir)
        self.work_dir = Path(work_dir) if work_dir else None
        self._separator: object | None = None
        self._loaded_model: str | None = None

    # -- SeparationBackend -------------------------------------------------- #

    def info(self) -> BackendInfo:
        import importlib.util

        available = importlib.util.find_spec("audio_separator") is not None
        return BackendInfo(
            backend_id=BACKEND_ID,
            display_name_he="מנוע הפרדה ראשי",
            available=available,
            unavailable_reason=None if available else "החבילה audio-separator אינה מותקנת.",
            supports_gpu=True,
            capabilities=frozenset({"roformer", "mdx", "demucs", "ensemble_source"}),
            model_root=self.models_dir,
        )

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.registry.for_backend(BACKEND_ID))

    def separate(
        self,
        audio: AudioBuffer,
        request: SeparationRequest,
        device: DeviceHint,
    ) -> Stems:
        spec = self._spec_for(request.model_id)
        self.downloader.check_space_for([spec])
        self.downloader.ensure_model(spec)

        if self.work_dir is not None:
            self.work_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=self.work_dir, prefix="svc_sep_") as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "input.wav"
            audio_io.save_wav(audio, source, bit_depth=32)

            # The output directory has to be final before `load_model`: the
            # architecture-specific separator copies it out of the config at
            # construction, so assigning it afterwards changes only where the
            # returned paths *claim* the files are.
            separator = self._build_separator(request, device, tmp_path)
            self._load(separator, spec, request)

            primary_name = getattr(separator.model_instance, "primary_stem_name", "Primary")
            secondary_name = getattr(
                separator.model_instance, "secondary_stem_name", "Secondary"
            )

            outputs = separator.separate(
                str(source),
                custom_output_names={primary_name: "primary", secondary_name: "secondary"},
            )
            produced = self._read_outputs(tmp_path, outputs, audio)

        parts = self._label(produced, spec, primary_name, secondary_name)
        parts = self._derive_missing(parts, audio, spec, request)
        return Stems(parts=parts, model_id=spec.id)

    def unload(self) -> None:
        separator, self._separator, self._loaded_model = self._separator, None, None
        if separator is None:
            return
        with suppress(Exception):
            # Drop the loaded weights so the next model does not have to wait
            # for Python to get around to collecting them.
            setattr(separator, "model_instance", None)  # noqa: B010
        self._empty_device_cache()

    # -- internals ---------------------------------------------------------- #

    def _spec_for(self, model_id: str) -> ModelSpec:
        if model_id in self.registry:
            return self.registry.get(model_id)
        found = self.registry.by_engine_model(BACKEND_ID, model_id)
        if found is None:
            raise EngineError(
                ErrorCode.MODEL_MISSING,
                f"{model_id} is not in the model catalogue",
            )
        return found

    def _build_separator(  # type: ignore[no-untyped-def]
        self, request: SeparationRequest, device: DeviceHint, output_dir: Path
    ):
        try:
            from audio_separator.separator import Separator
        except ImportError as exc:  # pragma: no cover - checked by `svc doctor`
            raise EngineError(ErrorCode.BACKEND_UNAVAILABLE, str(exc)) from exc

        self._ensure_offline_download_catalog()
        segment_size = request.segment_size or 256
        separator = Separator(
            log_level=logging.WARNING,
            model_file_dir=str(self.models_dir),
            output_dir=str(output_dir),
            output_format="WAV",
            sample_rate=44100,
            use_soundfile=True,
            # 1.0 turns peak normalisation into a plain anti-clip guard: stems
            # keep the level they were separated at, so summing them back
            # reconstructs the mix instead of a louder, wrong one.
            normalization_threshold=1.0,
            amplification_threshold=0.0,
            mdxc_params={
                "segment_size": segment_size,
                "override_model_segment_size": True,
                "batch_size": max(1, request.batch_size),
                # Placeholder. For a Roformer this number is not a divisor at
                # all -- see `_apply_overlap`, which rewrites it once the model
                # is loaded and its chunk length is known.
                "overlap": max(1, request.overlap),
                "pitch_shift": 0,
            },
            mdx_params={
                "hop_length": 1024,
                "segment_size": segment_size,
                "overlap": 0.25,
                "batch_size": max(1, request.batch_size),
                "enable_denoise": False,
            },
        )
        self._apply_device(separator, device)
        return separator

    def _ensure_offline_download_catalog(self) -> None:
        """Supply the locked library's required catalogue without a network call."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        catalog = self.models_dir / "download_checks.json"
        if catalog.is_file():
            return
        temporary = catalog.with_suffix(".tmp")
        try:
            temporary.write_text(
                json.dumps(_OFFLINE_DOWNLOAD_CATALOG, sort_keys=True), encoding="utf-8"
            )
            temporary.replace(catalog)
        finally:
            temporary.unlink(missing_ok=True)

    def _apply_device(self, separator, device: DeviceHint) -> None:  # type: ignore[no-untyped-def]
        """Point the library at the device the support matrix chose.

        Only XPU needs help: `setup_torch_device` already handles CUDA, and CPU
        is what it falls back to on its own.
        """
        if device.backend is not ComputeBackend.XPU:
            return
        try:
            import torch

            if not (hasattr(torch, "xpu") and torch.xpu.is_available()):
                log.warning("XPU requested but torch reports it unavailable; staying on CPU")
                return
            separator.torch_device = torch.device(f"xpu:{device.device_index}")
            separator.torch_device_cpu = torch.device("cpu")
        except Exception as exc:  # noqa: BLE001  never let device selection be fatal
            log.warning("could not select the Intel GPU, continuing on CPU: %s", exc)

    def _load(self, separator, spec: ModelSpec, request: SeparationRequest) -> None:  # type: ignore[no-untyped-def]
        try:
            separator.load_model(model_filename=spec.engine_model)
        except Exception as exc:
            if "checksum" in str(exc).lower() or "corrupt" in str(exc).lower():
                raise EngineError(ErrorCode.MODEL_CORRUPT, str(exc)) from exc
            raise
        self._apply_overlap(separator, request)
        self._separator = separator
        self._loaded_model = spec.id

    def _apply_overlap(self, separator, request: SeparationRequest) -> None:  # type: ignore[no-untyped-def]
        """Translate our overlap factor into whatever this model type means by it.

        `SeparationRequest.overlap` is the number of times each sample is
        covered: 1 is contiguous tiles, 4 is four passes averaged. That is the
        useful quantity -- more coverage costs proportionally more time and
        hides chunk seams.

        audio-separator's MDXC path uses two different conventions for the same
        field. Classic MDX23C treats it as a divisor (`hop = chunk / overlap`),
        which already matches ours. Roformer models instead read it as *step
        length in seconds*, clamped to the chunk, so a larger number means
        **less** overlap and a faster, seamier run -- the exact inverse. Left
        alone, "fast" would be slower and rougher than "balanced", which is
        what measurement showed before this translation existed.
        """
        model = getattr(separator, "model_instance", None)
        if model is None or not getattr(model, "is_roformer_model", False):
            return

        config = getattr(model, "model_data_cfgdict", None)
        if config is None:
            return

        try:
            hop = getattr(config.model, "stft_hop_length", None) or config.audio.hop_length
            segment = (
                model.segment_size
                if getattr(model, "override_model_segment_size", False)
                else config.inference.dim_t
            )
            chunk_seconds = int(hop) * (int(segment) - 1) / float(config.audio.sample_rate)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError) as exc:
            log.debug("could not derive chunk length, leaving overlap alone: %s", exc)
            return

        model.overlap = chunk_seconds / max(1, request.overlap)
        log.debug(
            "overlap x%d -> step %.3fs of a %.3fs chunk",
            request.overlap, model.overlap, chunk_seconds,
        )

    def _read_outputs(
        self, tmp_path: Path, outputs: list[str], reference: AudioBuffer
    ) -> dict[str, AudioBuffer]:
        """Load whatever the library wrote, keyed by our custom output names."""
        loaded: dict[str, AudioBuffer] = {}
        for out in outputs or []:
            path = Path(out)
            if not path.is_absolute():
                path = tmp_path / path
            if not path.exists():
                log.warning("separator reported a file it did not write: %s", path)
                continue
            role = "primary" if path.stem.startswith("primary") else (
                "secondary" if path.stem.startswith("secondary") else path.stem
            )
            loaded[role] = audio_io.load_audio(path, sample_rate=reference.sample_rate)
        if not loaded:
            raise EngineError(ErrorCode.INTERNAL, "separation produced no output files")
        return loaded

    def _label(
        self,
        produced: dict[str, AudioBuffer],
        spec: ModelSpec,
        primary_name: str,
        secondary_name: str,
    ) -> dict[StemKind, AudioBuffer]:
        """Name the two outputs.

        The catalogue entry decides, not the checkpoint: `stems` in models.json
        is `[primary, secondary]`. A model whose config calls its outputs
        "vocals"/"other" but which we use as a de-reverb still gets labelled the
        way we intend to use it. The model's own names are only the fallback.
        """
        roles: list[str] = list(spec.stems) or [primary_name, secondary_name]
        mapping = {"primary": roles[0], "secondary": roles[1] if len(roles) > 1 else ""}

        parts: dict[StemKind, AudioBuffer] = {}
        for role, buffer in produced.items():
            declared = mapping.get(role, role)
            kind = _canonical(declared) or _canonical(
                primary_name if role == "primary" else secondary_name
            )
            if kind is None:
                log.warning("unmapped stem name %r from %s", declared, spec.id)
                continue
            parts[kind] = buffer
        return parts

    def _derive_missing(
        self,
        parts: dict[StemKind, AudioBuffer],
        mix: AudioBuffer,
        spec: ModelSpec,
        request: SeparationRequest,
    ) -> dict[StemKind, AudioBuffer]:
        """Fill a requested stem the model did not return, by subtraction.

        Only valid because these models are trained to produce complementary
        pairs: mix - vocals is the instrumental by construction. It is not a
        general trick and is deliberately limited to the vocal/instrumental pair.
        """
        from svc_engine.audio.buffers import fit_length, subtract

        wanted = set(request.wanted)
        pair = (StemKind.VOCALS, StemKind.INSTRUMENTAL)
        for target, other in (pair, pair[::-1]):
            if target in wanted and target not in parts and other in parts:
                parts[target] = subtract(mix, parts[other])
                log.debug("derived %s by subtraction for %s", target.value, spec.id)

        return {
            kind: fit_length(buffer, mix.frames) for kind, buffer in parts.items()
        }

    def _empty_device_cache(self) -> None:
        try:
            import gc

            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                xpu.empty_cache()
        except Exception:  # noqa: BLE001  freeing memory must never raise
            pass
