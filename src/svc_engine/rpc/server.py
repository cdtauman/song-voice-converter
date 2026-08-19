"""Engine-side RPC loop.

The transport stays free of AI imports. Phase 7 adds durable project/job
inspection methods; concrete processing jobs remain engine-owned Python graphs,
never executable callables supplied over JSON.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Callable
from typing import Any, TextIO

from svc_engine.config import Paths, Settings, load_settings, paths, save_settings
from svc_engine.diag import run_all_checks
from svc_engine.diag.report import overall_status
from svc_engine.errors import EngineError, ErrorCode, message_for
from svc_engine.logging_setup import get_logger
from svc_engine.rpc.protocol import PROTOCOL_VERSION, Event, Request, Response, encode

log = get_logger(__name__)

Handler = Callable[[dict[str, Any]], Any]
EventSink = Callable[[Event], None]

__all__ = ["Server", "serve_stdio"]


class Server:
    def __init__(self, app_paths: Paths | None = None) -> None:
        self._paths = app_paths or paths()
        self._paths.ensure()
        self._handlers: dict[str, Handler] = {
            "ping": self._ping,
            "doctor": self._doctor,
            "jobs.recoverable": self._jobs_recoverable,
            "jobs.history": self._jobs_history,
            "jobs.cleanup": self._jobs_cleanup,
            "cache.stats": self._cache_stats,
            "projects.list": self._projects_list,
            "projects.load": self._projects_load,
            "projects.save": self._projects_save,
            "settings.get": self._settings_get,
            "settings.save": self._settings_save,
            "voices.list": self._voices_list,
            "voices.import": self._voices_import,
            "voices.remove": self._voices_remove,
            "voices.update": self._voices_update,
            "voices.health": self._voices_health,
            "training.create": self._training_create,
            "training.list": self._training_list,
            "training.inspect": self._training_inspect,
            "training.prepare": self._training_prepare,
            "training.start": self._training_start,
            "training.status": self._training_status,
            "training.pause": self._training_pause,
            "covers.preview": self._covers_preview,
            "covers.run": self._covers_run,
            "covers.resume": self._covers_resume,
        }
        self._request_id = ""
        self._event_sink: EventSink | None = None

    # -- methods ----------------------------------------------------------- #

    def _ping(self, params: dict[str, Any]) -> dict[str, Any]:
        return {"pong": True, "protocol": PROTOCOL_VERSION, "echo": params.get("echo")}

    def _doctor(self, params: dict[str, Any]) -> dict[str, Any]:
        results = run_all_checks(self._paths.work)
        return {
            "overall": overall_status(results).value,
            "checks": [
                {
                    "key": r.key,
                    "label_he": r.label_he,
                    "status": r.status.value,
                    "message_he": r.message_he,
                    "detail": r.detail,
                }
                for r in results
            ],
        }

    def _jobs_recoverable(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.jobs import RecoveryStore
        from svc_engine.workflows import load_cover_request

        snapshots = RecoveryStore(self._paths.root / "jobs").discover()
        results = []
        for item in snapshots:
            payload: dict[str, Any] = {
                "job_id": item.job_id,
                "name": item.name,
                "status": item.status.value,
                "updated_at": item.updated_at,
                "completed_steps": sum(
                    step.status.value in {"completed", "cached"} for step in item.steps.values()
                ),
                "total_steps": len(item.steps),
            }
            try:
                request = load_cover_request(self._paths, item.job_id)
            except (OSError, ValueError, KeyError, TypeError):
                request = None
            if request is not None:
                payload.update(
                    {
                        "kind": "cover",
                        "source": request.get("song"),
                        "voice_id": request.get("voice_id"),
                        "preview": bool(request.get("preview")),
                    }
                )
            results.append(payload)
        return results

    def _jobs_history(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.history import HistoryStore

        limit = int(params.get("limit", 100))
        return [item.to_dict() for item in HistoryStore(self._paths.db).list(limit=limit)]

    def _jobs_cleanup(self, params: dict[str, Any]) -> dict[str, int]:
        from svc_engine.jobs import JobRunner

        return JobRunner(self._paths, settings=load_settings(self._paths)).cleanup()

    def _cache_stats(self, params: dict[str, Any]) -> dict[str, int]:
        from svc_engine.jobs import StepCache

        stats = StepCache(self._paths.cache / "steps").stats()
        return {"entries": stats.entries, "size_bytes": stats.size_bytes}

    def _projects_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.projects import ProjectStore

        return [item.to_dict() for item in ProjectStore(self._paths.projects).list()]

    def _projects_load(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.projects import ProjectStore

        return ProjectStore(self._paths.projects).load(str(params["project_id"])).to_dict()

    def _projects_save(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.projects import ProjectStore

        data = params.get("data")
        if not isinstance(data, dict):
            raise ValueError("project data must be an object")
        return (
            ProjectStore(self._paths.projects)
            .save(str(params["project_id"]), name=str(params["name"]), data=data)
            .to_dict()
        )

    def _settings_get(self, params: dict[str, Any]) -> dict[str, Any]:
        return dataclasses.asdict(load_settings(self._paths))

    def _settings_save(self, params: dict[str, Any]) -> dict[str, Any]:
        current = dataclasses.asdict(load_settings(self._paths))
        allowed = {
            "quality",
            "prefer_gpu",
            "target_lufs",
            "output_dir",
            "keep_cache_gb",
            "check_updates",
            "allow_model_downloads",
            "language",
            "theme",
            "advanced_mode",
        }
        current.update({key: value for key, value in params.items() if key in allowed})
        settings = Settings(**current)
        if settings.quality not in {"fast", "balanced", "max"}:
            raise ValueError("invalid quality")
        if settings.theme not in {"system", "light", "dark"}:
            raise ValueError("invalid theme")
        if not -70.0 <= float(settings.target_lufs) <= -5.0:
            raise ValueError("invalid target loudness")
        if float(settings.keep_cache_gb) < 0:
            raise ValueError("invalid cache size")
        save_settings(settings, self._paths)
        return dataclasses.asdict(settings)

    def _voices_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        from svc_engine.voices import VoiceLibrary
        from svc_engine.voices.manifest import AVATAR_FILE, SAMPLE_FILE

        voices = []
        for entry in VoiceLibrary(self._paths).list():
            manifest = entry.manifest
            voices.append(
                {
                    **manifest.to_dict(),
                    "usable": manifest.usable and entry.profile_path is not None,
                    "health_note_he": manifest.health.note_he,
                    "has_profile": entry.profile_path is not None,
                    "sample_path": str(entry.root / SAMPLE_FILE)
                    if manifest.has_sample and (entry.root / SAMPLE_FILE).is_file()
                    else None,
                    "avatar_path": str(entry.root / AVATAR_FILE)
                    if manifest.has_avatar and (entry.root / AVATAR_FILE).is_file()
                    else None,
                }
            )
        return voices

    def _voices_import(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.voices import VoiceLibrary, import_voice_from_zip

        result = import_voice_from_zip(
            str(params["archive"]),
            display_name=str(params.get("display_name") or "קול חדש"),
            consent_confirmed=bool(params.get("consent_confirmed", False)),
            consent_note=str(params.get("consent_note") or ""),
            library=VoiceLibrary(self._paths),
        )
        return {
            "voice_id": result.voice_id,
            "summary_he": result.summary_he(),
        }

    def _voices_remove(self, params: dict[str, Any]) -> dict[str, bool]:
        from svc_engine.voices import VoiceLibrary

        VoiceLibrary(self._paths).remove(str(params["voice_id"]))
        return {"removed": True}

    def _voices_update(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.voices import VoiceLibrary

        entry = VoiceLibrary(self._paths).update(
            str(params["voice_id"]),
            display_name=(str(params["display_name"]) if params.get("display_name") else None),
            sample=(str(params["sample"]) if params.get("sample") else None),
            avatar=(str(params["avatar"]) if params.get("avatar") else None),
        )
        return {"id": entry.voice_id, "display_name": entry.manifest.display_name}

    def _voices_health(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.voices import VoiceLibrary

        entry = VoiceLibrary(self._paths).refresh_health(str(params["voice_id"]))
        return {
            "id": entry.voice_id,
            "usable": entry.manifest.usable and entry.profile_path is not None,
            "health": entry.manifest.health.to_dict(),
            "health_note_he": entry.manifest.health.note_he,
        }

    def _trainer(self):  # type: ignore[no-untyped-def]
        from svc_engine.training import TrainingCoordinator

        return TrainingCoordinator(self._paths)

    def _training_create(self, params: dict[str, Any]) -> dict[str, Any]:
        recordings = params.get("recordings")
        if not isinstance(recordings, list):
            raise ValueError("recordings must be a list")
        return self._trainer().create(
            str(params.get("display_name") or "קול חדש"),
            [str(path) for path in recordings],
            bool(params.get("consent_confirmed")),
            str(params.get("consent_note") or ""),
            int(params.get("total_epochs") or 200),
        )

    def _training_list(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        return self._trainer().list()

    def _training_inspect(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._trainer().inspect(str(params["session_id"]))

    def _training_prepare(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._trainer().prepare(
            str(params["session_id"]), separate_mix=bool(params.get("separate_mix", True))
        )

    def _training_start(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._trainer().start(str(params["session_id"]))

    def _training_status(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._trainer().status(str(params["session_id"]))

    def _training_pause(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._trainer().pause(str(params["session_id"]))

    def _covers_preview(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run_cover(params, preview=True)

    def _covers_run(self, params: dict[str, Any]) -> dict[str, Any]:
        return self._run_cover(params, preview=False)

    def _covers_resume(self, params: dict[str, Any]) -> dict[str, Any]:
        from svc_engine.workflows import resume_cover

        return resume_cover(
            self._paths,
            str(params["job_id"]),
            on_progress=lambda fraction, message: self._emit(
                "progress", {"fraction": fraction, "message_he": message}
            ),
            on_job=lambda job_id: self._emit("job", {"job_id": job_id}),
        )

    def _run_cover(self, params: dict[str, Any], *, preview: bool) -> dict[str, Any]:
        from svc_engine.workflows import run_cover

        return run_cover(
            self._paths,
            song=str(params["song"]),
            voice_id=str(params["voice_id"]),
            quality=str(params.get("quality") or "balanced"),
            output=None if preview else str(params["output"]),
            preview=preview,
            preview_seconds=float(params.get("preview_seconds") or 30.0),
            on_progress=lambda fraction, message: self._emit(
                "progress", {"fraction": fraction, "message_he": message}
            ),
            on_job=lambda job_id: self._emit("job", {"job_id": job_id}),
            job_id=str(params["job_id"]) if params.get("job_id") else None,
        )

    def _emit(self, event: str, data: dict[str, Any]) -> None:
        if self._event_sink is not None:
            self._event_sink(Event(id=self._request_id, event=event, data=data))

    # -- dispatch ---------------------------------------------------------- #

    def handle(self, req: Request, on_event: EventSink | None = None) -> Response:
        handler = self._handlers.get(req.method)
        if handler is None:
            msg = message_for(ErrorCode.INTERNAL)
            return Response(
                id=req.id,
                ok=False,
                error_code=ErrorCode.INTERNAL.value,
                error_message_he=msg.render(),
            )
        self._request_id = req.id
        self._event_sink = on_event
        try:
            return Response(id=req.id, ok=True, result=handler(req.params))
        except EngineError as exc:
            log.warning("method %s failed: %s", req.method, exc)
            return Response(
                id=req.id,
                ok=False,
                error_code=exc.code.value,
                error_message_he=exc.user_message.render(),
            )
        except Exception:  # noqa: BLE001  the engine must never die on one bad call
            log.exception("unhandled error in method %s", req.method)
            msg = message_for(ErrorCode.INTERNAL)
            return Response(
                id=req.id,
                ok=False,
                error_code=ErrorCode.INTERNAL.value,
                error_message_he=msg.render(),
            )
        finally:
            self._request_id = ""
            self._event_sink = None


def serve_stdio(stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
    """Read requests from stdin, write responses to stdout, until EOF."""
    from svc_engine.rpc.protocol import decode_request

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    server = Server()
    log.info("engine rpc ready (protocol v%d)", PROTOCOL_VERSION)

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = decode_request(line)
        except Exception:  # noqa: BLE001
            log.warning("dropped malformed request line")
            continue

        def send_event(event: Event) -> None:
            stdout.write(encode(event))
            stdout.flush()

        stdout.write(encode(server.handle(req, on_event=send_event)))
        stdout.flush()
