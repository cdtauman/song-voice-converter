"""Run command-backed variants and write a self-contained comparison result."""

from __future__ import annotations

import csv
import hashlib
import html
import json
import os
import platform
import random
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from svc_engine.benchmark.schema import ExperimentSpec, VariantSpec

__all__ = ["BenchmarkRunner"]


@dataclass(frozen=True)
class RunResult:
    variant_id: str
    repetition: int
    status: str
    seconds: float
    peak_ram_mb: float
    peak_vram_mb: float | None
    exit_code: int | None
    audio: str | None
    error: str | None
    settings: dict[str, Any]


class _ResourceSampler:
    def __init__(self, process: subprocess.Popen[str]) -> None:
        self.process = process
        self.peak_ram_mb = 0.0
        self.peak_vram_mb: float | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1.0)

    def _sample(self) -> None:
        while not self._stop.wait(0.1):
            ram = _working_set_mb(self.process.pid)
            if ram is not None:
                self.peak_ram_mb = max(self.peak_ram_mb, ram)
            vram = _nvidia_vram_mb(self.process.pid)
            if vram is not None:
                self.peak_vram_mb = max(self.peak_vram_mb or 0.0, vram)


class BenchmarkRunner:
    def run(self, spec: ExperimentSpec, out_dir: Path | str) -> Path:
        output = Path(out_dir).resolve()
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"benchmark output is not empty: {output}")
        audio_dir = output / "audio"
        logs_dir = output / "logs"
        audio_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        if not spec.input_audio.is_file():
            raise FileNotFoundError(spec.input_audio)

        results: list[RunResult] = []
        for variant in spec.variants:
            for repetition in range(1, spec.repetitions + 1):
                results.append(self._run_one(spec, variant, repetition, audio_dir, logs_dir))

        blind = list(dict.fromkeys(item.variant_id for item in results if item.audio))
        random.Random(spec.seed).shuffle(blind)
        blind_map = {f"גרסה {index + 1}": value for index, value in enumerate(blind)}
        self._write_csv(output / "results.csv", results)
        self._write_manifest(output / "manifest.json", spec, results, blind_map)
        self._write_html(output / "report.html", spec, results, blind_map)
        return output

    def _run_one(
        self,
        spec: ExperimentSpec,
        variant: VariantSpec,
        repetition: int,
        audio_dir: Path,
        logs_dir: Path,
    ) -> RunResult:
        destination = audio_dir / f"{variant.variant_id}-{repetition}.wav"
        command = [
            part.format(input=str(spec.input_audio), output=str(destination))
            for part in variant.command
        ]
        started = time.perf_counter()
        process: subprocess.Popen[str] | None = None
        error: str | None = None
        status = "failed"
        exit_code: int | None = None
        ram = 0.0
        vram: float | None = None
        stdout = ""
        stderr = ""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                shell=False,
            )
            sampler = _ResourceSampler(process)
            sampler.start()
            try:
                stdout, stderr = process.communicate(timeout=spec.timeout_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                error = f"timeout after {spec.timeout_seconds:g}s"
            finally:
                sampler.stop()
            ram, vram = sampler.peak_ram_mb, sampler.peak_vram_mb
            exit_code = process.returncode
            if exit_code == 0 and destination.is_file() and destination.stat().st_size > 0:
                status = "ok"
            elif error is None:
                error = "command did not produce the declared WAV output"
        except OSError as exc:
            error = str(exc)
        seconds = time.perf_counter() - started
        log = logs_dir / f"{variant.variant_id}-{repetition}.log"
        log.write_text(
            f"COMMAND: {json.dumps(command, ensure_ascii=False)}\n"
            f"EXIT: {exit_code}\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}\n",
            encoding="utf-8",
        )
        return RunResult(
            variant.variant_id,
            repetition,
            status,
            seconds,
            ram,
            vram,
            exit_code,
            str(destination.relative_to(audio_dir.parent)).replace(os.sep, "/")
            if status == "ok"
            else None,
            error,
            variant.settings,
        )

    @staticmethod
    def _write_csv(path: Path, results: list[RunResult]) -> None:
        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=[
                    "variant_id", "repetition", "status", "seconds", "peak_ram_mb",
                    "peak_vram_mb", "exit_code", "audio", "error", "settings",
                ],
            )
            writer.writeheader()
            for item in results:
                row = asdict(item)
                row["settings"] = json.dumps(row["settings"], ensure_ascii=False, sort_keys=True)
                writer.writerow(row)

    @staticmethod
    def _write_manifest(
        path: Path,
        spec: ExperimentSpec,
        results: list[RunResult],
        blind_map: dict[str, str],
    ) -> None:
        payload = {
            "schema": 1,
            "name": spec.name,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "input": {
                "path": str(spec.input_audio),
                "sha256": _sha256(spec.input_audio),
                "bytes": spec.input_audio.stat().st_size,
            },
            "host": {
                "platform": platform.platform(),
                "python": platform.python_version(),
                "processor": platform.processor(),
            },
            "seed": spec.seed,
            "repetitions": spec.repetitions,
            "variants": [
                {
                    "id": item.variant_id,
                    "label": item.label,
                    "backend": item.backend,
                    "command": list(item.command),
                    "settings": item.settings,
                    "license": item.license,
                    "environment": item.environment,
                }
                for item in spec.variants
            ],
            "blind_map": blind_map,
            "runs": [asdict(item) for item in results],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _write_html(
        path: Path,
        spec: ExperimentSpec,
        results: list[RunResult],
        blind_map: dict[str, str],
    ) -> None:
        rows = []
        by_id = {item.variant_id: item for item in spec.variants}
        for item in results:
            variant = by_id[item.variant_id]
            player = (
                f'<audio controls preload="metadata" src="{html.escape(item.audio)}"></audio>'
                if item.audio
                else "—"
            )
            rows.append(
                "<tr>"
                f"<td>{html.escape(variant.label)}</td><td>{item.repetition}</td>"
                f"<td>{item.status}</td><td>{item.seconds:.3f}</td>"
                f"<td>{item.peak_ram_mb:.1f}</td>"
                f"<td>{'—' if item.peak_vram_mb is None else f'{item.peak_vram_mb:.1f}'}</td>"
                f"<td>{player}</td><td>{html.escape(item.error or '')}</td></tr>"
            )
        blind_players = []
        first_audio = {item.variant_id: item.audio for item in results if item.audio}
        for alias, variant_id in blind_map.items():
            audio = first_audio.get(variant_id)
            if audio:
                blind_players.append(
                    f'<button data-src="{html.escape(audio)}">{html.escape(alias)}</button>'
                )
        document = f"""<!doctype html><html lang="he" dir="rtl"><meta charset="utf-8">
<title>{html.escape(spec.name)}</title><style>
body{{font-family:Segoe UI,Arial;margin:2rem;background:#f6f7fb;color:#172033}}
table{{border-collapse:collapse;width:100%;background:white}}
th,td{{padding:.6rem;border:1px solid #d9deea}}
button{{margin:.3rem;padding:.6rem 1rem}}audio{{max-width:260px}}
</style><h1>{html.escape(spec.name)}</h1>
<h2>השוואה עיוורת</h2><p>השמות ממופים רק ב־manifest.json. המעבר שומר נקודת זמן.</p>
<audio id="blind" controls></audio><div>{''.join(blind_players)}</div>
<p><button id="reveal">חשוף זהויות וטבלת תוצאות</button></p>
<section id="results" hidden><h2>תוצאות</h2><table><thead><tr><th>גרסה</th><th>חזרה</th><th>מצב</th>
<th>שניות</th><th>RAM MB</th><th>VRAM MB</th><th>שמע</th><th>שגיאה</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></section><script>
const p=document.querySelector('#blind');
document.querySelectorAll('button[data-src]').forEach(b=>b.onclick=()=>{{
 const t=p.currentTime; const playing=!p.paused; p.src=b.dataset.src;
 p.currentTime=t; if(playing)p.play(); }});
document.querySelector('#reveal').onclick=()=>{{document.querySelector('#results').hidden=false;}};
</script></html>"""
        path.write_text(document, encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _working_set_mb(pid: int) -> float | None:
    if os.name != "nt":
        status = Path(f"/proc/{pid}/status")
        if status.is_file():
            for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        return None
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        handle = ctypes.windll.kernel32.OpenProcess(0x1000 | 0x0400, False, pid)
        if not handle:
            return None
        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        ctypes.windll.kernel32.CloseHandle(handle)
        return counters.WorkingSetSize / (1024.0 * 1024.0) if ok else None
    except (AttributeError, OSError, ValueError):
        return None


def _nvidia_vram_mb(pid: int) -> float | None:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=1.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    total = 0.0
    found = False
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            try:
                total += float(parts[1])
                found = True
            except ValueError:
                pass
    return total if found else None
