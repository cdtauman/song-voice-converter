"""Platform process-tree boundary for one benchmark variant.

Windows uses a Job Object with ``KILL_ON_JOB_CLOSE``.  That is the production
path: adapter children inherit the job, telemetry can enumerate the complete
job, and a timeout terminates the job rather than only the adapter PID.  POSIX
uses a new session/process group and discovers descendants through ``/proc`` or
``ps`` as a best-effort fallback.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

__all__ = ["ProcessTree"]


class ProcessTree:
    def __init__(self, process: subprocess.Popen[str], windows_job: _WindowsJob | None) -> None:
        self.process = process
        self._windows_job = windows_job

    @classmethod
    def start(cls, command: Sequence[str], **kwargs: Any) -> ProcessTree:
        if os.name == "nt":
            job = _WindowsJob()
            process: subprocess.Popen[str] | None = None
            try:
                process = subprocess.Popen(
                    command,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    **kwargs,
                )
                job.assign(process)
            except BaseException:
                job.close()
                if process is not None:
                    process.kill()
                    process.wait(timeout=5)
                raise
            return cls(process, job)
        process = subprocess.Popen(command, start_new_session=True, **kwargs)
        return cls(process, None)

    def pids(self) -> set[int]:
        if self._windows_job is not None:
            return self._windows_job.pids()
        return _posix_descendants(self.process.pid)

    def terminate(self, timeout: float = 5.0) -> None:
        """Terminate the complete variant tree and wait for it to disappear."""
        if self._windows_job is not None:
            self._windows_job.terminate(1)
        else:
            with contextlib.suppress(ProcessLookupError):
                os.kill(-self.process.pid, 9)
        deadline = time.monotonic() + timeout
        try:
            self.process.wait(timeout=max(0.05, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=max(0.05, deadline - time.monotonic()))
        while self.pids() and time.monotonic() < deadline:
            time.sleep(0.02)

    def close(self) -> None:
        if self._windows_job is not None:
            self._windows_job.close()
            self._windows_job = None
        elif self.process.poll() is None:
            self.terminate()


class _WindowsJob:
    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        self._kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._kernel32.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self._kernel32.CreateJobObjectW.restype = wintypes.HANDLE
        self._kernel32.SetInformationJobObject.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD
        ]
        self._kernel32.SetInformationJobObject.restype = wintypes.BOOL
        self._kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self._kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
        self._kernel32.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self._kernel32.QueryInformationJobObject.restype = wintypes.BOOL
        self._kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self._kernel32.TerminateJobObject.restype = wintypes.BOOL
        self._kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
        self._kernel32.CloseHandle.restype = wintypes.BOOL
        self._handle = self._kernel32.CreateJobObjectW(None, None)
        if not self._handle:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            self._set_kill_on_close()
        except BaseException:
            self.close()
            raise

    def _set_kill_on_close(self) -> None:
        ctypes = self._ctypes
        info = _job_info_structures(ctypes)[2]()
        info.BasicLimitInformation.LimitFlags = 0x00002000
        ok = self._kernel32.SetInformationJobObject(
            self._handle,
            9,  # JobObjectExtendedLimitInformation
            ctypes.byref(info),
            ctypes.sizeof(info),
        )
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, process: subprocess.Popen[str]) -> None:
        handle = int(process._handle)  # type: ignore[attr-defined]
        if not self._kernel32.AssignProcessToJobObject(self._handle, handle):
            raise self._ctypes.WinError(self._ctypes.get_last_error())

    def pids(self) -> set[int]:
        ctypes = self._ctypes
        pointer_size = ctypes.sizeof(ctypes.c_size_t)
        capacity = 16
        while capacity <= 65536:
            size = 8 + capacity * pointer_size
            buffer = ctypes.create_string_buffer(size)
            returned = ctypes.c_ulong()
            ok = self._kernel32.QueryInformationJobObject(
                self._handle,
                3,  # JobObjectBasicProcessIdList
                buffer,
                size,
                ctypes.byref(returned),
            )
            assigned = ctypes.c_ulong.from_buffer(buffer, 0).value
            count = ctypes.c_ulong.from_buffer(buffer, 4).value
            if ok:
                return {
                    int(ctypes.c_size_t.from_buffer(buffer, 8 + index * pointer_size).value)
                    for index in range(min(count, capacity))
                }
            error = ctypes.get_last_error()
            if error != 234:  # ERROR_MORE_DATA
                raise ctypes.WinError(error)
            capacity = max(capacity * 2, assigned)
        raise RuntimeError("benchmark job contains too many processes")

    def terminate(self, exit_code: int) -> None:
        if self._handle and not self._kernel32.TerminateJobObject(self._handle, exit_code):
            error = self._ctypes.get_last_error()
            if error not in {5, 6}:  # already closed/terminated
                raise self._ctypes.WinError(error)

    def close(self) -> None:
        if self._handle:
            self._kernel32.CloseHandle(self._handle)
            self._handle = None


def _job_info_structures(ctypes: Any) -> tuple[type[Any], type[Any], type[Any]]:
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        ]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimit),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    return BasicLimit, IoCounters, ExtendedLimit


def _posix_descendants(root_pid: int) -> set[int]:
    parents: dict[int, int] = {}
    proc = "/proc"
    if os.path.isdir(proc):
        for name in os.listdir(proc):
            if not name.isdigit():
                continue
            try:
                fields = Path(f"{proc}/{name}/stat").read_text(encoding="utf-8").split()
                parents[int(name)] = int(fields[3])
            except (OSError, ValueError, IndexError):
                continue
    else:
        try:
            result = subprocess.run(
                ["ps", "-eo", "pid=,ppid="],
                capture_output=True,
                text=True,
                timeout=1.0,
                check=False,
            )
            for line in result.stdout.splitlines():
                pid, parent = (int(value) for value in line.split())
                parents[pid] = parent
        except (OSError, subprocess.TimeoutExpired, ValueError):
            return {root_pid} if _pid_exists(root_pid) else set()
    found = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, parent in parents.items():
            if parent in found and pid not in found:
                found.add(pid)
                changed = True
    return {pid for pid in found if _pid_exists(pid)}


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
