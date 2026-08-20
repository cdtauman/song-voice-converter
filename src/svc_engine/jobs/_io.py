"""Small, dependency-free filesystem primitives shared by durable stores."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def validate_identifier(value: str, *, label: str = "identifier") -> str:
    """Accept a portable, single-component identifier safe on Windows."""
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"invalid {label}: {value!r}")
    if value.rstrip(". ") != value or value.split(".", 1)[0].upper() in _WINDOWS_RESERVED:
        raise ValueError(f"invalid Windows {label}: {value!r}")
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    """Replace a JSON file atomically, flushing bytes before publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp = Path(temporary)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        _flush_directory(path.parent)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def read_json_object(path: Path) -> dict[str, Any]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"expected JSON object in {path}")
    return raw


def _flush_directory(path: Path) -> None:
    """Best-effort metadata durability; opening a directory is unsupported on Windows."""
    if os.name == "nt":
        return
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
