"""Rotating file log plus a console handler.

Logs record filenames and metrics only -- never audio content (privacy rule).
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

_CONFIGURED = False

__all__ = ["setup_logging", "get_logger"]

_FMT = "%(asctime)s %(levelname)-7s %(name)-24s %(message)s"


def setup_logging(log_dir: Path, level: int = logging.INFO, console: bool = True) -> Path:
    """Configure root logging once. Returns the active log file path."""
    global _CONFIGURED
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "songvoice.log"

    if _CONFIGURED:
        return log_file

    root = logging.getLogger()
    root.setLevel(level)

    fh = logging.handlers.RotatingFileHandler(
        log_file, maxBytes=4_000_000, backupCount=5, encoding="utf-8"
    )
    fh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)

    if console:
        ch = logging.StreamHandler()
        ch.setLevel(max(level, logging.INFO))
        ch.setFormatter(logging.Formatter("%(levelname)-7s %(message)s"))
        root.addHandler(ch)

    _CONFIGURED = True
    return log_file


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
