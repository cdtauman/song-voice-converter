"""Stable Windows launcher: apply a staged update, then start the GUI.

Installer shortcuts target this executable.  Update payloads intentionally do
not replace it, so it can update the rest of the installation before the GUI
has opened any files.
"""

from __future__ import annotations

import contextlib
import subprocess
import sys
from pathlib import Path

from svc_engine.config import paths
from svc_engine.updates import UpdateManager


def main() -> int:
    install_dir = Path(sys.executable).resolve().parent
    manager = UpdateManager(paths().root / "updates")
    with contextlib.suppress(Exception):
        manager.apply_pending(install_dir)
    # The manager rolls a partial transaction back. Starting the known-good
    # version is safer than stranding the user if an update cannot be applied.
    app = install_dir / "SongVoice.exe"
    if not app.is_file():
        return 2
    subprocess.Popen([str(app)], cwd=install_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
