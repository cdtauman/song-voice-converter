# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules

repo = Path(SPECPATH).parent
a = Analysis(
    [str(repo / "src" / "svc_app" / "launcher.py")],
    pathex=[str(repo / "src")],
    hiddenimports=collect_submodules("svc_engine.updates"),
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, a.binaries, a.datas, [], name="SongVoiceLauncher",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
