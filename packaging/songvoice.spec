# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

repo = Path(SPECPATH).parent
src = repo / "src"
deps = repo / "packaging" / "build" / "dependencies"

hiddenimports = [
    "audio_separator.separator.architectures.demucs_separator",
    "audio_separator.separator.architectures.mdx_separator",
    "audio_separator.separator.architectures.mdxc_separator",
    "audio_separator.separator.architectures.vr_separator",
    "transformers.models.hubert.modeling_hubert",
] + collect_submodules("audio_separator.separator.roformer")
datas = [
    (str(src / "svc_engine" / "data"), "svc_engine/data"),
    (str(repo / "docs" / "third-party.md"), "licenses"),
    (str(deps / "ffmpeg"), "runtime/ffmpeg"),
] + collect_data_files("audio_separator") + collect_data_files("pymss")
binaries = []
intel_runtime = Path(sys.prefix) / "Library" / "bin"
if intel_runtime.is_dir():
    binaries += [(str(path), ".") for path in intel_runtime.glob("*.dll")]

a = Analysis(
    [str(src / "svc_app" / "main.py")],
    pathex=[str(src)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(repo / "packaging" / "hooks")],
    excludes=["env-bench", "benchmark", "seed_vc", "DDSP_SVC"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="SongVoice",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False, name="SongVoice",
)
