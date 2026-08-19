"""Production-focused torch hook.

The upstream PyInstaller hook deliberately collects every torch test,
distributed, export and compiler module. SongVoice uses eager inference and
training only; collecting all submodules adds hundreds of irrelevant modules
and makes the Windows build impractical. Static analysis supplies the modules
we import, while this hook supplies torch's dynamically loaded DLLs and data.
"""

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

binaries = collect_dynamic_libs("torch")
datas = collect_data_files(
    "torch",
    excludes=["**/*.h", "**/*.hpp", "**/*.cuh", "**/*.lib", "**/*.cpp", "**/*.cmake"],
)
hiddenimports = [
    "torch._C",
    "torch.cuda",
    "torch.xpu",
    "torch.nn",
    "torch.fft",
    "torch.linalg",
]
module_collection_mode = "pyz+py"
