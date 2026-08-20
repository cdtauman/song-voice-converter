"""Preserve Torch's bundled third-party notices without Win32-long paths."""

from __future__ import annotations

import argparse
import zipfile
from importlib.metadata import distribution as installed_distribution
from pathlib import Path


def _license_root(distribution: Path) -> Path | None:
    """Find Torch's notices in the collected app or the build environment."""
    internal = distribution.resolve() / "_internal"
    if not (internal / "torch").is_dir():
        return None
    collected = sorted(internal.glob("torch-*.dist-info/licenses/third_party"))
    if len(collected) == 1:
        return collected[0]
    if len(collected) > 1:
        raise RuntimeError(
            f"expected at most one collected Torch license tree, found {len(collected)}"
        )

    # Recent CPU wheels do not always cause PyInstaller to collect the
    # dist-info directory.  The wheel still has the notices at build time, so
    # bundle them from the installed package instead of making a release
    # depend on a PyInstaller implementation detail.
    metadata = installed_distribution("torch")
    metadata_directory = Path(metadata._path)  # type: ignore[attr-defined]
    installed = metadata_directory / "licenses" / "third_party"
    if not installed.is_dir():
        raise RuntimeError(f"Torch third-party license tree is unavailable: {installed}")
    return installed


def bundle(distribution: Path) -> tuple[Path, int] | None:
    root = _license_root(distribution)
    if root is None:
        return None
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("Torch third-party license tree is empty")
    internal = distribution.resolve() / "_internal"
    output = internal / "licenses" / "torch-third-party-licenses.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if len(archive.infolist()) != len(files) or archive.testzip() is not None:
            raise RuntimeError("Torch third-party license archive verification failed")
    temporary.replace(output)
    return output, len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=Path, required=True)
    args = parser.parse_args()
    result = bundle(args.distribution)
    if result is None:
        print("Torch is not packaged; no Torch third-party license archive is needed")
        return 0
    output, count = result
    print(f"Bundled {count} Torch third-party license files: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
