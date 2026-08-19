"""Preserve Torch's bundled third-party notices without Win32-long paths."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path


def bundle(distribution: Path) -> tuple[Path, int]:
    internal = distribution.resolve() / "_internal"
    roots = sorted(internal.glob("torch-*.dist-info/licenses/third_party"))
    if len(roots) != 1:
        raise RuntimeError(f"expected one Torch third-party license tree, found {len(roots)}")
    files = sorted(path for path in roots[0].rglob("*") if path.is_file())
    if not files:
        raise RuntimeError("Torch third-party license tree is empty")
    output = internal / "licenses" / "torch-third-party-licenses.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(roots[0]).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if len(archive.infolist()) != len(files) or archive.testzip() is not None:
            raise RuntimeError("Torch third-party license archive verification failed")
    temporary.replace(output)
    return output, len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distribution", type=Path, required=True)
    args = parser.parse_args()
    output, count = bundle(args.distribution)
    print(f"Bundled {count} Torch third-party license files: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
