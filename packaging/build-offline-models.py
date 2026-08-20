"""Create the checksum-verified model payload for the offline installer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from svc_engine.provisioning import CORE_MODEL_IDS  # noqa: E402
from svc_engine.resources import DownloadManager, load_registry, sha256_of  # noqa: E402


def main() -> int:
    target = REPO / "packaging" / "build" / "offline" / "models"
    registry = load_registry()
    specs = [registry.get(model_id) for model_id in CORE_MODEL_IDS]
    manager = DownloadManager(target)
    manager.check_space_for(specs)
    for spec in specs:
        if not spec.license.is_redistributable:
            raise RuntimeError(f"refusing non-redistributable model: {spec.id}")
        for file_spec in spec.files:
            if not file_spec.sha256:
                raise RuntimeError(f"refusing unpinned file: {spec.id}/{file_spec.name}")
        print(f"Fetching {spec.id} ({spec.size_mb:.0f} MB)")
        manager.ensure_model(spec)
    files = []
    for path in sorted(item for item in target.rglob("*") if item.is_file()):
        files.append(
            {
                "path": path.relative_to(target).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_of(path),
            }
        )
    (target.parent / "manifest.json").write_text(
        json.dumps({"schema": 1, "models": list(CORE_MODEL_IDS), "files": files}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
