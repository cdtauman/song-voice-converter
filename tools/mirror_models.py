"""Mirror redistributable, pinned model files to the private release repository.

External writes require both ``HF_TOKEN`` and ``SONGVOICE_MODEL_REPO``.  Without
them, ``--audit`` still proves which files are eligible and refuses every
unlicensed or unpinned catalogue entry.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from svc_engine.resources import DownloadManager, load_registry  # noqa: E402


def eligible_files():  # type: ignore[no-untyped-def]
    registry = load_registry()
    for model in registry.models.values():
        if not model.license.is_redistributable:
            continue
        for file in model.files:
            if not file.sha256:
                raise RuntimeError(f"redistributable file is not pinned: {model.id}/{file.name}")
            yield model, file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args(argv)
    items = list(eligible_files())
    print(f"Eligible pinned files: {len(items)}")
    for model, file in items:
        print(f"  {model.id}/{file.name}  {file.sha256}")
    if args.audit:
        return 0

    token = os.environ.get("HF_TOKEN")
    repository = os.environ.get("SONGVOICE_MODEL_REPO")
    if not token or not repository:
        raise RuntimeError("HF_TOKEN and SONGVOICE_MODEL_REPO are required for mirroring")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("install huggingface_hub to perform the external mirror write") from exc

    api = HfApi(token=token)
    with tempfile.TemporaryDirectory(prefix="songvoice-model-mirror-") as temporary:
        root = Path(temporary)
        manager = DownloadManager(root)
        for model, file in items:
            local = manager.ensure_file(model, file)
            api.upload_file(
                path_or_fileobj=str(local),
                path_in_repo=f"models/{file.name}",
                repo_id=repository,
                repo_type="model",
                commit_message=f"Mirror verified {model.id}/{file.name}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
