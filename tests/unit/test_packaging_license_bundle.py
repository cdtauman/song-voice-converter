from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import SimpleNamespace


def _module():
    script = Path(__file__).resolve().parents[2] / "packaging" / "bundle-third-party-licenses.py"
    spec = importlib.util.spec_from_file_location("license_bundle", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_license_bundle_uses_collected_torch_notices(tmp_path: Path) -> None:
    distribution = tmp_path / "SongVoice"
    notices = distribution / "_internal" / "torch-2.13.0.dist-info" / "licenses" / "third_party"
    notices.mkdir(parents=True)
    (notices / "NOTICE").write_text("notice", encoding="utf-8")

    output, count = _module().bundle(distribution)

    assert count == 1
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["NOTICE"]
        assert archive.read("NOTICE") == b"notice"


def test_license_bundle_falls_back_to_installed_torch_notices(tmp_path: Path, monkeypatch) -> None:
    metadata = tmp_path / "site-packages" / "torch-2.13.0.dist-info"
    notices = metadata / "licenses" / "third_party"
    notices.mkdir(parents=True)
    (notices / "NOTICE").write_text("notice", encoding="utf-8")
    module = _module()
    monkeypatch.setattr(
        module, "installed_distribution", lambda _name: SimpleNamespace(_path=metadata)
    )

    output, count = module.bundle(tmp_path / "SongVoice")

    assert count == 1
    with zipfile.ZipFile(output) as archive:
        assert archive.namelist() == ["NOTICE"]
