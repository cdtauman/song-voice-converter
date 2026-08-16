"""Architectural boundary tests.

These enforce the two hard rules from docs/architecture.md:
  * svc_app must never import torch or any AI library
  * svc_engine must never import PySide6 or any GUI toolkit
and that the four required backend protocols exist.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

AI_MODULES = {"torch", "torchaudio", "librosa", "audio_separator", "pymss", "numpy"}
GUI_MODULES = {"PySide6", "PyQt5", "PyQt6", "tkinter", "wx"}


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def _files(package: str) -> list[Path]:
    return sorted((SRC / package).rglob("*.py"))


@pytest.mark.parametrize("path", _files("svc_app"), ids=lambda p: p.name)
def test_app_does_not_import_ai_libraries(path: Path) -> None:
    leaked = _imported_roots(path) & AI_MODULES
    assert not leaked, f"{path.relative_to(SRC)} imports AI library/ies: {sorted(leaked)}"


@pytest.mark.parametrize("path", _files("svc_engine"), ids=lambda p: p.name)
def test_engine_does_not_import_gui_libraries(path: Path) -> None:
    leaked = _imported_roots(path) & GUI_MODULES
    assert not leaked, f"{path.relative_to(SRC)} imports GUI library/ies: {sorted(leaked)}"


def test_source_layout_exists() -> None:
    assert (SRC / "svc_engine").is_dir()
    assert (SRC / "svc_app").is_dir()
    assert _files("svc_engine"), "engine package has no modules"
    assert _files("svc_app"), "app package has no modules"


def test_four_backend_protocols_are_defined() -> None:
    from svc_engine import backends

    assert backends.REQUIRED_PROTOCOLS == (
        "ConversionBackend",
        "SeparationBackend",
        "PitchShifter",
        "F0Extractor",
    )
    for name in backends.REQUIRED_PROTOCOLS:
        assert hasattr(backends, name), f"missing protocol: {name}"


def test_protocols_are_runtime_checkable() -> None:
    """A wrong object must fail isinstance -- otherwise the adapter buys us nothing."""
    from svc_engine.backends import ConversionBackend, PitchShifter

    assert not isinstance(object(), ConversionBackend)
    assert not isinstance(object(), PitchShifter)
