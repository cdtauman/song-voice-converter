"""Strict, dependency-free benchmark experiment schema (TOML or JSON)."""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["ExperimentSpec", "VariantSpec", "load_experiment"]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    label: str
    backend: str
    command: tuple[str, ...]
    settings: dict[str, Any] = field(default_factory=dict)
    license: str = "unknown"
    environment: str = "core"

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> VariantSpec:
        command = raw.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(part, str) and part for part in command
        ):
            raise ValueError("each variant requires a non-empty command array")
        variant_id = str(raw.get("id") or "")
        if not variant_id or not variant_id.replace("-", "").replace("_", "").isalnum():
            raise ValueError("variant id must contain only letters, numbers, '-' or '_'")
        settings = raw.get("settings") or {}
        if not isinstance(settings, dict):
            raise ValueError("variant settings must be an object")
        return cls(
            variant_id=variant_id,
            label=str(raw.get("label") or variant_id),
            backend=str(raw.get("backend") or variant_id),
            command=tuple(command),
            settings=dict(settings),
            license=str(raw.get("license") or "unknown"),
            environment=str(raw.get("environment") or "core"),
        )


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    input_audio: Path
    variants: tuple[VariantSpec, ...]
    repetitions: int = 1
    timeout_seconds: float = 1800.0
    seed: int = 10

    @classmethod
    def from_dict(cls, raw: dict[str, Any], base: Path) -> ExperimentSpec:
        variants_raw = raw.get("variants")
        if not isinstance(variants_raw, list) or len(variants_raw) < 2:
            raise ValueError("experiment requires at least two variants")
        repetitions = int(raw.get("repetitions", 1))
        if not 1 <= repetitions <= 20:
            raise ValueError("repetitions must be between 1 and 20")
        timeout = float(raw.get("timeout_seconds", 1800.0))
        if not 1.0 <= timeout <= 86400.0:
            raise ValueError("timeout_seconds must be between 1 and 86400")
        source = Path(str(raw.get("input_audio") or ""))
        if not source.is_absolute():
            source = base / source
        return cls(
            name=str(raw.get("name") or "benchmark"),
            input_audio=source.resolve(),
            variants=tuple(VariantSpec.from_dict(dict(item)) for item in variants_raw),
            repetitions=repetitions,
            timeout_seconds=timeout,
            seed=int(raw.get("seed", 10)),
        )


def load_experiment(path: Path | str) -> ExperimentSpec:
    source = Path(path)
    if source.suffix.lower() == ".toml":
        raw = tomllib.loads(source.read_text(encoding="utf-8"))
    elif source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
    else:
        raise ValueError("experiment files must be .toml or .json")
    if not isinstance(raw, dict):
        raise ValueError("experiment root must be an object")
    return ExperimentSpec.from_dict(raw, source.parent)
