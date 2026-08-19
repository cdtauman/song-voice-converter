from __future__ import annotations

import numpy as np
import pytest

from svc_engine.backends.base import AudioBuffer
from svc_engine.tuning import AdvancedConfig, auto_tune, candidate_grid, score_audio


def _audio(level: float, clipped: bool = False) -> AudioBuffer:
    wave = np.sin(np.linspace(0, 20, 4000, dtype=np.float32)) * level
    if clipped:
        wave[::5] = 1.0
    return AudioBuffer(wave.reshape(1, -1), 4000)


def test_advanced_config_validates_and_maps_every_engine_control() -> None:
    config = AdvancedConfig(
        index_rate=0.5,
        protect=0.4,
        rms_mix_rate=0.2,
        filter_radius=5,
        formant_shift=1.5,
        target_lufs=-12.0,
        ambience_strategy="C",
        playback_strategy="B",
        f0_method="rmvpe",
        deess_enabled=False,
        melody_correction=False,
        auto_tune=True,
    )
    assert config.conversion_params().index_rate == pytest.approx(0.5)
    assert config.postfx_config().target_lufs == pytest.approx(-12.0)
    assert config.playback.value == "B"
    assert AdvancedConfig.from_dict(config.to_dict()) == config
    with pytest.raises(ValueError):
        AdvancedConfig(index_rate=1.1)


def test_auto_tuning_is_four_bounded_candidates_and_keeps_manual_in_search() -> None:
    base = AdvancedConfig()
    assert len(candidate_grid(base)) == 4
    levels = iter([0.25, 0.25, 0.25, 0.25])
    result = auto_tune(base, lambda _config: _audio(next(levels)))
    assert len(result.candidates) == 4
    assert result.candidates[0].config == base
    assert result.winner.candidate_id == "candidate-1"


def test_objective_score_strongly_penalizes_clipping() -> None:
    clean, clean_metrics = score_audio(_audio(0.25))
    clipped, clipped_metrics = score_audio(_audio(0.25, clipped=True))
    assert clean > clipped
    assert clean_metrics["clipped_fraction"] == 0.0
    assert clipped_metrics["clipped_fraction"] > 0.0
