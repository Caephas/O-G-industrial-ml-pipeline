"""Settings contract tests."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pipeline.config import Settings, get_settings


def test_default_settings_are_fd002_scoped() -> None:
    settings = Settings()
    assert settings.subset == "FD002"
    assert settings.seed == 42
    assert settings.window_size == 5
    assert settings.features_per_sensor == 6
    assert settings.failure_horizon_cycles == 30
    assert settings.rul_clip == 125
    assert settings.psi_threshold == 0.20
    assert settings.shadow_eval_window == 50


def test_derived_paths() -> None:
    settings = Settings()
    assert settings.raw_dir == settings.data_dir / "raw"
    assert settings.processed_dir == settings.data_dir / "processed"
    assert settings.quality_dir == settings.data_dir / "quality"
    assert settings.runs_dir == settings.artifacts_dir / "runs"
    assert settings.registry_dir == settings.artifacts_dir / "registry"
    assert settings.lifecycle_dir == settings.artifacts_dir / "lifecycle"


def test_environment_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PIPELINE_SUBSET", "FD004")
    assert Settings().subset == "FD004"


def test_invalid_subset_rejected() -> None:
    with pytest.raises(ValidationError):
        Settings(subset="FD999")  # type: ignore[arg-type]


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
