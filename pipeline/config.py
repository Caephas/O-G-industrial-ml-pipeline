"""Central application settings (frozen contract).

Values are overridable through environment variables prefixed with
``PIPELINE_`` or a local ``.env`` file. FR-16: seeds, paths, and lifecycle
defaults are centralized so every run is reproducible from a single
configuration surface.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Subset = Literal["FD001", "FD002", "FD003", "FD004"]
ModelFamily = Literal["anomaly", "failure", "performance", "forecasting"]
ModelStage = Literal["staging", "production", "archived"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Runtime configuration for the pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="PIPELINE_",
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Data scope
    subset: Subset = "FD002"  # acceptance subset; loader supports FD001-FD004

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # Feature engineering
    window_size: int = 5
    features_per_sensor: int = 6

    # Labeling
    failure_horizon_cycles: int = 30
    rul_clip: int = 125

    # Forecasting
    forecast_horizon_cycles: int = 10
    forecast_sensor_count: int = 3

    # Lifecycle
    psi_threshold: float = 0.20
    shadow_eval_window: int = 50
    retrain_cooldown_windows: int = 100
    quality_improvement_margin: float = 0.01
    latency_budget_p50_ms: float = 50.0
    latency_budget_p95_ms: float = 150.0

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Reproducibility
    seed: int = 42

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def quality_dir(self) -> Path:
        return self.data_dir / "quality"

    @property
    def provenance_file(self) -> Path:
        return self.data_dir / "provenance.json"

    @property
    def runs_dir(self) -> Path:
        return self.artifacts_dir / "runs"

    @property
    def registry_dir(self) -> Path:
        return self.artifacts_dir / "registry"

    @property
    def lifecycle_dir(self) -> Path:
        return self.artifacts_dir / "lifecycle"


@lru_cache
def get_settings() -> Settings:
    return Settings()
