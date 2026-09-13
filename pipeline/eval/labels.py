"""Supervised label construction from run-to-failure trajectories."""

from __future__ import annotations

import pandas as pd

from pipeline.config import get_settings


def add_targets(features: pd.DataFrame, *, horizon_cycles: int | None = None) -> pd.DataFrame:
    """Add ``rul`` (clipped) and ``failure`` labels per row.

    RUL is computed from the engine's own final cycle, so labels never cross
    engine boundaries. FR-04.
    """
    settings = get_settings()
    horizon = horizon_cycles or settings.failure_horizon_cycles
    clip = settings.rul_clip
    labeled = features.copy()
    max_cycle = labeled.groupby("unit")["cycle"].transform("max")
    rul = (max_cycle - labeled["cycle"]).clip(upper=clip)
    labeled["rul"] = rul
    labeled["failure"] = (max_cycle - labeled["cycle"]) <= horizon
    return labeled
