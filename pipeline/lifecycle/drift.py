"""PSI drift detection with cooldown (FR-12)."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.config import get_settings


@dataclass(frozen=True)
class DriftReport:
    feature_psi: dict[str, float]
    max_psi: float
    top_features: list[str]
    triggered: bool
    cooldown_remaining: int
    breached_count: int
    breached_fraction: float


def psi_between(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """Population stability index between two 1-D samples."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    if ref.size == 0 or cur.size == 0:
        return 0.0
    if np.ptp(ref) == 0.0 and np.ptp(cur) == 0.0:
        return 0.0
    quantiles = np.unique(np.quantile(ref, np.linspace(0.0, 1.0, bins + 1)))
    edges = np.concatenate([[-np.inf], quantiles, [np.inf]])
    ref_hist, _ = np.histogram(ref, bins=edges)
    cur_hist, _ = np.histogram(cur, bins=edges)
    epsilon = 1e-6
    ref_share = ref_hist / ref_hist.sum()
    cur_share = cur_hist / cur_hist.sum()
    ref_share = np.where(ref_share == 0.0, epsilon, ref_share)
    cur_share = np.where(cur_share == 0.0, epsilon, cur_share)
    return float(np.sum((ref_share - cur_share) * np.log(ref_share / cur_share)))


class DriftDetector:
    """Compares recent streaming windows against the training reference.

    PSI is recomputed when ``buffer_size`` recent rows have accumulated; after
    a trigger the detector cools down for ``cooldown_updates`` update calls so
    retraining is not retriggered by the same shift.
    """

    def __init__(
        self,
        columns: list[str],
        *,
        threshold: float | None = None,
        buffer_size: int = 250,
        cooldown_updates: int | None = None,
        bins: int = 10,
        trigger_fraction: float = 0.15,
    ) -> None:
        settings = get_settings()
        self.columns = list(columns)
        self.threshold = threshold if threshold is not None else settings.psi_threshold
        self.buffer_size = buffer_size
        self.cooldown_updates = (
            cooldown_updates if cooldown_updates is not None else settings.retrain_cooldown_windows
        )
        self.bins = bins
        self.trigger_fraction = trigger_fraction
        self.reference: dict[str, np.ndarray] | None = None
        self.buffer: deque[pd.DataFrame] = deque(maxlen=buffer_size)
        self.cooldown_remaining = 0

    def fit_reference(self, frame: pd.DataFrame) -> None:
        self.reference = {
            column: frame[column].to_numpy(dtype=float) for column in self.columns
        }

    def update(self, frame: pd.DataFrame) -> DriftReport:
        if self.reference is None:
            raise RuntimeError("DriftDetector.fit_reference must be called before update")
        incoming = frame[self.columns]
        self.buffer.extend(incoming.to_dict("records"))
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
        if len(self.buffer) < self.buffer_size:
            return DriftReport({}, 0.0, [], False, self.cooldown_remaining, 0, 0.0)

        buffered = pd.DataFrame(list(self.buffer), columns=self.columns)
        feature_psi: dict[str, float] = {}
        for column in self.columns:
            feature_psi[column] = psi_between(
                self.reference[column],
                buffered[column].to_numpy(dtype=float),
                bins=self.bins,
            )
        ranked = sorted(feature_psi.items(), key=lambda item: item[1], reverse=True)
        max_psi = ranked[0][1] if ranked else 0.0
        top = [name for name, _ in ranked[:10]]
        breached = sum(value > self.threshold for value in feature_psi.values())
        fraction = breached / len(feature_psi) if feature_psi else 0.0
        triggered = bool(
            self.cooldown_remaining == 0
            and fraction >= self.trigger_fraction
        )
        if triggered:
            self.cooldown_remaining = self.cooldown_updates
        # Keep the newest rows as a warm start for the next window.
        warm_start = min(len(self.buffer), self.buffer_size // 5)
        self.buffer = deque(list(self.buffer)[-warm_start:], maxlen=self.buffer_size)
        return DriftReport(
            feature_psi=feature_psi,
            max_psi=max_psi,
            top_features=top,
            triggered=triggered,
            cooldown_remaining=self.cooldown_remaining,
            breached_count=int(breached),
            breached_fraction=float(fraction),
        )
