"""Isolation Forest anomaly detection with adaptive contamination (FR-05)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest


class AnomalyModel:
    """Unsupervised anomaly detector.

    Contamination is not hardcoded: it is estimated from the training split as
    the share of windows in the final ``degradation_window`` cycles of engine
    life (a label-derived proxy used only to size the contamination; the model
    itself is unsupervised at serve time).
    """

    def __init__(self, *, seed: int = 42, degradation_window: int | None = None) -> None:
        self.seed = seed
        self.degradation_window = degradation_window
        self.model: IsolationForest | None = None
        self.contamination: float | None = None

    def fit(self, X: np.ndarray, rul: np.ndarray, horizon_cycles: int) -> AnomalyModel:
        window = self.degradation_window or 3 * horizon_cycles
        self.contamination = float(np.mean(rul <= window))
        self.contamination = min(max(self.contamination, 1e-4), 0.5)
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.model.fit(X)
        return self

    def scores(self, X: np.ndarray) -> np.ndarray:
        """Higher score = more anomalous."""
        if self.model is None:
            raise RuntimeError("AnomalyModel must be fitted before scoring")
        return -self.model.decision_function(X)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (anomaly_score, flag) arrays."""
        if self.model is None:
            raise RuntimeError("AnomalyModel must be fitted before predicting")
        scores = self.scores(X)
        flags = self.model.predict(X) == -1
        return scores, flags

    @property
    def params(self) -> dict[str, object]:
        return {
            "family": "anomaly",
            "algorithm": "IsolationForest",
            "n_estimators": 200,
            "adaptive_contamination": self.contamination,
            "seed": self.seed,
        }
