"""Random Forest failure prediction with Platt calibration (FR-06)."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

_PROBA_CLIP = 1e-6


class FailureModel:
    """Binary "fails within horizon" classifier.

    The Random Forest is fit on one engine-disjoint split and the sigmoid
    (Platt) calibrator is fit on a separate engine-disjoint split, so no
    engine's cycles appear in both base-model training and calibration.
    """

    def __init__(self, *, seed: int = 42, n_estimators: int = 200) -> None:
        self.seed = seed
        self.n_estimators = n_estimators
        self.forest: RandomForestClassifier | None = None
        self.platt: LogisticRegression | None = None

    def fit(
        self,
        X_fit: np.ndarray,
        y_fit: np.ndarray,
        X_calibration: np.ndarray,
        y_calibration: np.ndarray,
    ) -> FailureModel:
        self.forest = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            random_state=self.seed,
            n_jobs=-1,
        )
        self.forest.fit(X_fit, y_fit)
        raw = self.forest.predict_proba(X_calibration)[:, 1]
        logits = self._logits(raw)
        self.platt = LogisticRegression(C=1e6)
        self.platt.fit(logits.reshape(-1, 1), y_calibration)
        return self

    @staticmethod
    def _logits(proba: np.ndarray) -> np.ndarray:
        clipped = np.clip(proba, _PROBA_CLIP, 1.0 - _PROBA_CLIP)
        return np.log(clipped / (1.0 - clipped))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.forest is None or self.platt is None:
            raise RuntimeError("FailureModel must be fitted before predicting")
        raw = self.forest.predict_proba(X)[:, 1]
        return self.platt.predict_proba(self._logits(raw).reshape(-1, 1))[:, 1]

    @property
    def params(self) -> dict[str, object]:
        return {
            "family": "failure",
            "algorithm": "RandomForest + Platt sigmoid calibration",
            "n_estimators": self.n_estimators,
            "class_weight": "balanced",
            "calibration_split": "engine-disjoint",
            "seed": self.seed,
        }
