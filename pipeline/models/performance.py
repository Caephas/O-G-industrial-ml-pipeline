"""RUL estimation: RF point estimate + gradient-boosted quantile interval."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


class RulModel:
    """Point RUL from Random Forest, uncertainty from quantile losses.

    The 5th and 95th percentile models use gradient-boosted quantile losses
    (``loss="quantile"``); predictions are clipped so lower <= point <= upper
    always holds. FR-07.
    """

    def __init__(
        self,
        *,
        seed: int = 42,
        n_estimators: int = 200,
        quantile_lower: float = 0.05,
        quantile_upper: float = 0.95,
        interval_level: float = 0.9,
    ) -> None:
        self.seed = seed
        self.interval_level = interval_level
        self.forest: RandomForestRegressor | None = None
        self.lower_model: HistGradientBoostingRegressor | None = None
        self.upper_model: HistGradientBoostingRegressor | None = None
        self.scale_factor: float = 1.0
        self._hgb_kwargs = dict(
            loss="quantile",
            max_iter=300,
            max_leaf_nodes=31,
            random_state=seed,
        )
        self._lower_quantile = quantile_lower
        self._upper_quantile = quantile_upper
        self.n_estimators = n_estimators

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_calibration: np.ndarray | None = None,
        y_calibration: np.ndarray | None = None,
    ) -> RulModel:
        self.forest = RandomForestRegressor(
            n_estimators=self.n_estimators,
            random_state=self.seed,
            n_jobs=-1,
        )
        self.forest.fit(X, y)
        self.lower_model = HistGradientBoostingRegressor(
            quantile=self._lower_quantile,
            **self._hgb_kwargs,
        )
        self.upper_model = HistGradientBoostingRegressor(
            quantile=self._upper_quantile,
            **self._hgb_kwargs,
        )
        self.lower_model.fit(X, y)
        self.upper_model.fit(X, y)
        if X_calibration is not None and y_calibration is not None:
            self._calibrate_scale(X_calibration, y_calibration)
        return self

    def _predict_raw(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.forest is None or self.lower_model is None or self.upper_model is None:
            raise RuntimeError("RulModel must be fitted before predicting")
        point = self.forest.predict(X)
        lower = np.maximum(self.lower_model.predict(X), 0.0)
        upper = self.upper_model.predict(X)
        upper = np.maximum(upper, lower)
        return {"point": point, "lower": lower, "upper": upper}

    def _calibrate_scale(self, X: np.ndarray, y: np.ndarray) -> None:
        """Find the smallest symmetric scale hitting nominal coverage."""
        raw = self._predict_raw(X)
        for factor in np.arange(1.0, 4.01, 0.05):
            lower = raw["point"] - factor * (raw["point"] - raw["lower"])
            upper = raw["point"] + factor * (raw["upper"] - raw["point"])
            coverage = float(np.mean((lower <= y) & (y <= upper)))
            if coverage >= self.interval_level:
                self.scale_factor = float(factor)
                return

    def predict(self, X: np.ndarray) -> dict[str, np.ndarray]:
        if self.forest is None or self.lower_model is None or self.upper_model is None:
            raise RuntimeError("RulModel must be fitted before predicting")
        raw = self._predict_raw(X)
        point = raw["point"]
        lower = point - self.scale_factor * (point - raw["lower"])
        upper = point + self.scale_factor * (raw["upper"] - point)
        lower = np.maximum(lower, 0.0)
        lower = np.minimum(lower, point)
        upper = np.maximum(np.maximum(upper, point), lower)
        return {
            "point": point,
            "lower": lower,
            "upper": upper,
            "interval_level": np.full(len(X), self.interval_level),
        }

    @property
    def params(self) -> dict[str, object]:
        return {
            "family": "performance",
            "point_algorithm": "RandomForestRegressor",
            "interval_algorithm": "HistGradientBoostingRegressor quantile loss",
            "quantiles": [self._lower_quantile, self._upper_quantile],
            "interval_level": self.interval_level,
            "n_estimators": self.n_estimators,
            "max_iter": 300,
            "interval_calibration": "multiplicative scale on engine-disjoint split",
            "calibration_scale_factor": self.scale_factor,
            "seed": self.seed,
        }
