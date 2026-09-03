"""Per-sensor ARIMA/SARIMA forecasting, independent DAG leg (FR-08)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.data.features import feature_columns
from pipeline.eval.baselines import persistence_mape
from pipeline.eval.metrics import evaluate_forecast


class ForecastingModel:
    """Configuration for the independent forecasting leg.

    Sensor selection is deterministic (highest mean |slope| on training
    features). Models are fit per engine/sensor at evaluation or serve time
    via pmdarima AutoARIMA, so the artifact stores configuration, not a single
    fitted estimator.
    """

    def __init__(
        self,
        *,
        sensors: list[str] | None = None,
        horizon: int = 10,
        seed: int = 42,
    ) -> None:
        self.sensors = sensors or []
        self.horizon = horizon
        self.seed = seed

    @staticmethod
    def select_sensors(train_features: pd.DataFrame, count: int) -> list[str]:
        slope_columns = [column for column in feature_columns() if column.endswith("_slope")]
        per_engine_means = train_features.groupby("unit")[slope_columns].mean()
        magnitudes = per_engine_means.abs().mean().sort_values(ascending=False)
        return [column.replace("_slope", "") for column in magnitudes.head(count).index]

    def _autoarima_kwargs(self) -> dict[str, object]:
        return {
            "seasonal": False,
            # Explicit stationarity/order bounds beat unconstrained search on
            # noisy sensor series: linear trend, differencing disabled, capped
            # p+q so forecasts cannot degrade into random-walk behavior.
            "d": 0,
            "trend": "c",
            "stepwise": True,
            "max_order": 4,
            "n_jobs": 1,
            "random": False,
            "random_state": self.seed,
            "suppress_warnings": True,
            "error_action": "ignore",
        }

    def fit_forecast(self, history: np.ndarray, steps: int) -> np.ndarray:
        """Fit AutoARIMA on history and return ``steps`` forecast values."""
        from pmdarima import auto_arima

        fitted = auto_arima(history, **self._autoarima_kwargs())
        return np.asarray(fitted.predict(n_periods=steps), dtype=float)

    def evaluate(
        self,
        raw_train: pd.DataFrame,
        train_features: pd.DataFrame,
        *,
        engine_sample: int = 30,
    ) -> dict[str, object]:
        """MAPE vs persistence over sampled train engines, per selected sensor."""
        if not self.sensors:
            raise RuntimeError("forecasting sensors must be selected before evaluation")
        eligible = [
            int(unit)
            for unit in sorted(raw_train["unit"].unique())
            if len(raw_train.query("unit == @unit")) > self.horizon + 5
        ]
        rng = np.random.default_rng(self.seed)
        sample = sorted(rng.choice(eligible, size=min(engine_sample, len(eligible)), replace=False))
        sensor_mapes: dict[str, float] = {}
        sensor_persistence: dict[str, float] = {}
        for sensor in self.sensors:
            model_values: list[float] = []
            baseline_values: list[float] = []
            for engine_id in sample:
                series = raw_train.loc[raw_train["unit"] == engine_id, sensor].to_numpy(dtype=float)
                context = series[: -self.horizon]
                forecast = self.fit_forecast(context, self.horizon)
                actual = series[-self.horizon :]
                model_values.append(evaluate_forecast(actual, forecast)["mape"])
                baseline_values.append(persistence_mape(series, self.horizon))
            sensor_mapes[sensor] = float(np.mean(model_values))
            sensor_persistence[sensor] = float(np.mean(baseline_values))
        mape = float(np.mean(list(sensor_mapes.values())))
        persistence = float(np.mean(list(sensor_persistence.values())))
        return {
            "sensors": self.sensors,
            "engine_count": len(sample),
            "mape": mape,
            "mape_by_sensor": sensor_mapes,
            "persistence_mape": persistence,
            "mape_ratio_vs_persistence": mape / persistence if persistence else float("inf"),
        }

    @property
    def params(self) -> dict[str, object]:
        return {
            "family": "forecasting",
            "algorithm": "pmdarima AutoARIMA (linear trend, capped order)",
            "sensors": self.sensors,
            "horizon_cycles": self.horizon,
            "seed": self.seed,
        }
