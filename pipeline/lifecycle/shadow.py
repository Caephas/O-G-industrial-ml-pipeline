"""Shadow-mode dual quality/latency gates (FR-11)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pipeline.config import get_settings
from pipeline.eval.metrics import evaluate_failure, evaluate_rul


@dataclass
class ShadowEvaluator:
    """Collects champion/challenger predictions and decides promotion.

    A candidate is evaluated over ``window_size`` scored predictions; the
    quality gate requires the challenger to be no worse than the champion on
    failure Brier (and RUL RMSE) within the configured tolerance, and the
    latency gate requires p50/p95 within budget.
    """

    window_size: int = 50
    min_labeled: int = 30
    tolerance: float = 0.02
    latency_budget_p50_ms: float = 50.0
    latency_budget_p95_ms: float = 150.0
    champion_proba: list[float] = field(default_factory=list)
    champion_rul: list[float] = field(default_factory=list)
    challenger_proba: list[float] = field(default_factory=list)
    challenger_rul: list[float] = field(default_factory=list)
    labeled: list[bool] = field(default_factory=list)
    failure_label: list[float] = field(default_factory=list)
    actual_rul: list[float] = field(default_factory=list)
    latencies_ms: list[float] = field(default_factory=list)

    def reset(self) -> None:
        self.champion_proba.clear()
        self.champion_rul.clear()
        self.challenger_proba.clear()
        self.challenger_rul.clear()
        self.labeled.clear()
        self.failure_label.clear()
        self.actual_rul.clear()
        self.latencies_ms.clear()

    def add(
        self,
        *,
        champion_proba: float,
        champion_rul: float,
        challenger_proba: float | None,
        challenger_rul: float | None,
        label_available: bool,
        failure_label: float | None = None,
        actual_rul: float | None = None,
        latency_ms: float,
    ) -> None:
        self.champion_proba.append(champion_proba)
        self.champion_rul.append(champion_rul)
        self.labeled.append(label_available)
        self.failure_label.append(failure_label if label_available else float("nan"))
        self.actual_rul.append(actual_rul if label_available else float("nan"))
        self.latencies_ms.append(latency_ms)
        if challenger_proba is not None and challenger_rul is not None:
            self.challenger_proba.append(challenger_proba)
            self.challenger_rul.append(challenger_rul)

    def scored_count(self) -> int:
        return len(self.champion_proba)

    def labeled_count(self) -> int:
        return sum(self.labeled)

    def ready(self) -> bool:
        return self.scored_count() >= self.window_size and self.labeled_count() >= self.min_labeled

    def evaluate(self) -> dict[str, object]:
        if not self.ready():
            return {
                "quality_pass": False,
                "latency_pass": False,
                "promote": False,
                "reason": "insufficient labeled predictions",
            }
        labeled_idx = np.array([index for index, flag in enumerate(self.labeled) if flag])
        y_proba_champion = np.asarray(self.champion_proba)[labeled_idx]
        y_proba_challenger = np.asarray(self.challenger_proba)[labeled_idx]
        rul_champion = np.asarray(self.champion_rul)[labeled_idx]
        rul_challenger = np.asarray(self.challenger_rul)[labeled_idx]
        failure_labels = np.asarray(self.failure_label)[labeled_idx]
        actual_rul_values = np.asarray(self.actual_rul)[labeled_idx]

        quality = self._quality_gate(
            y_proba_champion,
            y_proba_challenger,
            rul_champion,
            rul_challenger,
            failure_labels,
            actual_rul_values,
        )
        latency = self._latency_gate(np.asarray(self.latencies_ms))
        promote = bool(quality["quality_pass"] and latency["latency_pass"])
        return {
            "quality_pass": bool(quality["quality_pass"]),
            "latency_pass": latency["latency_pass"],
            "promote": promote,
            "labeled_predictions": int(len(labeled_idx)),
            "reason": "dual gates passed" if promote else "quality or latency gate failed",
            **quality,
            **latency,
        }

    def _quality_gate(
        self,
        champion_proba: np.ndarray,
        challenger_proba: np.ndarray,
        champion_rul: np.ndarray,
        challenger_rul: np.ndarray,
        failure_labels: np.ndarray,
        actual_rul_values: np.ndarray,
    ) -> dict[str, object]:
        if len(np.unique(failure_labels)) < 2:
            return {
                "quality_pass": False,
                "reason": "labeled windows lack both classes",
                "champion_brier": float("nan"),
                "challenger_brier": float("nan"),
            }
        champion_metrics = evaluate_failure(failure_labels, champion_proba)
        challenger_metrics = evaluate_failure(failure_labels, challenger_proba)
        rul_metrics_champion = evaluate_rul(actual_rul_values, champion_rul)
        rul_metrics_challenger = evaluate_rul(actual_rul_values, challenger_rul)
        pass_brier = (
            challenger_metrics["brier"]
            <= champion_metrics["brier"] * (1.0 + self.tolerance)
        )
        pass_rmse = (
            rul_metrics_challenger["rmse"]
            <= rul_metrics_champion["rmse"] * (1.0 + self.tolerance)
        )
        return {
            "quality_pass": bool(pass_brier and pass_rmse),
            "champion_brier": float(champion_metrics["brier"]),
            "challenger_brier": float(challenger_metrics["brier"]),
            "champion_rmse": float(rul_metrics_champion["rmse"]),
            "challenger_rmse": float(rul_metrics_challenger["rmse"]),
        }

    def _latency_gate(self, latencies: np.ndarray) -> dict[str, object]:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        return {
            "latency_pass": bool(
                p50 <= self.latency_budget_p50_ms and p95 <= self.latency_budget_p95_ms
            ),
            "p50_ms": p50,
            "p95_ms": p95,
        }

    @classmethod
    def from_settings(cls) -> ShadowEvaluator:
        settings = get_settings()
        return cls(
            window_size=settings.shadow_eval_window,
            tolerance=settings.quality_improvement_margin,
            latency_budget_p50_ms=settings.latency_budget_p50_ms,
            latency_budget_p95_ms=settings.latency_budget_p95_ms,
        )
