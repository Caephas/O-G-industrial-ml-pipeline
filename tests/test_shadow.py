"""Shadow-mode dual gate tests (FR-11)."""

from __future__ import annotations

import numpy as np

from pipeline.lifecycle.shadow import ShadowEvaluator


def _fill(evaluator: ShadowEvaluator, *, n: int = 50) -> None:
    rng = np.random.default_rng(0)
    actual_rul = np.sort(rng.uniform(1.0, 120.0, n))
    labels = actual_rul <= 30.0
    champion_proba = np.where(labels, 0.7, 0.2) + rng.normal(0, 0.05, n)
    champion_proba = np.clip(champion_proba, 0.01, 0.99)
    for index in range(n):
        champion_rul_value = float(actual_rul[index] + rng.normal(0, 5))
        evaluator.add(
            champion_proba=float(champion_proba[index]),
            champion_rul=champion_rul_value,
            challenger_proba=float(champion_proba[index]),
            challenger_rul=champion_rul_value,
            label_available=True,
            failure_label=float(labels[index]),
            actual_rul=float(actual_rul[index]),
            latency_ms=2.0,
        )


def test_dual_gates_promote_comparable_candidate() -> None:
    evaluator = ShadowEvaluator(window_size=50, min_labeled=30)
    _fill(evaluator)
    assert evaluator.ready()
    gate = evaluator.evaluate()
    assert gate["quality_pass"] is True
    assert gate["latency_pass"] is True
    assert gate["promote"] is True


def test_latency_gate_blocks_slow_candidate() -> None:
    evaluator = ShadowEvaluator(
        window_size=50,
        min_labeled=30,
        latency_budget_p50_ms=1.0,
        latency_budget_p95_ms=2.0,
    )
    _fill(evaluator)
    gate = evaluator.evaluate()
    assert gate["latency_pass"] is False
    assert gate["promote"] is False


def test_insufficient_labels_defers_gate() -> None:
    evaluator = ShadowEvaluator(window_size=50, min_labeled=55)
    _fill(evaluator, n=50)
    assert evaluator.ready() is False
    assert evaluator.evaluate()["promote"] is False


def test_worse_candidate_fails_quality_gate() -> None:
    evaluator = ShadowEvaluator(window_size=50, min_labeled=30)
    for index in range(50):
        label = 1.0 if index < 15 else 0.0
        evaluator.add(
            champion_proba=0.8 if label else 0.2,
            champion_rul=float(50.0 - index),
            challenger_proba=0.5,
            challenger_rul=100.0,
            label_available=True,
            failure_label=label,
            actual_rul=float(50.0 - index),
            latency_ms=1.0,
        )
    gate = evaluator.evaluate()
    assert gate["quality_pass"] is False
    assert gate["promote"] is False
