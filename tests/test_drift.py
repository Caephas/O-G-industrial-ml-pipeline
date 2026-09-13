"""PSI drift detector tests (FR-12)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.lifecycle.drift import DriftDetector, psi_between


def _frame(values: np.ndarray) -> dict:
    return {"a": values, "b": values * 2.0 + 1.0}


def test_psi_is_zero_for_identical_distributions() -> None:
    assert psi_between(np.ones(500), np.ones(500)) == 0.0
    assert psi_between(np.random.default_rng(1).normal(size=500), np.zeros(0)) == 0.0


def test_psi_increases_with_shift() -> None:
    reference = np.random.default_rng(2).normal(loc=0.0, size=1000)
    shifted = reference + 4.0
    assert psi_between(reference, shifted) > psi_between(reference, reference)


def test_drift_triggers_and_cooldown_holds() -> None:
    rng = np.random.default_rng(3)
    reference = pd.DataFrame({"a": rng.normal(0, 1, 1000), "b": rng.normal(0, 1, 1000)})
    detector = DriftDetector(
        ["a", "b"],
        threshold=0.20,
        buffer_size=250,
        cooldown_updates=3,
    )
    detector.fit_reference(reference)
    shifted = pd.DataFrame({"a": rng.normal(3, 1, 250), "b": rng.normal(3, 1, 250)})
    first = detector.update(shifted)
    assert first.triggered is True
    assert first.cooldown_remaining == 3
    second = detector.update(shifted)
    assert second.triggered is False
    assert second.cooldown_remaining == 2

    # Drain cooldown with more shifted windows; a trigger may fire again once
    # the cooldown expires and the buffer is fully shifted.
    third = detector.update(shifted)
    fourth = detector.update(shifted)
    assert third.triggered is False
    assert fourth.triggered is True
    assert fourth.cooldown_remaining == 3
