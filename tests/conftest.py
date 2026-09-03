"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

from tests.fixtures.cmapss_synthetic import SyntheticCmapss, generate_synthetic_cmapss


@pytest.fixture(scope="session")
def synthetic_cmapss() -> SyntheticCmapss:
    """Medium FD002-like dataset shared read-only across the session."""
    return generate_synthetic_cmapss(
        subset="FD002",
        n_train_engines=6,
        n_test_engines=3,
        seed=7,
    )


@pytest.fixture()
def small_synthetic_cmapss() -> SyntheticCmapss:
    """Smaller FD002-like dataset for fast per-test use."""
    return generate_synthetic_cmapss(
        subset="FD002",
        n_train_engines=4,
        n_test_engines=2,
        seed=0,
    )
