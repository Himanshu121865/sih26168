"""Tests for trajectory metrics (need numpy — run on Colab)."""

from __future__ import annotations

import numpy as np
import pytest

from python.eval.metrics import ate, coverage, drift_pct, rte, total_distance


@pytest.fixture
def straight_line() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GT straight line + noisy estimate + timestamps."""
    rng = np.random.default_rng(0)
    gt = np.stack([np.linspace(0, 100, 50), np.zeros(50)], axis=1)
    est = gt + rng.normal(0, 0.5, size=gt.shape)
    return est, gt, np.arange(50) * 0.1


def test_ate_identical_is_zero(straight_line: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Identical trajectories score ~0 with and without alignment."""
    _, gt, _ = straight_line
    rmse, err = ate(gt.copy(), gt, align=False)
    assert rmse == pytest.approx(0.0)
    assert err.shape == (50,)


def test_ate_aligned_removes_offset(
    straight_line: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """SE(2) alignment absorbs a constant offset."""
    _, gt, _ = straight_line
    rmse, _ = ate(gt + np.array([5.0, 0.0]), gt, align=True)
    assert rmse < 1.0


def test_ate_rejects_bad_shapes() -> None:
    """Mismatched shapes raise ValueError, not assert."""
    with pytest.raises(ValueError, match=r"\(N,2\)"):
        ate(np.zeros((10, 3)), np.zeros((10, 2)))


def test_rte_short_track_returns_zero() -> None:
    """No valid 60 s window yields 0.0 instead of NaN."""
    gt = np.zeros((5, 2))
    assert rte(gt, gt, np.arange(5) * 0.1, window_s=60.0) == 0.0


def test_rte_detects_drift(straight_line: tuple[np.ndarray, np.ndarray, np.ndarray]) -> None:
    """Biased estimate has larger RTE than unbiased noise."""
    est, gt, t = straight_line
    biased = est + np.linspace(0, 10, 50)[:, None] * np.array([1.0, 0.0])
    assert rte(biased, gt, t, window_s=2.0) > rte(est, gt, t, window_s=2.0)


def test_drift_pct_and_degenerate() -> None:
    """5% case plus zero-distance guard."""
    assert drift_pct(5.0, 100.0) == pytest.approx(5.0)
    assert drift_pct(5.0, 0.0) == 0.0


def test_total_distance_and_coverage() -> None:
    """100 m line measures 100 m; 1σ coverage of Gaussians ≈ 0.68."""
    gt = np.stack([np.linspace(0, 100, 101), np.zeros(101)], axis=1)
    assert total_distance(gt) == pytest.approx(100.0)
    rng = np.random.default_rng(0)
    assert coverage(rng.normal(size=(2000, 1)), np.ones((2000, 1))) == pytest.approx(0.68, abs=0.03)
