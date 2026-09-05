"""Tests for shared signal helpers (need numpy/pandas — run on Colab)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.core.signal import (
    find_column,
    gravity_align_linear,
    is_window_stationary,
    resample_uniform,
)


@pytest.fixture
def sensor_df() -> pd.DataFrame:
    """Minimal IO-VNBD-like frame with cp1252-style headers."""
    return pd.DataFrame(
        {
            "ACCELEROMETER X (m/s²)": [0.1, 0.2],
            "TIME SINCE START (ms)": [0.0, 100.0],
        }
    )


def test_find_column_case_insensitive(sensor_df: pd.DataFrame) -> None:
    """Regex matches across case and returns None when absent."""
    assert find_column(sensor_df, r"accelerometer.*x") == "ACCELEROMETER X (m/s²)"
    assert find_column(sensor_df, r"gyroscope.*yaw") is None


def test_resample_uniform_1d_linear() -> None:
    """1D linear signal interpolates exactly at the midpoint."""
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    _, v = resample_uniform(t, np.array([0.0, 10.0]), 2)
    assert v.tolist() == pytest.approx([0.0, 5.0, 10.0])


def test_resample_uniform_2d_edges_are_nan() -> None:
    """Out-of-overlap edges are NaN so callers can finite-mask them."""
    t = np.array([0, 1_000_000_000], dtype=np.int64)
    vals = np.array([[0.0, 1.0], [10.0, 11.0]])
    t_new, v_new = resample_uniform(t, vals, 1)
    assert t_new.shape == (2,)
    assert v_new.shape == (2, 2)
    assert np.isfinite(v_new).all()


def test_resample_uniform_rejects_single_sample() -> None:
    """Fewer than 2 samples raises instead of extrapolating."""
    with pytest.raises(ValueError, match=">=2 samples"):
        resample_uniform(np.array([0], dtype=np.int64), np.array([1.0]), 100)


def test_gravity_align_linear_subtracts() -> None:
    """Linear acc equals raw minus gravity estimate."""
    acc = np.array([[0.0, 0.0, 10.0]])
    grav = np.array([[0.0, 0.0, 9.81]])
    assert gravity_align_linear(acc, grav).tolist() == pytest.approx([[0.0, 0.0, 0.19]])


@pytest.mark.parametrize("hz", [50, 100])
def test_stationary_gate(hz: int) -> None:
    """Still window passes; shaking window fails at any supported rate."""
    still = np.zeros((200, 6))
    still[:, 2] = 0.05  # tiny vertical jitter
    assert is_window_stationary(still, hz=hz) is True
    rng = np.random.default_rng(0)
    moving = rng.normal(0, 2.0, size=(200, 6))
    assert is_window_stationary(moving, hz=hz) is False
