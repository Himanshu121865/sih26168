"""Tests for the ZUPT stationary detector (need numpy — run on Colab)."""

from __future__ import annotations

import numpy as np
import pytest

from python.utils.zupt import (
    StationaryConfig,
    StationaryDetector,
    detect_stationary_windows,
)


@pytest.fixture
def still_stream() -> tuple[np.ndarray, np.ndarray]:
    """Low-variance accel/gyro resembling a parked vehicle."""
    rng = np.random.default_rng(0)
    acc = np.tile([0.0, 0.0, 0.2], (100, 1)) + rng.normal(0, 0.02, (100, 3))
    gyro = rng.normal(0, 0.005, (100, 3))
    return acc, gyro


@pytest.fixture
def moving_stream() -> tuple[np.ndarray, np.ndarray]:
    """High-variance stream resembling driving."""
    rng = np.random.default_rng(1)
    return rng.normal(0, 2.0, (100, 3)), rng.normal(0, 0.5, (100, 3))


def test_becomes_stationary_after_min_duration(still_stream: tuple[np.ndarray, np.ndarray]) -> None:
    """Still stream reports stationary once persistence elapses."""
    acc, gyro = still_stream
    det = StationaryDetector(rate_hz=100.0)
    states = [det.update(a, w, i * 10_000_000) for i, (a, w) in enumerate(zip(acc, gyro))]
    assert states[-1] is True
    assert det.is_stationary is True
    assert det.stationary_duration_s >= 0.3


def test_moving_never_stationary(moving_stream: tuple[np.ndarray, np.ndarray]) -> None:
    """Driving stream never latches, and measurements stay None."""
    acc, gyro = moving_stream
    det = StationaryDetector(rate_hz=100.0)
    assert not any(det.update(a, w, i * 10_000_000) for i, (a, w) in enumerate(zip(acc, gyro)))
    assert det.get_zupt_measurement() is None
    assert det.get_zaru_measurement(np.zeros(3)) is None


def test_speed_gate_blocks_stationary(still_stream: tuple[np.ndarray, np.ndarray]) -> None:
    """Low variance at speed still counts as moving."""
    acc, gyro = still_stream
    det = StationaryDetector(rate_hz=100.0)
    assert not any(
        det.update(a, w, i * 10_000_000, speed_mps=10.0) for i, (a, w) in enumerate(zip(acc, gyro))
    )


def test_zupt_measurement_shape(still_stream: tuple[np.ndarray, np.ndarray]) -> None:
    """ZUPT returns (2,) zeros with tight diag covariance."""
    acc, gyro = still_stream
    det = StationaryDetector(rate_hz=100.0)
    for i, (a, w) in enumerate(zip(acc, gyro)):
        det.update(a, w, i * 10_000_000)
    meas = det.get_zupt_measurement()
    assert meas is not None
    y, R = meas
    assert y.shape == (2,) and R.shape == (2, 2)


def test_custom_config_respected() -> None:
    """Tighter thresholds need more stillness than defaults."""
    strict = StationaryDetector(config=StationaryConfig(accel_var_threshold=1e-9), rate_hz=100.0)
    rng = np.random.default_rng(0)
    a = np.tile([0.0, 0.0, 0.2], (100, 1)) + rng.normal(0, 0.02, (100, 3))
    w = rng.normal(0, 0.005, (100, 3))
    assert not any(strict.update(x, y, i * 10_000_000) for i, (x, y) in enumerate(zip(a, w)))


def test_detect_stationary_windows_shape(
    still_stream: tuple[np.ndarray, np.ndarray],
) -> None:
    """Offline helper returns a bool array with late-True flags."""
    acc, gyro = still_stream
    out = detect_stationary_windows(acc, gyro, hz=100)
    assert out.dtype == bool and out.shape == (100,)
    assert out[-1] == True  # noqa: E712
