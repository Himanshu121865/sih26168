"""Shared resample + gravity + column helpers.

Single source of truth for train and live preprocessing
(harsh/pipeline.py:30-68, sivaraman/prepare_training_data.py:205).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def find_column(df: pd.DataFrame, pattern: str) -> str | None:
    """Find the first column whose name matches a regex pattern.

    Args:
        df: DataFrame with IO-VNBD sensor columns.
        pattern: Case-insensitive regex, e.g. ``r"accelerometer.*x"``.

    Returns:
        Matching column name, or None if no column matches.
    """
    regex = re.compile(pattern, re.IGNORECASE)
    for c in df.columns:
        if regex.search(c):
            return c
    return None


def resample_uniform(
    t_ns: np.ndarray, values: np.ndarray, rate_hz: float
) -> tuple[np.ndarray, np.ndarray]:
    """Resample a signal onto a uniform time grid with linear interpolation.

    Args:
        t_ns: Source timestamps in nanoseconds, shape (N,), strictly increasing.
        values: Source values, shape (N,) or (N, K).
        rate_hz: Target uniform sample rate in Hz.

    Returns:
        Tuple of (t_new_ns, values_new) on the uniform grid. Samples outside
        the source overlap are NaN (caller drops them with a finite mask).

    Raises:
        ValueError: If fewer than 2 source samples are given.
    """
    if len(t_ns) < 2:
        raise ValueError(f"need >=2 samples, got {len(t_ns)}")
    period_ns = round(1e9 / rate_hz)
    t_start, t_end = t_ns[0], t_ns[-1]
    n = int((t_end - t_start) // period_ns) + 1
    t_new = t_start + np.arange(n) * period_ns
    if values.ndim == 1:
        v_new = np.interp(t_new, t_ns, values, left=np.nan, right=np.nan)
        return t_new, v_new
    v_new = np.empty((n, values.shape[1]))
    for k in range(values.shape[1]):
        v_new[:, k] = np.interp(t_new, t_ns, values[:, k], left=np.nan, right=np.nan)
    return t_new, v_new


def gravity_align_linear(acc_raw: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """Remove gravity to obtain linear acceleration.

    Args:
        acc_raw: Raw accelerometer readings, shape (N, 3).
        gravity: Low-pass gravity estimate, shape (N, 3)
            (IO-VNBD provides GRAVITY X/Y/Z columns).

    Returns:
        Linear acceleration ``acc_raw - gravity``, shape (N, 3).
    """
    return acc_raw - gravity


def is_window_stationary(
    window_6: np.ndarray,
    hz: float = 100,
    acc_var_thr: float = 0.05,
    gyro_var_thr: float = 0.01,
) -> bool:
    """Check whether an IMU window looks stationary (ZUPT gate).

    Uses variance of the accel/gyro norms over the last 0.5 s
    (harsh thresholds 0.05 / 0.01).

    Args:
        window_6: IMU window, shape (T, 6), linear-acc (3) + gyro (3).
        hz: Sample rate of the window in Hz.
        acc_var_thr: Accelerometer-norm variance threshold (m/s²)².
        gyro_var_thr: Gyroscope-norm variance threshold (rad/s)².

    Returns:
        True if both variances are below threshold.
    """
    acc = window_6[:, :3]
    gyro = window_6[:, 3:6]
    tail = int(0.5 * hz)
    a_var = (
        np.var(np.linalg.norm(acc[-tail:], axis=1))
        if len(acc) >= tail
        else np.var(np.linalg.norm(acc, axis=1))
    )
    w_var = (
        np.var(np.linalg.norm(gyro[-tail:], axis=1))
        if len(gyro) >= tail
        else np.var(np.linalg.norm(gyro, axis=1))
    )
    return bool(a_var < acc_var_thr and w_var < gyro_var_thr)
