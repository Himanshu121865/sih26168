"""
core/signal.py — Shared resample + gravity + column helpers (extracted from preprocess.py)
Single source for train and live (harsh/pipeline.py:30-68, sivaraman/prepare_training_data.py:205).
"""
import re
import numpy as np
import pandas as pd

def find_column(df: pd.DataFrame, pattern: str):
    regex = re.compile(pattern, re.IGNORECASE)
    for c in df.columns:
        if regex.search(c):
            return c
    return None

def resample_uniform(t_ns: np.ndarray, values: np.ndarray, rate_hz: float):
    """Shared uniform resample — period_ns=round(1e9/rate), np.interp left/right=np.nan."""
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

def gravity_align_linear(acc_raw: np.ndarray, gravity: np.ndarray):
    """Linear acc = ACC - GRAVITY (IO-VNBD provides gravity)."""
    return acc_raw - gravity

def is_window_stationary(window_6: np.ndarray, hz=100, acc_var_thr=0.05, gyro_var_thr=0.01):
    """ZUPT per window — last 0.5s var check (harsh 0.05/0.01). window_6 (200,6) linear+gyro."""
    acc = window_6[:, :3]
    gyro = window_6[:, 3:6]
    tail = int(0.5 * hz)
    a_var = np.var(np.linalg.norm(acc[-tail:], axis=1)) if len(acc) >= tail else np.var(np.linalg.norm(acc, axis=1))
    w_var = np.var(np.linalg.norm(gyro[-tail:], axis=1)) if len(gyro) >= tail else np.var(np.linalg.norm(gyro, axis=1))
    return a_var < acc_var_thr and w_var < gyro_var_thr
