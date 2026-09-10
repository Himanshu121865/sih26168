"""IO-VNBD file pipeline: CSV → resample → gravity-remove → windows → labels.

Single source of truth for the window path (spec v2, docs/WINDOW_SPEC.md).
Previously THREE hand-rolled copies existed (preprocess.py,
datasets/iovnbd_dataset.py, eval_per_file.py) — one of which (the per-file
audit) silently still used spec-v1 dataset gravity columns. Every consumer
now routes through this module, so train / streaming / audit paths are
guaranteed identical channel-for-channel.

The canonical path (each step testable, each a pure function where possible):

1. ``load_phone_csv``       — cp1252 decode + stripped headers
2. ``resolve_columns``      — regex column map with exact-name fallback
3. ``timestamp_audit``      — P4 dt discipline (median/gaps/rate flag)
4. ``monotonic_time_ns``    — sort-if-needed + ms→ns conversion
5. ``estimate_gravity_lowpass`` — spec v2 live gravity (core.signal)
6. ``gravity_align_linear`` — linear acc = raw − gravity
7. ``resample_uniform``     — 10 Hz → 100 Hz linear interp (edges NaN)
8. ``windowize``            — sliding (200, 6) windows, oldest-first
9. ``velocity_labels``      — GPS speed at window tail (km/h → m/s)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from python.config import QUALITY_GATES, QualityGates
from python.core.signal import (
    estimate_gravity_lowpass,
    find_column,
    gravity_align_linear,
    is_window_stationary,
    resample_uniform,
)

# --- column patterns (IO-VNBD S-file headers, cp1252 ² characters) ----------

TIME_PAT = r"time since start"
GPS_SPEED_PAT = r"gps speed"
ACC_PAT = r"accelerometer"
GRAV_PAT = r"gravity"
GYRO_PATTERNS = (r"gyroscope.*yaw", r"gyroscope.*pitch", r"gyroscope.*roll")

ACC_FALLBACK = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
GRAV_FALLBACK = ["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]
GYRO_FALLBACK = ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]
TIME_FALLBACK = "TIME SINCE START (ms)"
GPS_SPEED_FALLBACK = "GPS SPEED (Kmh)"

KMH_TO_MS = 1.0 / 3.6

# IMU channel order (model + scaler + Android all pin this):
# [linear acc x/y/z, gyro yaw/pitch/roll]
N_CHANNELS = 6


@dataclass(frozen=True, slots=True)
class ColumnMap:
    """Resolved IO-VNBD column names for one file.

    Attributes:
        acc: Accelerometer x/y/z column names.
        gravity: Gravity x/y/z column names (cross-check only, spec v2).
        gyro: Gyro yaw/pitch/roll column names.
        time: Timestamp (ms) column name.
        gps_speed: GPS speed (km/h) column name.
    """

    acc: tuple[str, str, str]
    gravity: tuple[str, str, str]
    gyro: tuple[str, str, str]
    time: str
    gps_speed: str

    @property
    def gravity_present(self) -> bool:
        """Whether all gravity columns actually exist in the frame."""
        return all(c for c in self.gravity)


def load_phone_csv(path: str | Path) -> pd.DataFrame:
    """Read an IO-VNBD S-file with cp1252 decoding and stripped headers.

    Args:
        path: CSV file path.

    Returns:
        DataFrame with whitespace-stripped column names.
    """
    df = pd.read_csv(path, encoding="cp1252")
    df.columns = [c.strip() for c in df.columns]
    return df


def resolve_columns(df: pd.DataFrame) -> ColumnMap:
    """Map IO-VNBD column names via regex with exact-name fallback.

    Args:
        df: Frame with IO-VNBD headers (already stripped).

    Returns:
        ColumnMap; missing regex matches fall back to the canonical names,
        and gravity columns fall back to empty strings when absent (they
        are a spec-v2 cross-check only, never truth).
    """
    def xyz(pattern: str) -> list[str | None]:
        return [find_column(df, rf"{pattern}.*{ax}") for ax in "xyz"]

    acc = xyz(ACC_PAT)
    grav = xyz(GRAV_PAT)
    gyro = [find_column(df, p) for p in GYRO_PATTERNS]

    acc_resolved = [c or f for c, f in zip(acc, ACC_FALLBACK, strict=False)]
    grav_resolved = [(c if c is not None and c in df.columns else "") for c in grav]
    gyro_resolved = [c or f for c, f in zip(gyro, GYRO_FALLBACK, strict=False)]
    time_col = find_column(df, TIME_PAT) or TIME_FALLBACK
    gps_col = find_column(df, GPS_SPEED_PAT) or GPS_SPEED_FALLBACK

    return ColumnMap(
        acc=tuple(acc_resolved),  # type: ignore[arg-type]
        gravity=tuple(grav_resolved),  # type: ignore[arg-type]
        gyro=tuple(gyro_resolved),  # type: ignore[arg-type]
        time=time_col,
        gps_speed=gps_col,
    )


@dataclass(frozen=True, slots=True)
class TimestampAudit:
    """Per-file dt statistics (P4 timestamp discipline).

    Attributes:
        median_dt_ms: Median dt in milliseconds.
        gap_fraction: Fraction of dt samples above the gap threshold.
        rate_flag: ``"odd-rate"`` when median dt deviates from nominal,
            else ``""``.
        rejected: True when ``gap_fraction`` exceeds the reject threshold.
    """

    median_dt_ms: float
    gap_fraction: float
    rate_flag: str
    rejected: bool


def timestamp_audit(t_ms: np.ndarray, gates: QualityGates = QUALITY_GATES) -> TimestampAudit:
    """Audit dt statistics before resampling.

    Args:
        t_ms: Timestamps in milliseconds (monotonic or not — diffs of the
            raw sequence are used; callers sort first for the median).
        gates: Quality thresholds.

    Returns:
        TimestampAudit with median/gap stats and the reject verdict.
    """
    dts = np.diff(np.asarray(t_ms, dtype=np.float64))
    finite = dts[np.isfinite(dts)]
    if len(finite) == 0:
        return TimestampAudit(float("nan"), 0.0, "", False)
    median_dt = float(np.median(finite))
    gap_thr = max(gates.gap_factor * median_dt, gates.gap_min_ms)
    gap_frac = float(np.mean(finite > gap_thr))
    rate_flag = (
        "odd-rate" if abs(median_dt - gates.expected_dt_ms) > gates.rate_tolerance_ms else ""
    )
    return TimestampAudit(
        median_dt_ms=median_dt,
        gap_fraction=gap_frac,
        rate_flag=rate_flag,
        rejected=gap_frac > gates.gap_fraction_max,
    )


def monotonic_time_ns(df: pd.DataFrame, time_col: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (sorted df, t_ns) with time monotonically increasing.

    Args:
        df: Source frame (devices occasionally emit reordered rows).
        time_col: Timestamp (ms) column name.

    Returns:
        Tuple of (df sorted by time with reset index, t_ns int64 array).
    """
    t_ms = np.asarray(df[time_col].values, dtype=np.float64)
    if np.any(np.diff(t_ms) <= 0):
        df = df.iloc[np.argsort(t_ms)].reset_index(drop=True)
        t_ms = np.asarray(df[time_col].values, dtype=np.float64)
    return df, np.asarray(t_ms * 1e6, dtype=np.int64)


@dataclass(frozen=True, slots=True)
class FileWindows:
    """Windowed output for one source file.

    Attributes:
        windows: Normalized-ready windows, shape (N, window, 6), float32.
        v_ms: Speed label per window in m/s (ZUPT-forced 0 when stationary).
        stationary: ZUPT flag per window.
        audit: Timestamp audit of the source file.
        gravity_max_diff: Max |gEst − dataset gravity| (NaN when the file
            has no gravity columns — cross-check only, spec v2).
        n_resampled: Number of samples after resample+finite-mask.
    """

    windows: np.ndarray
    v_ms: np.ndarray
    stationary: np.ndarray
    audit: TimestampAudit
    gravity_max_diff: float
    n_resampled: int


def windowize(
    arr: np.ndarray, window: int = 200, stride: int = 10
) -> np.ndarray:
    """Sliding windows along axis 0, oldest-first.

    Args:
        arr: Signal of shape (T, C).
        window: Window length in samples.
        stride: Hop between window starts.

    Returns:
        Windows of shape (N, window, C) where N = (T − window)//stride + 1,
        or an empty (0, window, C) array when the signal is too short.
    """
    T = arr.shape[0]
    if T < window:
        return np.empty((0, window, arr.shape[-1]), dtype=arr.dtype)
    # sliding_window_view(arr, window, axis=0) -> (T-w+1, C, window);
    # transpose to (N, window, C) — parity with the legacy stack-of-slices.
    views = np.lib.stride_tricks.sliding_window_view(arr, window, axis=0)
    return views[::stride].transpose(0, 2, 1).copy()


def velocity_labels(
    v_ms_resampled: np.ndarray, n_windows: int, window: int, stride: int
) -> np.ndarray:
    """Speed label per window = resampled speed at the window's last sample."""
    tail_idx = np.arange(n_windows) * stride + (window - 1)
    return v_ms_resampled[tail_idx]


def resample_stream(
    df: pd.DataFrame, hz: int = 100
) -> tuple[np.ndarray, np.ndarray, TimestampAudit, float]:
    """Resample one S-file to a uniform-hz IMU stream (spec v2 path, steps 3-7).

    Args:
        df: Frame from :func:`load_phone_csv`.
        hz: Uniform resample rate.

    Returns:
        Tuple of (imu_6 (T,6) [linear acc, gyro], v_ms (T,) GPS speed m/s,
        audit, gravity_max_diff). Samples outside the source overlap are
        dropped via the finite mask.

    Raises:
        ValueError: When required columns are missing from the frame.
    """
    cols = resolve_columns(df)
    missing = [c for c in (*cols.acc, *cols.gyro, cols.time) if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")

    df, t_ns = monotonic_time_ns(df, cols.time)
    t_ms = np.asarray(df[cols.time].values, dtype=np.float64)
    audit = timestamp_audit(t_ms)

    acc_raw = np.asarray(df[list(cols.acc)].values, dtype=np.float64)
    gyro = np.asarray(df[list(cols.gyro)].values, dtype=np.float64)

    # Spec v2 (P1): gravity via the live low-pass — identical to the Kotlin
    # LeanDetector filter. Dataset GRAVITY columns are a cross-check only.
    grav_est = estimate_gravity_lowpass(acc_raw)
    if cols.gravity_present:
        grav_col_vals = np.asarray(df[list(cols.gravity)].values, dtype=np.float64)
        grav_diff = float(np.abs(grav_est - grav_col_vals).max())
    else:
        grav_diff = float("nan")

    linear_acc = gravity_align_linear(acc_raw, grav_est)
    imu_6 = np.concatenate([linear_acc, gyro], axis=1)

    _, imu_new = resample_uniform(t_ns, imu_6, hz)
    if cols.gps_speed in df.columns:
        gps_raw = np.asarray(df[cols.gps_speed].values, dtype=np.float64)
    else:
        gps_raw = np.zeros(len(df))
    _, gps_new = resample_uniform(t_ns, gps_raw, hz)

    finite_mask = np.isfinite(imu_new).all(axis=1) & np.isfinite(gps_new)
    return imu_new[finite_mask], gps_new[finite_mask] * KMH_TO_MS, audit, grav_diff


def process_file(
    df: pd.DataFrame,
    hz: int = 100,
    window: int = 200,
    stride: int = 10,
    gates: QualityGates = QUALITY_GATES,
    zupt_speed_gate: float = 0.5,
) -> FileWindows | None:
    """Run the full window path on one loaded S-file (steps 1-9).

    Args:
        df: Frame from :func:`load_phone_csv`.
        hz: Uniform resample rate.
        window: Window length in samples.
        stride: Window hop in samples.
        gates: Timestamp quality gates (reject when gap fraction exceeds).
        zupt_speed_gate: Windows with speed below this (m/s) and low IMU
            variance are labeled stationary (v forced to 0).

    Returns:
        FileWindows with normalized-READY (not yet normalized) windows, or
        None when the file is rejected (excessive gaps) or too short.
    """
    imu_new, v_ms, audit, grav_diff = resample_stream(df, hz)
    if audit.rejected or len(imu_new) < window:
        return None

    windows = windowize(imu_new.astype(np.float32), window=window, stride=stride)
    if len(windows) == 0:
        return None
    v_labels = velocity_labels(v_ms, len(windows), window, stride)

    # ZUPT stationary flags: IMU variance + speed gate (vehicle moving but
    # smooth, e.g. highway, must not count as stationary).
    stationary = np.array([is_window_stationary(w, hz=hz) for w in windows])
    stationary = stationary & (v_labels < zupt_speed_gate)
    v_labels = np.where(stationary, 0.0, v_labels)

    return FileWindows(
        windows=windows,
        v_ms=v_labels,
        stationary=stationary,
        audit=audit,
        gravity_max_diff=grav_diff,
        n_resampled=int(len(imu_new)),
    )
