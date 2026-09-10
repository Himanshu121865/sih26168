"""Outage replay: naive vs AVNet-only vs InEKF along-track drift.

Replays a 60 s val segment (600 windows @10 Hz). Windows are stored
normalized; the LAST sample of window i is raw 100 Hz sample ``i*stride+199``
→ consecutive window tails form the 10 Hz measurement stream, the same rate
as AVNet output.

This module is the pure computation; CLI wiring lives in
``python/inekf_harness.py`` and metrics land in ``reports/inekf_vs_avnet.csv``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from python.config import PROCESSED_DIR, SCALER_PATH
from python.fusion.ine_kf import InEKF, gravity_align_R  # noqa: F401 — gravity_align_R re-exported for parity
from python.models.avnet import AVNetLite
from python.models.lean_estimator import LeanEstimator
from python.utils.zupt import StationaryDetector

DT_10HZ = 0.1


def load_scaler(path: str | Path = SCALER_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Load (mean, std) from a scaler JSON as float64 arrays."""
    with open(path) as f:
        sc = json.load(f)
    return np.array(sc["mean"], dtype=np.float64), np.array(sc["std"], dtype=np.float64)


def predict_batch(
    model: AVNetLite,
    seg: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> dict[str, np.ndarray]:
    """Batch model inference over a segment of normalized windows.

    Args:
        model: Loaded AVNetLite in eval mode.
        seg: Windows (N, 200, 6), normalized, float64 OK.
        mean / std: Scaler stats (denormalize for lean physics).

    Returns:
        Dict with ``v_pred``, ``sigma_v``, ``phi``, ``p_bike`` — each (N,).
    """
    lean = LeanEstimator()
    lean.eval()
    with torch.no_grad():
        xb = torch.from_numpy(np.array(seg, dtype=np.float32))  # copy: mmap is non-writable
        v_pred, ls_v, _, _, _ = model(xb)
        phi_arr, p_bike_arr = lean(xb, scaler_mean=mean, scaler_std=std)
    return {
        "v_pred": v_pred.squeeze(-1).numpy().astype(np.float64),
        "sigma_v": torch.exp(ls_v.squeeze(-1)).numpy().astype(np.float64),
        "phi": phi_arr.numpy().astype(np.float64),
        "p_bike": p_bike_arr.numpy().astype(np.float64),
    }


def _zupt_decision(
    i: int,
    zupt_det: StationaryDetector | None,
    acc_stream: np.ndarray,
    gyro_stream: np.ndarray,
    v_pred: np.ndarray,
    start: int,
    zupt_speed_gate: float,
) -> bool:
    """Variance ZUPT (primary) + first-5-samples speed heuristic (fallback).

    The heuristic only bridges the gap until the 0.5 s variance window
    fills — afterwards the variance detector alone decides (parity with
    Android DrPipeline; the model-speed heuristic alone regressed stop-go
    to 25%).
    """
    still_var = False
    if zupt_det is not None:
        t_ns = int((start + i) * 100_000_000)  # 10 Hz ticks; monotonic is enough
        still_var = bool(
            zupt_det.update(
                np.asarray(acc_stream[i]),
                np.asarray(gyro_stream[i]),
                t_ns,
                speed_mps=float(max(float(v_pred[i]), 0.0)),
            )
        )
    if zupt_det is not None and i >= 5:
        return still_var
    v_fwd_raw = max(float(v_pred[i]), 0.0)
    return bool(still_var or v_fwd_raw < zupt_speed_gate)


def run_replay(
    model_path: str | Path,
    n_windows: int = 600,
    start: int | None = None,
    lean_mode: str = "auto",
    verbose: bool = True,
    q_acc: float = 30.0,
    zupt_speed_gate: float = 0.3,
    use_variance_zupt: bool = True,
    val_windows: str | Path | None = None,
    val_v: str | Path | None = None,
) -> dict[str, float | int]:
    """Replay a 60 s outage on a val segment and score three estimators.

    Args:
        model_path: AVNetLite checkpoint (state_dict).
        n_windows: Segment length in windows (600 = 60 s @10 Hz).
        start: First window index (default: middle of the val set).
        lean_mode: ``auto`` | ``car`` | ``bike`` (forces p_bike to 1/0).
        verbose: Log per-segment metrics via loguru.
        q_acc: Accelerometer process noise (30.0 for the 10 Hz proxy).
        zupt_speed_gate: Speed-gate fallback for the first 5 samples.
        use_variance_zupt: Enable the variance stationary detector.
        val_windows / val_v: Overrides for the processed-npy paths.

    Returns:
        Metrics dict (drift % per estimator, distances, ZUPT count, ...).
    """
    from loguru import logger  # late: keeps module importable without loguru for tests

    mean, std = load_scaler()
    windows_path = Path(val_windows) if val_windows else PROCESSED_DIR / "val_windows.npy"
    v_path = Path(val_v) if val_v else PROCESSED_DIR / "val_v.npy"
    X = np.load(windows_path, mmap_mode="r")  # (N,200,6) normalized
    v_gt_all = np.load(v_path)                # (N,) m/s

    n_windows = min(n_windows, len(X) - 4)
    if start is None:
        start = len(X) // 2
    seg = np.array(X[start : start + n_windows], dtype=np.float64)
    v_gt = np.asarray(v_gt_all[start : start + n_windows], dtype=np.float64)
    dt = DT_10HZ

    # Denormalize tail samples → physical 10 Hz stream (window last sample).
    tails = seg[:, -1, :] * std + mean
    acc_stream = tails[:, :3]
    gyro_stream = tails[:, 3:6]

    # AVNet predictions + learned σ + lean
    model = AVNetLite()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    pred = predict_batch(model, seg, mean, std)
    v_pred, sig_v = pred["v_pred"], pred["sigma_v"]
    phi_arr, p_bike_arr = pred["phi"], pred["p_bike"]
    if lean_mode == "bike":
        p_bike_arr[:] = 1.0
    elif lean_mode == "car":
        p_bike_arr[:] = 0.0

    # GT along-track distance
    dist_gt = np.cumsum(v_gt * dt)
    total_dist = float(dist_gt[-1])
    # Degenerate stop segments: drift% is undefined at 0 distance — report
    # absolute error instead of crashing (stop-go replay is a core case).
    drift_scale = total_dist if total_dist > 1e-6 else 1.0

    # --- naive: integrate raw accelerometer (gravity already removed;
    # nav frame = identity, body x treated as forward)
    R0 = torch.eye(3, dtype=torch.float64)
    fwd0 = R0[:, 0].numpy()
    v_naive = v_gt[0] + np.cumsum(acc_stream @ fwd0 * dt)
    dist_naive = np.cumsum(np.clip(v_naive, 0, None) * dt)

    # --- AVNet-only integration
    dist_avnet = np.cumsum(np.clip(v_pred, 0, None) * dt)

    # --- InEKF
    ekf = InEKF(R0, q_acc=q_acc)
    ekf.v = torch.tensor([v_gt[0], 0.0, 0.0], dtype=torch.float64)
    dist_ekf = np.zeros(n_windows)
    r_floor = 0.3**2  # m/s² floor on velocity-measurement variance
    zupt_det = StationaryDetector(rate_hz=10.0) if use_variance_zupt else None
    n_zupt = 0
    for i in range(1, n_windows):
        zupt = _zupt_decision(
            i, zupt_det, acc_stream, gyro_stream, v_pred, start, zupt_speed_gate
        )
        if zupt:
            n_zupt += 1
        # Skip propagation while stationary so accel noise isn't integrated
        # into position (parity with DrPipeline.kt `if (!still) propagate`).
        if not zupt:
            ekf.propagate(
                torch.from_numpy(gyro_stream[i - 1]),
                torch.from_numpy(acc_stream[i - 1]),
                dt,
            )

        # Velocity measurement from AVNet + adaptive NHC (+ ZUPT v=0)
        v_fwd = max(float(v_pred[i]), 0.0)
        phi = float(phi_arr[i])
        is_bike = float(p_bike_arr[i]) > 0.5
        v_lat = v_fwd * math.sin(phi) if (is_bike and not zupt) else 0.0
        r_scale = (1.0 + 2.0 * abs(phi)) if is_bike else 1.0

        R_fwd = 0.05**2 if zupt else max(sig_v[i] ** 2, r_floor)
        R_meas = torch.diag(torch.tensor(
            [R_fwd, R_fwd * r_scale, 25.0],  # vertical soft: linear acc has real vertical dynamics
            dtype=torch.float64,
        ))
        z = torch.tensor([0.0 if zupt else v_fwd, v_lat, 0.0], dtype=torch.float64)
        ekf.update_velocity(z, R_meas)

        # Along-track = forward speed in CAR frame (robust to attitude drift)
        v_car = ekf.R.t() @ ekf.v
        dist_ekf[i] = dist_ekf[i - 1] + max(float(v_car[0]), 0.0) * dt

    def pct(d: np.ndarray) -> float:
        return abs(float(d[-1]) - float(dist_gt[-1])) / drift_scale * 100

    metrics: dict[str, float | int] = {
        "segment_windows": int(n_windows),
        "start": int(start),
        "total_dist_m": float(total_dist),
        "naive_final_m": float(abs(dist_naive[-1] - dist_gt[-1])),
        "avnet_final_m": float(abs(dist_avnet[-1] - dist_gt[-1])),
        "inekf_final_m": float(abs(dist_ekf[-1] - dist_gt[-1])),
        "naive_drift_pct": pct(dist_naive),
        "avnet_drift_pct": pct(dist_avnet),
        "inekf_drift_pct": pct(dist_ekf),
        "q_acc": float(q_acc),
        "n_zupt": int(n_zupt),
    }
    if verbose:
        logger.info(
            f"[harness] segment {n_windows} windows (60s) total_dist {total_dist:.1f}m "
            f"mode={lean_mode} q_acc={q_acc}"
        )
        logger.info(f"  naive  final {metrics['naive_final_m']:7.1f}m  {metrics['naive_drift_pct']:6.1f}%")
        logger.info(f"  avnet  final {metrics['avnet_final_m']:7.1f}m  {metrics['avnet_drift_pct']:6.1f}%")
        logger.info(f"  inekf  final {metrics['inekf_final_m']:7.1f}m  {metrics['inekf_drift_pct']:6.1f}%")
        logger.info(
            f"  mean |v_pred-v_gt| {np.abs(v_pred - v_gt).mean():.3f} m/s | mean σ_v {sig_v.mean():.3f} "
            f"| mean φ {np.degrees(np.abs(phi_arr)).mean():.1f}° | mean p_bike {p_bike_arr.mean():.2f} "
            f"| P_trace {float(torch.trace(ekf.P)):.3g} | zupt {n_zupt}/{n_windows}"
        )
    return metrics


def test_lean() -> None:
    """Synthetic bike turn φ=30°: NHC must use v_fwd·sinφ, not 0 (Step 9.4).

    Raises:
        AssertionError: When the bike/car NHC branches drift from spec.
    """
    from loguru import logger

    lean = LeanEstimator()
    v_fwd = torch.tensor([5.0])
    phi = torch.tensor([math.radians(30.0)])
    v_y, scale = lean.nhc_correction(v_fwd, phi, torch.tensor([0.9]))
    expected = 5.0 * math.sin(math.radians(30.0))
    assert abs(float(v_y[0]) - expected) < 1e-5, f"NHC bike branch wrong: {v_y} != {expected}"
    assert abs(float(scale[0]) - (1 + 2 * math.radians(30.0))) < 1e-5
    v_y_car, scale_car = lean.nhc_correction(v_fwd, phi, torch.tensor([0.2]))
    assert float(v_y_car[0]) == 0.0, "car branch must keep v_lat=0"
    assert float(scale_car[0]) == 1.0
    logger.info(f"[test-lean] PASS — bike φ=30°: v_lat={float(v_y[0]):.3f} m/s (expected {expected:.3f}), R_scale={float(scale[0]):.3f}")
    logger.info("[test-lean] PASS — car fallback: v_lat=0, R_scale=1.0")
