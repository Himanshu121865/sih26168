#!/usr/bin/env python3
"""
inekf_harness.py — Step 9 — Validate InEKF in Python before Kotlin port.

Faithful port of ref/QAIIMU filter_propagate_improved + filter_update_improved
(21-DOF state: R_nav, v, p, b_g, b_a, R_car, p_car; right-invariant, SE2(3), float64).

Additions over reference (AGENTS.md Step 9.2 / 5.2):
- AVNet velocity measurement: v_fwd from model, R from learned logσ_v head.
- Adaptive NHC branch (lean_estimator.py):
    car:  v_car_x = 0
    bike: v_car_x = v_fwd * sin(φ), R_lat *= (1 + 2|φ|)
- ZUPT: when stationary (utils/zupt.py), measure v = 0 with tight R.
- Bias clamping (AGENTS.md risk table: clamp b_g ±0.5 rad/s, b_a ±2 m/s²).

Replay: val windows (600 = 60s outage @10Hz). Windows are stored normalized;
denormalize with scaler.json to recover physical acc/gyro. The LAST sample of
window i is raw 100Hz sample i*stride+199 → consecutive window tails form the
10Hz measurement stream (dt=0.1s), same rate as AVNet output.

Comparison (along-track drift vs GT):
    naive:      integrate raw accelerometer directly
    AVNet-only: integrate v_pred per window
    InEKF:      this filter (propagate IMU, update with v_pred + NHC)

Usage:
  PYTHONPATH=. python python/inekf_harness.py --model experiments/checkpoints/model_avnet_stage1.p
  PYTHONPATH=. python python/inekf_harness.py --test-lean   # Step 9.4 synthetic φ=30° check
Output: reports/inekf_vs_avnet.csv
"""
import argparse, json, math
from pathlib import Path
import numpy as np
import torch

from python.models.avnet import AVNetLite
from python.models.lean_estimator import LeanEstimator
from python.utils.zupt import StationaryDetector

# Single source of truth for Lie group math (F2): canonical impl lives in
# python/utils/lie_group.py. Harness re-exports the same symbols so the
# Kotlin port and any `from python.inekf_harness import skew` callers keep working.
from python.utils.lie_group import skew, so3exp, sen3exp

# ---------------------------------------------------------------- state index
# [R_nav(0:3) | v(3:6) | p(6:9) | b_g(9:12) | b_a(12:15) | R_car(15:18) | p_car(18:21)]
DIM_STATE = 21
DIM_NOISE = 18
IDX_V = slice(3, 6)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)

G = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)
# NOTE: IO-VNBD preprocessed windows contain GRAVITY-REMOVED linear acceleration,
# so the harness replays with G disabled (use_gravity=False). Raw phone IMU on
# Android keeps gravity and uses R0 from gravity_align_R().
# NOTE (F2): skew/so3exp/sen3exp imported from python.utils.lie_group (exact
# left Jacobian). Do NOT redefine locally — parity with Kotlin LieGroup.kt.


def gravity_align_R(acc_mean: torch.Tensor) -> torch.Tensor:
    """Initial R_nav from mean acc: align gravity axis, yaw=0 (body x = nav x)."""
    a = acc_mean / torch.norm(acc_mean)
    z_body = a  # reaction acc points up in body frame when level
    x_body = torch.tensor([1.0, 0, 0], dtype=torch.float64)
    x_body = x_body - (x_body @ z_body) * z_body
    x_body = x_body / torch.norm(x_body)
    y_body = torch.linalg.cross(z_body, x_body)
    return torch.stack([x_body, y_body, z_body], dim=1)  # columns = body axes in nav

class InEKF:
    """21-DOF right-invariant EKF (improved path), float64, CPU torch.

    q_acc divergence note (F3): Python default 30.0 matches the 10Hz proxy
    replay where one tail sample per 0.1s has huge variance (optimal weight≈0).
    Android InEKFEngine.kt uses qAcc=0.5 because live 100Hz propagation is
    smoother and P exploded to 3e26 when stationary at 30.0 (commit 23cc3dd).
    Pass --q-acc to sweep; log P trace to compare.
    """

    BG_CLAMP = 0.5    # rad/s
    BA_CLAMP = 2.0    # m/s^2

    def __init__(self, R0, q_acc=30.0, use_gravity=False):
        self.q_acc = float(q_acc)
        self.R = R0.clone()
        self.g = G.clone() if use_gravity else torch.zeros(3, dtype=torch.float64)
        self.v = torch.zeros(3, dtype=torch.float64)
        self.p = torch.zeros(3, dtype=torch.float64)
        self.b_g = torch.zeros(3, dtype=torch.float64)
        self.b_a = torch.zeros(3, dtype=torch.float64)
        self.R_car = torch.eye(3, dtype=torch.float64)
        self.p_car = torch.zeros(3, dtype=torch.float64)
        self.P = torch.diag(torch.tensor(
            [1e-2] * 3 + [0.5] * 3 + [0.5] * 3 + [1e-4] * 3 + [1e-2] * 3 + [1e-6] * 3 + [1e-6] * 3,
            dtype=torch.float64))
        # noise cov: [gyro(3) acc(3) bg_walk(3) ba_walk(3) R_car(3) p_car(3)]
        self.Q = torch.diag(torch.tensor(
            [1e-4] * 3 + [q_acc] * 3 + [1e-6] * 3 + [1e-4] * 3 + [1e-8] * 3 + [1e-8] * 3,
            dtype=torch.float64))

    def _clamp_biases(self):
        self.b_g = self.b_g.clamp(-self.BG_CLAMP, self.BG_CLAMP)
        self.b_a = self.b_a.clamp(-self.BA_CLAMP, self.BA_CLAMP)

    def propagate(self, gyro: torch.Tensor, acc: torch.Tensor, dt: float):
        """Port of filter_propagate_improved (ref lines 357-405)."""
        w = gyro - self.b_g
        dR = so3exp(w * dt)
        R_prop = self.R @ dR
        a = acc - self.b_a
        a_nav = self.R @ a + self.g
        v_prop = self.v + a_nav * dt
        p_prop = self.p + (self.v + v_prop) * dt * 0.5

        # state Jacobian (ref lines 373-383)
        F = torch.zeros(DIM_STATE, DIM_STATE, dtype=torch.float64)
        F[3:6, 0:3] = skew(self.g)
        F[6:9, 3:6] = torch.eye(3, dtype=torch.float64)
        F[0:3, 9:12] = -self.R
        F[3:6, 9:12] = skew(self.v) @ self.R
        F[6:9, 9:12] = skew(self.p) @ self.R
        F[3:6, 12:15] = self.R
        F = F * dt
        Phi = torch.eye(DIM_STATE, dtype=torch.float64) + F + 0.5 * (F @ F) + (1.0 / 6.0) * (F @ F @ F)

        # noise Jacobian (ref lines 385-390)
        Gn = torch.zeros(DIM_STATE, DIM_NOISE, dtype=torch.float64)
        Gn[0:3, 0:3] = -self.R
        Gn[3:6, 3:6] = self.R
        Gn[9:15, 6:12] = torch.eye(6, dtype=torch.float64)
        Gn[15:18, 12:15] = self.R_car.t()
        Gn[18:21, 15:18] = torch.eye(3, dtype=torch.float64)
        Gn = Gn * dt

        self.P = Phi @ (self.P + Gn @ self.Q @ Gn.t()) @ Phi.t()
        self.R, self.v, self.p = R_prop, v_prop, p_prop
        self._clamp_biases()

    def update_velocity(self, v_car_meas: torch.Tensor, R_meas: torch.Tensor):
        """Port of filter_update_improved (ref lines 408-467) with lean NHC extension.

        Frame convention (differs from ref!): our gravity-aligned frame has
        body X = forward, Y = lateral, Z = up (ref uses car-frame Y forward).

        v_car_meas: (3,) car-frame velocity measurement [v_fwd, v_lat, v_vert]
            car:  v_lat = 0 (NHC)
            bike: v_lat = v_fwd * sin(φ) (lean)
        R_meas: (3,3) measurement covariance diag (matches v_car_meas order).
        """
        v_imu = self.R.t() @ self.v
        v_car_pred = self.R_car @ v_imu  # p_car=0, ω-term vanishes

        H = torch.zeros(3, DIM_STATE, dtype=torch.float64)
        H[:, IDX_V] = self.R_car @ self.R.t()

        S = H @ self.P @ H.t() + R_meas
        K = torch.linalg.solve(S, (self.P @ H.t()).t()).t()

        innov = v_car_meas - v_car_pred
        dx = K @ innov

        dR, dv, dp = sen3exp(dx[:9])
        self.R = dR @ self.R
        self.v = dR @ self.v + dv
        self.p = dR @ self.p + dp
        self.b_g = self.b_g + dx[IDX_BG]
        self.b_a = self.b_a + dx[IDX_BA]
        dR_car = so3exp(dx[15:18])
        self.R_car = dR_car @ self.R_car
        self.p_car = self.p_car + dx[18:21]

        IKH = torch.eye(DIM_STATE, dtype=torch.float64) - K @ H
        P_new = IKH @ self.P @ IKH.t() + K @ R_meas @ K.t()
        self.P = (P_new + P_new.t()) * 0.5
        self._clamp_biases()


def load_scaler(path="python/scaler.json"):
    with open(path) as f:
        sc = json.load(f)
    return np.array(sc["mean"], dtype=np.float64), np.array(sc["std"], dtype=np.float64)


def run_replay(model_path, n_windows=600, start=None, lean_mode="auto", verbose=True,
               q_acc=30.0, zupt_speed_gate=0.3, use_variance_zupt=True):
    """Replay 60s outage on val segment. Returns dict of metrics."""
    mean, std = load_scaler()
    X = np.load("data/processed/val_windows.npy", mmap_mode="r")  # (N,200,6) normalized
    v_gt_all = np.load("data/processed/val_v.npy")                # (N,) m/s
    n_windows = min(n_windows, len(X) - 4)
    if start is None:
        start = len(X) // 2
    seg = np.array(X[start:start + n_windows], dtype=np.float64)  # (N,200,6) normalized
    v_gt = np.asarray(v_gt_all[start:start + n_windows], dtype=np.float64)
    dt = 0.1

    # denormalize tail samples → physical 10Hz stream (window last sample)
    tails = seg[:, -1, :] * std + mean       # (N,6) linear-acc(3)+gyro(3)
    acc_stream = tails[:, :3]
    gyro_stream = tails[:, 3:6]

    # AVNet predictions + learned σ + lean
    model = AVNetLite()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    lean = LeanEstimator()
    lean.eval()
    with torch.no_grad():
        xb = torch.from_numpy(seg.astype(np.float32))
        v_pred, ls_v, _, _, _ = model(xb)
        v_pred = v_pred.squeeze(-1).numpy().astype(np.float64)
        sig_v = torch.exp(ls_v.squeeze(-1)).numpy().astype(np.float64)
        phi_arr, p_bike_arr = lean(xb)
        phi_arr = phi_arr.numpy().astype(np.float64)
        p_bike_arr = p_bike_arr.numpy().astype(np.float64)
    if lean_mode == "bike":
        p_bike_arr[:] = 1.0
    elif lean_mode == "car":
        p_bike_arr[:] = 0.0

    # GT along-track distance
    dist_gt = np.cumsum(v_gt * dt)
    total_dist = dist_gt[-1]

    # --- naive: integrate raw accelerometer (linear acc, gravity already removed;
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
    r_floor = 0.3 ** 2  # m/s² floor on velocity-measurement variance
    # Variance-based stationary detector on the 10Hz proxy stream (F4).
    # Mirrors Android DrPipeline (ZuptDetector @100Hz): low accel/gyro variance
    # + speed gate → freeze propagate + v=0 update. At 10Hz, window_s=0.5 → 5 samples.
    zupt_det = StationaryDetector(rate_hz=10.0) if use_variance_zupt else None
    n_zupt = 0
    for i in range(1, n_windows):
        # variance ZUPT uses current sample + model speed as proxy for vehicle speed
        still_var = False
        if zupt_det is not None:
            t_ns = int((start + i) * 100_000_000)  # 10Hz ticks, monotonic is enough
            still_var = bool(zupt_det.update(
                np.asarray(acc_stream[i]), np.asarray(gyro_stream[i]), t_ns,
                speed_mps=float(max(float(v_pred[i]), 0.0))))
        # legacy model-speed heuristic as fallback/OR (keeps old behavior when
        # variance window not yet filled, e.g. first 0.5s of replay)
        v_fwd_raw = max(float(v_pred[i]), 0.0)
        zupt_heur = v_fwd_raw < zupt_speed_gate
        zupt = bool(still_var or (zupt_heur and i < 5))
        if zupt:
            # first samples: heuristic only until detector fills; afterwards variance only
            pass
        if zupt_det is not None and i >= 5:
            zupt = still_var
        if zupt:
            n_zupt += 1
        # Skip propagation while stationary so accel noise isn't integrated
        # into position (parity with DrPipeline.kt:109 `if (!still) propagate`).
        if not zupt:
            ekf.propagate(torch.from_numpy(gyro_stream[i - 1]),
                          torch.from_numpy(acc_stream[i - 1]), dt)

        # velocity measurement from AVNet + adaptive NHC + ZUPT
        # (zupt already decided above via variance detector; do NOT overwrite
        # with model-speed heuristic — that regressed stop-go to 25%.)
        v_fwd = v_fwd_raw
        phi = float(phi_arr[i])
        is_bike = float(p_bike_arr[i]) > 0.5
        v_lat = v_fwd * math.sin(phi) if (is_bike and not zupt) else 0.0
        r_scale = (1.0 + 2.0 * abs(phi)) if is_bike else 1.0

        if zupt:
            R_fwd = 0.05 ** 2
        else:
            R_fwd = max(sig_v[i] ** 2, r_floor)
        R_meas = torch.diag(torch.tensor([
            R_fwd,
            R_fwd * r_scale,  # NHC lateral (bike: scaled by lean)
            25.0,  # vertical soft (linear acc contains real vertical dynamics)
        ], dtype=torch.float64))
        z = torch.tensor([0.0 if zupt else v_fwd, v_lat, 0.0], dtype=torch.float64)
        ekf.update_velocity(z, R_meas)

        # along-track = forward speed in CAR frame (matches the measurement model,
        # robust to attitude drift): v_fwd = (R_nav^T v_nav)[x]
        v_car = ekf.R.t() @ ekf.v
        dist_ekf[i] = dist_ekf[i - 1] + max(float(v_car[0]), 0.0) * dt

    def pct(d):
        return abs(float(d[-1]) - float(dist_gt[-1])) / total_dist * 100

    metrics = {
        "segment_windows": int(n_windows),
        "start": int(start),
        "total_dist_m": float(total_dist),
        "naive_final_m": float(abs(dist_naive[-1] - dist_gt[-1])),
        "avnet_final_m": float(abs(dist_avnet[-1] - dist_gt[-1])),
        "inekf_final_m": float(abs(dist_ekf[-1] - dist_gt[-1])),
        "naive_drift_pct": pct(dist_naive),
        "avnet_drift_pct": pct(dist_avnet),
        "inekf_drift_pct": pct(dist_ekf),
    }
    if verbose:
        print(f"[harness] segment {n_windows} windows (60s) total_dist {total_dist:.1f}m mode={lean_mode} q_acc={q_acc}")
        print(f"  naive  final {metrics['naive_final_m']:7.1f}m  {metrics['naive_drift_pct']:6.1f}%")
        print(f"  avnet  final {metrics['avnet_final_m']:7.1f}m  {metrics['avnet_drift_pct']:6.1f}%")
        print(f"  inekf  final {metrics['inekf_final_m']:7.1f}m  {metrics['inekf_drift_pct']:6.1f}%")
        print(f"  mean |v_pred-v_gt| {np.abs(v_pred - v_gt).mean():.3f} m/s | mean σ_v {sig_v.mean():.3f} "
              f"| mean φ {np.degrees(np.abs(phi_arr)).mean():.1f}° | mean p_bike {p_bike_arr.mean():.2f} "
              f"| P_trace {float(torch.trace(ekf.P)):.3g} | zupt {n_zupt}/{n_windows}")
    metrics["q_acc"] = float(q_acc)
    metrics["n_zupt"] = int(n_zupt)
    return metrics


def test_lean():
    """Step 9.4 — synthetic bike turn φ=30°: NHC must use v_fwd·sinφ, not 0."""
    import torch as th
    lean = LeanEstimator()
    v_fwd = th.tensor([5.0])
    phi = th.tensor([math.radians(30.0)])
    v_y, scale = lean.nhc_correction(v_fwd, phi, th.tensor([0.9]))
    expected = 5.0 * math.sin(math.radians(30.0))
    assert abs(float(v_y[0]) - expected) < 1e-5, f"NHC bike branch wrong: {v_y} != {expected}"
    # R_scale spec (AGENTS.md 5.2): (1 + 2*|φ|) with φ in radians
    assert abs(float(scale[0]) - (1 + 2 * math.radians(30.0))) < 1e-5
    v_y_car, scale_car = lean.nhc_correction(v_fwd, phi, th.tensor([0.2]))
    assert float(v_y_car[0]) == 0.0, "car branch must keep v_lat=0"
    assert float(scale_car[0]) == 1.0
    print(f"[test-lean] PASS — bike φ=30°: v_lat={float(v_y[0]):.3f} m/s (expected {expected:.3f}), R_scale={float(scale[0]):.3f}")
    print(f"[test-lean] PASS — car fallback: v_lat=0, R_scale=1.0")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p")
    ap.add_argument("--windows", type=int, default=600)
    ap.add_argument("--lean-mode", choices=["auto", "car", "bike"], default="auto")
    ap.add_argument("--start", type=int, default=None, help="window index to start segment")
    ap.add_argument("--q-acc", type=float, default=30.0,
                    help="accel process noise (Python 10Hz proxy default 30.0; Android live uses 0.5)")
    ap.add_argument("--zupt-gate", type=float, default=0.3, help="v_fwd below this → ZUPT v=0 (first 5 samples fallback)")
    ap.add_argument("--no-variance-zupt", action="store_true", help="disable variance detector, use speed heuristic only")
    ap.add_argument("--test-lean", action="store_true")
    ap.add_argument("--csv", default="reports/inekf_vs_avnet.csv")
    args = ap.parse_args()

    if args.test_lean:
        test_lean()
        return

    metrics = run_replay(args.model, n_windows=args.windows, start=args.start,
                         lean_mode=args.lean_mode, q_acc=args.q_acc,
                         zupt_speed_gate=args.zupt_gate,
                         use_variance_zupt=not args.no_variance_zupt)
    Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
    write_header = not Path(args.csv).exists()
    with open(args.csv, "a") as f:
        if write_header:
            f.write("trajectory,naive_drift_pct,avnet_drift_pct,inekf_drift_pct\n")
        f.write(f"val_start{metrics['start']}_{metrics['segment_windows']}w_{metrics['total_dist_m']:.0f}m,{metrics['naive_drift_pct']:.2f},"
                f"{metrics['avnet_drift_pct']:.2f},{metrics['inekf_drift_pct']:.2f}\n")
    print(f"[csv] appended {args.csv}")


if __name__ == "__main__":
    main()

