"""21-DOF right-invariant EKF (InEKF) — validated Python port.

Faithful port of QAIIMU ``filter_propagate_improved`` / ``filter_update_improved``
(commit f8a18d9 harness — the reference for the Kotlin ``InEKFEngine.kt``).

State (21): ``[R_nav(3) | v(3) | p(3) | b_g(3) | b_a(3) | R_car(3) | p_car(3)]``
on SE2(3) with right-invariant error, float64 throughout (paper convention:
float64 filter, float32 CNN).

Frame convention (differs from the reference repo!): the gravity-aligned
frame has body **X = forward, Y = lateral, Z = up**. Velocity measurements
are therefore ordered ``z = [v_fwd, v_lat, v_vert]``.

Additions over the reference filter:
- AVNet velocity update with learned σ_v (R from the logσ head).
- Adaptive NHC — car ``v_lat = 0``; bike ``v_lat = v_fwd·sin φ`` with
  ``R_lat × (1 + 2|φ|)`` (python/models/lean_estimator.py).
- ZUPT — stationary → measure v = 0 with tight R.
- Bias clamping — ``b_g ±0.5 rad/s``, ``b_a ±2 m/s²`` (divergence guard).
"""

from __future__ import annotations

import torch

from python.utils.lie_group import sen3exp, skew, so3exp

# --- state index map ---------------------------------------------------------
DIM_STATE = 21
DIM_NOISE = 18
IDX_V = slice(3, 6)
IDX_BG = slice(9, 12)
IDX_BA = slice(12, 15)

GRAVITY = torch.tensor([0.0, 0.0, -9.80665], dtype=torch.float64)
# IO-VNBD preprocessed windows are gravity-REMOVED linear acceleration → the
# Python replay disables gravity (use_gravity=False). Raw Android IMU keeps
# gravity and initializes R0 from gravity_align_R() instead.


def gravity_align_R(acc_mean: torch.Tensor) -> torch.Tensor:
    """Initial R_nav from mean accel: align gravity axis, yaw = 0.

    Args:
        acc_mean: Mean raw accel (3,), gravity-dominated.

    Returns:
        Rotation (3,3) whose columns are body axes expressed in the nav
        frame (body x = nav x at start).
    """
    a = acc_mean / torch.norm(acc_mean)
    x_body = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    x_body = x_body - (x_body @ a) * a
    x_body = x_body / torch.norm(x_body)
    y_body = torch.linalg.cross(a, x_body)
    return torch.stack([x_body, y_body, a], dim=1)


class InEKF:
    """21-DOF right-invariant EKF (improved path), float64, CPU torch.

    q_acc divergence note (F3): the Python default 30.0 matches the 10 Hz
    proxy replay where one tail sample per 0.1 s carries huge variance
    (optimal weight ≈ 0). Android ``InEKFEngine.kt`` uses qAcc = 0.5 because
    live 100 Hz propagation is smoother (P exploded to 3e26 when stationary
    at 30.0 — commit 23cc3dd). Sweep with ``q_acc`` and log the P trace.
    """

    BG_CLAMP = 0.5  # rad/s
    BA_CLAMP = 2.0  # m/s^2

    def __init__(self, R0: torch.Tensor, q_acc: float = 30.0, use_gravity: bool = False) -> None:
        """Initialize the filter at identity velocity/position and given attitude.

        Args:
            R0: Initial attitude (3,3), typically :func:`gravity_align_R` or I.
            q_acc: Accelerometer process noise (see class docstring).
            use_gravity: False for gravity-removed linear-acc streams
                (IO-VNBD replay); True for raw phone IMU.
        """
        self.q_acc = float(q_acc)
        self.R = R0.clone()
        self.g = GRAVITY.clone() if use_gravity else torch.zeros(3, dtype=torch.float64)
        self.v = torch.zeros(3, dtype=torch.float64)
        self.p = torch.zeros(3, dtype=torch.float64)
        self.b_g = torch.zeros(3, dtype=torch.float64)
        self.b_a = torch.zeros(3, dtype=torch.float64)
        self.R_car = torch.eye(3, dtype=torch.float64)
        self.p_car = torch.zeros(3, dtype=torch.float64)
        self.P = torch.diag(torch.tensor(
            [1e-2] * 3 + [0.5] * 3 + [0.5] * 3 + [1e-4] * 3
            + [1e-2] * 3 + [1e-6] * 3 + [1e-6] * 3,
            dtype=torch.float64,
        ))
        # [gyro(3) acc(3) bg_walk(3) ba_walk(3) R_car(3) p_car(3)]
        self.Q = torch.diag(torch.tensor(
            [1e-4] * 3 + [q_acc] * 3 + [1e-6] * 3 + [1e-4] * 3 + [1e-8] * 3 + [1e-8] * 3,
            dtype=torch.float64,
        ))

    def _clamp_biases(self) -> None:
        """Clamp bias estimates to physically plausible ranges."""
        self.b_g = self.b_g.clamp(-self.BG_CLAMP, self.BG_CLAMP)
        self.b_a = self.b_a.clamp(-self.BA_CLAMP, self.BA_CLAMP)

    def propagate(self, gyro: torch.Tensor, acc: torch.Tensor, dt: float) -> None:
        """IMU propagation step (port of filter_propagate_improved).

        Args:
            gyro: Angular rate (3,), rad/s (bias-corrected internally).
            acc: Linear acceleration (3,), m/s² (gravity handling per init).
            dt: Elapsed time in seconds.
        """
        w = gyro - self.b_g
        dR = so3exp(w * dt)
        R_prop = self.R @ dR
        a = acc - self.b_a
        a_nav = self.R @ a + self.g
        v_prop = self.v + a_nav * dt
        p_prop = self.p + (self.v + v_prop) * dt * 0.5

        # state Jacobian + 3rd-order Phi series
        F = torch.zeros(DIM_STATE, DIM_STATE, dtype=torch.float64)
        F[3:6, 0:3] = skew(self.g)
        F[6:9, 3:6] = torch.eye(3, dtype=torch.float64)
        F[0:3, 9:12] = -self.R
        F[3:6, 9:12] = skew(self.v) @ self.R
        F[6:9, 9:12] = skew(self.p) @ self.R
        F[3:6, 12:15] = self.R
        F = F * dt
        Phi = (
            torch.eye(DIM_STATE, dtype=torch.float64)
            + F + 0.5 * (F @ F) + (1.0 / 6.0) * (F @ F @ F)
        )

        # noise Jacobian
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

    def update_velocity(self, v_car_meas: torch.Tensor, R_meas: torch.Tensor) -> None:
        """Car-frame velocity update (port of filter_update_improved) + NHC.

        Measurement order: ``z = [v_fwd, v_lat, v_vert]`` (X-forward frame).

        Args:
            v_car_meas: Car-frame velocity measurement (3,).
                car: ``v_lat = 0`` (NHC); bike: ``v_lat = v_fwd·sin φ``.
            R_meas: Measurement covariance (3,3) diag, same order as z.
        """
        v_imu = self.R.t() @ self.v
        v_car_pred = self.R_car @ v_imu  # p_car = 0, lever ω-term vanishes

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
        self.R_car = so3exp(dx[15:18]) @ self.R_car
        self.p_car = self.p_car + dx[18:21]

        IKH = torch.eye(DIM_STATE, dtype=torch.float64) - K @ H
        P_new = IKH @ self.P @ IKH.t() + K @ R_meas @ K.t()
        self.P = (P_new + P_new.t()) * 0.5
        self._clamp_biases()
