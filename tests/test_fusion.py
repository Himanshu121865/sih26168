"""Tests for the InEKF fusion layer (propagate/update physics + lean NHC)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from python.fusion.ine_kf import DIM_STATE, InEKF, gravity_align_R
from python.utils.lie_group import sen3exp, skew, so3exp


def _eye_filter(**kw) -> InEKF:
    return InEKF(torch.eye(3, dtype=torch.float64), **kw)


class TestLieGroup:
    def test_skew_antisymmetric(self) -> None:
        """skew(w)ᵀ = −skew(w) and skew maps back via vee."""
        w = torch.tensor([0.1, -0.2, 0.3], dtype=torch.float64)
        W = skew(w)
        assert torch.allclose(W.t(), -W)
        assert torch.allclose(torch.stack([W[2, 1], W[0, 2], W[1, 0]]), w)

    def test_so3exp_identity_and_rotation(self) -> None:
        """Zero rotation → I; 90° about z rotates x→y."""
        assert torch.allclose(so3exp(torch.zeros(3, dtype=torch.float64)), torch.eye(3, dtype=torch.float64))
        R = so3exp(torch.tensor([0.0, 0.0, math.pi / 2], dtype=torch.float64))
        x = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        y = R @ x
        assert y[0] == pytest.approx(0.0, abs=1e-10)
        assert y[1] == pytest.approx(1.0, abs=1e-10)

    def test_so3exp_orthonormal(self) -> None:
        """Random rotations stay in SO(3): RᵀR = I, det = +1."""
        rng = np.random.default_rng(0)
        for _ in range(5):
            w = torch.tensor(rng.normal(size=3), dtype=torch.float64)
            R = so3exp(w)
            assert torch.allclose(R.t() @ R, torch.eye(3, dtype=torch.float64), atol=1e-10)
            assert torch.linalg.det(R) == pytest.approx(1.0, abs=1e-10)

    def test_sen3exp_small_angle(self) -> None:
        """Near-zero xi: R ≈ I + K, v/p ≈ their rho parts (J ≈ I)."""
        xi = torch.randn(9, dtype=torch.float64) * 1e-6
        R, v, p = sen3exp(xi)
        assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-5)
        assert torch.allclose(v, xi[3:6], atol=1e-11)
        assert torch.allclose(p, xi[6:9], atol=1e-11)


class TestInEKF:
    def test_state_dims(self) -> None:
        """21-DOF state, 18-dim noise (QAIIMU parity)."""
        assert DIM_STATE == 21
        ekf = _eye_filter()
        assert ekf.P.shape == (21, 21)
        assert ekf.Q.shape == (18, 18)

    def test_zero_imu_holds_state(self) -> None:
        """Zero gyro/acc + no gravity: position and velocity stay at zero."""
        ekf = _eye_filter(use_gravity=False)
        z3 = torch.zeros(3, dtype=torch.float64)
        for _ in range(100):
            ekf.propagate(z3, z3, 0.01)
        assert torch.allclose(ekf.v, z3)
        assert torch.allclose(ekf.p, z3)

    def test_constant_accel_parabola(self) -> None:
        """Constant nav-frame accel → p = ½at² (midpoint integration)."""
        ekf = _eye_filter(use_gravity=False)
        a = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
        z3 = torch.zeros(3, dtype=torch.float64)
        dt, n = 0.01, 200  # 2 s
        for _ in range(n):
            ekf.propagate(z3, a, dt)
        t = n * dt
        assert ekf.v[0] == pytest.approx(a[0] * t)
        assert ekf.p[0] == pytest.approx(0.5 * a[0] * t * t)

    def test_velocity_update_pulls_state(self) -> None:
        """A v_meas = 5 m/s forward measurement moves v toward 5."""
        ekf = _eye_filter(use_gravity=False)
        R = torch.diag(torch.tensor([1e-4, 1e-2, 1e-2], dtype=torch.float64))
        z5 = torch.tensor([5.0, 0.0, 0.0], dtype=torch.float64)
        for _ in range(50):
            ekf.update_velocity(z5, R)
        assert float(ekf.v[0]) == pytest.approx(5.0, abs=1e-3)

    def test_bias_clamps(self) -> None:
        """Biases clamp to ±0.5 (gyro) / ±2.0 (acc) — divergence guard."""
        ekf = _eye_filter()
        ekf.b_g = torch.tensor([10.0, -10.0, 0.0], dtype=torch.float64)
        ekf.b_a = torch.tensor([50.0, 0.0, 0.0], dtype=torch.float64)
        ekf._clamp_biases()
        assert float(ekf.b_g.abs().max()) <= 0.5
        assert float(ekf.b_a.abs().max()) <= 2.0

    def test_covariance_stays_psd(self) -> None:
        """P remains symmetric positive-semidefinite over many steps."""
        ekf = _eye_filter(q_acc=1.0, use_gravity=False)
        rng = np.random.default_rng(0)
        R = torch.diag(torch.full((3,), 0.1, dtype=torch.float64))
        for _i in range(200):
            gyro = torch.tensor(rng.normal(0, 0.05, 3), dtype=torch.float64)
            acc = torch.tensor(rng.normal(0, 0.5, 3), dtype=torch.float64)
            ekf.propagate(gyro, acc, 0.1)
            ekf.update_velocity(torch.tensor([2.0, 0.0, 0.0], dtype=torch.float64), R)
        eigs = torch.linalg.eigvalsh(ekf.P)
        assert bool((eigs >= -1e-9).all()), f"P lost PSD: min eig {float(eigs.min()):.3g}"
        assert torch.allclose(ekf.P, ekf.P.t(), atol=1e-9)

    def test_gravity_align_R_level(self) -> None:
        """Level gravity (z up) → identity attitude (yaw 0, x forward)."""
        acc = torch.tensor([0.0, 0.0, 9.81], dtype=torch.float64)
        R = gravity_align_R(acc)
        assert torch.allclose(R, torch.eye(3, dtype=torch.float64), atol=1e-9)

    def test_gravity_align_R_pitched(self) -> None:
        """30° pitch: R rotates about y by 30°, det = 1."""
        pitch = math.radians(30.0)
        acc = torch.tensor([9.81 * math.sin(pitch), 0.0, 9.81 * math.cos(pitch)], dtype=torch.float64)
        R = gravity_align_R(acc)
        assert torch.linalg.det(R) == pytest.approx(1.0)
        # body x in nav = [cos30, 0, -sin30]? columns: x_body, y_body, z_body
        # z_body must equal acc/|acc| (gravity up)
        assert R[:, 2][2] == pytest.approx(math.cos(pitch))


class TestReplayHelpers:
    def test_zupt_decision_first_samples_heuristic(self) -> None:
        """Before the variance window fills, low model speed triggers ZUPT."""
        from python.fusion.replay import _zupt_decision

        v_pred = np.zeros(10)
        acc = np.zeros((10, 3))
        gyro = np.zeros((10, 3))
        assert _zupt_decision(1, None, acc, gyro, v_pred, start=0, zupt_speed_gate=0.3) is True
        assert _zupt_decision(1, None, acc, gyro, v_pred + 5.0, start=0, zupt_speed_gate=0.3) is False

    def test_run_replay_test_lean(self) -> None:
        """The synthetic φ=30° NHC check passes (Step 9.4)."""
        from python.fusion.replay import test_lean

        test_lean()  # raises AssertionError on spec drift
