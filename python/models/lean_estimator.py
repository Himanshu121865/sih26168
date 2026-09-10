"""Bike lean detector + adaptive NHC (Step 5).

Solves the 95% blocker for 2-wheelers: the car NHC assumption ``v_y ≈ 0``
fails when a bike leans into a turn. The lateral velocity is then
``v_y = v_fwd·sin φ`` and its uncertainty grows with |φ|.

Input: (B, 200, 6) normalized ``[linAcc xyz, gyro yaw/pitch/roll]``.
Outputs: ``phi (B,)`` lean angle rad, ``p_bike (B,)`` bike probability.

Physics: φ = atan2(mean acc_y, mean acc_z) low-passed over the window
(clamped to ±40°). The classifier head (Conv1d → GAP → sigmoid) keys on
20–80 Hz engine harmonics that cars lack.

CRITICAL: φ must be computed from DENORMALIZED acc — normalized means
(~0,0,0) give atan2 ≈ ±45° garbage (the observed 38.4° mean-|φ| bug).
Pass ``scaler_mean``/``scaler_std`` (or raw ``acc``) whenever input is
normalized.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

WINDOW = 200
PHI_CLAMP_RAD = math.radians(40.0)
NHC_BIKE_THRESHOLD = 0.5


class LeanEstimator(nn.Module):
    """Lean-angle physics + bike-vs-car classifier for adaptive NHC."""

    def __init__(self, window: int = WINDOW) -> None:
        """Build the tiny harmonic classifier.

        Args:
            window: Input window length (200).
        """
        super().__init__()
        self.window = window
        self.conv = nn.Conv1d(6, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(
        self,
        x: torch.Tensor,
        acc_raw: torch.Tensor | None = None,
        scaler_mean: torch.Tensor | np.ndarray | None = None,
        scaler_std: torch.Tensor | np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Estimate lean angle and bike probability per window.

        Args:
            x: (B, 200, 6) normalized windows.
            acc_raw: Optional (B, 200, 3) UNNORMALIZED acc — used directly
                for φ when given (preferred).
            scaler_mean / scaler_std: Optional per-channel stats (6,) to
                denormalize the first 3 channels when acc_raw is absent.
                Required for correct φ on normalized input (see module
                docstring).

        Returns:
            (phi (B,) rad clamped ±40°, p_bike (B,) in [0,1]).
        """
        if x.dim() != 3:
            raise ValueError(f"expected (B,T,6), got {x.shape}")
        # canonical (B, 6, T) for the conv; keep a (B, T, 6) view for acc
        x_t = x.permute(0, 2, 1) if x.shape[2] == 6 else x

        if acc_raw is not None:
            acc = acc_raw
        else:
            acc_n = x if x.shape[-1] == 6 else x_t.permute(0, 2, 1)
            if scaler_mean is not None and scaler_std is not None:
                m = torch.as_tensor(scaler_mean, dtype=acc_n.dtype, device=acc_n.device)[:3]
                s = torch.as_tensor(scaler_std, dtype=acc_n.dtype, device=acc_n.device)[:3]
                acc = acc_n[:, :, :3] * s + m
            else:
                # legacy path: normalized units — φ is garbage but shapes
                # stay valid (kept only for backward compat)
                acc = acc_n[:, :, :3]

        g_est = acc.mean(dim=1)          # (B, 3)
        phi = torch.atan2(g_est[:, 1], g_est[:, 2])
        phi = torch.clamp(phi, -PHI_CLAMP_RAD, PHI_CLAMP_RAD)

        h = self.gap(self.relu(self.conv(x_t))).squeeze(-1)  # (B, 16)
        p_bike = torch.sigmoid(self.fc(h).squeeze(-1))       # (B,)
        return phi, p_bike

    def nhc_correction(
        self,
        v_fwd: torch.Tensor,
        phi: torch.Tensor,
        p_bike: torch.Tensor,
        threshold: float = NHC_BIKE_THRESHOLD,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Adaptive NHC targets + covariance scaling.

        Args:
            v_fwd: Forward speed (B,), m/s.
            phi: Lean angle (B,), rad.
            p_bike: Bike probability (B,).
            threshold: p_bike above this selects the bike branch.

        Returns:
            (v_y_target (B,), R_scale (B,)) — car: (0, 1); bike:
            (``v_fwd·sin φ``, ``1 + 2|φ|``).
        """
        v_y_bike = v_fwd * torch.sin(phi)
        is_bike = (p_bike > threshold).to(v_fwd.dtype)
        v_y_target = is_bike * v_y_bike  # car branch contributes 0
        r_scale = 1.0 + 2.0 * phi.abs() * is_bike
        return v_y_target, r_scale
