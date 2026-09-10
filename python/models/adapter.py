"""QAIIMU adaptive-covariance adapter port — DEPRECATED (F6 decision).

Status: STUB. Not used by train_avnet.py, the fusion package, or Android.
Uncertainty comes solely from the AVNetLite ``head_logsig_vel`` (σ_v head).

Why stub: the adapter is untrained and expects W=20 @200 Hz windows we
don't have at the 10 Hz proxy replay; wiring random covariance into R_meas
would destabilize the validated harness (f8a18d9). Revisit after stage-2
bike data: train the adapter on a 200 Hz live stream, then fuse its output
as R_scale alongside σ_v.

Kept for reference + shape test only. Do NOT import in training/inference.
Importing issues a DeprecationWarning (lazily via __getattr__, not at
module import — see PEP 562).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def __getattr__(name: str):
    if name == "AdaptiveParameterAdjustmentModel":
        import warnings

        warnings.warn(
            "python.models.adapter is DEPRECATED (F6): σ_v head is the sole "
            "uncertainty source; do not wire into InEKF until stage-2. "
            "See module docstring.",
            DeprecationWarning,
            stacklevel=2,
        )
        return AdaptiveParameterAdjustmentModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class AdaptiveParameterAdjustmentModel(nn.Module):
    """Learned measurement-covariance adapter (10^(β·tanh) × base)."""

    def __init__(self, beta: float = 1.0, base_cov: tuple = (3, 3, 3)) -> None:
        """Build the adapter.

        Args:
            beta: Exponent scale (paper β=1, output range 1e-3–1e3 × base).
            base_cov: Base covariance diagonal (3,).
        """
        super().__init__()
        self.beta = beta
        self.register_buffer("base", torch.tensor(base_cov, dtype=torch.float32))
        self.cov_net = nn.Sequential(
            nn.Conv1d(6, 32, 5),
            nn.ReplicationPad1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Conv1d(32, 32, 5, dilation=3),
            nn.ReplicationPad1d(4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        )
        self.cov_lin = nn.Sequential(
            nn.Linear(32, 3),
            nn.Tanh(),
        )
        # small init as in the reference
        self.cov_lin[0].weight.data /= 100
        self.cov_lin[0].bias.data /= 100

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map an IMU window to a covariance diagonal.

        Args:
            x: (B, 20, 6) or (B, 6, 20) @200 Hz window.

        Returns:
            Covariance diagonal (B, 3), ``base × 10^(β·mean_t tanh(h))``.
        """
        if x.dim() == 3 and x.shape[1] == 20 and x.shape[2] == 6:
            x = x.permute(0, 2, 1)  # (B,6,20)
        h = self.cov_net(x)          # (B,32,20)
        h = h.permute(0, 2, 1)       # (B,20,32)
        h = self.cov_lin(h)          # (B,20,3)
        h = h.mean(dim=1)            # (B,3) — average over the window
        return self.base.unsqueeze(0) * (10 ** (self.beta * h))
