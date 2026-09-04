"""
adapter.py — Step 4.2 / QAIIMU adapter port — DEPRECATED (F6 decision).

Status: STUB. Not used by train_avnet.py, inekf_harness.py, or Android.
Uncertainty comes solely from AVNetLite `head_logsig_vel` (σ_v head).

Why stub (best option): adapter is untrained and expects W=20 @200Hz windows
we don't have at the 10Hz proxy replay; wiring random cov into R_meas would
destabilize the validated harness (f8a18d9). Revisit after stage-2 bike data:
train adapter on 200Hz live stream, then fuse as R_scale alongside σ_v.

Kept for reference + shape test only. Do NOT import in training/inference.
"""
import warnings
import torch
import torch.nn as nn

warnings.warn(
    "python.models.adapter is DEPRECATED (F6): σ_v head is the sole uncertainty; "
    "do not wire into InEKF until stage-2. See module docstring.",
    DeprecationWarning, stacklevel=2)

class AdaptiveParameterAdjustmentModel(nn.Module):
    def __init__(self, beta=1.0, base_cov=(3,3,3)):
        super().__init__()
        self.beta = beta
        self.register_buffer("base", torch.tensor(base_cov, dtype=torch.float32))  # (3,)
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
        # small init as in ref
        self.cov_lin[0].weight.data /= 100
        self.cov_lin[0].bias.data /= 100

    def forward(self, x):
        """
        x: (B, 20, 6) or (B, 6, 20)
        returns: (B, 3) cov diag
        """
        if x.dim() == 3 and x.shape[1] == 20 and x.shape[2] == 6:
            x = x.permute(0, 2, 1)  # (B,6,20)
        # (B,6,20) -> cov_net -> (B,32,20) -> transpose -> (B,20,32) -> lin -> (B,20,3) -> mean over time?
        # ref does transpose(0,2).squeeze() which is per-sample, not batch. For batch, we average.
        h = self.cov_net(x)  # (B,32,20)
        h = h.permute(0, 2, 1)  # (B,20,32)
        h = self.cov_lin(h)  # (B,20,3)
        h = h.mean(dim=1)  # (B,3) — average over window
        cov = self.base.unsqueeze(0) * (10 ** (self.beta * h))
        return cov

if __name__ == "__main__":
    m = AdaptiveParameterAdjustmentModel()
    m.eval()
    x = torch.randn(4, 20, 6)
    cov = m(x)
    print(f"cov {cov.shape} {cov[0]}")
    print(f"params {sum(p.numel() for p in m.parameters())}")
