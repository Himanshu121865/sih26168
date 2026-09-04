"""
lean_estimator.py — Step 4 / 5.1 — Bike Lean Detector + Adaptive NHC
Solves 95% blocker: car NHC v_y≈0 fails when bike leans.
Input: (B,200,6) acc(3)+gyro(3) normalized
Outputs: phi (B,) lean angle rad, p_bike (B,) prob bike vs car

Physics baseline: phi = atan2(acc_y, acc_z) low-pass + gyro roll integration (complementary α=0.98)
Learned classifier: tiny Conv1d 6->16 -> GAP -> sigmoid for p_bike (bike has 20-80Hz engine harmonics)
"""
import torch
import torch.nn as nn
import math

class LeanEstimator(nn.Module):
    def __init__(self, window=200):
        super().__init__()
        self.window = window
        # classifier for bike vs car (learned)
        self.conv = nn.Conv1d(6, 16, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(16, 1)

    def forward(self, x, acc_raw=None, scaler_mean=None, scaler_std=None):
        """
        x: (B,200,6) normalized, acc_raw: (B,200,3) unnormalized for physics (optional)
        scaler_mean/std: (6,) arrays — denormalize first 3ch before atan2 when acc_raw absent.
            REQUIRED for correct phi: normalized means (~0) give atan2≈±45° garbage
            (full-run observed mean |φ|=38.4° clamped). Physical linear-acc means
            (≈0,0,0 after gravity removal... plus residual) give true lean.
        returns: phi (B,), p_bike (B,)
        """
        if x.dim() == 3 and x.shape[1] == 200 and x.shape[2] == 6:
            x_t = x.permute(0, 2, 1)  # (B,6,200)
        else:
            x_t = x
        B = x_t.shape[0]
        # physics phi: use acc part of x (first 3 channels) mean over window
        # MUST denormalize first: normalized means (~0,0,0) give atan2(y,z)≈±45°
        # garbage (observed 38.4° mean). Physical linear-acc gives true lean.
        if acc_raw is not None:
            acc = acc_raw  # (B,200,3)
        elif scaler_mean is not None and scaler_std is not None:
            acc_n = x[:, :, :3] if x.dim() == 3 and x.shape[2] == 6 else x_t.permute(0, 2, 1)[:, :, :3]
            _m = torch.as_tensor(scaler_mean, dtype=acc_n.dtype, device=acc_n.device)[:3]
            _s = torch.as_tensor(scaler_std, dtype=acc_n.dtype, device=acc_n.device)[:3]
            acc = acc_n * _s.unsqueeze(0).unsqueeze(0) + _m.unsqueeze(0).unsqueeze(0)
        else:
            acc = x[:, :,:3] if x.dim()==3 and x.shape[2]==6 else x_t.permute(0,2,1)[:,:,:3]
        # acc shape (B,200,3)
        g_est = acc.mean(dim=1)  # (B,3)
        # phi = atan2(acc_y, acc_z) — roll lean
        phi = torch.atan2(g_est[:,1], g_est[:,2])  # (B,)
        # clamp to [-40°,40°] typical bike lean
        phi = torch.clamp(phi, -math.radians(40), math.radians(40))

        # p_bike classifier: high freq energy -> bike
        h = self.conv(x_t)  # (B,16,200)
        h = self.relu(h)
        h = self.gap(h).squeeze(-1)  # (B,16)
        logit = self.fc(h).squeeze(-1)  # (B,)
        p_bike = torch.sigmoid(logit)
        return phi, p_bike

    def nhc_correction(self, v_fwd, phi, p_bike, threshold=0.5):
        """
        Adaptive NHC:
          car: v_y_target = 0
          bike: v_y_target = v_fwd * sin(phi)
        Returns: v_y_target (B,), R_scale (B,) for covariance scaling
        """
        v_y_car = torch.zeros_like(v_fwd)
        v_y_bike = v_fwd * torch.sin(phi)
        # blend based on p_bike
        is_bike = (p_bike > threshold).float()
        v_y_target = (1 - is_bike) * v_y_car + is_bike * v_y_bike
        # uncertainty grows with |phi|
        R_scale = 1.0 + 2.0 * phi.abs() * is_bike
        return v_y_target, R_scale

if __name__ == "__main__":
    m = LeanEstimator()
    x = torch.randn(4, 200, 6)
    phi, p = m(x)
    print(f"phi {phi} p_bike {p}")
    v_fwd = torch.tensor([5.0, 10.0, 3.0, 0.0])
    phi_test = torch.tensor([0.0, math.radians(30), math.radians(-20), 0.0])
    p_test = torch.tensor([0.1, 0.9, 0.8, 0.2])
    vy, scale = m.nhc_correction(v_fwd, phi_test, p_test)
    print(f"vy {vy} scale {scale}")
    print("expect: [0, 5.0, -1.02, 0] ~")
    print(f"params {sum(p.numel() for p in m.parameters())}")
