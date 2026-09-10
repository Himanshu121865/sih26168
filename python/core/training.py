"""Shared training loop: augmentations, losses, epoch helpers.

Single implementation shared by ``train_avnet.py`` (stage-1) and any
fine-tune entrypoints — previously duplicated in ``train_avnet.py``.

Augmentations (operate on NORMALIZED (B, 200, 6)):
- ``augment_random_yaw``      — heading-agnostic rotation of horizontal
  acc x/y + gyro pitch/roll (harsh tcn.py:133-193).
- ``augment_synthetic_bike``  — pothole / engine-harmonic / lean-rotation
  robustness (F5 Plan B — no real bike data yet). Amplitudes in std units
  match observed IO-VNBD extremes (raw ±16 m/s²).

Both are gated per-batch at 50%: always-on shifts the train distribution
away from the unaugmented val set and raises the val floor (observed
train 1.17 vs val 1.77 on the full run).
"""

from __future__ import annotations

import math
import random

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    """Seed every RNG in play (python/numpy/torch + deterministic cuDNN)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment_random_yaw(x: torch.Tensor) -> torch.Tensor:
    """Rotate horizontal acc x,y (ch 0,1) and gyro pitch/roll (ch 4,5).

    acc_z (ch 2) and gyro yaw (ch 3) are vertical and stay invariant.

    Args:
        x: (B, 200, 6) normalized windows.

    Returns:
        Augmented copy; labels are unchanged (heading-agnostic).
    """
    B = x.shape[0]
    thetas = torch.rand(B, device=x.device) * 2 * math.pi
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    xa = x.clone()
    ax, ay = x[:, :, 0], x[:, :, 1]
    xa[:, :, 0] = ax * cos_t.unsqueeze(-1) - ay * sin_t.unsqueeze(-1)
    xa[:, :, 1] = ax * sin_t.unsqueeze(-1) + ay * cos_t.unsqueeze(-1)
    gx, gy = x[:, :, 5], x[:, :, 4]
    xa[:, :, 5] = gx * cos_t.unsqueeze(-1) - gy * sin_t.unsqueeze(-1)
    xa[:, :, 4] = gx * sin_t.unsqueeze(-1) + gy * cos_t.unsqueeze(-1)
    return xa


def gaussian_nll_loss(
    v_pred: torch.Tensor, v_gt: torch.Tensor, log_sig: torch.Tensor, min_sigma: float = 1e-3
) -> torch.Tensor:
    """Gaussian NLL for scalar velocity with a softplus σ floor.

    Args:
        v_pred: (B, 1) mean prediction.
        v_gt: (B,) target speed m/s.
        log_sig: (B, 1) raw head output (softplus → σ, floor ``min_sigma``).
        min_sigma: Numerical floor on σ.

    Returns:
        Mean NLL ``0.5·((err/σ)²) + log σ + 0.5·log 2π``.
    """
    sigma = nn.functional.softplus(log_sig.squeeze(-1)) + min_sigma
    err = v_pred.squeeze(-1) - v_gt
    nll = 0.5 * (err / sigma) ** 2 + torch.log(sigma) + 0.5 * math.log(2 * math.pi)
    return nll.mean()


def augment_synthetic_bike(
    x: torch.Tensor,
    pothole_prob: float = 0.2,
    engine_prob: float = 0.3,
    lean_prob: float = 0.2,
) -> torch.Tensor:
    """Synthetic bike robustness (F5 Plan B — no real bike data yet).

    Operates in NORMALIZED units (std ≈ 1). Amplitudes chosen to match
    observed IO-VNBD extremes (raw ±16 m/s²):

    - pothole: Gaussian pulse amp 2–8 on acc_z (ch 2), width 3–8 samples
      @100 Hz. Speed label unchanged — the model must learn to ignore.
    - engine: 30 Hz + 55 Hz sines amp 0.15–0.4 on gyro (ch 3–5). Models
      20–80 Hz bike engine harmonics vs the car's smoother spectrum.
    - lean: rotate acc y/z (ch 1,2) + gyro yaw/pitch (ch 3,4) by φ ±25°
      about forward X. Speed label unchanged.

    Args:
        x: (B, 200, 6) normalized windows.
        pothole_prob / engine_prob / lean_prob: Per-sample event rates.

    Returns:
        Augmented copy (non-3D or non-6ch input is returned unchanged).
    """
    if x.dim() != 3 or x.shape[2] != 6:
        return x
    B, T, _ = x.shape
    device = x.device
    xa = x.clone()
    t = torch.arange(T, device=device, dtype=x.dtype)  # 0..199 @100 Hz

    # --- pothole pulse on acc_z ---
    if pothole_prob > 0:
        m = torch.rand(B, device=device) < pothole_prob
        if bool(m.any()):
            center = torch.rand(B, device=device) * 160.0 + 20.0  # 20..180
            width = torch.rand(B, device=device) * 5.0 + 3.0      # 3..8 samples
            amp = torch.rand(B, device=device) * 6.0 + 2.0        # 2..8 std
            amp = amp * torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
            pulse = amp.unsqueeze(-1) * torch.exp(
                -0.5 * ((t.unsqueeze(0) - center.unsqueeze(-1)) / width.unsqueeze(-1)) ** 2
            )
            xa[m, :, 2] = xa[m, :, 2] + pulse[m]

    # --- engine harmonic on gyro ---
    if engine_prob > 0:
        m = torch.rand(B, device=device) < engine_prob
        if bool(m.any()):
            amp_e = torch.rand(B, device=device) * 0.25 + 0.15   # 0.15..0.4
            ph1 = torch.rand(B, device=device) * 2 * math.pi
            ph2 = torch.rand(B, device=device) * 2 * math.pi
            h = (
                torch.sin(2 * math.pi * 30.0 * t.unsqueeze(0) / 100.0 + ph1.unsqueeze(-1))
                + 0.5 * torch.sin(2 * math.pi * 55.0 * t.unsqueeze(0) / 100.0 + ph2.unsqueeze(-1))
            )
            h = amp_e.unsqueeze(-1) * h
            for c in (3, 4, 5):
                xa[m, :, c] = xa[m, :, c] + h[m]

    # --- lean rotation about forward X ---
    if lean_prob > 0:
        m = torch.rand(B, device=device) < lean_prob
        if bool(m.any()):
            phi = (torch.rand(B, device=device) * 2 - 1) * math.radians(25.0)
            cos_p, sin_p = torch.cos(phi), torch.sin(phi)
            ay = xa[:, :, 1].clone()
            az = xa[:, :, 2].clone()
            xa[:, :, 1] = torch.where(m.unsqueeze(-1), ay * cos_p.unsqueeze(-1) - az * sin_p.unsqueeze(-1), ay)
            xa[:, :, 2] = torch.where(m.unsqueeze(-1), ay * sin_p.unsqueeze(-1) + az * cos_p.unsqueeze(-1), az)
            gy = xa[:, :, 3].clone()
            gz = xa[:, :, 4].clone()
            xa[:, :, 3] = torch.where(m.unsqueeze(-1), gy * cos_p.unsqueeze(-1) - gz * sin_p.unsqueeze(-1), gy)
            xa[:, :, 4] = torch.where(m.unsqueeze(-1), gy * sin_p.unsqueeze(-1) + gz * cos_p.unsqueeze(-1), gz)
    return xa


def train_one_epoch(
    model,
    loader,
    optim,
    device,
    lambda_nll: float = 0.1,
    augment_yaw: bool = False,
    augment_bike: bool = False,
) -> tuple[float, float]:
    """One training epoch: MSE + λ·NLL, 50%-gated augmentations, grad clip.

    Args:
        model: AVNet/AVNetLite (any model returning the 5-tuple).
        loader: Yields ``(x (B,200,6), v (B,), att)``.
        optim: Optimizer (zero_grad + step handled here).
        device: torch device.
        lambda_nll: NLL weight (0 → plain MSE).
        augment_yaw / augment_bike: Enable per-batch 50%-gated augmentations.

    Returns:
        (mean combined loss, mean MSE) over the epoch.
    """
    model.train()
    total = total_mse = 0.0
    n = 0
    for x, v, _ in loader:
        x = x.to(device)
        v = v.to(device)
        if augment_yaw and random.random() < 0.5:
            x = augment_random_yaw(x)
        if augment_bike and random.random() < 0.5:
            x = augment_synthetic_bike(x)
        optim.zero_grad()
        v_pred, log_sig, _, _, _ = model(x)
        mse = nn.functional.mse_loss(v_pred.squeeze(-1), v)
        loss = (1 - lambda_nll) * mse + lambda_nll * gaussian_nll_loss(v_pred, v, log_sig) if lambda_nll > 0 else mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += loss.item() * len(x)
        total_mse += mse.item() * len(x)
        n += len(x)
    return total / n, total_mse / n


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    """Mean MSE over a loader (no augmentations, eval mode)."""
    model.eval()
    total = n = 0
    for x, v, _ in loader:
        x = x.to(device)
        v = v.to(device)
        v_pred, _, _, _, _ = model(x)
        total += nn.functional.mse_loss(v_pred.squeeze(-1), v, reduction="sum").item()
        n += len(x)
    return total / n
