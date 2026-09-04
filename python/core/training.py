"""core/training.py — Shared train loop + NLL + yaw augment (extracted from train_avnet.py)."""
import math, random
import torch
import torch.nn as nn

def set_seed(seed=42):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    import numpy as np; np.random.seed(seed)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def augment_random_yaw(x: torch.Tensor):
    """Rotate horizontal acc x,y (0,1) and gyro pitch/roll (4,5). x (B,200,6)"""
    B=x.shape[0]
    thetas=torch.rand(B, device=x.device)*2*math.pi
    cos_t=torch.cos(thetas); sin_t=torch.sin(thetas)
    xa=x.clone()
    ax=x[:,:,0]; ay=x[:,:,1]
    xa[:,:,0]=ax*cos_t.unsqueeze(-1) - ay*sin_t.unsqueeze(-1)
    xa[:,:,1]=ax*sin_t.unsqueeze(-1) + ay*cos_t.unsqueeze(-1)
    gx=x[:,:,5]; gy=x[:,:,4]
    xa[:,:,5]=gx*cos_t.unsqueeze(-1) - gy*sin_t.unsqueeze(-1)
    xa[:,:,4]=gx*sin_t.unsqueeze(-1) + gy*cos_t.unsqueeze(-1)
    return xa

def gaussian_nll_loss(v_pred, v_gt, log_sig, min_sigma=1e-3):
    sigma=torch.nn.functional.softplus(log_sig.squeeze(-1))+min_sigma
    err=v_pred.squeeze(-1)-v_gt
    nll=0.5*(err/sigma)**2 + torch.log(sigma) + 0.5*math.log(2*math.pi)
    return nll.mean()

def augment_synthetic_bike(x: torch.Tensor, pothole_prob=0.2, engine_prob=0.3,
                             lean_prob=0.2):
    """Synthetic bike robustness (F5, Plan B — no real bike data yet).

    Operates in NORMALIZED units (std≈1). Amplitudes chosen to match observed
    IO-VNBD extremes (raw ±16 m/s² → normalized ±16):
    - pothole: Gaussian pulse amp 2-8 on acc_z (ch2), width 3-8 samples @100Hz.
      Models 2-8 m/s² vertical spikes (broken road). Speed label unchanged —
      model must learn to ignore.
    - engine: 30Hz + 55Hz sine amp 0.15-0.4 on gyro (ch3-5). Models 20-80Hz
      bike engine harmonics vs car smoother spectrum.
    - lean: rotate acc y/z (ch1,2) + gyro yaw/pitch (ch3,4) by φ ±25° about
      forward X. Models bike roll; speed label unchanged.
    x: (B,200,6) normalized. Returns augmented copy.
    """
    if x.dim() != 3 or x.shape[2] != 6:
        return x
    B, T, _ = x.shape
    device = x.device
    xa = x.clone()
    t = torch.arange(T, device=device, dtype=x.dtype)  # 0..199 @100Hz

    # --- pothole pulse on acc_z ---
    if pothole_prob > 0:
        m = torch.rand(B, device=device) < pothole_prob
        if bool(m.any()):
            center = torch.rand(B, device=device) * 160.0 + 20.0  # 20..180
            width = torch.rand(B, device=device) * 5.0 + 3.0      # 3..8 samples
            amp = (torch.rand(B, device=device) * 6.0 + 2.0)      # 2..8 std
            amp = amp * torch.where(torch.rand(B, device=device) < 0.5, -1.0, 1.0)
            pulse = amp.unsqueeze(-1) * torch.exp(
                -0.5 * ((t.unsqueeze(0) - center.unsqueeze(-1)) / width.unsqueeze(-1)) ** 2)
            xa[m, :, 2] = xa[m, :, 2] + pulse[m]

    # --- engine harmonic on gyro ---
    if engine_prob > 0:
        m = torch.rand(B, device=device) < engine_prob
        if bool(m.any()):
            amp_e = torch.rand(B, device=device) * 0.25 + 0.15   # 0.15..0.4
            ph1 = torch.rand(B, device=device) * 2 * math.pi
            ph2 = torch.rand(B, device=device) * 2 * math.pi
            h = (torch.sin(2 * math.pi * 30.0 * t.unsqueeze(0) / 100.0 + ph1.unsqueeze(-1))
                 + 0.5 * torch.sin(2 * math.pi * 55.0 * t.unsqueeze(0) / 100.0 + ph2.unsqueeze(-1)))
            h = amp_e.unsqueeze(-1) * h
            for c in (3, 4, 5):
                xa[m, :, c] = xa[m, :, c] + h[m]

    # --- lean rotation about forward X ---
    if lean_prob > 0:
        m = torch.rand(B, device=device) < lean_prob
        if bool(m.any()):
            phi = (torch.rand(B, device=device) * 2 - 1) * math.radians(25.0)
            cos_p = torch.cos(phi); sin_p = torch.sin(phi)
            # acc y/z
            ay = xa[:, :, 1].clone(); az = xa[:, :, 2].clone()
            xa[:, :, 1] = torch.where(m.unsqueeze(-1), ay * cos_p.unsqueeze(-1) - az * sin_p.unsqueeze(-1), ay)
            xa[:, :, 2] = torch.where(m.unsqueeze(-1), ay * sin_p.unsqueeze(-1) + az * cos_p.unsqueeze(-1), az)
            # gyro yaw(Z ch3)/pitch(Y ch4) mix under roll
            gy = xa[:, :, 3].clone(); gz = xa[:, :, 4].clone()
            xa[:, :, 3] = torch.where(m.unsqueeze(-1), gy * cos_p.unsqueeze(-1) - gz * sin_p.unsqueeze(-1), gy)
            xa[:, :, 4] = torch.where(m.unsqueeze(-1), gy * sin_p.unsqueeze(-1) + gz * cos_p.unsqueeze(-1), gz)
    return xa

def train_one_epoch(model, loader, optim, device, lambda_nll=0.1, augment_yaw=False,
                    augment_bike=False):
    model.train()
    total=total_mse=0; n=0
    for x,v,_ in loader:
        x=x.to(device); v=v.to(device)
        if augment_yaw and random.random()<0.5:
            x=augment_random_yaw(x)
        if augment_bike:
            x=augment_synthetic_bike(x)
        optim.zero_grad()
        v_pred, log_sig,_ ,_ ,_=model(x)
        mse=nn.functional.mse_loss(v_pred.squeeze(-1), v)
        loss=(1-lambda_nll)*mse + lambda_nll*gaussian_nll_loss(v_pred, v, log_sig) if lambda_nll>0 else mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optim.step()
        total+=loss.item()*len(x); total_mse+=mse.item()*len(x); n+=len(x)
    return total/n, total_mse/n

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total=n=0
    for x,v,_ in loader:
        x=x.to(device); v=v.to(device)
        v_pred,_,_,_,_=model(x)
        total+=nn.functional.mse_loss(v_pred.squeeze(-1), v, reduction="sum").item()
        n+=len(x)
    return total/n
