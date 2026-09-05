#!/usr/bin/env python3
"""
train_avnet.py — Step 6 / Stage-1: Train AVNetLite on IO-VNBD phone data
P0 #3 FIXED: random-yaw augmentation + joint NLL cov (harsh tcn.py:133-193)

Usage:
  python python/train_avnet.py --epochs 5 --batch 64 --lr 1e-3           # smoke
  python python/train_avnet.py --epochs 50 --batch 128 --lr 1e-3 --device cuda --augment-yaw --lambda-nll 0.1
"""
import argparse, os, json, time, math, random
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from python.datasets.iovnbd_dataset import IOVNBDWindowDataset
from python.models.avnet import AVNetLite

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda"])
    ap.add_argument("--train-windows", default="data/processed/train_windows.npy")
    ap.add_argument("--train-v", default="data/processed/train_v.npy")
    ap.add_argument("--val-windows", default="data/processed/val_windows.npy")
    ap.add_argument("--val-v", default="data/processed/val_v.npy")
    ap.add_argument("--out", default="experiments/checkpoints/model_avnet_stage1.p")
    ap.add_argument("--lambda-nll", type=float, default=0.1, help="NLL weight, 0= MSE only")
    ap.add_argument("--augment-yaw", action="store_true", help="random yaw rotation (heading-agnostic)")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def augment_random_yaw(x: torch.Tensor):
    """
    Rotate horizontal acc x,y (ch 0,1) and gyro pitch/roll (ch 4,5) by random theta.
    x: (B,200,6) with 6: acc x,y,z (0,1,2) + gyro yaw,pitch,roll (3,4,5)
    keep acc_z (2) and gyro yaw (3) invariant (vertical).
    """
    B = x.shape[0]
    thetas = torch.rand(B, device=x.device) * 2 * math.pi  # (B,)
    cos_t = torch.cos(thetas)
    sin_t = torch.sin(thetas)
    # copy
    x_aug = x.clone()
    # rotate acc x,y
    ax = x[:, :, 0]
    ay = x[:, :, 1]
    x_aug[:, :, 0] = ax * cos_t.unsqueeze(-1) - ay * sin_t.unsqueeze(-1)
    x_aug[:, :, 1] = ax * sin_t.unsqueeze(-1) + ay * cos_t.unsqueeze(-1)
    # rotate gyro pitch (4=Y) and roll (5=X) — horizontal gyro
    # Note: gyro yaw (3=Z) is vertical, keep
    gx = x[:, :, 5]  # roll X
    gy = x[:, :, 4]  # pitch Y
    x_aug[:, :, 5] = gx * cos_t.unsqueeze(-1) - gy * sin_t.unsqueeze(-1)
    x_aug[:, :, 4] = gx * sin_t.unsqueeze(-1) + gy * cos_t.unsqueeze(-1)
    return x_aug

def gaussian_nll_loss(v_pred, v_gt, log_sig, min_sigma=1e-3):
    """
    Gaussian NLL for scalar velocity: N(v_gt | v_pred, sigma^2), sigma=exp(log_sig)
    NLL = 0.5* ((v_pred - v_gt)/sigma)^2 + log sigma + 0.5*log2pi
    Clamp sigma >= min_sigma via softplus as in harsh tcn.py:155
    """
    # softplus to ensure positive sigma, plus floor
    sigma = torch.nn.functional.softplus(log_sig.squeeze(-1)) + min_sigma  # (B,)
    # actually log_sig is direct log sigma, but we use softplus on it for floor
    # simpler: sigma = exp(log_sig) + min_sigma
    # use exp for direct
    # sigma = torch.exp(log_sig.squeeze(-1)) + min_sigma
    err = v_pred.squeeze(-1) - v_gt
    nll = 0.5 * (err / sigma) ** 2 + torch.log(sigma) + 0.5 * math.log(2*math.pi)
    return nll.mean()

def train_one_epoch(model, loader, optim, device, lambda_nll=0.1, augment_yaw=False):
    model.train()
    total_loss = 0
    total_mse = 0
    total_n = 0
    for x, v, att in loader:
        x = x.to(device)  # (B,200,6)
        v = v.to(device)  # (B,)
        if augment_yaw and random.random() < 0.5:
            x = augment_random_yaw(x)
        optim.zero_grad()
        v_pred, log_sig_v, att_pred, log_sig_att, _ = model(x)
        v_pred_s = v_pred.squeeze(-1)  # (B,)
        mse = nn.functional.mse_loss(v_pred_s, v)
        if lambda_nll > 0:
            nll = gaussian_nll_loss(v_pred, v, log_sig_v)
            loss = (1 - lambda_nll) * mse + lambda_nll * nll
        else:
            loss = mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item() * len(x)
        total_mse += mse.item() * len(x)
        total_n += len(x)
    return total_loss / total_n, total_mse / total_n

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total_mse = 0
    total_n = 0
    for x, v, att in loader:
        x = x.to(device); v = v.to(device)
        v_pred, _, _, _, _ = model(x)
        v_pred = v_pred.squeeze(-1)
        mse = nn.functional.mse_loss(v_pred, v)
        total_mse += mse.item() * len(x)
        total_n += len(x)
    return total_mse / total_n

def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    if args.device=="cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available, using cpu")
        device = torch.device("cpu")
    print(f"[train] device {device} epochs {args.epochs} batch {args.batch} lr {args.lr} augment_yaw={args.augment_yaw} lambda_nll={args.lambda_nll}")

    train_ds = IOVNBDWindowDataset(args.train_windows, args.train_v)
    val_ds = IOVNBDWindowDataset(args.val_windows, args.val_v)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"[data] train {len(train_ds)} val {len(val_ds)}")

    model = AVNetLite().to(device)
    print(f"[model] params {sum(p.numel() for p in model.parameters()):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)

    best_val = float("inf")
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss, tr_mse = train_one_epoch(model, train_loader, optim, device, args.lambda_nll, args.augment_yaw)
        val_mse = eval_loss(model, val_loader, device)
        sched.step(val_mse)
        dt = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] train loss {tr_loss:.4f} (mse {tr_mse:.4f}) val MSE {val_mse:.4f} lr {optim.param_groups[0]['lr']:.2e} {dt:.1f}s")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), out_path)
            print(f"  [save] {out_path} best {best_val:.4f}")

    print(f"[done] best val {best_val:.4f} saved to {out_path}")

if __name__ == "__main__":
    main()
