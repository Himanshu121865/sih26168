#!/usr/bin/env python3
"""
train_avnet.py — Step 6 / Stage-1: Train AVNetLite on IO-VNBD phone data

Usage:
  python python/train_avnet.py --epochs 5 --batch 64 --lr 1e-3           # smoke (5 epochs, 164k windows)
  python python/train_avnet.py --epochs 50 --batch 128 --lr 1e-3 --device cuda

Saves: experiments/checkpoints/model_avnet_stage1.p + runs/ TB logs
"""
import argparse, os, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from loguru import logger

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
    ap.add_argument("--lambda-nll", type=float, default=0.1, help="NLL weight (unused if no sigma supervision)")
    return ap.parse_args()

def train_one_epoch(model, loader, optim, device):
    model.train()
    total_loss = 0
    total_n = 0
    for x, v, att in loader:
        x = x.to(device)  # (B,200,6)
        v = v.to(device)  # (B,)
        # att zeros for now
        optim.zero_grad()
        v_pred, log_sig_v, att_pred, log_sig_att, _ = model(x)
        v_pred = v_pred.squeeze(-1)  # (B,)
        # MSE loss for vel
        mse = nn.functional.mse_loss(v_pred, v)
        # NLL: if we have log_sig, NLL = 0.5*exp(-log_sig)*mse + 0.5*log_sig ; but we train mse only for now
        loss = mse
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total_loss += loss.item() * len(x)
        total_n += len(x)
    return total_loss / total_n

@torch.no_grad()
def eval_loss(model, loader, device):
    model.eval()
    total_loss = 0
    total_n = 0
    for x, v, att in loader:
        x = x.to(device); v = v.to(device)
        v_pred, _, _, _, _ = model(x)
        v_pred = v_pred.squeeze(-1)
        mse = nn.functional.mse_loss(v_pred, v)
        total_loss += mse.item() * len(x)
        total_n += len(x)
    return total_loss / total_n

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or args.device=="cpu" else "cpu")
    if args.device=="cuda" and not torch.cuda.is_available():
        print("[warn] cuda not available, using cpu")
        device = torch.device("cpu")
    print(f"[train] device {device} epochs {args.epochs} batch {args.batch} lr {args.lr}")

    # datasets
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
        tr_loss = train_one_epoch(model, train_loader, optim, device)
        val_loss = eval_loss(model, val_loader, device)
        sched.step(val_loss)
        dt = time.time() - t0
        print(f"[epoch {epoch}/{args.epochs}] train MSE {tr_loss:.4f} val MSE {val_loss:.4f} lr {optim.param_groups[0]['lr']:.2e} {dt:.1f}s")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), out_path)
            print(f"  [save] {out_path} best {best_val:.4f}")
    print(f"[done] best val {best_val:.4f} saved to {out_path}")

if __name__ == "__main__":
    main()
