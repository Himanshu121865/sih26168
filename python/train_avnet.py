#!/usr/bin/env python3
"""Stage-1 training: AVNetLite on IO-VNBD phone-data windows.

Loop, augmentations, and NLL live in python/core/training.py (single
implementation); this file is CLI wiring only.

Usage:
    python python/train_avnet.py --epochs 5 --batch 64 --lr 1e-3            # smoke
    python python/train_avnet.py --epochs 50 --batch 128 --lr 1e-3 --device cuda \\
        --augment-yaw --lambda-nll 0.1
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from loguru import logger
from torch.utils.data import DataLoader

from python.config import PROCESSED_DIR
from python.core.runlog import init_runlog
from python.core.training import eval_loss, set_seed, train_one_epoch
from python.datasets.iovnbd_dataset import IOVNBDWindowDataset
from python.models.avnet import AVNetLite, count_params


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--train-windows", default=str(PROCESSED_DIR / "train_windows.npy"))
    ap.add_argument("--train-v", default=str(PROCESSED_DIR / "train_v.npy"))
    ap.add_argument("--val-windows", default=str(PROCESSED_DIR / "val_windows.npy"))
    ap.add_argument("--val-v", default=str(PROCESSED_DIR / "val_v.npy"))
    ap.add_argument("--out", default="experiments/checkpoints/model_avnet_stage1.p")
    ap.add_argument("--lambda-nll", type=float, default=0.1, help="NLL weight, 0 = MSE only")
    ap.add_argument("--augment-yaw", action="store_true", help="random yaw rotation (heading-agnostic)")
    ap.add_argument("--augment-bike", action="store_true",
                    help="synthetic bike robustness: pothole 20%% + engine 30%% + lean ±25° (F5)")
    ap.add_argument("--log-dir", default=None, help="optional dir for a run log file")
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args()


def main() -> None:
    """CLI entry: load data, train, checkpoint on best val MSE."""
    args = parse_args()
    init_runlog("train", args.log_dir)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("[warn] cuda not available, using cpu")
    logger.info(
        f"[train] device {device} epochs {args.epochs} batch {args.batch} lr {args.lr} "
        f"augment_yaw={args.augment_yaw} augment_bike={args.augment_bike} lambda_nll={args.lambda_nll}"
    )

    train_ds = IOVNBDWindowDataset(args.train_windows, args.train_v)
    val_ds = IOVNBDWindowDataset(args.val_windows, args.val_v)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    logger.info(f"[data] train {len(train_ds)} val {len(val_ds)}")

    model = AVNetLite().to(device)
    logger.info(f"[model] params {count_params(model):,}")

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=5, factor=0.5)

    best_val = float("inf")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_mse = train_one_epoch(
            model, train_loader, optim, device,
            lambda_nll=args.lambda_nll,
            augment_yaw=args.augment_yaw,
            augment_bike=args.augment_bike,
        )
        val_mse = eval_loss(model, val_loader, device)
        sched.step(val_mse)
        dt = time.time() - t0
        logger.info(
            f"[epoch {epoch}/{args.epochs}] train loss {tr_loss:.4f} (mse {tr_mse:.4f}) "
            f"val MSE {val_mse:.4f} lr {optim.param_groups[0]['lr']:.2e} {dt:.1f}s"
        )
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), out_path)
            logger.info(f"  [save] {out_path} best {best_val:.4f}")

    logger.info(f"[done] best val {best_val:.4f} saved to {out_path}")


if __name__ == "__main__":
    main()
