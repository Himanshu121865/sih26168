#!/usr/bin/env python3
"""Drift evaluation CLI (Step 11.4) — produces reports/drift_plot.png.

All logic lives in python/eval/drift.py; this is CLI wiring. Modes (F7):

- ``--mode 1d`` (default) — legacy screening demo (byte-identical): naive
  red, AVNet blue, map-placeholder green. THE plot for the proposal PPT.
- ``--mode 2d`` — real 2D ATE/RTE, optional ``--gps-track`` CSV.

Usage:
    python python/eval_drift.py --model experiments/checkpoints/model_avnet_stage1.p --plot reports/drift_plot.png
    python python/eval_drift.py --mode 2d --model experiments/checkpoints/model_avnet_stage1.p --plot reports/drift_2d.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from loguru import logger

from python.config import PROCESSED_DIR, SCALER_PATH
from python.core.runlog import init_runlog
from python.eval.drift import eval_1d, eval_2d, eval_mse


def main() -> None:
    """CLI entry: optional val-MSE report, then the 1d/2d plot."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="path to model_avnet_stage1.p, or none for demo")
    ap.add_argument("--plot", default="reports/drift_plot.png")
    ap.add_argument("--val-windows", default=str(PROCESSED_DIR / "val_windows.npy"))
    ap.add_argument("--val-v", default=str(PROCESSED_DIR / "val_v.npy"))
    ap.add_argument("--mode", choices=["1d", "2d"], default="1d",
                    help="1d=legacy screening demo (default); 2d=real ATE/RTE, no fake map")
    ap.add_argument("--gps-track", default=None, help="optional CSV with lat/lon for true 2D GT")
    ap.add_argument("--scaler", default=str(SCALER_PATH))
    ap.add_argument("--log-dir", default=None, help="optional dir for a run log file")
    args = ap.parse_args()

    init_runlog(f"eval-{args.mode}", args.log_dir)

    if args.model and Path(args.model).exists():
        from torch.utils.data import DataLoader

        from python.datasets.iovnbd_dataset import IOVNBDWindowDataset
        from python.models.avnet import AVNetLite

        ds = IOVNBDWindowDataset(args.val_windows, args.val_v)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        model = AVNetLite().to("cpu")
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
        mse = eval_mse(model, loader, torch.device("cpu"))
        logger.info(f"[mse] val MSE {mse:.4f} RMSE {mse**0.5:.4f} m/s")

    if args.mode == "2d":
        eval_2d(args.val_v, args.model, args.plot, args.gps_track,
               args.val_windows, args.scaler)
    else:
        eval_1d(args.val_v, args.model, args.plot, args.val_windows)


if __name__ == "__main__":
    main()
