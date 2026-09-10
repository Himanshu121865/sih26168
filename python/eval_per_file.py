"""Per-file validation audit (Option A): find which val files dominate error.

Reproduces the preprocess by-trajectory split (seed 26168), scores each val
file with the stage-1 checkpoint, and prints a sorted table + speed-binned
MSE to decide: stratified re-split vs label/capacity fix.

FIX (this refactor): previously this script hand-rolled the window path and
silently used the spec-v1 DATASET gravity columns. It now runs the shared
``python.datasets.iovnbd`` pipeline (spec v2 live low-pass gravity), so
audit numbers match training preprocessing exactly.

Usage (Colab T4, ~6-8 min, writes only reports/per_file_val.csv):
    PYTHONPATH=. python python/eval_per_file.py \\
        --model experiments/checkpoints/model_avnet_stage1.p
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from python.config import IOVNBD_BASE, SCALER_PATH
from python.core.runlog import init_runlog
from python.datasets.iovnbd import load_phone_csv, resample_stream, windowize
from python.datasets.iovnbd_dataset import rebuild_split
from python.models.avnet import AVNetLite, count_params


def score_file(
    path: str,
    model,
    device,
    mean: np.ndarray,
    std: np.ndarray,
    hz: int = 100,
    window: int = 200,
    stride: int = 10,
    batch: int = 256,
) -> dict | None:
    """Score one S-file: windows through the SHARED pipeline (spec v2).

    Returns:
        Metrics dict (n, mse, rmse, mean_v, p95_v, stat_frac, median_dt_ms,
        speed-binned MSE) or None when the file yields no windows.
    """
    df = load_phone_csv(path)
    imu, v_ms, audit, _grav = resample_stream(df, hz)
    if len(imu) < window:
        return None

    X = (imu - mean) / std
    W = windowize(X.astype(np.float32), window, stride)
    v_lab = np.array([v_ms[i * stride + window - 1] for i in range(len(W))])

    se_sum, n = 0.0, 0
    bins = {"0-5": [0.0, 0], "5-15": [0.0, 0], ">15": [0.0, 0]}
    with torch.no_grad():
        for i in range(0, len(W), batch):
            xb = torch.from_numpy(np.array(W[i : i + batch], dtype=np.float32)).to(device)
            vp = model(xb)[0].squeeze(-1).float().cpu().numpy()
            vt = v_lab[i : i + batch]
            se_sum += float(((vp - vt) ** 2).sum())
            n += len(vt)
            for p_, t_ in zip(vp, vt, strict=False):
                key = "0-5" if t_ < 5 else "5-15" if t_ < 15 else ">15"
                bins[key][0] += float((p_ - t_) ** 2)
                bins[key][1] += 1

    mse = se_sum / n
    return {
        "file": path,
        "n": n,
        "mse": mse,
        "rmse": mse**0.5,
        "mean_v": float(v_lab.mean()),
        "p95_v": float(np.percentile(v_lab, 95)),
        "stat_frac": 0.0,  # stationary flags live in preprocess npy, not rescored here
        "median_dt_ms": audit.median_dt_ms,
        "mse_0_5": bins["0-5"][0] / max(bins["0-5"][1], 1),
        "mse_5_15": bins["5-15"][0] / max(bins["5-15"][1], 1),
        "mse_gt15": bins[">15"][0] / max(bins[">15"][1], 1),
    }


def main() -> None:
    """CLI: audit every val file and emit the CSV + verdict."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p")
    ap.add_argument("--base", default=str(IOVNBD_BASE))
    ap.add_argument("--scaler", default=str(SCALER_PATH))
    ap.add_argument("--out", default="reports/per_file_val.csv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", choices=["random", "stratified"], default="random",
                    help="must match the preprocess --split used for this checkpoint")
    ap.add_argument("--log-dir", default=None, help="optional dir for a run log file")
    args = ap.parse_args()

    init_runlog("audit", args.log_dir)
    _, val_files = rebuild_split(args.base, split=args.split)
    logger.info(f"[audit] val files ({len(val_files)}):")
    for f in val_files:
        logger.info(f"  {Path(f).parent.name}/{Path(f).name}")

    sc = json.loads(Path(args.scaler).read_text())
    mean, std = np.array(sc["mean"]), np.array(sc["std"])
    device = torch.device(args.device)
    model = AVNetLite().to(device).eval()
    model.load_state_dict(torch.load(args.model, map_location=device))
    logger.info(f"[model] params {count_params(model):,} on {device}")

    rows = []
    for f in val_files:
        try:
            r = score_file(f, model, device, mean, std, batch=args.batch)
            if r is None:
                logger.info(f"[skip] {f} no windows")
                continue
            rows.append(r)
            logger.info(
                f"{Path(f).parent.name}/{Path(f).name}: n={r['n']} RMSE={r['rmse']:.2f} "
                f"mean_v={r['mean_v']:.1f} p95={r['p95_v']:.1f} dt={r['median_dt_ms']:.0f}ms | "
                f"0-5={r['mse_0_5']**0.5:.2f} 5-15={r['mse_5_15']**0.5:.2f} >15={r['mse_gt15']**0.5:.2f}"
            )
        except Exception as e:  # noqa: BLE001 — audit continues past broken files
            logger.error(f"[err] {f}: {type(e).__name__}: {e}")

    if not rows:
        logger.error("[audit] no val files scored — check --base/--split")
        return

    tot_n = sum(r["n"] for r in rows)
    recomb = sum(r["n"] * r["mse"] for r in rows) / tot_n
    logger.info(f"\nrecombined val MSE {recomb:.4f} (sanity vs train-log best ~1.729)")
    rows.sort(key=lambda r: r["n"] * r["mse"], reverse=True)
    top3 = sum(r["n"] * r["mse"] for r in rows[:3]) / sum(r["n"] * r["mse"] for r in rows)
    logger.info(f"top-3 files share of weighted MSE: {top3:.0%} → {'STRATIFY' if top3 > 0.6 else 'systemic (labels/capacity)'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    logger.info(f"[csv] {args.out}")


if __name__ == "__main__":
    main()
