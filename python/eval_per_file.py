#!/usr/bin/env python3
"""
eval_per_file.py — Option A per-file val audit (Colab-run, read-only).

Reproduces the preprocess 80/20 by-trajectory split (seed 26168), scores each
val file with the stage-1 checkpoint, and prints a sorted table + speed-binned
MSE to decide: stratified re-split vs label/capacity fix.

Usage (Colab T4, ~6-8 min, no repo writes except reports/per_file_val.csv):
  PYTHONPATH=. python python/eval_per_file.py --model experiments/checkpoints/model_avnet_stage1.p

Outputs:
  reports/per_file_val.csv — file,n,rmse,mean_v,p95_v,stat_frac,median_dt_ms,mse_0_5,mse_5_15,mse_gt15
  stdout — recombined val MSE sanity + top-3 share verdict (STRATIFY vs systemic)
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from python.core.signal import (
    find_column,
    gravity_align_linear,
    is_window_stationary,
    resample_uniform,
)
from python.models.avnet import AVNetLite
from python.preprocess import make_windows, stratified_split


def rebuild_split(base: str, train_ratio: float = 0.8, seed: int = 26168, split: str = "random") -> tuple[list, list]:
    """Reproduce the preprocess split (random default; stratified for ADR-010 branch A)."""
    s_files = sorted(glob.glob(str(Path(base) / "**/S-*.csv"), recursive=True))
    if split == "stratified":
        return stratified_split(s_files, train_ratio, seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(s_files))
    train_idx = set(perm[: int(len(s_files) * train_ratio)])
    train_files = [s_files[i] for i in range(len(s_files)) if i in train_idx]
    val_files = [s_files[i] for i in range(len(s_files)) if i not in train_idx]
    return train_files, val_files


def score_file(path, model, device, mean, std, hz=100, window=200, stride=10, batch=256):
    df = pd.read_csv(path, encoding="cp1252")
    df.columns = [c.strip() for c in df.columns]
    acc_c = [find_column(df, rf"accelerometer.*{a}") for a in "xyz"]
    grav_c = [find_column(df, rf"gravity.*{a}") for a in "xyz"]
    gyro_c = [
        find_column(df, r"gyroscope.*yaw"),
        find_column(df, r"gyroscope.*pitch"),
        find_column(df, r"gyroscope.*roll"),
    ]
    t_c = find_column(df, r"time since start") or "TIME SINCE START (ms)"
    v_c = find_column(df, r"gps speed") or "GPS SPEED (Kmh)"
    t_ms = df[t_c].values.astype(float)
    if np.any(np.diff(t_ms) <= 0):
        df = df.iloc[np.argsort(t_ms)].reset_index(drop=True)
        t_ms = df[t_c].values.astype(float)
    median_dt = float(np.median(np.diff(t_ms)))
    t_ns = (t_ms * 1e6).astype(np.int64)

    imu = np.concatenate(
        [
            gravity_align_linear(
                df[acc_c].values.astype(float), df[grav_c].values.astype(float)
            ),
            df[gyro_c].values.astype(float),
        ],
        axis=1,
    )
    _, imu_r = resample_uniform(t_ns, imu, hz)
    _, v_r = resample_uniform(
        t_ns,
        df[v_c].values.astype(float) if v_c in df.columns else np.zeros(len(df)),
        hz,
    )
    mask = np.isfinite(imu_r).all(axis=1) & np.isfinite(v_r)
    imu_r, v_r = imu_r[mask], v_r[mask] / 3.6
    if len(imu_r) < window:
        return None
    X = (imu_r - mean) / std
    W = make_windows(X.astype(np.float32), window, stride)
    v_lab = np.array([v_r[i * stride + window - 1] for i in range(len(W))])

    se_sum, n = 0.0, 0
    bins = {"0-5": [0.0, 0], "5-15": [0.0, 0], ">15": [0.0, 0]}
    stat = 0
    with torch.no_grad():
        for i in range(0, len(W), batch):
            xb = torch.from_numpy(np.array(W[i : i + batch], dtype=np.float32)).to(device)
            vp = model(xb)[0].squeeze(-1).float().cpu().numpy()
            vt = v_lab[i : i + batch]
            se_sum += float((((vp - vt) ** 2).sum()))
            n += len(vt)
            for p_, t_ in zip(vp, vt):
                key = "0-5" if t_ < 5 else "5-15" if t_ < 15 else ">15"
                bins[key][0] += float((p_ - t_) ** 2)
                bins[key][1] += 1
            stat += sum(
                1 for w, t_ in zip(W[i : i + batch], vt) if is_window_stationary(w) and t_ < 0.5
            )
    mse = se_sum / n
    return {
        "file": path,
        "n": n,
        "mse": mse,
        "rmse": mse**0.5,
        "mean_v": float(v_lab.mean()),
        "p95_v": float(np.percentile(v_lab, 95)),
        "stat_frac": stat / n,
        "median_dt_ms": median_dt,
        "mse_0_5": bins["0-5"][0] / max(bins["0-5"][1], 1),
        "mse_5_15": bins["5-15"][0] / max(bins["5-15"][1], 1),
        "mse_gt15": bins[">15"][0] / max(bins[">15"][1], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p")
    ap.add_argument("--base", default="data/iovnbd/Synchronised V abd S datasets/Categorised IOVNB Dataset")
    ap.add_argument("--scaler", default="python/scaler.json")
    ap.add_argument("--out", default="reports/per_file_val.csv")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--split", choices=["random", "stratified"], default="random",
                    help="must match the preprocess --split used for this checkpoint")
    args = ap.parse_args()

    _, val_files = rebuild_split(args.base, split=args.split)
    print(f"[audit] val files ({len(val_files)}):")
    for f in val_files:
        print(f"  {Path(f).parent.name}/{Path(f).name}")

    sc = json.load(open(args.scaler))
    mean, std = np.array(sc["mean"]), np.array(sc["std"])
    device = torch.device(args.device)
    model = AVNetLite().to(device).eval()
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"[model] params {sum(p.numel() for p in model.parameters()):,} on {device}")

    rows = []
    for f in val_files:
        try:
            r = score_file(f, model, device, mean, std, batch=args.batch)
            if r is None:
                print(f"[skip] {f} no windows")
                continue
            rows.append(r)
            short = f"{Path(f).parent.name}/{Path(f).name}"
            print(
                f"{short}: n={r['n']} RMSE={r['rmse']:.2f} mean_v={r['mean_v']:.1f} "
                f"p95={r['p95_v']:.1f} stat={r['stat_frac']:.1%} dt={r['median_dt_ms']:.0f}ms | "
                f"0-5={r['mse_0_5']**0.5:.2f} 5-15={r['mse_5_15']**0.5:.2f} >15={r['mse_gt15']**0.5:.2f}"
            )
        except Exception as e:
            print(f"[err] {f}: {type(e).__name__}: {e}")

    tot_n = sum(r["n"] for r in rows)
    recomb = sum(r["n"] * r["mse"] for r in rows) / tot_n
    print(f"\nrecombined val MSE {recomb:.4f} (sanity vs train-log best ~1.729)")
    rows.sort(key=lambda r: r["n"] * r["mse"], reverse=True)
    top3 = sum(r["n"] * r["mse"] for r in rows[:3]) / sum(r["n"] * r["mse"] for r in rows)
    print(f"top-3 files share of weighted MSE: {top3:.0%} → {'STRATIFY' if top3 > 0.6 else 'systemic (labels/capacity)'}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] {args.out}")


if __name__ == "__main__":
    main()
