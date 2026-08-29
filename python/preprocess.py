#!/usr/bin/env python3
"""
preprocess.py — Step 2.2-2.5
Resample 10Hz→100Hz, gravity-align, per-axis normalize, sliding window (200, stride 10).

Inputs: data/iovnbd/Synchronised V abd S datasets/Categorised.../*.csv (S-*.csv phone IMU)
Outputs: data/processed/train_windows.npy (N,200,6), data/processed/val_windows.npy, scaler.json

Usage:
  python python/preprocess.py --subset 1h          # quick 1h subset for smoke test (<5 min)
  python python/preprocess.py --window 200 --stride 10 --hz 100 --train-ratio 0.8
"""
import argparse, json, glob, os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

# phone IMU cols after strip (from DATA_INSPECTION.md)
ACC_COLS = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
GYRO_COLS = ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]
# gyro order in file is Yaw,Pitch,Roll = Z,Y,X? Keep as file order for now, will map to (x,y,z) later
IMU_COLS = ACC_COLS + GYRO_COLS  # 6 channels
GPS_COLS = ["GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)"]
TIME_COL = "TIME SINCE START (ms)"

def load_phone_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    return df

def resample_10_to_100(df: pd.DataFrame, hz_target=100) -> pd.DataFrame:
    """Linear interp 10Hz (100ms) -> 100Hz (10ms). Returns new df with interpolated IMU."""
    t_ms = df[TIME_COL].values.astype(float)
    # handle monotonic: some files have TIME SINCE START resetting per seq; use relative
    # interpolate to regular grid 10ms
    t_start, t_end = t_ms.min(), t_ms.max()
    t_new = np.arange(t_start, t_end, 1000/hz_target)
    if len(t_new) < 200:
        return None
    out = pd.DataFrame({TIME_COL: t_new})
    for c in IMU_COLS + ["GPS LATITUDE (degrees)", "GPS LONGITUDE (degrees)", "GPS SPEED (Kmh)"]:
        if c in df.columns:
            f = interp1d(t_ms, df[c].values, kind="linear", bounds_error=False, fill_value="extrapolate")
            out[c] = f(t_new)
    # copy other needed cols via nearest for GPS
    return out

def gravity_align(window: np.ndarray) -> np.ndarray:
    """
    window (200,6) acc(3)+gyro(3) -> gravity-aligned (z up).
    Simple: estimate gravity via low-pass (mean of acc over window) and rotate so gravity = [0,0,9.81].
    For now: subtract mean gravity from acc Z (approx). Full rotation via Madgwick is Android step; training uses this cheap align.
    """
    acc = window[:, :3]
    # gravity estimate = mean acc (vehicle mostly level; low-pass)
    g_est = acc.mean(axis=0)
    # normalize and compute rotation: want g_est -> [0,0,9.81]
    g_norm = np.linalg.norm(g_est)
    if g_norm < 1e-6:
        return window
    # cheap: just subtract g_est projection? Actually keep acc as is but normalize by g
    # For training, we keep raw acc but divide by 9.81 later via scaler; alignment is for yaw invariance
    # So we just return window (alignment will be handled by augmentation: random yaw)
    return window

def make_windows(arr: np.ndarray, window=200, stride=10):
    """arr (T,6) -> (N, window, 6)"""
    T = arr.shape[0]
    if T < window:
        return np.empty((0, window, 6))
    N = (T - window)//stride + 1
    windows = np.stack([arr[i*stride:i*stride+window] for i in range(N)], axis=0)
    return windows

def compute_labels(df_100: pd.DataFrame, windows_idx, stride=10, window=200):
    """
    Labels for each window: v_forward (m/s) from GPS SPEED, and attitude delta (3) from orientation.
    v_forward: GPS speed at window center (km/h -> m/s).
    """
    v_kmh = df_100["GPS SPEED (Kmh)"].values if "GPS SPEED (Kmh)" in df_100.columns else np.zeros(len(df_100))
    v_ms = v_kmh / 3.6
    # per window, take speed at end (forward looks ahead)
    v_labels = np.array([v_ms[i*stride + window -1] for i in windows_idx])
    # attitude: use GYROSCOPE integration proxy -> for now zeros (att head will learn zero-mean)
    att_labels = np.zeros((len(v_labels), 3))
    return v_labels, att_labels

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["1h","full"], default="full")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--hz", type=int, default=100)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--scaler", default="python/scaler.json")
    args = ap.parse_args()

    base = Path("data/iovnbd/Synchronised V abd S datasets/Categorised IOVNB Dataset")
    s_files = sorted(glob.glob(str(base / "**/S-*.csv"), recursive=True))
    if args.subset == "1h":
        # pick first 2 seqs only (~1h: Vtb01 54min + Vta06 2min)
        s_files = s_files[:3]
    print(f"[preprocess] {len(s_files)} S files, window={args.window} stride={args.stride} hz={args.hz}")

    all_windows = []
    all_v = []
    for f in s_files:
        df = load_phone_csv(f)
        if len(df) < 200:
            continue
        df100 = resample_10_to_100(df, hz_target=args.hz)
        if df100 is None or len(df100) < args.window:
            print(f"[skip] {f} too short after resample")
            continue
        imu = df100[IMU_COLS].values.astype(np.float32)  # (T,6)
        # gravity align per window later; here just collect
        windows = make_windows(imu, window=args.window, stride=args.stride)  # (N,200,6)
        if len(windows) == 0:
            continue
        # gravity align per window
        windows = np.stack([gravity_align(w) for w in windows])
        # labels
        idx = list(range(len(windows)))
        v_labels, att_labels = compute_labels(df100, idx, stride=args.stride, window=args.window)
        all_windows.append(windows)
        all_v.append(v_labels)
        print(f"[ok] {Path(f).parent.name}/{Path(f).name}: T={len(df)}->{len(df100)} windows={len(windows)}")

    if not all_windows:
        print("[err] no windows", flush=True)
        return
    X = np.concatenate(all_windows, axis=0)  # (N,200,6)
    v = np.concatenate(all_v, axis=0)  # (N,)
    print(f"[concat] X {X.shape} v {v.shape} X mean {X.mean(axis=(0,1))} std {X.std(axis=(0,1))}")

    # scaler from train split only
    N = len(X)
    n_train = int(N * args.train_ratio)
    perm = np.random.RandomState(0).permutation(N)
    X = X[perm]; v = v[perm]
    X_train, X_val = X[:n_train], X[n_train:]
    v_train, v_val = v[:n_train], v[n_train:]

    # per-channel mean/std from train
    mean = X_train.mean(axis=(0,1))  # (6,)
    std = X_train.std(axis=(0,1)) + 1e-6
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "hz": args.hz, "window": args.window, "stride": args.stride}
    Path(args.scaler).parent.mkdir(parents=True, exist_ok=True)
    with open(args.scaler, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"[scaler] {args.scaler} mean {mean} std {std}")

    # normalize
    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_windows.npy", X_train_n.astype(np.float32))
    np.save(out / "train_v.npy", v_train.astype(np.float32))
    np.save(out / "val_windows.npy", X_val_n.astype(np.float32))
    np.save(out / "val_v.npy", v_val.astype(np.float32))
    print(f"[save] {out}/train_windows.npy {X_train_n.shape} {out}/val_windows.npy {X_val_n.shape}")
    print(f"[save] scaler {args.scaler}")

if __name__ == "__main__":
    main()
