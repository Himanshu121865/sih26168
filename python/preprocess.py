#!/usr/bin/env python3
"""
preprocess.py — Step 2.2-2.5 (P0 #1 FIXED 2026-08-30)
Resample 10Hz→100Hz, gravity-align, per-axis normalize, sliding window (200, stride 10).

Fixes from competitor audit (docs/IMPROVEMENTS_FROM_COMPETITORS.md P0 #1):
- find_column regex (sivaraman/prepare_training_data.py:205) for (m/s²) vs (m/s^2)
- resample_uniform shared (harsh/pipeline.py:30-68) with period_ns=round(1e9/rate), np.interp left=np.nan, finite_mask
- gravity_align via ACC - GRAVITY (IO-VNBD provides gravity), not no-op
- split_by_trajectory (harsh/loaders.py:287) not window shuffle leak
- encoding cp1252 + strip, dt validation

Usage:
  python python/preprocess.py --subset 1h          # quick 1h subset for smoke test (<5 min)
  python python/preprocess.py --window 200 --stride 10 --hz 100 --train-ratio 0.8
"""
import argparse, json, glob, re
from pathlib import Path
import numpy as np
import pandas as pd

# robust column finder (sivaraman/prepare_training_data.py:205)
def find_column(df, pattern: str):
    regex = re.compile(pattern, re.IGNORECASE)
    for c in df.columns:
        if regex.search(c):
            return c
    return None

def find_columns_xyz(df, base_pattern):
    # e.g., base "accelerometer" -> X,Y,Z cols
    cols = {}
    for axis in ["x","y","z"]:
        pat = rf"{base_pattern}.*{axis}"
        col = find_column(df, pat)
        if col is None:
            # fallback: try base without axis then positional
            pass
        cols[axis] = col
    return cols

# shared resample_uniform (harsh/pipeline.py:30-68)
def resample_uniform(t_ns: np.ndarray, values: np.ndarray, rate_hz: float):
    """
    t_ns (N,) int64 ns, values (N,K) or (N,), rate_hz
    Returns t_new_ns (M,), values_new (M,K)
    Uses period_ns=round(1e9/rate), linear np.interp per channel, left/right=np.nan
    Error if <2 samples.
    """
    if len(t_ns) < 2:
        raise ValueError(f"need >=2 samples, got {len(t_ns)}")
    period_ns = round(1e9 / rate_hz)
    t_start, t_end = t_ns[0], t_ns[-1]
    n = int((t_end - t_start) // period_ns) + 1
    t_new = t_start + np.arange(n) * period_ns
    if values.ndim == 1:
        v_new = np.interp(t_new, t_ns, values, left=np.nan, right=np.nan)
        return t_new, v_new
    else:
        v_new = np.empty((n, values.shape[1]))
        for k in range(values.shape[1]):
            v_new[:, k] = np.interp(t_new, t_ns, values[:, k], left=np.nan, right=np.nan)
        return t_new, v_new

def load_phone_csv(path: str) -> pd.DataFrame:
    # cp1252 handles ² byte 0xb2 better than latin1, plus strip
    df = pd.read_csv(path, encoding="cp1252")
    df.columns = [c.strip() for c in df.columns]
    return df

def gravity_align_linear(acc_raw: np.ndarray, gravity: np.ndarray) -> np.ndarray:
    """
    IO-VNBD provides GRAVITY X/Y/Z (already low-pass). Linear acc = ACC - GRAVITY.
    This removes gravity before scaling (harsh pipeline.py:105-108 R_wb@a - [0,0,9.80665]).
    acc_raw (N,3), gravity (N,3) -> linear (N,3)
    """
    return acc_raw - gravity

def make_windows(arr: np.ndarray, window=200, stride=10):
    """arr (T,6) -> (N, window, 6)"""
    T = arr.shape[0]
    if T < window:
        return np.empty((0, window, 6))
    N = (T - window)//stride + 1
    windows = np.stack([arr[i*stride:i*stride+window] for i in range(N)], axis=0)
    return windows

def compute_labels(df_100: pd.DataFrame, windows_idx, stride=10, window=200, gps_speed_col=None):
    v_kmh = df_100[gps_speed_col].values if gps_speed_col and gps_speed_col in df_100.columns else np.zeros(len(df_100))
    v_ms = v_kmh / 3.6
    v_labels = np.array([v_ms[i*stride + window -1] for i in windows_idx])
    att_labels = np.zeros((len(v_labels), 3))
    return v_labels, att_labels

def is_window_stationary(window_6: np.ndarray, hz=100):
    """
    ZUPT check per window (harsh zupt.py:39-40 thresholds, scaled for vehicle 100Hz).
    window_6 (200,6) linear_acc(3)+gyro(3) @100Hz
    Returns True if stationary (acc var <0.05 and gyro var <0.01 for 0.5s)
    """
    # use norms
    acc = window_6[:, :3]
    gyro = window_6[:, 3:6]
    a_norm = np.linalg.norm(acc, axis=1)
    w_norm = np.linalg.norm(gyro, axis=1)
    # variance over window (200 samples =2s, but detector uses 0.5s sliding — we check whole window var)
    # For vehicle, use 0.5s sub-window at end
    tail = 50  # last 0.5s @100Hz
    a_var = np.var(a_norm[-tail:])
    w_var = np.var(w_norm[-tail:])
    # speed gate will be applied via v_label <0.5 later, here just IMU
    return a_var < 0.05 and w_var < 0.01

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["1h","full"], default="full")
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--hz", type=int, default=100)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--scaler", default="python/scaler.json")
    ap.add_argument("--resume", action="store_true", help="resume from existing npy if interrupted (skip completed files)")
    args = ap.parse_args()

    base = Path("data/iovnbd/Synchronised V abd S datasets/Categorised IOVNB Dataset")
    s_files = sorted(glob.glob(str(base / "**/S-*.csv"), recursive=True))
    if args.subset == "1h":
        s_files = s_files[:3]
    print(f"[preprocess] {len(s_files)} S files, window={args.window} stride={args.stride} hz={args.hz} resume={args.resume}")

    # resume: if output exists and is recent, skip (user can rm -rf data/processed to force)
    out_path = Path(args.out)
    if args.resume and (out_path / "train_windows.npy").exists() and (out_path / "val_windows.npy").exists():
        print(f"[resume] {out_path}/train_windows.npy exists ({(out_path/'train_windows.npy').stat().st_size/1e9:.2f}GB), skipping preprocess. Use --no-resume or rm -rf {out_path} to force.")
        return

    # split by trajectory (file), not window — prevents leakage (harsh/loaders.py:287)
    n_train_files = int(len(s_files) * args.train_ratio)
    rng = np.random.default_rng(26168)
    perm_files = rng.permutation(len(s_files))
    train_files_idx = set(perm_files[:n_train_files])
    train_files = [s_files[i] for i in range(len(s_files)) if i in train_files_idx]
    val_files = [s_files[i] for i in range(len(s_files)) if i not in train_files_idx]
    print(f"[split] train files {len(train_files)} val files {len(val_files)} (by trajectory, seed 26168)")

    # handle Ctrl-C gracefully: save what we have so far
    import signal
    interrupted = {"flag": False}
    def handle_sigint(sig, frame):
        interrupted["flag"] = True
        print("\n[interrupt] Ctrl-C detected, will save partial progress and exit...")
    orig_handler = signal.signal(signal.SIGINT, handle_sigint)

    def process_file_list(file_list):
        all_windows = []
        all_v = []
        for idx_f, f in enumerate(file_list):
            if interrupted["flag"]:
                print(f"[interrupt] stopping after {idx_f}/{len(file_list)} files, saving partial...")
                break
            try:
                df = load_phone_csv(f)
                # robust column mapping
                acc_cols = [find_column(df, rf"accelerometer.*{axis}") for axis in ["x","y","z"]]
                grav_cols = [find_column(df, rf"gravity.*{axis}") for axis in ["x","y","z"]]
                gyro_cols = [find_column(df, r"gyroscope.*yaw"), find_column(df, r"gyroscope.*pitch"), find_column(df, r"gyroscope.*roll")]
                # fallback to exact names if regex fails
                if None in acc_cols:
                    acc_cols = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
                if None in grav_cols:
                    grav_cols = ["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]
                if None in gyro_cols:
                    gyro_cols = ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]
                time_col = find_column(df, r"time since start")
                gps_speed_col = find_column(df, r"gps speed")
                if time_col is None:
                    time_col = "TIME SINCE START (ms)"
                if gps_speed_col is None:
                    gps_speed_col = "GPS SPEED (Kmh)"

                # validate time monotonic (sivaraman/ekf:65)
                t_ms_raw = df[time_col].values.astype(float)
                # check monotonic (allow resets per file: just check diff within file)
                diffs = np.diff(t_ms_raw)
                if np.any(diffs <= 0):
                    # fix: sort by time
                    order = np.argsort(t_ms_raw)
                    df = df.iloc[order].reset_index(drop=True)
                    t_ms_raw = df[time_col].values.astype(float)

                # build t_ns
                t_ns = (t_ms_raw * 1e6).astype(np.int64)  # ms -> ns
                # verify median interval ~100ms for 10Hz
                median_dt_ms = np.median(np.diff(t_ms_raw))
                # print(f"  median dt {median_dt_ms:.1f} ms")

                acc_raw = df[acc_cols].values.astype(np.float64)
                grav = df[grav_cols].values.astype(np.float64)
                gyro = df[gyro_cols].values.astype(np.float64)

                # gravity align: linear acc = acc - gravity (removes 9.81 before scaler)
                linear_acc = gravity_align_linear(acc_raw, grav)  # (N,3)

                # stack IMU: linear_acc (3) + gyro (3) = 6ch
                imu_6 = np.concatenate([linear_acc, gyro], axis=1)  # (N,6)

                # resample via resample_uniform per channel with ns
                # need to handle NaN from interp left/right
                t_new_ns, imu_new = resample_uniform(t_ns, imu_6, args.hz)
                # also resample gps speed for labels
                gps_speed_raw = df[gps_speed_col].values.astype(np.float64) if gps_speed_col in df.columns else np.zeros(len(df))
                _, gps_speed_new = resample_uniform(t_ns, gps_speed_raw, args.hz)

                # drop NaN rows from resample (outside overlap)
                finite_mask = np.isfinite(imu_new).all(axis=1) & np.isfinite(gps_speed_new)
                t_new_ns = t_new_ns[finite_mask]
                imu_new = imu_new[finite_mask]
                gps_speed_new = gps_speed_new[finite_mask]

                if len(imu_new) < args.window:
                    print(f"[skip] {Path(f).parent.name}/{Path(f).name}: too short after resample {len(imu_new)}")
                    continue

                windows = make_windows(imu_new.astype(np.float32), window=args.window, stride=args.stride)
                if len(windows) == 0:
                    continue

                # labels from resampled gps_speed_new
                df_100_dummy = pd.DataFrame({gps_speed_col: gps_speed_new})
                idx = list(range(len(windows)))
                v_labels, _ = compute_labels(df_100_dummy, idx, stride=args.stride, window=args.window, gps_speed_col=gps_speed_col)

                # ZUPT stationary flags per window (P1 wiring, harsh zupt.py:39-40)
                # Use IMU variance + speed gate (<0.5 m/s)
                stationary = np.array([is_window_stationary(w, hz=args.hz) for w in windows])
                # also gate by speed: if v_label <0.5, keep stationary, else False (vehicle moving but low var e.g. smooth highway)
                stationary = stationary & (v_labels < 0.5)
                # for stationary windows, force v_label=0 (ZUPT)
                v_labels = np.where(stationary, 0.0, v_labels)

                all_windows.append(windows)
                all_v.append(v_labels)
                # also collect stationary for saving (optional)
                if not hasattr(process_file_list, "all_stationary"):
                    process_file_list.all_stationary = []
                process_file_list.all_stationary.append(stationary)
                print(f"[ok] {Path(f).parent.name}/{Path(f).name}: T={len(df)}->{len(imu_new)} windows={len(windows)} stationary={stationary.sum()} median_dt={median_dt_ms:.1f}ms")
            except Exception as e:
                print(f"[err] {f}: {e}")
                import traceback; traceback.print_exc()
                continue
        if not all_windows:
            return np.empty((0, args.window, 6)), np.empty((0,)), np.empty((0,), dtype=bool)
        X = np.concatenate(all_windows, axis=0)
        v = np.concatenate(all_v, axis=0)
        # collect stationary from attribute
        if hasattr(process_file_list, "all_stationary"):
            stationary = np.concatenate(process_file_list.all_stationary, axis=0)
            # clear for next call
            delattr(process_file_list, "all_stationary")
        else:
            stationary = np.zeros(len(X), dtype=bool)
        return X, v, stationary

    try:
        X_train, v_train, stat_train = process_file_list(train_files)
        X_val, v_val, stat_val = process_file_list(val_files)
    finally:
        signal.signal(signal.SIGINT, orig_handler)

    if interrupted["flag"]:
        # save partial even if val empty, so resume can detect
        if 'X_train' in locals() and len(X_train) > 0:
            print(f"[interrupt] saving partial train {X_train.shape} val {X_val.shape if 'X_val' in locals() and len(X_val)>0 else 'none'}")
            # still need scaler from what we have
            mean = X_train.mean(axis=(0,1)); std = X_train.std(axis=(0,1)) + 1e-6
            for i in range(len(std)):
                if std[i] < 1e-8:
                    mean[i]=0; std[i]=1
            scaler = {"mean": mean.tolist(), "std": std.tolist(), "hz": args.hz, "window": args.window, "stride": args.stride, "train_files": [str(Path(f).name) for f in train_files], "partial": True}
            Path(args.scaler).parent.mkdir(parents=True, exist_ok=True)
            with open(args.scaler, "w") as f:
                json.dump(scaler, f, indent=2)
            out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
            np.save(out / "train_windows.npy", ((X_train - mean)/std).astype(np.float32))
            np.save(out / "train_v.npy", v_train.astype(np.float32))
            if 'X_val' in locals() and len(X_val)>0:
                np.save(out / "val_windows.npy", ((X_val - mean)/std).astype(np.float32))
                np.save(out / "val_v.npy", v_val.astype(np.float32))
            print(f"[interrupt] partial saved to {args.out}, re-run with --resume to skip or rm -rf to restart")
        return

    print(f"[concat] train X {X_train.shape} v {v_train.shape} stationary {stat_train.sum()}/{len(stat_train)} | val X {X_val.shape} v {v_val.shape} stationary {stat_val.sum()}/{len(stat_val)}")
    if len(X_train) == 0 or len(X_val) == 0:
        print("[err] no windows")
        return
    print(f"  train mean {X_train.mean(axis=(0,1))} std {X_train.std(axis=(0,1))}")
    print(f"  val mean {X_val.mean(axis=(0,1))} std {X_val.std(axis=(0,1))}")

    # scaler from train only (train-only, sivaraman/agastya)
    mean = X_train.mean(axis=(0,1))
    std = X_train.std(axis=(0,1)) + 1e-6
    # PASS_THROUGH for any future binary flags would be mean=0,std=1 if std<1e-8 — not needed for 6ch, but keep logic
    for i in range(len(std)):
        if std[i] < 1e-8:
            mean[i] = 0; std[i] = 1
    scaler = {"mean": mean.tolist(), "std": std.tolist(), "hz": args.hz, "window": args.window, "stride": args.stride, "train_files": [str(Path(f).name) for f in train_files]}
    Path(args.scaler).parent.mkdir(parents=True, exist_ok=True)
    with open(args.scaler, "w") as f:
        json.dump(scaler, f, indent=2)
    print(f"[scaler] {args.scaler} mean {mean} std {std}")

    X_train_n = (X_train - mean) / std
    X_val_n = (X_val - mean) / std

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_windows.npy", X_train_n.astype(np.float32))
    np.save(out / "train_v.npy", v_train.astype(np.float32))
    np.save(out / "val_windows.npy", X_val_n.astype(np.float32))
    np.save(out / "val_v.npy", v_val.astype(np.float32))
    print(f"[save] {out}/train_windows.npy {X_train_n.shape} {out}/val_windows.npy {X_val_n.shape}")

if __name__ == "__main__":
    main()
