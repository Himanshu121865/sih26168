"""
iovnbd_dataset.py — Step 2.4 / 3.4
Streaming Dataset for IO-VNBD windows. Supports both precomputed npy (small subsets)
and on-the-fly windowing (full 58h) to avoid 90GB RAM.

Parity (F1): streaming path now uses python.core.signal.resample_uniform
(period_ns, np.interp left/right=nan + finite_mask), gravity removal
(ACC - GRAVITY), cp1252 + robust find_column, real t_ns from TIME col with
monotonic sort — identical to preprocess.py.

Usage:
  from python.datasets.iovnbd_dataset import IOVNBDWindowDataset
  ds = IOVNBDWindowDataset("data/processed/train_windows.npy", "data/processed/train_v.npy")  # npy mode
  ds = IOVNBDWindowDataset(files=["data/iovnbd/.../S-Vtb1.csv"], window=200, stride=10, scaler="python/scaler.json")  # streaming
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from python.core.signal import find_column, resample_uniform, gravity_align_linear

TIME_PAT = r"time since start"
GPS_SPEED_PAT = r"gps speed"
ACC_PAT = r"accelerometer"
GRAV_PAT = r"gravity"
GYRO_YAW_PAT = r"gyroscope.*yaw"
GYRO_PITCH_PAT = r"gyroscope.*pitch"
GYRO_ROLL_PAT = r"gyroscope.*roll"


def _resolve_columns(df):
    """Robust IO-VNBD S-file column map with exact-name fallback (cp1252 ²)."""
    acc_cols = [find_column(df, rf"{ACC_PAT}.*{ax}") for ax in ["x", "y", "z"]]
    grav_cols = [find_column(df, rf"{GRAV_PAT}.*{ax}") for ax in ["x", "y", "z"]]
    gyro_cols = [
        find_column(df, GYRO_YAW_PAT),
        find_column(df, GYRO_PITCH_PAT),
        find_column(df, GYRO_ROLL_PAT),
    ]
    if None in acc_cols:
        acc_cols = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
    if None in grav_cols:
        grav_cols = ["GRAVITY X (m/s²)", "GRAVITY Y (m/s²)", "GRAVITY Z (m/s²)"]
    if None in gyro_cols:
        gyro_cols = ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]
    time_col = find_column(df, TIME_PAT) or "TIME SINCE START (ms)"
    gps_col = find_column(df, GPS_SPEED_PAT) or "GPS SPEED (Kmh)"
    return acc_cols, grav_cols, gyro_cols, time_col, gps_col


def resample_file_to_hz(df, hz):
    """Resample one S-file df to uniform hz. Returns (imu_6 (T,6) linear+gyro, v_ms (T,), t_new_ns)."""
    acc_cols, grav_cols, gyro_cols, time_col, gps_col = _resolve_columns(df)
    # monotonic time (sort if device reordered)
    t_ms_raw = df[time_col].values.astype(float)
    if np.any(np.diff(t_ms_raw) <= 0):
        order = np.argsort(t_ms_raw)
        df = df.iloc[order].reset_index(drop=True)
        t_ms_raw = df[time_col].values.astype(float)
    t_ns = (t_ms_raw * 1e6).astype(np.int64)

    acc_raw = df[acc_cols].values.astype(np.float64)
    grav = df[grav_cols].values.astype(np.float64)
    gyro = df[gyro_cols].values.astype(np.float64)
    linear_acc = gravity_align_linear(acc_raw, grav)
    imu_6 = np.concatenate([linear_acc, gyro], axis=1)

    t_new_ns, imu_new = resample_uniform(t_ns, imu_6, hz)
    gps_raw = df[gps_col].values.astype(np.float64) if gps_col in df.columns else np.zeros(len(df))
    _, gps_new = resample_uniform(t_ns, gps_raw, hz)

    finite_mask = np.isfinite(imu_new).all(axis=1) & np.isfinite(gps_new)
    return imu_new[finite_mask], (gps_new[finite_mask] / 3.6), t_new_ns[finite_mask]


class IOVNBDWindowDataset(Dataset):
    def __init__(self, windows_path=None, v_path=None, files=None, window=200, stride=10, hz=100, scaler_path="python/scaler.json"):
        """
        Two modes:
        1) windows_path + v_path: load precomputed npy (from preprocess.py)
        2) files: list of S-*.csv paths, window/stride/hz, scaler
        """
        self.window = window; self.stride = stride; self.hz = hz
        if windows_path and Path(windows_path).exists():
            print(f"[dataset] npy mode {windows_path}")
            self.X = np.load(windows_path, mmap_mode="r")  # (N,200,6) float32, mmapped
            self.v = np.load(v_path, mmap_mode="r") if v_path and Path(v_path).exists() else np.zeros(len(self.X), dtype=np.float32)
            self.mode = "npy"
            self.N = len(self.X)
        elif files:
            print(f"[dataset] streaming mode {len(files)} files")
            self.files = list(files)
            self.mode = "stream"
            # load scaler
            with open(scaler_path) as f:
                sc = json.load(f)
            self.mean = np.array(sc["mean"], dtype=np.float32)
            self.std = np.array(sc["std"], dtype=np.float32)
            # build index via real resampled lengths (parity with preprocess)
            self.index = []
            self._resampled_cache = {}
            for fi, fp in enumerate(self.files):
                df = pd.read_csv(fp, encoding="cp1252")
                df.columns = [c.strip() for c in df.columns]
                imu_new, _, _ = resample_file_to_hz(df, hz)
                n_windows = max(0, (len(imu_new) - window) // stride + 1)
                for wi in range(n_windows):
                    self.index.append((fi, wi))
            self.N = len(self.index)
            print(f"[dataset] streaming N={self.N} windows")
        else:
            # Check if windows_path was provided but file missing (common after rm -rf or Ctrl-C)
            if windows_path:
                raise FileNotFoundError(
                    f"Dataset not found: {windows_path} (and {v_path}). "
                    f"Run `python python/preprocess.py --subset {'full' if 'full' in str(windows_path) else '1h'} --window {window} --stride {stride} --hz {hz}` first, "
                    f"or use streaming mode: IOVNBDWindowDataset(files=[...])"
                )
            raise ValueError("need windows_path or files")

    def _get_resampled(self, fi):
        if fi not in self._resampled_cache:
            df = pd.read_csv(self.files[fi], encoding="cp1252")
            df.columns = [c.strip() for c in df.columns]
            imu_new, v_new, _ = resample_file_to_hz(df, self.hz)
            imu_new = (imu_new - self.mean.astype(np.float64)) / self.std.astype(np.float64)
            self._resampled_cache[fi] = (imu_new.astype(np.float32), v_new.astype(np.float32))
            # bound cache to last 4 files (full 72-seq streaming)
            if len(self._resampled_cache) > 4:
                oldest = next(iter(self._resampled_cache))
                if oldest != fi:
                    del self._resampled_cache[oldest]
        return self._resampled_cache[fi]

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        if self.mode == "npy":
            # X is (N,200,6) already normalized
            x = torch.from_numpy(np.array(self.X[idx]))  # copy from mmap
            v = torch.tensor(float(self.v[idx]), dtype=torch.float32)
            # att label dummy zeros (3,)
            att = torch.zeros(3, dtype=torch.float32)
            return x, v, att
        else:
            fi, wi = self.index[idx]
            imu_new, v_new = self._get_resampled(fi)
            start = wi * self.stride
            x = imu_new[start:start + self.window]  # (200,6) normalized, gravity-removed
            v_ms = float(v_new[start + self.window - 1])
            return torch.from_numpy(np.array(x, dtype=np.float32)), torch.tensor(v_ms, dtype=torch.float32), torch.zeros(3)

if __name__ == "__main__":
    # smoke test npy mode
    ds = IOVNBDWindowDataset("data/processed/train_windows.npy", "data/processed/train_v.npy")
    print(f"N={len(ds)} sample X {ds[0][0].shape} v {ds[0][1]}")
    # streaming test on 1 file
    import glob as g
    files = sorted(g.glob("data/iovnbd/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vtb (Driver E)/Vtb01/S-*.csv"))
    ds2 = IOVNBDWindowDataset(files=files[:1], scaler_path="python/scaler.json")
    print(f"stream N={len(ds2)} sample {ds2[0][0].shape} {ds2[0][1]}")
