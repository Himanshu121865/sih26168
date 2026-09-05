"""
iovnbd_dataset.py — Step 2.4 / 3.4
Streaming Dataset for IO-VNBD windows. Supports both precomputed npy (small subsets)
and on-the-fly windowing (full 58h) to avoid 90GB RAM.

Usage:
  from python.datasets.iovnbd_dataset import IOVNBDWindowDataset
  ds = IOVNBDWindowDataset("data/processed/train_windows.npy", "data/processed/train_v.npy")  # npy mode
  ds = IOVNBDWindowDataset(files=["data/iovnbd/.../S-Vtb1.csv"], window=200, stride=10, scaler="python/scaler.json")  # streaming
"""
import json, glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import torch
from torch.utils.data import Dataset

ACC_COLS = ["ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)"]
GYRO_COLS = ["GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)"]
IMU_COLS = ACC_COLS + GYRO_COLS
TIME_COL = "TIME SINCE START (ms)"

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
            self.files = files
            self.mode = "stream"
            # load scaler
            with open(scaler_path) as f:
                sc = json.load(f)
            self.mean = np.array(sc["mean"], dtype=np.float32)
            self.std = np.array(sc["std"], dtype=np.float32)
            # build index: list of (file_idx, window_start)
            self.index = []
            for fi, fp in enumerate(files):
                df = pd.read_csv(fp, encoding="latin1")
                df.columns = [c.strip() for c in df.columns]
                # resample count
                t_ms = df[TIME_COL].values.astype(float)
                t_new_len = int((t_ms.max() - t_ms.min()) / (1000/hz))
                n_windows = max(0, (t_new_len - window)//stride + 1)
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
            # streaming: load file, resample, window
            fi, wi = self.index[idx]
            df = pd.read_csv(self.files[fi], encoding="latin1")
            df.columns = [c.strip() for c in df.columns]
            t_ms = df[TIME_COL].values.astype(float)
            t_start, t_end = t_ms.min(), t_ms.max()
            t_new = np.arange(t_start, t_end, 1000/self.hz)
            # interp
            imu_new = []
            for c in IMU_COLS:
                f = interp1d(t_ms, df[c].values, kind="linear", bounds_error=False, fill_value="extrapolate")
                imu_new.append(f(t_new))
            imu_new = np.stack(imu_new, axis=1)  # (T,6)
            # normalize
            imu_new = (imu_new - self.mean) / self.std
            start = wi * self.stride
            x = imu_new[start:start+self.window]  # (200,6)
            # label: GPS speed at window end
            v_kmh = df["GPS SPEED (Kmh)"].values if "GPS SPEED (Kmh)" in df.columns else np.zeros(len(df))
            # interp v
            f_v = interp1d(t_ms, v_kmh, kind="linear", bounds_error=False, fill_value="extrapolate")
            v_new = f_v(t_new)
            v_ms = v_new[start+self.window-1] / 3.6
            return torch.from_numpy(x.astype(np.float32)), torch.tensor(float(v_ms), dtype=torch.float32), torch.zeros(3)

if __name__ == "__main__":
    # smoke test npy mode
    ds = IOVNBDWindowDataset("data/processed/train_windows.npy", "data/processed/train_v.npy")
    print(f"N={len(ds)} sample X {ds[0][0].shape} v {ds[0][1]}")
    # streaming test on 1 file
    import glob as g
    files = sorted(g.glob("data/iovnbd/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vtb (Driver E)/Vtb01/S-*.csv"))
    ds2 = IOVNBDWindowDataset(files=files[:1], scaler_path="python/scaler.json")
    print(f"stream N={len(ds2)} sample {ds2[0][0].shape} {ds2[0][1]}")
