"""Streaming Dataset for IO-VNBD windows.

Two modes:

- **npy** — precomputed ``(N, 200, 6)`` float32 memmaps from preprocess
  (small subsets, instant startup).
- **streaming** — on-the-fly windowing from raw S-CSVs via the shared
  :mod:`python.datasets.iovnbd` pipeline (full 58 h without 90 GB RAM).
  The resample→gravity→normalize path is IDENTICAL to preprocess because
  both call :func:`python.datasets.iovnbd.process_file`.

Usage:
    from python.datasets.iovnbd_dataset import IOVNBDWindowDataset
    ds = IOVNBDWindowDataset("data/processed/train_windows.npy", "data/processed/train_v.npy")
    ds = IOVNBDWindowDataset(files=[...], scaler_path="python/scaler.json")
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from python.datasets.iovnbd import load_phone_csv, process_file, resample_stream

# Bound the streaming resample cache: 4 files ≈ tens of MB (full 72-seq runs).
_CACHE_LIMIT = 4


class IOVNBDWindowDataset(Dataset[Any]):
    """Torch Dataset over IO-VNBD windows (npy memmap or streamed).

    Attributes:
        mode: ``"npy"`` or ``"stream"``.
        N: Number of windows.
        window / stride / hz: Window geometry (spec v2 constants).
    """

    def __init__(
        self,
        windows_path: str | Path | None = None,
        v_path: str | Path | None = None,
        files: list[str] | None = None,
        window: int = 200,
        stride: int = 10,
        hz: int = 100,
        scaler_path: str | Path = "python/scaler.json",
    ) -> None:
        """Create the dataset in npy or streaming mode.

        Args:
            windows_path: Precomputed ``(N, window, 6)`` npy (normalized).
            v_path: Per-window speed labels npy; zeros when absent.
            files: Raw S-CSV paths → streaming mode (takes precedence only
                when the npy path is missing/None).
            window / stride / hz: Window geometry for streaming mode.
            scaler_path: Scaler JSON for streaming normalization (npy windows
                are already normalized).

        Raises:
            FileNotFoundError: When windows_path is given but missing (a
                common post-``rm -rf`` state — the message says how to fix).
            ValueError: When neither mode's inputs are provided.
        """
        self.window = window
        self.stride = stride
        self.hz = hz

        if windows_path is not None and Path(windows_path).exists():
            self.mode = "npy"
            self.X = np.load(windows_path, mmap_mode="r")
            if v_path is not None and Path(v_path).exists():
                self.v = np.load(v_path, mmap_mode="r")
            else:
                self.v = np.zeros(len(self.X), dtype=np.float32)
            self.N = len(self.X)
            print(f"[dataset] npy mode {windows_path}")
        elif files:
            self.mode = "stream"
            self.files = list(files)
            with open(scaler_path) as f:
                sc = json.load(f)
            self._mean = np.array(sc["mean"], dtype=np.float64)
            self._std = np.array(sc["std"], dtype=np.float64)
            # Build the (file_idx, window_idx) index from real resampled
            # lengths — parity with preprocess window counts.
            self._resampled_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            self.index: list[tuple[int, int]] = []
            for fi, fp in enumerate(self.files):
                df = load_phone_csv(fp)
                result = process_file(df, hz=hz, window=window, stride=stride)
                n_windows = 0 if result is None else len(result.windows)
                for wi in range(n_windows):
                    self.index.append((fi, wi))
            self.N = len(self.index)
            print(f"[dataset] streaming mode {len(self.files)} files, N={self.N} windows")
        else:
            if windows_path is not None:
                subset = "full" if "full" in str(windows_path) else "1h"
                raise FileNotFoundError(
                    f"Dataset not found: {windows_path} (and {v_path}). "
                    f"Run `python python/preprocess.py --subset {subset} "
                    f"--window {window} --stride {stride} --hz {hz}` first, "
                    "or use streaming mode: IOVNBDWindowDataset(files=[...])"
                )
            raise ValueError("need windows_path or files")

    def _get_resampled(self, fi: int) -> tuple[np.ndarray, np.ndarray]:
        """Normalized (imu_stream (T,6), v_stream (T,)) for file fi, cached (LRU≈4)."""
        if fi not in self._resampled_cache:
            df = load_phone_csv(self.files[fi])
            imu, v_ms, _audit, _gdiff = resample_stream(df, hz=self.hz)
            imu_norm = (imu - self._mean) / self._std
            self._resampled_cache[fi] = (
                imu_norm.astype(np.float32),
                v_ms.astype(np.float32),
            )
            if len(self._resampled_cache) > _CACHE_LIMIT:
                oldest = next(iter(self._resampled_cache))
                if oldest != fi:
                    del self._resampled_cache[oldest]
        return self._resampled_cache[fi]

    def __len__(self) -> int:
        return self.N

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return ``(window (window,6), v (,), att (3,))`` — att is a dummy
        zero placeholder retained for loader-interface parity."""
        if self.mode == "npy":
            x = torch.from_numpy(np.array(self.X[idx]))  # copy: mmap is read-only
            v = torch.tensor(float(self.v[idx]), dtype=torch.float32)
            return x, v, torch.zeros(3, dtype=torch.float32)

        fi, wi = self.index[idx]
        imu_norm, v_ms = self._get_resampled(fi)
        start = wi * self.stride
        x_seg = imu_norm[start : start + self.window]  # (200,6) normalized
        v_seg = float(v_ms[start + self.window - 1])
        return (
            torch.from_numpy(np.ascontiguousarray(x_seg, dtype=np.float32)),
            torch.tensor(v_seg, dtype=torch.float32),
            torch.zeros(3, dtype=torch.float32),
        )


def rebuild_split(
    base: str | Path,
    train_ratio: float = 0.8,
    seed: int = 26168,
    split: str = "random",
) -> tuple[list[str], list[str]]:
    """Reproduce a preprocess split from a dataset base dir (audit parity).

    Args:
        base: Directory containing ``**/S-*.csv``.
        train_ratio / seed / split: Must match the preprocess invocation
            that produced the checkpoint under audit.

    Returns:
        (train_files, val_files) — same lists preprocess chose.
    """
    import glob as _glob

    from python.datasets.split import split_files

    s_files = sorted(_glob.glob(str(Path(base) / "**/S-*.csv"), recursive=True))
    return split_files(s_files, strategy=split, train_ratio=train_ratio, seed=seed)
