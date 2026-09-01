"""
core/scaler.py — Train-only Z-score with PASS_THROUGH (Agastya scaler.py:53-56, harsh)
"""
import json
import numpy as np
from pathlib import Path

class TrainOnlyScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        self.fitted = False
        self.train_files = []

    def fit(self, X_train: np.ndarray, train_files=None):
        """X_train (N,window,6) or (N,6) — per-channel mean/std from train only."""
        # reduce over N and window
        axes = tuple(range(X_train.ndim - 1))
        self.mean = X_train.mean(axis=axes)
        self.std = X_train.std(axis=axes) + 1e-6
        # PASS_THROUGH for binary flags (std<1e-8)
        for i in range(len(self.std)):
            if self.std[i] < 1e-8:
                self.mean[i] = 0
                self.std[i] = 1
        self.fitted = True
        if train_files:
            self.train_files = [str(Path(f).name) for f in train_files]

    def transform(self, X: np.ndarray):
        assert self.fitted, "call fit first"
        return (X - self.mean) / self.std

    def inverse(self, X_norm: np.ndarray):
        return X_norm * self.std + self.mean

    def save(self, path: str):
        assert self.fitted
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump({"mean": self.mean.tolist(), "std": self.std.tolist(), "train_files": self.train_files}, f, indent=2)

    @classmethod
    def load(cls, path: str):
        s = cls()
        with open(path) as f:
            d = json.load(f)
        s.mean = np.array(d["mean"])
        s.std = np.array(d["std"])
        s.train_files = d.get("train_files", [])
        s.fitted = True
        return s
