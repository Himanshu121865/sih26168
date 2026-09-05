"""Train-only Z-score scaler with PASS_THROUGH (Agastya scaler.py:53-56, harsh)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class TrainOnlyScaler:
    """Per-channel mean/std fitted on train data only (no test leakage).

    Attributes:
        mean: Per-channel means, shape (C,).
        std: Per-channel standard deviations, shape (C,).
        fitted: Whether :meth:`fit` has been called.
        train_files: Basenames of the training files used for fitting.
    """

    def __init__(self) -> None:
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.fitted: bool = False
        self.train_files: list[str] = []

    def fit(self, X_train: np.ndarray, train_files: list[str | Path] | None = None) -> None:
        """Fit per-channel statistics, reducing over all but the last axis.

        Args:
            X_train: Training array, shape (N, window, C) or (N, C).
            train_files: Optional source file list recorded for provenance.

        Raises:
            ValueError: If the input has fewer than 2 dimensions.
        """
        if X_train.ndim < 2:
            raise ValueError(f"expected >=2 dims, got shape {X_train.shape}")
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

    def _require_fitted(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (mean, std), raising if not fitted.

        Raises:
            RuntimeError: If :meth:`fit` has not been called yet.
        """
        if not self.fitted or self.mean is None or self.std is None:
            raise RuntimeError("call fit() before transform/inverse/save")
        return self.mean, self.std

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Standardize ``(X - mean) / std``.

        Args:
            X: Input array with trailing channel dim C.

        Returns:
            Standardized array of the same shape.

        Raises:
            RuntimeError: If the scaler is not fitted.
        """
        mean, std = self._require_fitted()
        return (X - mean) / std

    def inverse(self, X_norm: np.ndarray) -> np.ndarray:
        """Invert :meth:`transform`.

        Args:
            X_norm: Standardized array.

        Returns:
            Array in original units.

        Raises:
            RuntimeError: If the scaler is not fitted.
        """
        mean, std = self._require_fitted()
        return X_norm * std + mean

    def save(self, path: str | Path) -> None:
        """Save fitted statistics as JSON.

        Args:
            path: Destination ``.json`` path (parents are created).

        Raises:
            RuntimeError: If the scaler is not fitted.
        """
        self._require_fitted()
        assert self.mean is not None and self.std is not None  # for mypy
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("w") as f:
            json.dump(
                {
                    "mean": self.mean.tolist(),
                    "std": self.std.tolist(),
                    "train_files": self.train_files,
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> TrainOnlyScaler:
        """Load statistics saved with :meth:`save`.

        Args:
            path: Source ``.json`` path.

        Returns:
            Fitted scaler instance.

        Raises:
            FileNotFoundError: If the path does not exist.
            KeyError: If required keys are missing.
        """
        s = cls()
        with Path(path).open() as f:
            d = json.load(f)
        s.mean = np.array(d["mean"])
        s.std = np.array(d["std"])
        s.train_files = list(d.get("train_files", []))
        s.fitted = True
        return s
