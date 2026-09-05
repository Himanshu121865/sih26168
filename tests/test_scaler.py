"""Tests for the train-only scaler (need numpy — run on Colab)."""

from __future__ import annotations

import numpy as np
import pytest
from pathlib import Path

from python.core.scaler import TrainOnlyScaler


@pytest.fixture
def train_array() -> np.ndarray:
    """Small (N, window, C) array with known statistics."""
    rng = np.random.default_rng(26168)
    return rng.normal(loc=[1.0, 2.0], scale=[2.0, 0.5], size=(50, 10, 2))


def test_fit_transform_roundtrip(train_array: np.ndarray) -> None:
    """Standardized data has ~zero mean and inverts exactly."""
    sc = TrainOnlyScaler()
    sc.fit(train_array, train_files=["S-M.csv"])
    assert sc.train_files == ["S-M.csv"]
    z = sc.transform(train_array)
    assert abs(float(z.mean(axis=(0, 1))[0])) < 0.1
    assert np.allclose(sc.inverse(z), train_array)


def test_transform_before_fit_raises(train_array: np.ndarray) -> None:
    """Unfitted scaler raises RuntimeError instead of asserting."""
    with pytest.raises(RuntimeError, match="fit"):
        TrainOnlyScaler().transform(train_array)


def test_save_load_roundtrip(train_array: np.ndarray, tmp_path: Path) -> None:
    """Saved JSON reloads to an identical transform."""
    sc = TrainOnlyScaler()
    sc.fit(train_array)
    dest = tmp_path / "scaler.json"
    sc.save(dest)
    sc2 = TrainOnlyScaler.load(dest)
    assert np.allclose(sc2.transform(train_array), sc.transform(train_array))


def test_constant_channel_pass_through() -> None:
    """Zero-variance channel becomes mean=0, std=1 (binary-flag safe)."""
    X = np.ones((10, 5, 2))
    X[:, :, 1] = np.arange(50).reshape(10, 5)  # varying channel
    sc = TrainOnlyScaler()
    sc.fit(X)
    assert sc.mean is not None and sc.std is not None
    assert sc.mean[0] == 0 and sc.std[0] == 1


def test_fit_rejects_1d() -> None:
    """1D input raises a clear ValueError."""
    with pytest.raises(ValueError, match=">=2 dims"):
        TrainOnlyScaler().fit(np.ones(6))
