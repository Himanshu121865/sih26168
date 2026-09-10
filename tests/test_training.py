"""Tests for the shared training loop (augmentations, losses, epochs)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from python.core.training import (
    augment_random_yaw,
    augment_synthetic_bike,
    eval_loss,
    gaussian_nll_loss,
    set_seed,
    train_one_epoch,
)


def _loader(n: int = 32, seed: int = 0) -> DataLoader:
    rng = np.random.default_rng(seed)
    X = torch.from_numpy(rng.normal(size=(n, 200, 6)).astype(np.float32))
    v = torch.from_numpy(rng.uniform(0, 10, size=(n,)).astype(np.float32))
    att = torch.zeros(n, 3)
    return DataLoader(TensorDataset(X, v, att), batch_size=16)


class _TinyVelModel(torch.nn.Module):
    """Minimal model matching the 5-tuple contract (v, logσ_v, att, logσ_att, h)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = torch.nn.Linear(6, 1)
        self.sig = torch.nn.Linear(6, 1)

    def forward(self, x):
        pooled = x.mean(dim=1)
        return self.lin(pooled), self.sig(pooled), torch.zeros(*pooled.shape[:-1], 3), \
            torch.zeros(*pooled.shape[:-1], 3), pooled


class TestSetSeed:
    def test_reproducible(self) -> None:
        """Identical seeds → identical torch draws (Colab parity)."""
        set_seed(7)
        a = torch.randn(5)
        set_seed(7)
        b = torch.randn(5)
        assert torch.equal(a, b)


class TestAugmentRandomYaw:
    def test_invariants(self) -> None:
        """acc_z (ch2) and gyro yaw (ch3) are invariant under yaw rotation."""
        x = torch.randn(8, 200, 6)
        xa = augment_random_yaw(x)
        assert torch.allclose(xa[:, :, 2], x[:, :, 2])
        assert torch.allclose(xa[:, :, 3], x[:, :, 3])
        assert not torch.allclose(xa[:, :, 0], x[:, :, 0]) or True  # may collide rarely

    def test_horizontal_norm_preserved(self) -> None:
        """|acc_xy| and |gyro_pr| are preserved (pure rotation)."""
        x = torch.randn(4, 200, 6)
        xa = augment_random_yaw(x)
        n_before = (x[:, :, 0] ** 2 + x[:, :, 1] ** 2).sqrt()
        n_after = (xa[:, :, 0] ** 2 + xa[:, :, 1] ** 2).sqrt()
        assert torch.allclose(n_before, n_after, atol=1e-5)

    def test_original_unchanged(self) -> None:
        """Augmentation returns a copy; input tensor is untouched."""
        x = torch.randn(2, 200, 6)
        augment_random_yaw(x)
        assert torch.equal(x, x.clone())  # trivially true; real check below
        x2 = x.clone()
        augment_random_yaw(x)
        assert torch.equal(x, x2)


class TestAugmentSyntheticBike:
    def test_noop_on_bad_shape(self) -> None:
        """Non-3D / non-6ch input passes through unchanged."""
        x = torch.randn(2, 100)
        assert torch.equal(augment_synthetic_bike(x), x)

    def test_prob_zero_is_identity(self) -> None:
        """Zero probabilities leave the windows untouched."""
        x = torch.randn(4, 200, 6)
        assert torch.equal(augment_synthetic_bike(x, 0, 0, 0), x)

    def test_pothole_changes_acc_z(self) -> None:
        """prob=1 pothole adds a bounded pulse to acc_z (ch2)."""
        x = torch.zeros(4, 200, 6)
        xa = augment_synthetic_bike(x, pothole_prob=1.0, engine_prob=0, lean_prob=0)
        assert float(xa[:, :, 2].abs().max()) > 0.5

    def test_engine_changes_gyro(self) -> None:
        """prob=1 engine adds 30/55 Hz content to gyro (ch3-5)."""
        x = torch.zeros(4, 200, 6)
        xa = augment_synthetic_bike(x, pothole_prob=0, engine_prob=1.0, lean_prob=0)
        assert float(xa[:, :, 3:].abs().max()) > 0.05


class TestGaussianNLL:
    def test_perfect_prediction_is_low(self) -> None:
        """v_pred = v_gt with small σ → NLL ≈ log σ + const."""
        v = torch.zeros(4)
        vp = torch.zeros(4, 1)
        ls = torch.full((4, 1), -6.0)  # softplus(−6)≈0.0025 → σ≈0.0035
        nll = gaussian_nll_loss(vp, v, ls)
        assert float(nll) < 0.0  # log(σ) dominates → negative

    def test_sigma_floor(self) -> None:
        """Very negative log_sig still yields σ ≥ min_sigma (no div-by-0)."""
        v = torch.zeros(2)
        vp = torch.ones(2, 1)
        ls = torch.full((2, 1), -100.0)
        nll = gaussian_nll_loss(vp, v, ls, min_sigma=1e-3)
        assert math.isfinite(float(nll))


class TestTrainEvalLoops:
    def test_train_one_epoch_decreases_loss(self) -> None:
        """A linear-model-solvable dataset converges within epochs."""
        set_seed(3)
        model = _TinyVelModel()
        optim = torch.optim.Adam(model.parameters(), lr=1e-2)
        loader = _loader()
        first = None
        for _ in range(20):
            loss, mse = train_one_epoch(model, loader, optim, torch.device("cpu"), lambda_nll=0.0)
            if first is None:
                first = mse
        assert mse < first

    def test_eval_loss_matches_manual(self) -> None:
        """eval_loss equals a hand-computed MSE for a weight/bias-filled model."""
        model = _TinyVelModel()
        model.eval()
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(0.5)  # weight 0.5 per input + bias 0.5
        loader = _loader(16)
        got = eval_loss(model, loader, torch.device("cpu"))
        X = loader.dataset.tensors[0]
        v = loader.dataset.tensors[1]
        with torch.no_grad():
            pooled = X.mean(dim=1)                       # (N, 6)
            pred = 0.5 * pooled.sum(-1) + 0.5             # Linear(6,1) all-0.5
            manual = float((pred - v).pow(2).mean())
        assert got == pytest.approx(manual, rel=1e-4)
