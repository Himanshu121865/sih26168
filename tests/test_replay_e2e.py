"""End-to-end replay test with a tiny trained-ish model + synthetic val data.

Validates run_replay wiring (denormalize → ZUPT → NHC → InEKF → metrics)
without needing the 822k-window dataset — the checkpoint contract and the
scaler JSON shape are exercised exactly as in production.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from python.fusion.replay import run_replay
from python.models.avnet import AVNetLite


@pytest.fixture
def tiny_val(tmp_path: Path):
    """Synthetic val set: 700 windows, constant 5 m/s, mild accel noise."""
    n = 700
    rng = np.random.default_rng(42)
    X = np.zeros((n, 200, 6), dtype=np.float32)
    X[:, :, 0] = rng.normal(0, 0.3, (n, 200))  # forward accel noise
    X[:, :, 3] = rng.normal(0, 0.01, (n, 200))  # yaw noise
    v = np.full(n, 5.0, dtype=np.float32)  # constant 5 m/s
    np.save(tmp_path / "val_windows.npy", X)
    np.save(tmp_path / "val_v.npy", v)
    scaler = {
        "mean": [0.0] * 6,
        "std": [1.0] * 6,
        "hz": 100,
        "window": 200,
        "stride":  10,
        "spec_version": 2,
        "spec_sha256": "x" * 64,
    }
    sp = tmp_path / "scaler.json"
    sp.write_text(json.dumps(scaler))
    return tmp_path, sp


def _near_constant_model(tmp_path: Path, v0: float = 5.0) -> Path:
    """Checkpoint whose velocity head ≈ constant v0 regardless of input."""
    torch.manual_seed(0)
    m = AVNetLite()
    with torch.no_grad():
        m.head_vel.weight.zero_()
        m.head_vel.bias.fill_(v0)
        m.head_logsig_vel.weight.zero_()
        m.head_logsig_vel.bias.fill_(math.log(0.5))
        # freeze-ish: tiny weights so BN stats barely matter
        for p in m.parameters():
            if p.dim() > 1 and p is not m.head_vel.weight and p is not m.head_logsig_vel.weight:
                p.mul_(0.01)
    ckpt = tmp_path / "const.p"
    torch.save(m.state_dict(), ckpt)
    return ckpt


class TestRunReplay:
    def test_metrics_contract(self, tiny_val, monkeypatch) -> None:
        """run_replay returns the full metric dict with sane invariants."""
        tmp, scaler = tiny_val
        # point the default scaler path at our fixture
        monkeypatch.setattr("python.fusion.replay.load_scaler", lambda path=None: (
            np.zeros(6), np.ones(6)
        ))
        ckpt = _near_constant_model(tmp)
        metrics = run_replay(
            ckpt, n_windows=100, start=10, lean_mode="car", verbose=False,
            val_windows=tmp / "val_windows.npy", val_v=tmp / "val_v.npy",
        )
        for key in ("segment_windows", "start", "total_dist_m",
                    "naive_final_m", "avnet_final_m", "inekf_final_m",
                    "naive_drift_pct", "avnet_drift_pct", "inekf_drift_pct",
                    "q_acc", "n_zupt"):
            assert key in metrics, f"missing {key}"
        assert metrics["segment_windows"] == 100
        assert metrics["total_dist_m"] == pytest.approx(100 * 5.0 * 0.1, rel=0.05)
        # AVNet v≈5 m/s → drift ≈ 0; naive integrates zero-mean noise → small
        assert metrics["avnet_drift_pct"] < 5.0
        assert metrics["inekf_final_m"] >= 0.0

    def test_zupt_on_stopped_segment(self, tiny_val, monkeypatch) -> None:
        """A fully-stopped segment: ZUPT fires and InEKF holds near zero drift."""
        tmp, _ = tiny_val
        n = 300
        X = np.zeros((n, 200, 6), dtype=np.float32)
        v = np.zeros(n, dtype=np.float32)
        np.save(tmp / "val_windows.npy", X)
        np.save(tmp / "val_v.npy", v)
        monkeypatch.setattr("python.fusion.replay.load_scaler", lambda path=None: (
            np.zeros(6), np.ones(6)
        ))
        ckpt = _near_constant_model(tmp, v0=0.0)  # model says: stopped
        metrics = run_replay(
            ckpt, n_windows=200, start=0, lean_mode="car", verbose=False,
            val_windows=tmp / "val_windows.npy", val_v=tmp / "val_v.npy",
        )
        assert metrics["n_zupt"] > 100  # variance detector fires on flat stream
        assert metrics["inekf_drift_pct"] < 5.0
        assert metrics["avnet_drift_pct"] < 5.0
