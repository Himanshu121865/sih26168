"""Tests for drift evaluation (1D/2D) with synthetic npy fixtures."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from python.eval.drift import eval_1d, eval_2d, latlon_to_enu, predict_segment
from python.eval.metrics import ate, drift_pct, total_distance


@pytest.fixture
def val_npys(tmp_path: Path):
    """Deterministic val set: 800 windows, ramping speed 0→10 m/s."""
    n = 800
    v = np.linspace(0.0, 10.0, n).astype(np.float32)
    X = np.zeros((n, 200, 6), dtype=np.float32)
    X[:, :, 3] = 0.01  # mild yaw rate
    np.save(tmp_path / "val_v.npy", v)
    np.save(tmp_path / "val_windows.npy", X)
    return tmp_path / "val_v.npy", tmp_path / "val_windows.npy", v


class TestPredictSegment:
    def test_synthetic_fallback_when_no_model(self, tmp_path: Path) -> None:
        """Missing model path → None (caller uses v_gt + seeded noise)."""
        X = np.zeros((10, 200, 6), dtype=np.float32)
        assert predict_segment(None, X, 0, 10) is None
        assert predict_segment(tmp_path / "nope.p", X, 0, 10) is None


class TestEval1D:
    def test_writes_plot_and_json(self, val_npys, tmp_path: Path) -> None:
        """Output contract: PNG + JSON sidecar with the metric fields."""
        v_path, X_path, v = val_npys
        plot = tmp_path / "drift_plot.png"
        m = eval_1d(v_path, model_path=None, plot_path=plot, val_windows_path=X_path)
        assert plot.exists() and plot.stat().st_size > 1000
        sidecar = json.loads((tmp_path / "drift_plot.json").read_text())
        for key in ("mode", "total_dist_m", "naive_final_m", "ai_final_m", "ai_drift_pct"):
            assert key in sidecar
        assert m["mode"] == "1d"
        # legacy start clamp: start = min(N//2, N - seg_len) = 200 → seg = v[200:800]
        seg = v[len(v) - 600 :]
        assert m["total_dist_m"] == pytest.approx(float(np.cumsum(seg * 0.1)[-1]))

    def test_ai_drift_small_with_good_predictions(self, val_npys, tmp_path: Path) -> None:
        """Perfect predictions → AI final drift ≈ 0 (sanity of the math)."""
        v_path, X_path, v = val_npys
        # monkeypatch-free check: model=None uses v_gt + noise(0.1) — still small
        m = eval_1d(v_path, model_path=None, plot_path=tmp_path / "p.png", val_windows_path=X_path)
        assert m["ai_drift_pct"] < 5.0


class TestEval2D:
    def test_writes_plot_and_json(self, val_npys, tmp_path: Path) -> None:
        """2D mode: ATE/RTE fields present, trajectory finite."""
        v_path, X_path, _ = val_npys
        scaler = tmp_path / "scaler.json"
        scaler.write_text(json.dumps({"mean": [0] * 6, "std": [1] * 6}))
        plot = tmp_path / "drift_2d.png"
        m = eval_2d(v_path, model_path=None, plot_path=plot,
                    val_windows_path=X_path, scaler_path=scaler)
        assert plot.exists()
        for key in ("total_dist_m", "ai_ate_m", "ai_rte60_m", "ai_drift_pct"):
            assert key in m
        assert m["mode"] == "2d"
        assert np.isfinite(m["ai_ate_m"])


class TestGeoHelpers:
    def test_latlon_to_enu_identity(self) -> None:
        """The reference point maps to (0, 0)."""
        xy = latlon_to_enu(28.6, 77.2, 28.6, 77.2)
        assert np.allclose(xy, 0.0, atol=1e-9)

    def test_latlon_to_enu_north_direction(self) -> None:
        """+1° latitude ≈ 111 km north (y > x)."""
        xy = latlon_to_enu(29.6, 77.2, 28.6, 77.2)
        assert xy[1] == pytest.approx(111_194, abs=500)
        assert abs(xy[0]) < 1.0

    def test_ate_zero_for_identical(self) -> None:
        """Identical est/gt → ATE 0 without alignment."""
        gt = np.stack([np.linspace(0, 10, 50), np.zeros(50)], 1)
        rmse, _ = ate(gt, gt, align=False)
        assert rmse == pytest.approx(0.0)

    def test_drift_pct_degenerate(self) -> None:
        """Zero distance → 0% drift, not division-by-zero."""
        assert drift_pct(5.0, 0.0) == 0.0

    def test_total_distance_straight(self) -> None:
        """A 10×1 straight line has length 10."""
        gt = np.stack([np.linspace(0, 10, 11), np.zeros(11)], 1)
        assert total_distance(gt) == pytest.approx(10.0)
