"""Tests for the AVNet model family (shapes, checkpoint compat, determinism)."""

from __future__ import annotations

import pytest
import torch

from python.models.avnet import AVNet, AVNetLite, count_params
from python.models.lean_estimator import LeanEstimator

CKPT = "experiments/checkpoints/model_avnet_stage1.p"


class TestAVNetLite:
    def test_forward_shapes(self) -> None:
        """(B,200,6) → 5-tuple with v (B,1), att (B,3), h (B,64)."""
        m = AVNetLite().eval()
        v, ls_v, att, ls_att, h = m(torch.randn(4, 200, 6))
        assert v.shape == (4, 1)
        assert ls_v.shape == (4, 1)
        assert att.shape == (4, 3)
        assert ls_att.shape == (4, 3)
        assert h.shape == (4, 64)

    def test_param_count_matches_checkpoint(self) -> None:
        """460,136 params — pinned; a change means the checkpoint won't load."""
        assert count_params(AVNetLite()) == 460_136

    def test_channels_first_input_accepted(self) -> None:
        """(B,6,200) input is permuted automatically (ONNX/TFLite parity)."""
        m = AVNetLite().eval()
        assert m(torch.randn(2, 6, 200))[0].shape == (2, 1)

    def test_bad_shape_raises(self) -> None:
        """Anything not (B,200,6)/(B,6,200) fails loud."""
        with pytest.raises(ValueError, match="expected"):
            AVNetLite().eval()(torch.randn(2, 100, 6))

    @pytest.mark.skipif(not torch.cuda.is_available() and not __import__("pathlib").Path(CKPT).exists(),
                        reason="checkpoint not present")
    def test_checkpoint_loads(self) -> None:
        """The stage-1 checkpoint loads into the refactored model unchanged."""
        m = AVNetLite()
        m.load_state_dict(torch.load(CKPT, map_location="cpu"))

    def test_deterministic_eval(self) -> None:
        """Same input + eval mode → identical output (ZUPT/σ_v stability)."""
        m = AVNetLite().eval()
        x = torch.randn(2, 200, 6)
        with torch.no_grad():
            assert torch.allclose(m(x)[0], m(x)[0])


class TestAVNet:
    def test_forward_shapes(self) -> None:
        """Full model: v (B,1), att (B,3), h_next (B,512)."""
        m = AVNet().eval()
        v, ls_v, att, ls_att, h = m(torch.randn(2, 200, 6))
        assert v.shape == (2, 1)
        assert att.shape == (2, 3)
        assert h.shape == (2, 512)

    def test_gru_state_passthrough(self) -> None:
        """Explicit hx flows to h_next (temporal chaining support)."""
        m = AVNet().eval()
        x = torch.randn(2, 200, 6)
        with torch.no_grad():
            _, _, _, _, h1 = m(x)
            _, _, _, _, h2 = m(x, hx=h1)
        assert not torch.allclose(h1, h2)


class TestLeanEstimator:
    def test_forward_shapes(self) -> None:
        """(B,200,6) → phi (B,), p_bike (B,) in range."""
        m = LeanEstimator().eval()
        phi, p_bike = m(torch.randn(3, 200, 6))
        assert phi.shape == (3,)
        assert p_bike.shape == (3,)
        assert bool((p_bike >= 0).all() and (p_bike <= 1).all())

    def test_phi_from_raw_acc(self) -> None:
        """φ ≈ atan2(mean acc_y, mean acc_z) when raw acc is given (5° roll)."""
        import math

        m = LeanEstimator().eval()
        roll = math.radians(5.0)
        acc = torch.zeros(1, 200, 3)
        acc[:, :, 1] = 9.81 * math.sin(roll)
        acc[:, :, 2] = 9.81 * math.cos(roll)
        phi, _ = m(torch.zeros(1, 200, 6), acc_raw=acc)
        assert float(phi[0]) == pytest.approx(roll, abs=0.05)

    def test_phi_denormalized_via_scaler(self) -> None:
        """Normalized input + scaler stats recovers physical φ (38° bug guard)."""
        import math

        m = LeanEstimator().eval()
        roll = math.radians(8.0)
        acc = torch.zeros(1, 200, 3)
        acc[:, :, 1] = 9.81 * math.sin(roll)
        acc[:, :, 2] = 9.81 * math.cos(roll)
        mean = torch.zeros(6)
        std = torch.ones(6) * 0.5
        x = torch.zeros(1, 200, 6)
        x[:, :, :3] = (acc - mean[:3]) / std[:3]
        phi, _ = m(x, scaler_mean=mean, scaler_std=std)
        assert abs(float(phi[0])) < math.radians(10)  # tripwire <10°

    def test_nhc_bike_vs_car(self) -> None:
        """Bike: v_lat = v·sinφ, R×(1+2|φ|); car: 0 and 1 (Step 9.4 spec)."""
        import math

        m = LeanEstimator()
        v = torch.tensor([5.0, 5.0])
        phi = torch.tensor([math.radians(30.0), 0.0])
        p = torch.tensor([0.9, 0.2])
        vy, scale = m.nhc_correction(v, phi, p)
        assert vy[0] == pytest.approx(5.0 * math.sin(math.radians(30.0)))
        assert scale[0] == pytest.approx(1 + 2 * math.radians(30.0))
        assert float(vy[1]) == 0.0
        assert float(scale[1]) == 1.0

    def test_phi_clamped(self) -> None:
        """Extreme attitudes clamp to ±40° (bike range)."""
        m = LeanEstimator().eval()
        acc = torch.zeros(1, 200, 3)
        acc[:, :, 0] = 9.81  # 90° roll — unphysical for a bike
        acc[:, :, 2] = 0.0
        phi, _ = m(torch.zeros(1, 200, 6), acc_raw=acc)
        assert float(phi[0]) <= 40.0 / 180.0 * 3.14159 + 1e-6
