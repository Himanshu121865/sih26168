"""Tests for shared contracts (need numpy — run on Colab)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from python.core.types import GnssFix, ImuSample


def test_imu_sample_fields() -> None:
    """Sample stores timestamped body-frame vectors."""
    s = ImuSample(t_ns=10_000_000, a_body=np.zeros(3), w_body=np.ones(3))
    assert s.t_ns == 10_000_000
    assert s.a_body.shape == (3,)


def test_dataclasses_are_frozen() -> None:
    """Contracts are immutable value objects."""
    s = ImuSample(t_ns=0, a_body=np.zeros(3), w_body=np.zeros(3))
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.t_ns = 1  # type: ignore[misc]
    g = GnssFix(t_ns=0, lat=28.6, lon=77.2, acc_m=5.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        g.lat = 0.0  # type: ignore[misc]
