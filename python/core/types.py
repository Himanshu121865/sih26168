"""Shared contracts (harsh/types.py:89-106)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ImuSample:
    """One IMU sample in the body frame.

    Attributes:
        t_ns: Timestamp in nanoseconds.
        a_body: Acceleration in m/s², shape (3,).
        w_body: Angular rate in rad/s, shape (3,).
    """

    t_ns: int
    a_body: np.ndarray  # (3,) m/s2
    w_body: np.ndarray  # (3,) rad/s


@dataclass(frozen=True, slots=True)
class GnssFix:
    """One GNSS position fix.

    Attributes:
        t_ns: Timestamp in nanoseconds.
        lat: Latitude in degrees.
        lon: Longitude in degrees.
        acc_m: 1-sigma horizontal accuracy in meters.
    """

    t_ns: int
    lat: float
    lon: float
    acc_m: float
