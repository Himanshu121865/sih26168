"""core/types.py — Shared contracts (harsh/types.py:89-106)."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class ImuSample:
    t_ns: int
    a_body: np.ndarray  # (3,) m/s2
    w_body: np.ndarray  # (3,) rad/s

@dataclass(frozen=True, slots=True)
class GnssFix:
    t_ns: int
    lat: float
    lon: float
    acc_m: float
