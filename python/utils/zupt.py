"""
zupt.py — Adapted from ref/competitors/harsh/src/dr_core/fusion/zupt.py
Stationary detection for vehicle ZUPT + ZARU.
For vehicle: speed <0.5 m/s + low variance → stationary.
Thresholds tuned for car/bike 100Hz.

Usage:
  det = StationaryDetector(rate_hz=100)
  for sample in imu_stream:  # ImuSample(a_body, w_body, t_ns)
      if det.update(sample):  # True → apply ZUPT (v=0) + ZARU (gyro bias)
          ekf.correct_zupt()
"""
from collections import deque
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class StationaryConfig:
    window_s: float = 0.5
    accel_var_threshold: float = 0.05  # (m/s2)^2 — from harsh, works for vehicle
    gyro_var_threshold: float = 0.01   # (rad/s)^2
    min_duration_s: float = 0.3
    # vehicle-specific: also check speed
    speed_threshold: float = 0.5  # m/s — if GPS/vel <0.5, likely stopped

class StationaryDetector:
    def __init__(self, config=None, rate_hz=100.0):
        self.config = config if config else StationaryConfig()
        self.rate_hz = rate_hz
        self.window_size = max(2, round(self.config.window_s * rate_hz))
        self._a_norms = deque(maxlen=self.window_size)
        self._w_norms = deque(maxlen=self.window_size)
        self._candidate_start_ns = None
        self._current_t_ns = 0
        self._is_stationary = False

    def update(self, a_body: np.ndarray, w_body: np.ndarray, t_ns: int, speed_mps: float = None) -> bool:
        """a_body (3,) m/s2, w_body (3,) rad/s, t_ns, optional speed"""
        self._current_t_ns = t_ns
        self._a_norms.append(float(np.linalg.norm(a_body)))
        self._w_norms.append(float(np.linalg.norm(w_body)))
        if len(self._a_norms) < self.window_size:
            self._is_stationary = False
            return False
        a_var = float(np.var(self._a_norms))
        w_var = float(np.var(self._w_norms))
        # also check speed if provided
        speed_ok = True
        if speed_mps is not None:
            speed_ok = speed_mps < self.config.speed_threshold
        if a_var < self.config.accel_var_threshold and w_var < self.config.gyro_var_threshold and speed_ok:
            if self._candidate_start_ns is None:
                self._candidate_start_ns = t_ns
            duration_s = (t_ns - self._candidate_start_ns) * 1e-9
            self._is_stationary = duration_s >= self.config.min_duration_s
        else:
            self._candidate_start_ns = None
            self._is_stationary = False
        return self._is_stationary

    @property
    def is_stationary(self) -> bool:
        return self._is_stationary

    @property
    def stationary_duration_s(self) -> float:
        if self._candidate_start_ns is None:
            return 0.0
        return max(0.0, (self._current_t_ns - self._candidate_start_ns) * 1e-9)

    # For ESKF wiring (harsh eskf.py:237-297): ZUPT y=-v_world, H=[1 at dv], R=diag(0.02^2); ZARU y=-(gyro-bg), H[5]=-1, R=0.005
    def get_zupt_measurement(self):
        """Returns (y, R) for ZUPT when stationary: y=-v_world (2D), R=diag(0.02^2)"""
        if not self._is_stationary:
            return None
        y = np.zeros(2)  # device expects v=0, innovation = -v_pred
        R = np.diag([0.02**2, 0.02**2])
        return y, R

    def get_zaru_measurement(self, gyro_bias):
        """ZARU: y=-(gyro - bg), H at db_g, R=0.005^2"""
        if not self._is_stationary:
            return None
        # caller should compute y = -(w_body - bg_pred)
        R = np.diag([0.005**2])
        return R

# Simple function for offline: detect stationary windows in numpy arrays
def detect_stationary_windows(acc: np.ndarray, gyro: np.ndarray, hz=100, **kwargs):
    """
    acc (N,3), gyro (N,3) @hz → bool array (N,) stationary
    """
    det = StationaryDetector(rate_hz=hz, **kwargs)
    out = np.zeros(len(acc), dtype=bool)
    for i in range(len(acc)):
        t_ns = int(i * (1e9/hz))
        out[i] = det.update(acc[i], gyro[i], t_ns)
    return out

if __name__ == "__main__":
    # test: stationary vs moving
    det = StationaryDetector(rate_hz=100)
    # stationary: low variance
    for i in range(100):
        a = np.array([0,0,9.81]) + np.random.normal(0,0.02,3)
        w = np.random.normal(0,0.005,3)
        print(f"{i} stat={det.update(a,w, i*10_000_000)}", end="\r")
    print("\n--- moving ---")
    det2 = StationaryDetector(rate_hz=100)
    for i in range(100):
        a = np.array([0,0,9.81]) + np.random.normal(0,0.5,3)
        w = np.random.normal(0,0.1,3)
        print(f"{i} stat={det2.update(a,w, i*10_000_000)}", end="\r")
    print("\ndone")
