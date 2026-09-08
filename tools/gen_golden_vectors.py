#!/usr/bin/env python3
"""gen_golden_vectors.py — Window-Path Hardening P3.

Generates `window_golden.json`: a FIXED raw CSV snippet -> expected normalized
window bytes, checked by BOTH `tests/test_window_golden.py` (Python) AND Kotlin
`WindowGoldenTest` (Android) against the SAME asset file. Any drift in the
window path (resample, gravity, normalization) or between the two languages
fails loudly instead of silently skewing drift metrics.

The lean tripwire rides along: the fixture includes a synthetic lean
sequence whose mean |phi| must stay < 10 deg in BOTH implementations (guards
the 38-deg-bug regression where gravity init made atan2 explode).

Usage:
  python tools/gen_golden_vectors.py [--out android/app/src/main/assets/window_golden.json]

Idempotent: deterministic inputs, deterministic outputs (seeded RNG, no time
or platform dependence; float64 math throughout, JSON rounded to 1e-9).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.core.scaler import TrainOnlyScaler  # noqa: E402
from python.core.signal import estimate_gravity_lowpass  # noqa: E402
from python.core.spec import SPEC_VERSION, spec_sha256  # noqa: E402


def make_fixture() -> dict:
    """Build the deterministic raw snippet + expected normalized window.

    Synthetic signal (10 Hz, 3 s = 30 rows): stopped -> gentle accel -> cruise,
    with a constant gravity vector and mild yaw rotation. Numbers chosen to
    exercise: gravity low-pass settle, interp edges, per-channel scaling,
    and the lean tripwire (mean |phi| < 10 deg).
    """
    rng = np.random.default_rng(26168)
    n = 30  # 3 s @ 10 Hz
    t_ms = np.arange(n, dtype=np.float64) * 100.0

    # Gravity (device on bike holder, slight 5-deg roll): |g| ~ 9.81
    roll = np.deg2rad(5.0)
    g_vec = np.array([0.0, 9.81 * np.sin(roll), 9.81 * np.cos(roll)])

    # Forward accel profile: 0 for 1 s, ramp to 1.5 m/s^2, cruise with noise
    fwd = np.zeros(n)
    fwd[10:20] = np.linspace(0.2, 1.5, 10)
    fwd[20:] = 1.5
    noise = rng.normal(0, 0.05, size=(n, 3))

    acc_raw = np.zeros((n, 3))
    acc_raw[:, 0] = fwd + noise[:, 0]
    acc_raw[:, 1] = g_vec[1] + noise[:, 1]
    acc_raw[:, 2] = g_vec[2] + noise[:, 2]

    # Gyro: mild yaw (turn), near-zero pitch/roll rates
    gyro = np.zeros((n, 3))
    gyro[:, 0] = 0.02 + rng.normal(0, 0.005, n)  # yaw
    gyro[:, 1] = rng.normal(0, 0.002, n)  # pitch
    gyro[:, 2] = np.deg2rad(2.0) * np.ones(n) + rng.normal(0, 0.003, n)  # roll rate (lean turn)

    # Expected gravity via the single implementation (spec v2)
    grav = estimate_gravity_lowpass(acc_raw)

    return {
        "t_ms": t_ms,
        "acc_raw": acc_raw,
        "gyro": gyro,
        "grav_expected": grav,
    }


def build_golden() -> dict:
    """Run the canonical window path on the fixture and snapshot outputs."""
    fx = make_fixture()
    t_ms, acc_raw, gyro = fx["t_ms"], fx["acc_raw"], fx["gyro"]

    # 1) gravity remove (spec v2 live low-pass)
    grav = estimate_gravity_lowpass(acc_raw)
    lin_acc = acc_raw - grav

    # 2) resample 10Hz -> 100Hz on the uniform ns grid (preprocess parity)
    t_ns = (t_ms * 1e6).astype(np.int64)
    imu_6 = np.concatenate([lin_acc, gyro], axis=1)
    from python.core.signal import resample_uniform

    _, imu_new = resample_uniform(t_ns, imu_6, 100)

    # 3) scaler fitted on THIS deterministic fixture (train-only semantics
    #    hold trivially; the point is a fixed mean/std, not statistics)
    sc = TrainOnlyScaler()
    X = imu_new[np.newaxis, :, :]  # (1, T, 6)
    sc.fit(X)
    norm = sc.transform(imu_new)  # (T, 6)

    # 4) lean tripwire: mean |phi| over the fixture, via the same atan2(g_y, g_z)
    phi = np.arctan2(grav[:, 1], grav[:, 2])
    mean_abs_phi_deg = float(np.degrees(np.mean(np.abs(phi))))

    return {
        "spec_version": SPEC_VERSION,
        "spec_sha256": spec_sha256(),
        "fixture": {
            "t_ms": _round(t_ms),
            "acc_raw": _round(acc_raw),
            "gyro": _round(gyro),
        },
        "scaler": {"mean": _round(sc.mean), "std": _round(sc.std)},
        "window_100hz_normalized": _round(norm),
        "grav_expected": _round(grav),
        "mean_abs_phi_deg": round(mean_abs_phi_deg, 6),
        "phi_limit_deg": 10.0,
    }


def _round(a: np.ndarray) -> list:
    """Round to 9 decimals so JSON text is byte-stable across platforms."""
    return np.round(np.asarray(a, dtype=np.float64), 9).tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="android/app/src/main/assets/window_golden.json",
        help="destination asset (also read by tests/test_window_golden.py)",
    )
    args = ap.parse_args()

    golden = build_golden()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(golden, indent=2))

    n = len(golden["window_100hz_normalized"])
    print(f"golden vectors: {n} samples @100Hz, phi={golden['mean_abs_phi_deg']:.3f}deg "
          f"(limit {golden['phi_limit_deg']}), spec v{golden['spec_version']}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
