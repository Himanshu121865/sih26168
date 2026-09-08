"""Cross-language golden vectors (Window-Path Hardening P3).

Recomputes the full window path (gravity low-pass -> resample -> normalize)
from the FIXED raw fixture inside `window_golden.json` and asserts byte-level
parity with the snapshot in the same file. The same asset is read by Kotlin
`WindowGoldenTest` on the device — any drift between the Python training path
and the Kotlin live path fails here first.

Also trips on the 38-degree lean bug: the fixture's mean |phi| must stay
below 10 degrees.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from python.core.scaler import TrainOnlyScaler
from python.core.signal import estimate_gravity_lowpass, resample_uniform

ASSET = Path(__file__).resolve().parents[1] / "android/app/src/main/assets/window_golden.json"
TOL = 1e-6  # snapshots rounded to 1e-9; allow small fp slack


@pytest.fixture(scope="module")
def golden() -> dict:
    import json

    return json.loads(ASSET.read_text())


def test_asset_exists_and_current_spec(golden: dict) -> None:
    """The checked-in snapshot must carry the CURRENT window spec."""
    from python.core.spec import SPEC_VERSION, spec_sha256

    assert golden["spec_version"] == SPEC_VERSION
    assert golden["spec_sha256"] == spec_sha256()


def test_window_path_parity(golden: dict) -> None:
    """Raw fixture -> normalized window matches the snapshot exactly."""
    t_ms = np.array(golden["fixture"]["t_ms"], dtype=np.float64)
    acc_raw = np.array(golden["fixture"]["acc_raw"], dtype=np.float64)
    gyro = np.array(golden["fixture"]["gyro"], dtype=np.float64)

    # 1) gravity removal via the SINGLE implementation (spec v2)
    grav = estimate_gravity_lowpass(acc_raw)
    lin_acc = acc_raw - grav
    imu_6 = np.concatenate([lin_acc, gyro], axis=1)

    # 2) resample 10Hz -> 100Hz
    t_ns = (t_ms * 1e6).astype(np.int64)
    _, imu_new = resample_uniform(t_ns, imu_6, 100)

    # 3) fixed fixture scaler (mean/std are part of the snapshot)
    sc = TrainOnlyScaler()
    sc.mean = np.array(golden["scaler"]["mean"], dtype=np.float64)
    sc.std = np.array(golden["scaler"]["std"], dtype=np.float64)
    sc.fitted = True
    norm = sc.transform(imu_new)

    snap = np.array(golden["window_100hz_normalized"], dtype=np.float64)
    assert norm.shape == snap.shape
    assert np.max(np.abs(norm - snap)) < TOL


def test_gravity_snapshot_parity(golden: dict) -> None:
    """The low-pass gravity estimate itself matches the snapshot."""
    acc_raw = np.array(golden["fixture"]["acc_raw"], dtype=np.float64)
    grav = estimate_gravity_lowpass(acc_raw)
    snap = np.array(golden["grav_expected"], dtype=np.float64)
    assert grav.shape == snap.shape
    assert np.max(np.abs(grav - snap)) < TOL


def test_lean_tripwire(golden: dict) -> None:
    """Mean |phi| stays under the limit — the 38-deg-bug tripwire."""
    phi_deg = golden["mean_abs_phi_deg"]
    assert phi_deg < golden["phi_limit_deg"]


def test_lean_tripwire_recomputed(golden: dict) -> None:
    """Recompute phi from raw (don't trust the snapshot number alone)."""
    acc_raw = np.array(golden["fixture"]["acc_raw"], dtype=np.float64)
    grav = estimate_gravity_lowpass(acc_raw)
    phi = np.arctan2(grav[:, 1], grav[:, 2])
    mean_abs_phi_deg = float(np.degrees(np.mean(np.abs(phi))))
    assert mean_abs_phi_deg < golden["phi_limit_deg"]
    assert abs(mean_abs_phi_deg - golden["mean_abs_phi_deg"]) < 1e-6
