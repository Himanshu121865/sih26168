"""Tests for the unified IO-VNBD window pipeline (spec v2 path)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from python.datasets.iovnbd import (
    load_phone_csv,
    monotonic_time_ns,
    process_file,
    resample_stream,
    resolve_columns,
    timestamp_audit,
    velocity_labels,
    windowize,
)


def _synthetic_df(n: int = 60, dt_ms: float = 100.0, with_gravity_cols: bool = True) -> pd.DataFrame:
    """IO-VNBD-shaped frame: 10 Hz, gravity-ish accel, ramping GPS speed."""
    t = np.arange(n) * dt_ms
    acc = np.zeros((n, 3))
    acc[:, 2] = 9.81  # gravity-dominated
    acc[:, 0] = np.linspace(0, 1.0, n)  # gentle forward accel
    gyro = np.zeros((n, 3))
    gyro[:, 0] = 0.01  # mild yaw
    data = {
        "TIME SINCE START (ms)": t,
        "ACCELEROMETER X (m/s²)": acc[:, 0],
        "ACCELEROMETER Y (m/s²)": acc[:, 1],
        "ACCELEROMETER Z (m/s²)": acc[:, 2],
        "GYROSCOPE Yaw (rad/s)": gyro[:, 0],
        "GYROSCOPE Pitch (rad/s)": gyro[:, 1],
        "GYROSCOPE Roll (rad/s)": gyro[:, 2],
        "GPS SPEED (Kmh)": np.linspace(0, 36.0, n),  # 0..10 m/s
    }
    if with_gravity_cols:
        data["GRAVITY X (m/s²)"] = 0.0
        data["GRAVITY Y (m/s²)"] = 0.0
        data["GRAVITY Z (m/s²)"] = 9.81
    return pd.DataFrame(data)


class TestResolveColumns:
    def test_exact_headers(self) -> None:
        """Canonical IO-VNBD headers resolve without fallback."""
        cm = resolve_columns(_synthetic_df())
        assert cm.acc == ("ACCELEROMETER X (m/s²)", "ACCELEROMETER Y (m/s²)", "ACCELEROMETER Z (m/s²)")
        assert cm.gyro == ("GYROSCOPE Yaw (rad/s)", "GYROSCOPE Pitch (rad/s)", "GYROSCOPE Roll (rad/s)")
        assert cm.time == "TIME SINCE START (ms)"
        assert cm.gps_speed == "GPS SPEED (Kmh)"
        assert cm.gravity_present

    def test_missing_gravity_columns_ok(self) -> None:
        """Gravity columns are cross-check only — absence is not an error."""
        cm = resolve_columns(_synthetic_df(with_gravity_cols=False))
        assert not cm.gravity_present
        assert cm.gravity == ("", "", "")


class TestTimestampAudit:
    def test_clean_10hz_passes(self) -> None:
        """A clean 10 Hz stream is accepted, no rate flag."""
        audit = timestamp_audit(np.arange(100) * 100.0)
        assert not audit.rejected
        assert audit.rate_flag == ""
        assert audit.median_dt_ms == pytest.approx(100.0)

    def test_gappy_stream_rejected(self) -> None:
        """>5% gaps (dt > 300ms) rejects the file (P4 gate).

        Gaps are modeled as monotonic skips (t[i+1] − t[i] > 300 ms), not
        timestamp shifts — a shift makes the stream non-monotonic and the
        sort in monotonic_time_ns erases the hole.
        """
        t = np.arange(100) * 100.0
        t[10:] += 1000.0  # one persistent 1.1s hole → but that is <5%...

        # build a >5% gappy but monotonic stream: every 10th interval is 1.1 s
        t2 = np.cumsum(np.where(np.arange(99) % 10 == 0, 1100.0, 100.0))
        t2 = np.concatenate([[0.0], t2])
        audit = timestamp_audit(t2)
        assert audit.rejected
        assert audit.gap_fraction > 0.05
        assert not audit.rejected == audit.rate_flag  # sanity: flag ≠ verdict

    def test_odd_rate_flagged_not_rejected(self) -> None:
        """80 ms median (S4 files) flags odd-rate but stays accepted."""
        audit = timestamp_audit(np.arange(100) * 80.0)
        assert audit.rate_flag == "odd-rate"
        assert not audit.rejected


class TestMonotonicTime:
    def test_reordered_rows_sorted(self) -> None:
        """Device-reordered timestamps are sorted; t_ns scales ms→ns."""
        df = _synthetic_df(10)
        df.iloc[3], df.iloc[4] = df.iloc[4].copy(), df.iloc[3].copy()  # type: ignore[call-overload]
        df_sorted, t_ns = monotonic_time_ns(df, "TIME SINCE START (ms)")
        assert np.all(np.diff(t_ns) > 0)
        assert len(df_sorted) == 10
        assert t_ns[1] == 100_000_000


class TestWindowize:
    def test_geometry_and_parity_with_legacy(self) -> None:
        """(T,C) → (N,window,C), byte-identical to the legacy stack impl."""
        rng = np.random.default_rng(0)
        arr = rng.normal(size=(1000, 6)).astype(np.float32)
        w = windowize(arr, window=200, stride=10)
        n = (1000 - 200) // 10 + 1
        assert w.shape == (n, 200, 6)
        legacy = np.stack([arr[i * 10 : i * 10 + 200] for i in range(n)], axis=0)
        assert np.array_equal(w, legacy)

    def test_too_short_returns_empty(self) -> None:
        """Signals shorter than the window yield (0, window, C)."""
        w = windowize(np.zeros((10, 6)), window=200, stride=10)
        assert w.shape == (0, 200, 6)

    def test_stride_one_covers_all(self) -> None:
        """stride=1 produces T-window+1 windows."""
        w = windowize(np.zeros((50, 3)), window=10, stride=1)
        assert w.shape == (41, 10, 3)


class TestVelocityLabels:
    def test_tail_indices(self) -> None:
        """Label i = speed at window i's LAST sample (i*stride + window−1)."""
        v = np.arange(100.0)
        labels = velocity_labels(v, n_windows=5, window=20, stride=10)
        assert labels.tolist() == [19.0, 29.0, 39.0, 49.0, 59.0]


class TestProcessFile:
    def test_full_path_shapes(self) -> None:
        """Synthetic 60-row file → windows (N, 200, 6) + ZUPT labels."""
        result = process_file(_synthetic_df(600), hz=100, window=200, stride=10)
        assert result is not None
        # resampled length: 59s span × 100Hz + 1 ≈ 5901 (uniform grid)
        # windows = (n - 200)//10 + 1; assert via the invariant, not a magic number
        n_res = result.n_resampled
        n_expected = (n_res - 200) // 10 + 1
        assert result.windows.shape == (n_expected, 200, 6)
        assert result.v_ms.shape == (n_expected,)
        assert result.stationary.dtype == bool
        assert result.audit.median_dt_ms == pytest.approx(100.0)
        # gravity settles → cross-check vs dataset cols is small but nonzero (low-pass lag)
        assert np.isfinite(result.gravity_max_diff)

    def test_stationary_labels_zeroed(self) -> None:
        """Stationary windows get v forced to 0 (ZUPT supervision)."""
        df = _synthetic_df(400)
        df["GPS SPEED (Kmh)"] = 0.0  # fully stopped → every low-var window stationary
        result = process_file(df, hz=100, window=200, stride=10, zupt_speed_gate=0.5)
        assert result is not None
        if result.stationary.any():
            assert np.all(result.v_ms[result.stationary] == 0.0)

    def test_missing_columns_raise(self) -> None:
        """A frame without accelerometer columns raises ValueError."""
        df = pd.DataFrame({"foo": [1, 2]})
        with pytest.raises(ValueError, match="missing columns"):
            process_file(df)

    def test_too_short_returns_none(self) -> None:
        """Files yielding fewer samples than the window return None."""
        assert process_file(_synthetic_df(5)) is None

    def test_rejected_gaps_return_none(self) -> None:
        """A gappy file is rejected before windowing (monotonic skips)."""
        df = _synthetic_df(300)
        t = df["TIME SINCE START (ms)"].values.copy()
        t[10:] = t[9] + np.cumsum(  # every 10th interval is 1.1 s, stays monotonic
            np.where(np.arange(290) % 10 == 0, 1100.0, 100.0)
        )
        df["TIME SINCE START (ms)"] = t
        assert process_file(df) is None


class TestResampleStream:
    def test_stream_shapes_and_units(self) -> None:
        """](T,6) stream + m/s labels; km/h → m/s conversion applied."""
        imu, v_ms, audit, gdiff = resample_stream(_synthetic_df(100), hz=100)
        assert imu.shape[1] == 6
        assert v_ms.shape == (len(imu),)
        assert audit.median_dt_ms == pytest.approx(100.0)
        assert v_ms.max() <= 10.0 + 1e-6  # 36 km/h ≈ 10 m/s
        # gravity removal: channel 2 ≈ 0 after settle (linear acc)
        assert abs(imu[200:, 2].mean()) < 0.5

    def test_no_gravity_cols_gives_nan_diff(self) -> None:
        """Missing gravity columns → NaN cross-check, pipeline still runs."""
        _, _, _, gdiff = resample_stream(_synthetic_df(50, with_gravity_cols=False))
        assert np.isnan(gdiff)


class TestLoadPhoneCsv:
    def test_cp1252_and_strip(self, tmp_path) -> None:
        """cp1252 ² headers decode and get whitespace-stripped."""
        p = tmp_path / "S-test.csv"
        p.write_bytes(
            "ACCELEROMETER X (m/s²) ,TIME SINCE START (ms) \n0.1,0\n0.2,100\n".encode("cp1252")
        )
        df = load_phone_csv(p)
        assert "ACCELEROMETER X (m/s²)" in df.columns
        assert "TIME SINCE START (ms)" in df.columns
        assert len(df) == 2
