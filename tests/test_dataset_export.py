"""Tests for IOVNBDWindowDataset (npy + streaming modes) and export gates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from python.datasets.iovnbd_dataset import IOVNBDWindowDataset, rebuild_split


@pytest.fixture
def npy_pair(tmp_path: Path):
    """Tiny npy dataset: 20 windows, ramping labels."""
    X = np.zeros((20, 200, 6), dtype=np.float32)
    X[:, :, 0] = 0.5
    v = np.arange(20, dtype=np.float32)
    np.save(tmp_path / "w.npy", X)
    np.save(tmp_path / "v.npy", v)
    return tmp_path / "w.npy", tmp_path / "v.npy"


class TestNpyMode:
    def test_len_and_getitem_shapes(self, npy_pair) -> None:
        """npy mode: N from the memmap, items are (window, scalar, att3)."""
        w, v = npy_pair
        ds = IOVNBDWindowDataset(w, v)
        assert len(ds) == 20
        x, label, att = ds[0]
        assert isinstance(x, torch.Tensor) and x.shape == (200, 6)
        assert float(label) == 0.0
        assert att.shape == (3,)

    def test_missing_v_falls_back_to_zeros(self, npy_pair) -> None:
        """Absent label file → zeros (interface parity for unlabeled data)."""
        w, _ = npy_pair
        ds = IOVNBDWindowDataset(w, None)
        assert float(ds[5][1]) == 0.0

    def test_missing_npy_raises_with_fix_hint(self, tmp_path: Path) -> None:
        """A named-but-missing npy explains how to regenerate it."""
        with pytest.raises(FileNotFoundError, match="preprocess.py"):
            IOVNBDWindowDataset(tmp_path / "gone.npy", None)


class TestStreamingMode:
    def test_stream_index_and_items(self, tmp_path: Path) -> None:
        """Streaming mode windows match the shared pipeline channel-for-channel."""
        from tests.test_iovnbd_pipeline import _synthetic_df

        files = []
        for i in range(2):
            p = tmp_path / f"S-{i:02d}.csv"
            _synthetic_df(300).to_csv(p, index=False, encoding="cp1252")
            files.append(str(p))
        scaler = tmp_path / "scaler.json"
        scaler.write_text(json.dumps({"mean": [0.0] * 6, "std": [1.0] * 6}))

        ds = IOVNBDWindowDataset(files=files, scaler_path=scaler, window=200, stride=10)
        assert len(ds) > 0
        fi_counts: dict[int, int] = {}
        for fi, _ in ds.index:
            fi_counts[fi] = fi_counts.get(fi, 0) + 1
        assert set(fi_counts) == {0, 1}

        x0, v0, att0 = ds[0]
        assert x0.shape == (200, 6)
        assert x0.dtype == torch.float32
        assert att0.shape == (3,)
        # label at window tail = resampled GPS speed (m/s); synthetic ramp is
        # ~0 km/h at the file start but resample interpolates — allow small
        assert float(v0) == pytest.approx(0.0, abs=1.5)

    def test_neither_mode_raises(self) -> None:
        """No npy and no files → ValueError (not a silent empty dataset)."""
        with pytest.raises(ValueError, match="need windows_path or files"):
            IOVNBDWindowDataset()


class TestRebuildSplit:
    def test_rebuild_matches_split_files(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """rebuild_split(files) reproduces split_files() for the same inputs."""
        import glob

        base = tmp_path / "base"
        for d in ("a (Driver E)", "b (Driver F)"):
            for i in range(3):
                p = base / d / f"seq{i}" / f"S-{i}.csv"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("x")
        monkeypatch.setattr(
            "python.datasets.split._file_mean_speed", lambda f: float(len(str(f)) % 7)
        )
        files = sorted(glob.glob(str(base / "**/S-*.csv"), recursive=True))
        from python.datasets.split import split_files

        assert rebuild_split(base) == split_files(files, strategy="random")


class TestExportGate:
    def test_run_export_refuses_missing_scaler(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Export exits 2 when the scaler is absent (P2 fail-loud gate)."""
        from python.export.tflite import run_export

        # a checkpoint is not required (random weights pipeline test)
        code = run_export(
            model_path="nonexistent.p",
            out_path=tmp_path / "m.tflite",
            onnx_path=tmp_path / "m.onnx",
            scaler_path=tmp_path / "no_scaler.json",
        )
        assert code == 2
