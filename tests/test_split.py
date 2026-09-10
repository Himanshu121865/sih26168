"""Tests for the by-trajectory split strategies (leakage guard)."""

from __future__ import annotations

import pytest

from python.datasets.split import random_split, split_files, stratified_split


def _fake_files(n: int = 20) -> list[str]:
    """Synthetic S-file paths across 3 driver dirs."""
    out = []
    for d in ("Vtb (Driver E)", "Vtc (Driver F)", "Vtd (Driver G)"):
        out += [f"data/iovnbd/x/{d}/seq{i:02d}/S-{d[-1]}{i:02d}.csv" for i in range(n // 3)]
    return sorted(out)


class TestRandomSplit:
    def test_no_overlap_and_coverage(self) -> None:
        """Train ∩ val = ∅ and train ∪ val = all files."""
        files = _fake_files()
        train, val = random_split(files)
        assert set(train) & set(val) == set()
        assert sorted(train + val) == files

    def test_deterministic(self) -> None:
        """Same seed → same split (Colab reproducibility)."""
        files = _fake_files()
        assert random_split(files) == random_split(files)

    def test_ratio(self) -> None:
        """Train count = int(N × ratio) (legacy int() semantics)."""
        files = _fake_files(20)
        assert len(files) == 18  # 3 drivers × 6 (20//3)
        train, _ = random_split(files, train_ratio=0.8)
        assert len(train) == int(18 * 0.8)  # 14, floor semantics preserved


class TestStratifiedSplit:
    def test_no_overlap_and_coverage(self, tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Stratified split also covers all files exactly once."""
        files = _fake_files()
        # stub speed reading (no dataset on CI)
        monkeypatch.setattr(
            "python.datasets.split._file_mean_speed", lambda p: float(hash(p) % 30)
        )
        train, val = stratified_split(files)
        assert set(train) & set(val) == set()
        assert sorted(train + val) == files

    def test_deterministic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same input + seed → same stratified split."""
        monkeypatch.setattr(
            "python.datasets.split._file_mean_speed", lambda p: float(len(p) % 25)
        )
        files = _fake_files()
        assert stratified_split(files) == stratified_split(files)


class TestSplitFiles:
    def test_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """split_files routes to the chosen strategy."""
        monkeypatch.setattr(
            "python.datasets.split._file_mean_speed", lambda p: 10.0
        )
        files = _fake_files()
        assert split_files(files, "random") == random_split(files)

    def test_unknown_strategy_raises(self) -> None:
        """An unknown strategy fails loud, never silently falls back."""
        with pytest.raises(ValueError, match="unknown split strategy"):
            split_files(["a"], "typo")
