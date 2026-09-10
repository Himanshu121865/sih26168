"""By-trajectory train/val split (never by window — leakage guard).

Two strategies, one seed family (26168) so runs are reproducible:

- ``random``       — seeded file-level shuffle (legacy default; the
  notebook/checkpoints use this).
- ``stratified``   — 80/20 within (driver × speed-tercile) buckets
  (ADR-010 branch A), for when the per-file audit shows a few files
  dominating weighted val MSE.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from loguru import logger

from python.config import SPLIT_SEED, TRAIN_RATIO


def _driver_of(path: str | Path) -> str:
    """Driver letter from the category dir, e.g. 'Vtb (Driver E)' → 'E'."""
    m = re.search(r"Driver ([A-Z])", Path(path).parent.parent.name)
    return m.group(1) if m else "?"


def _file_mean_speed(path: str | Path) -> float:
    """Mean GPS speed (km/h) of one file — 0.0 on any read failure."""
    # Late import keeps pandas out of module import time.
    from python.datasets.iovnbd import load_phone_csv, resolve_columns

    try:
        df = load_phone_csv(path)
        cols = resolve_columns(df)
        if cols.gps_speed in df.columns:
            vals = np.asarray(df[cols.gps_speed].values, dtype=np.float64)
            return float(np.nanmean(vals))
    except Exception:  # noqa: BLE001 — audit must survive broken files
        pass
    return 0.0


def random_split(
    files: list[str], train_ratio: float = TRAIN_RATIO, seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """Seeded by-file shuffle (legacy behavior, backward compatible).

    Args:
        files: Sorted S-file paths.
        train_ratio: Fraction of files assigned to train.
        seed: Split seed (26168 — same family as the Colab notebook).

    Returns:
        (train_files, val_files); deterministic for identical inputs.
    """
    n_train = int(len(files) * train_ratio)
    rng = np.random.default_rng(seed)
    train_idx = set(rng.permutation(len(files))[:n_train].tolist())
    train = [f for i, f in enumerate(files) if i in train_idx]
    val = [f for i, f in enumerate(files) if i not in train_idx]
    return train, val


def stratified_split(
    files: list[str], train_ratio: float = TRAIN_RATIO, seed: int = SPLIT_SEED
) -> tuple[list[str], list[str]]:
    """Split 80/20 within (driver × speed-tercile) buckets.

    Buckets with a single file go to train (a one-file bucket cannot split).
    Speed terciles are computed from mean GPS speed across all files, so
    the split is deterministic given the same file set.

    Args:
        files: S-file paths.
        train_ratio: Fraction of each bucket assigned to train.
        seed: Base seed; each bucket draws ``seed*1000 + bucket_index``.

    Returns:
        (train_files, val_files).
    """
    means = {f: _file_mean_speed(f) for f in files}
    ordered = sorted(means.values())
    lo, hi = ordered[len(ordered) // 3], ordered[2 * len(ordered) // 3]
    buckets: dict[tuple[str, str], list[str]] = {}
    for f in files:
        band = "lo" if means[f] < lo else ("hi" if means[f] > hi else "mid")
        buckets.setdefault((_driver_of(f), band), []).append(f)

    train: list[str] = []
    val: list[str] = []
    for bi, key in enumerate(sorted(buckets)):
        members = sorted(buckets[key])
        if len(members) == 1:
            train.extend(members)
            logger.info(f"[strat] bucket {key}: n=1 -> train (singleton)")
            continue
        rng = np.random.default_rng(seed * 1000 + bi)
        perm = rng.permutation(len(members))
        n_tr = max(1, int(len(members) * train_ratio))
        tr_idx = set(perm[:n_tr].tolist())
        for i, m in enumerate(members):
            (train if i in tr_idx else val).append(m)
        logger.info(f"[strat] bucket {key}: n={len(members)} -> train {n_tr} val {len(members) - n_tr}")
    return train, val


def split_files(
    files: list[str],
    strategy: str = "random",
    train_ratio: float = TRAIN_RATIO,
    seed: int = SPLIT_SEED,
) -> tuple[list[str], list[str]]:
    """Dispatch to the chosen split strategy (never window-level)."""
    if strategy == "stratified":
        return stratified_split(files, train_ratio, seed)
    if strategy == "random":
        return random_split(files, train_ratio, seed)
    raise ValueError(f"unknown split strategy {strategy!r} (random|stratified)")
