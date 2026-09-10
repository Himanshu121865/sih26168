"""Centralized configuration: every default path/constant in one place.

Scripts previously hardcoded `data/processed/val_windows.npy`,
`python/scaler.json`, etc. in a dozen places — this module is the single
source of truth. Every value is overridable via the corresponding CLI flag,
the constants here are just the frozen defaults (docs/WINDOW_SPEC.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# --- repository layout (defaults; override per-script when needed) ---------

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IOVNBD_BASE = DATA_DIR / "iovnbd" / "Synchronised V abd S datasets" / "Categorised IOVNB Dataset"
SCALER_PATH = REPO_ROOT / "python" / "scaler.json"
CHECKPOINT_PATH = REPO_ROOT / "experiments" / "checkpoints" / "model_avnet_stage1.p"
REPORTS_DIR = REPO_ROOT / "reports"
GOLDEN_VECTOR_ASSET = REPO_ROOT / "android" / "app" / "src" / "main" / "assets" / "window_golden.json"

# --- window spec (frozen — mirrors docs/WINDOW_SPEC.md; see core/spec.py) ---

WINDOW = 200          # samples @100 Hz = 2 s (paper West=200)
STRIDE = 10            # 10 Hz output cadence
HZ = 100              # uniform resample rate
TRAIN_RATIO = 0.8
SPLIT_SEED = 26168    # also used by the Colab notebook — do not change


# --- data quality gates (Window-Path Hardening P4) --------------------------

@dataclass(frozen=True, slots=True)
class QualityGates:
    """Timestamp discipline thresholds applied per source file.

    Attributes:
        gap_fraction_max: Reject a file when more than this fraction of dt
            samples exceed the gap threshold (holey interpolation poisons
            windows).
        gap_factor: A dt counts as a gap when it exceeds ``gap_factor *
            median_dt`` (and at least ``gap_min_ms``).
        gap_min_ms: Absolute floor for the gap threshold in milliseconds.
        expected_dt_ms: Nominal IO-VNBD S-file sample period (10 Hz).
        rate_tolerance_ms: Median dt further than this from expected flags
            the file as odd-rate (logged, not rejected).
    """

    gap_fraction_max: float = 0.05
    gap_factor: float = 3.0
    gap_min_ms: float = 150.0
    expected_dt_ms: float = 100.0
    rate_tolerance_ms: float = 15.0


QUALITY_GATES = QualityGates()


def processed_paths(out_dir: str | Path = PROCESSED_DIR) -> dict[str, Path]:
    """Return the conventional processed-artifact paths under ``out_dir``.

    Args:
        out_dir: Base directory for npy artifacts (default ``data/processed``).

    Returns:
        Dict with keys ``train_windows``, ``train_v``, ``val_windows``,
        ``val_v`` mapping to file paths inside ``out_dir``.
    """
    base = Path(out_dir)
    return {
        "train_windows": base / "train_windows.npy",
        "train_v": base / "train_v.npy",
        "val_windows": base / "val_windows.npy",
        "val_v": base / "val_v.npy",
    }
