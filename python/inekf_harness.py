#!/usr/bin/env python3
"""InEKF replay harness CLI (Step 9) — validate the filter before Kotlin.

All logic lives in python/fusion (InEKF + replay); this is CLI wiring:
``--test-lean`` runs the synthetic φ=30° NHC check, the default replays a
val segment and appends to reports/inekf_vs_avnet.csv.

Usage:
    PYTHONPATH=. python python/inekf_harness.py --model experiments/checkpoints/model_avnet_stage1.p
    PYTHONPATH=. python python/inekf_harness.py --test-lean
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from python.core.runlog import init_runlog
from python.fusion.ine_kf import InEKF, gravity_align_R  # noqa: F401
from python.fusion.replay import run_replay, test_lean

# Re-exported for the Kotlin port and any `from python.inekf_harness import
# skew` callers (F2: single source of truth is python/utils/lie_group.py).
from python.utils.lie_group import sen3exp, skew, so3exp  # noqa: F401


def main() -> None:
    """CLI entry: test-lean or a replay segment → CSV append."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p")
    ap.add_argument("--windows", type=int, default=600)
    ap.add_argument("--lean-mode", choices=["auto", "car", "bike"], default="auto")
    ap.add_argument("--start", type=int, default=None, help="window index to start segment")
    ap.add_argument("--q-acc", type=float, default=30.0,
                    help="accel process noise (Python 10Hz proxy default 30.0; Android live uses 0.5)")
    ap.add_argument("--zupt-gate", type=float, default=0.3,
                    help="v_fwd below this → ZUPT v=0 (first 5 samples fallback)")
    ap.add_argument("--no-variance-zupt", action="store_true",
                    help="disable variance detector, use speed heuristic only")
    ap.add_argument("--test-lean", action="store_true")
    ap.add_argument("--csv", default="reports/inekf_vs_avnet.csv")
    ap.add_argument("--val-windows", default=None,
                    help="override data/processed/val_windows.npy")
    ap.add_argument("--val-v", default=None,
                    help="override data/processed/val_v.npy")
    ap.add_argument("--scaler", default=None,
                    help="override python/scaler.json")
    ap.add_argument("--log-dir", default=None, help="optional dir for a run log file")
    args = ap.parse_args()

    init_runlog("harness", args.log_dir)

    if args.test_lean:
        test_lean()
        return

    if args.scaler is not None:
        import json

        import numpy as np

        import python.fusion.replay as _replay

        with open(args.scaler) as f:
            sc = json.load(f)
        stats = (
            np.array(sc["mean"], dtype=np.float64),
            np.array(sc["std"], dtype=np.float64),
        )
        _replay.load_scaler = lambda path=None: stats  # type: ignore[assignment]

    metrics = run_replay(
        args.model,
        n_windows=args.windows,
        start=args.start,
        lean_mode=args.lean_mode,
        q_acc=args.q_acc,
        zupt_speed_gate=args.zupt_gate,
        use_variance_zupt=not args.no_variance_zupt,
        val_windows=args.val_windows,
        val_v=args.val_v,
    )
    csv_path = Path(args.csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a") as f:
        if write_header:
            f.write("trajectory,naive_drift_pct,avnet_drift_pct,inekf_drift_pct\n")
        f.write(
            f"val_start{metrics['start']}_{metrics['segment_windows']}w_{metrics['total_dist_m']:.0f}m,"
            f"{metrics['naive_drift_pct']:.2f},{metrics['avnet_drift_pct']:.2f},{metrics['inekf_drift_pct']:.2f}\n"
        )
    logger.info(f"[csv] appended {csv_path}")


if __name__ == "__main__":
    main()
