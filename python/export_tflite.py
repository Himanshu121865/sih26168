#!/usr/bin/env python3
"""Export CLI (Step 11): PyTorch → ONNX → TFLite + validation gate.

All logic lives in python/export/tflite.py; this is CLI wiring. The P2
gate REFUSES (exit 2) to pair a model with an unversioned or spec-mismatched
scaler, then stamps model_manifest.json binding model ↔ scaler ↔ spec.

Usage:
    python python/export_tflite.py --model experiments/checkpoints/model_avnet_stage1.p --out model.tflite --validate 1000
"""

from __future__ import annotations

import argparse
import sys

from python.core.runlog import init_runlog


def main() -> None:
    """CLI entry: verify scaler spec, export, validate, stamp manifest."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p",
                    help="PyTorch checkpoint (or 'none' for random weights pipeline test)")
    ap.add_argument("--out", default="model.tflite")
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--validate", type=int, default=1000, help="num windows to validate")
    ap.add_argument("--val-windows", default="data/processed/val_windows.npy")
    ap.add_argument("--scaler", default="python/scaler.json",
                    help="scaler.json to pair (spec gate refuses mismatches)")
    ap.add_argument("--quant", choices=["fp32", "fp16"], default="fp16")
    ap.add_argument("--log-dir", default=None, help="optional dir for a run log file")
    args = ap.parse_args()

    init_runlog("export", args.log_dir)

    from python.export.tflite import run_export

    sys.exit(run_export(
        model_path=args.model,
        out_path=args.out,
        onnx_path=args.onnx,
        n_validate=args.validate,
        val_windows=args.val_windows,
        quant=args.quant,
        scaler_path=args.scaler,
    ))


if __name__ == "__main__":
    main()
