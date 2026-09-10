#!/usr/bin/env python3
"""Preprocess IO-VNBD S-files into normalized window npys + scaler.json.

Pipeline per file (single implementation in python/datasets/iovnbd.py):
``load → resolve → dt-audit → sort → gravity(spec v2) → resample 10→100Hz →
window (200, stride 10) → ZUPT labels``; then train-only scaler, chunked
memmap save (no OOM on the full 822k-window set), spec fingerprint stamp.

Usage:
    python python/preprocess.py --subset 1h          # smoke (<5 min)
    python python/preprocess.py --window 200 --stride 10 --hz 100
    python python/preprocess.py --split stratified   # ADR-010 branch A
"""

from __future__ import annotations

import argparse
import glob
import json
import signal
import sys
from pathlib import Path

import numpy as np
from loguru import logger

from python.config import (
    HZ,
    IOVNBD_BASE,
    PROCESSED_DIR,
    SCALER_PATH,
    STRIDE,
    TRAIN_RATIO,
    WINDOW,
    processed_paths,
)
from python.core.runlog import init_runlog
from python.core.scaler import TrainOnlyScaler
from python.core.spec import attach_spec
from python.datasets.iovnbd import load_phone_csv, process_file
from python.datasets.split import split_files

_CHUNK = 50_000  # memmap save chunk (windows per write)


def _process_file_list(
    file_list: list[str], hz: int, window: int, stride: int,
    interrupted: dict, subset_label: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Window every non-rejected file; stop early on Ctrl-C (partial save)."""
    all_windows: list[np.ndarray] = []
    all_v: list[np.ndarray] = []
    for idx_f, f in enumerate(file_list):
        if interrupted["flag"]:
            logger.warning(f"[interrupt] stopping after {idx_f}/{len(file_list)} files, saving partial...")
            break
        try:
            df = load_phone_csv(f)
            result = process_file(df, hz=hz, window=window, stride=stride)
            short_name = f"{Path(f).parent.name}/{Path(f).name}"
            if result is None:
                # Distinguish reject (gaps) vs too-short by re-running audit
                logger.info(f"[skip] {short_name}: rejected or too short for window={window}")
                continue
            grav_str = "n/a" if np.isnan(result.gravity_max_diff) else f"{result.gravity_max_diff:.2f}"
            logger.info(
                f"[ok] {short_name}: T->{result.n_resampled} windows={len(result.windows)} "
                f"stationary={result.stationary.sum()} median_dt={result.audit.median_dt_ms:.1f}ms "
                f"gap_frac={result.audit.gap_fraction:.1%} {result.audit.rate_flag} grav_xdiff={grav_str}"
            )
            all_windows.append(result.windows)
            all_v.append(result.v_ms)
        except Exception as e:  # noqa: BLE001 — one bad file must not kill the run
            logger.error(f"[err] {f}: {e}")
            import traceback

            traceback.print_exc()
            continue
    if not all_windows:
        return np.empty((0, window, 6)), np.empty((0,))
    return np.concatenate(all_windows, axis=0), np.concatenate(all_v, axis=0)


def _save_windows_memmap(
    path: Path, X: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> None:
    """Chunked memmap save: never hold X and X_normalized in RAM at once.

    This is what OOM-killed the Colab run on the full 822k set (7.8 GB
    peak) — np.lib.format.open_memmap writes a proper .npy header that
    np.load(mmap_mode='r') accepts.
    """
    fp = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=X.shape)
    for start in range(0, X.shape[0], _CHUNK):
        end = min(X.shape[0], start + _CHUNK)
        fp[start:end] = ((X[start:end].astype(np.float64) - mean) / std).astype(np.float32)
    del fp  # flush via __del__


def main() -> None:
    """CLI entry: parse, split, window, scale, save npys + scaler.json."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", choices=["1h", "full"], default="full")
    ap.add_argument("--window", type=int, default=WINDOW)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--hz", type=int, default=HZ)
    ap.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    ap.add_argument("--split", choices=["random", "stratified"], default="random",
                    help="random=seeded shuffle (default, backward compat); stratified=80/20 within driver×speed buckets (ADR-010 branch A)")
    ap.add_argument("--out", default=str(PROCESSED_DIR))
    ap.add_argument("--scaler", default=str(SCALER_PATH))
    ap.add_argument("--base", default=str(IOVNBD_BASE),
                    help="dataset base dir containing **/S-*.csv")
    ap.add_argument("--resume", action="store_true",
                    help="skip if output npys already exist (rm -rf to force)")
    ap.add_argument("--log-dir", default=None, help="optional dir for a run log file")
    args = ap.parse_args()

    init_runlog("preprocess", args.log_dir)

    s_files = sorted(glob.glob(str(Path(args.base) / "**/S-*.csv"), recursive=True))
    if args.subset == "1h":
        s_files = s_files[:3]
    logger.info(
        f"[preprocess] {len(s_files)} S files, window={args.window} "
        f"stride={args.stride} hz={args.hz} resume={args.resume}"
    )
    if not s_files:
        logger.error(f"[err] no S-*.csv under {args.base} — run download_iovnbd.py first")
        sys.exit(1)

    out = Path(args.out)
    paths = processed_paths(out)
    if args.resume and paths["train_windows"].exists() and paths["val_windows"].exists():
        gb = paths["train_windows"].stat().st_size / 1e9
        logger.info(
            f"[resume] {paths['train_windows']} exists ({gb:.2f}GB), skipping. "
            f"Use --no-resume or rm -rf {out} to force."
        )
        return

    # Split by trajectory (file), never by window — leakage guard.
    train_files, val_files = split_files(s_files, strategy=args.split, train_ratio=args.train_ratio)
    logger.info(f"[split:{args.split}] train files {len(train_files)} val files {len(val_files)} (by trajectory)")

    # Graceful Ctrl-C: save partial progress, scaler from what we have.
    interrupted = {"flag": False}

    def handle_sigint(sig, frame) -> None:
        interrupted["flag"] = True
        logger.warning("\n[interrupt] Ctrl-C detected, will save partial progress and exit...")

    orig_handler = signal.signal(signal.SIGINT, handle_sigint)
    try:
        X_train, v_train = _process_file_list(train_files, args.hz, args.window, args.stride, interrupted, args.subset)
        X_val, v_val = _process_file_list(val_files, args.hz, args.window, args.stride, interrupted, args.subset)
    finally:
        signal.signal(signal.SIGINT, orig_handler)

    if interrupted["flag"]:
        if len(X_train) > 0:
            _finalize(args, out, paths, X_train, v_train, X_val, v_val, partial=True)
        return

    logger.info(
        f"[concat] train X {X_train.shape} v {v_train.shape} | "
        f"val X {X_val.shape} v {v_val.shape}"
    )
    if len(X_train) == 0 or len(X_val) == 0:
        logger.error("[err] no windows — check dataset/download + quality gates")
        sys.exit(1)
    logger.info(f"  train mean {X_train.mean(axis=(0, 1))} std {X_train.std(axis=(0, 1))}")
    logger.info(f"  val mean {X_val.mean(axis=(0, 1))} std {X_val.std(axis=(0, 1))}")

    _finalize(args, out, paths, X_train, v_train, X_val, v_val, partial=False)


def _finalize(args, out: Path, paths: dict[str, Path],
               X_train, v_train, X_val, v_val, partial: bool) -> None:
    """Fit train-only scaler, save npys (chunked memmap), stamp spec."""
    scaler_path = Path(args.scaler)
    scaler = TrainOnlyScaler()
    scaler.fit(X_train, train_files=[f"train:{args.split}"])
    scaler.save(scaler_path)

    # Attach run metadata alongside train-only stats, then the spec stamp (P2).
    with open(scaler_path) as f:
        sj = json.load(f)
    sj.update({"hz": args.hz, "window": args.window, "stride": args.stride})
    if partial:
        sj["partial"] = True
    with open(scaler_path, "w") as f:
        json.dump(sj, f, indent=2)
    attach_spec(scaler_path)
    mean, std = scaler.mean, scaler.std
    logger.info(f"[scaler] {scaler_path} mean {mean} std {std}")

    out.mkdir(parents=True, exist_ok=True)
    try:
        np.save(paths["train_v"], v_train.astype(np.float32))
        np.save(paths["val_v"], v_val.astype(np.float32))
        _save_windows_memmap(paths["train_windows"], X_train, mean, std)
        if len(X_val) > 0:
            _save_windows_memmap(paths["val_windows"], X_val, mean, std)
        logger.info(f"[save] {out}/train_windows.npy via memmap (no OOM)")
    except Exception as e:  # noqa: BLE001 — fallback keeps the scaler usable for streaming
        logger.warning(f"[warn] memmap save failed ({e}), falling back to scaler-only + streaming.")
        import traceback

        traceback.print_exc()
        (out / ".streaming").touch()
        logger.info(f"[save] scaler only at {scaler_path}, train with streaming")
        return
    if partial:
        logger.warning(f"[interrupt] partial saved to {out}, re-run with --resume or rm -rf to restart")


if __name__ == "__main__":
    main()
