"""Window-spec versioning + asset fingerprints (Window-Path Hardening P2).

The window path (raw CSV → resample → gravity-remove → normalize → (200,6))
changes silently unless every artifact declares which spec it was built under.
This module is the single implementation of:

- ``SPEC_VERSION``: 1 = dataset-gravity-columns era, 2 = unified live
  low-pass gravity (P1). Bump ONLY with a full retrain + re-export.
- ``window_sha256``: content hash of the normalized window pipeline.
- ``attach_spec`` / ``verify_scaler``: scaler.json gains ``spec_version`` +
  ``spec_sha256``; consumers refuse mismatched pairs (fail loud, not log-only).

Consumers:
- ``python/preprocess.py`` stamps scaler.json at save time.
- ``python/export_tflite.py`` stamps + verifies model_manifest.json and
  refuses to ship a model whose scaler doesn't match its training spec.
- Android ``WindowSpecGuard.kt`` reads the same fields and refuses
  inference on mismatch (see docs/WINDOW_SPEC.md).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# v1: gravity from dataset GRAVITY X/Y/Z columns (pre-P1).
# v2: gravity via live low-pass gEst (P1, matches LeanDetector.kt).
SPEC_VERSION = 2

# Canonical spec description hashed into spec_sha256. Any change to the
# window path (column order, resample, gravity, normalization) MUST update
# this string and bump SPEC_VERSION.
SPEC_TEXT = (
    "window-spec v2: channels [linAcc xyz, gyro yaw/pitch/roll]; "
    "resample linear-interp to uniform grid hz (period_ns=round(1e9/hz), "
    "edges NaN+finite-mask); gravity REMOVED via live low-pass "
    "gEst+=0.02*(acc-gEst) init [0,0,9.81] BEFORE resample (dataset gravity "
    "columns are cross-check only); normalize (x-mean)/std train-only per "
    "channel; sliding window 200 stride 10 oldest-first; float32."
)

_GRAVITY_ALPHA = 0.02  # LeanDetector.kt / estimate_gravity_lowpass parity


def spec_sha256() -> str:
    """SHA-256 of the canonical spec text (stable across runs/platforms)."""
    return hashlib.sha256(SPEC_TEXT.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (streamed, 1 MiB chunks)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def attach_spec(scaler_path: str | Path, spec_version: int = SPEC_VERSION) -> dict:
    """Stamp a scaler.json with ``spec_version`` + ``spec_sha256`` in place.

    Args:
        scaler_path: Path to the scaler.json written by preprocess.
        spec_version: Spec the fitting run used (default: current).

    Returns:
        The updated JSON dict (also written back to the file).

    Raises:
        FileNotFoundError: If scaler_path does not exist.
    """
    p = Path(scaler_path)
    d = json.loads(p.read_text())
    d["spec_version"] = spec_version
    d["spec_sha256"] = spec_sha256()
    d["gravity_alpha"] = _GRAVITY_ALPHA
    p.write_text(json.dumps(d, indent=2))
    return d


def verify_scaler(
    scaler_path: str | Path, expect_version: int = SPEC_VERSION
) -> dict:
    """Load + verify a scaler.json against the expected window spec.

    Args:
        scaler_path: Path to a stamped scaler.json.
        expect_version: Required spec version (refuse older/newer).

    Returns:
        The parsed JSON dict when the fingerprint matches.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If spec fields are missing, stale, or the hash mismatches.
    """
    p = Path(scaler_path)
    d = json.loads(p.read_text())
    got_ver = d.get("spec_version")
    got_hash = d.get("spec_sha256")
    if got_ver is None or got_hash is None:
        raise ValueError(
            f"{p}: scaler.json has NO spec fingerprint (spec_version/spec_sha256 "
            f"missing) — regenerate with preprocess.py (spec v{SPEC_VERSION}). "
            "Refusing to pair an unversioned scaler with a model."
        )
    if got_ver != expect_version:
        raise ValueError(
            f"{p}: spec_version {got_ver} != expected {expect_version} — "
            "retrain or re-export before pairing."
        )
    if got_hash != spec_sha256():
        raise ValueError(
            f"{p}: spec_sha256 mismatch — window path drifted from the "
            "canonical spec (docs/WINDOW_SPEC.md). Refusing."
        )
    return d


def write_model_manifest(
    out_path: str | Path,
    model_path: str | Path,
    scaler_path: str | Path,
    extra: dict | None = None,
) -> dict:
    """Write model_manifest.json binding model ↔ scaler ↔ spec.

    The export gate and the Android startup guard both read this file (or
    recompute its fields), so a model can never silently ship against a
    scaler from a different window spec.

    Args:
        out_path: Destination manifest path.
        model_path: Exported model file (hashed, not copied).
        scaler_path: The scaler.json this model was trained under.
        extra: Optional additional fields (e.g. val MSE, drift numbers).

    Returns:
        The manifest dict (also written to out_path).
    """
    scaler = verify_scaler(scaler_path)
    manifest = {
        "spec_version": SPEC_VERSION,
        "spec_sha256": spec_sha256(),
        "model_sha256": file_sha256(model_path),
        "scaler_sha256": file_sha256(scaler_path),
        "scaler_spec_version": scaler["spec_version"],
        "gravity_alpha": _GRAVITY_ALPHA,
    }
    if extra:
        manifest.update(extra)
    Path(out_path).write_text(json.dumps(manifest, indent=2))
    return manifest


def verify_model_manifest(
    manifest_path: str | Path, model_path: str | Path, scaler_path: str | Path
) -> dict:
    """Verify a manifest against the actual model + scaler files on disk.

    Raises:
        ValueError: On any mismatch (fail loud — the caller refuses to ship).
    """
    m = json.loads(Path(manifest_path).read_text())
    for key, path in (("model_sha256", model_path), ("scaler_sha256", scaler_path)):
        got = file_sha256(path)
        if m.get(key) != got:
            raise ValueError(
                f"{manifest_path}: {key} mismatch — {path} changed after export. "
                "Re-run export_tflite.py."
            )
    verify_scaler(scaler_path, expect_version=m.get("spec_version", SPEC_VERSION))
    return m
