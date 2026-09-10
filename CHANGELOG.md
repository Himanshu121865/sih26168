# Changelog

Keep-a-Changelog format. `main` is unreleased by default; releases are cut by
pushing a `v*` tag, which triggers the release workflow (APK + screening
bundle attached to the GitHub Release).

## [Unreleased]

### Changed — Production refactor (2026-09-10)
- **Single window pipeline:** the three hand-rolled copies (preprocess,
  streaming dataset, per-file audit) merged into
  `python/datasets/iovnbd.py` (`process_file`/`resample_stream`). The
  per-file audit previously used spec-v1 dataset gravity columns — it now
  runs the spec-v2 live low-pass path, so audit numbers match training.
- **Layered architecture:** new `python/fusion/` (InEKF + replay), moved
  out of the 383-line harness; new `python/export/` (export logic + thin
  CLI); `python/eval/drift.py` extracted from eval_drift.py. CLIs are now
  argparse wiring only over importable modules.
- **Single training implementation:** `python/core/training.py` is the only
  copy of the loop/augmentations/NLL (train_avnet.py duplicated it).
- **Centralized config:** `python/config.py` — every default path/constant
  (previously hardcoded in a dozen places) incl. P4 quality gates.
- **Unified splits:** `python/datasets/split.py`
  (random/stratified), consumed by preprocess and the audit via
  `rebuild_split`.
- Typed + docstringed all public modules; ruff clean (E/F/I/UP/B);
  strict-mypy clean on the pure core; `__init__` exports everywhere.
- Adapter deprecation now lazy (`__getattr__` warning, not import-time).
- Fixed `run_replay` ZeroDivisionError on fully-stopped segments (drift% is
  reported against 1 m scale on degenerate distances).

### Added
- Test suite 43 → 125 tests (coverage 17% → 59%; core/fusion/models/pipeline
  at 96–100%): unified-pipeline parity, split determinism/leakage, Lie-group
  orthonormality, InEKF physics (parabola, PSD, clamps), model shape/checkpoint
  compat, training-loop convergence, drift contract, streaming dataset,
  end-to-end replay with synthetic checkpoints (caught the ZeroDivision bug),
  export spec-gate refusal.
- `python/config.py` `QualityGates` dataclass (P4 thresholds named, not inline).

### Window-Path Hardening P1–P4 (2026-09-08)
- IMPLEMENTED: unified live low-pass gravity (`estimate_gravity_lowpass`,
  dataset GRAVITY columns now
  cross-check only), versioned window spec + fingerprints
  (`docs/WINDOW_SPEC.md`, `python/core/spec.py`, scaler.json stamps,
  export refuses unversioned/mismatched pairs, `model_manifest.json`),
  cross-language golden vectors (`tools/gen_golden_vectors.py` +
  `tests/test_window_golden.py` + Kotlin `WindowGoldenTest`, parity ≤1e-6,
  lean <10° tripwire), timestamp discipline (preprocess gap audit + >5%
  reject, Android dt>50ms ring guard). Remaining: spec-v2 retrain (one
  Colab run) then flip STRICT_SPEC / WindowSpecGuard.strict.
- Window-path hardening plan P1–P4 recorded in AGENTS.md (gravity unification,
  versioned spec + fingerprints, golden vectors, timestamp discipline).
- Team scaffolding: CONTRIBUTING.md, CODEOWNERS, CI (python + android),
  interface contracts, per-branch Step-2 training plan.
- Python Pro pass: type-hardened pure modules, `pyproject.toml`, pytest suite.
- Kotlin pass: sealed `FusionMode`, `!!` removal, KDoc, handler/detector tests.
- ADRs 001–010 (`docs/adr/`); ADR-008 accepted (loss window 300 ms → 1500 ms).
- Per-file val audit script (`python/eval_per_file.py`) + stratified split flag.

## [0.1.0] — 2026-09-05 (retrospective; no tag cut yet)

### Added
- Full 72-seq Colab pipeline: 822,928 train / 178,369 val windows, 15-epoch
  AVNetLite run (best val 1.729), 1D + 2D drift eval, ONNX export gate.
- Python gaps F1–F7: resample/gravity/scaler parity, exact SE2(3) Jacobian,
  variance ZUPT in harness, synthetic bike aug, adapter deprecation.
- Android skeleton: InEKF port, AVNet TFLite inference, HMM matcher, offline
  tiles, CSV logger, debug APK in `releases/`.

### Fixed
- EKF covariance explosion 3e26 → 0 (`qAcc` 30 → 0.5 live, freeze-when-still,
  linear-acc propagation, motion confirm + deadband).
- Walking-suppressed and over-reactive-motion regressions in the DR gate.
- NPY memmap header + full-822k OOM during preprocess save.
- Missing `PYTHONPATH=` in the Colab preprocess cell.
