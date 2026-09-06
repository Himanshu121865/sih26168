# Changelog

Keep-a-Changelog format. `main` is unreleased by default; releases are cut by
pushing a `v*` tag, which triggers the release workflow (APK + screening
bundle attached to the GitHub Release).

## [Unreleased]

### Added
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
