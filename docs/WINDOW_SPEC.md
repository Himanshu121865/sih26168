# Window Spec (versioned)

The IMU window path — `raw CSV → resample → gravity-remove → normalize →
(200,6)` — is the frozen core of the project. Every past silent breakage
lived here, never in the modeling ideas. This document is the versioned
contract; `python/core/spec.py` is its single implementation and
`WindowSpecGuard.kt` its on-device enforcer.

**Rule: any change to the window path bumps `SPEC_VERSION` and requires a
full retrain + re-export. No exceptions, no silent "small" tweaks.**

## v1 — dataset gravity columns (legacy)

- Channels `[linAcc xyz, gyro yaw/pitch/roll]`, float32.
- Resample 10 Hz → 100 Hz, linear interp on uniform ns grid
  (`period_ns=round(1e9/hz)`), edges NaN + finite-mask.
- Gravity removed from dataset `GRAVITY X/Y/Z` columns (`acc − gravity`).
- Normalize `(x − mean)/std`, train-only stats, per channel.
- Sliding window 200, stride 10, oldest-first.

**v1 defect:** the dataset gravity column and the on-device live low-pass
(`gEst += 0.02·(acc−gEst)`) are different filters — train/live mismatch.

## v2 — unified live low-pass gravity (current)

Identical to v1 except:

- Gravity is COMPUTED with the live low-pass
  `gEst += 0.02·(acc−gEst)`, init `[0,0,9.81]` —
  `python/core/signal.estimate_gravity_lowpass()` (single implementation,
  mirrors `LeanDetector.gEst` in Kotlin), applied to the raw signal BEFORE
  resample.
- Dataset `GRAVITY X/Y/Z` columns become a **cross-check only**
  (`grav_xdiff` logged per file in preprocess; large values flag a frozen
  column, never poison windows).
- Timestamp discipline (P4): preprocess logs per-file `median_dt` +
  `gap_frac`, rejects files with >5% gaps (`>max(3·median_dt, 150ms)`), and
  flags odd rates (S-M 51 ms / S4 80 ms known). Android `onImu` rejects
  `dt` spikes >50 ms instead of feeding the ring.
- `scaler.json` gains `spec_version` + `spec_sha256` (+ `gravity_alpha`).
  `export_tflite.py` REFUSES unversioned/mismatched scalers and stamps
  `model_manifest.json` (model ↔ scaler ↔ spec hashes).
- Android startup (`WindowSpecGuard`) compares the asset fingerprint
  against the compiled-in spec and refuses inference on mismatch
  (hard-refuse once `strict`/STRICT_SPEC is on — staged rollout below).

Spec hash (SHA-256 of the canonical text in `python/core/spec.py`):
`063703d0fd897b5280b2c64257dad28fea6743dc03ac05b9ab62f175a6ed42bd`

## Fingerprints

| Artifact | Fields | Enforced by |
|----------|--------|-------------|
| `scaler.json` | `spec_version`, `spec_sha256`, `gravity_alpha` | `verify_scaler()` (export), `WindowSpecGuard` (Android) |
| `model_manifest.json` | `model_sha256`, `scaler_sha256`, `spec_version` | `verify_model_manifest()`, release CI |
| `window_golden.json` | `spec_version`, `spec_sha256` | `tests/test_window_golden.py` + Kotlin `WindowGoldenTest` |

## Golden vectors (P3)

`tools/gen_golden_vectors.py` snapshots a fixed raw snippet → expected
normalized window. Both `tests/test_window_golden.py` (Python) and
`WindowGoldenTest` (Kotlin) read the SAME asset file
(`android/app/src/main/assets/window_golden.json`) and recompute the path
in their own language — cross-language parity must hold to 1e-6. The lean
tripwire (`mean|φ| < 10°`) rides along as the 38°-bug regression guard.

## Staged rollout (P1 needs a retrain — accepted)

1. **Done (no retrain):** P2 plumbing (spec module, stamps, export gate,
   gradle warning, Kotlin guard in warn mode), P3 golden vectors from a
   synthetic fixture, P4 audit + Android dt guard.
2. **Next Colab run (one cycle):** P1 retrain — `preprocess.py` now computes
   gravity with the live low-pass; run the Step-2 branch plan
   (`docs/STEP2_TRAINING_PLAN.md`) which produces a v2 scaler + model.
   Then `export_tflite.py` stamps a v2 manifest and the root `scaler.json`
   upgrades from its current honest v1 stamp.
3. **Studio gate:** flip `WindowSpecGuard.strict` (and `STRICT_SPEC=1` in
   gradle) → mismatched pairs crash at startup / fail the build. Until the
   retrain lands, mismatches warn loudly (Log.w / gradle lifecycle) so the
   current v1 APKs keep working.

## Regenerating golden vectors after a spec bump

```bash
python tools/gen_golden_vectors.py            # writes android assets
PYTHONPATH=. python -m pytest tests/test_window_golden.py -q
cd android && gradle :app:testDebugUnitTest --project-dir .
```

The Kotlin `WindowSpecGuardTest` pins the spec hash prefix — update it in
the same PR as `python/core/spec.py`.
