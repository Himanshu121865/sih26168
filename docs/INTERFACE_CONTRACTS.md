# Interface Contracts (cross-track — both crews review changes here)

Three files plus one schema cross the Android ↔ AI boundary. Change any of them
only with a reviewer from the other track (CODEOWNERS enforces this).

## 1. `model.tflite` — model I/O

- **Producer:** `python/export_tflite.py` (from `AVNetLite` checkpoint).
- **Consumer:** `android/.../engine/AVNetInference.kt` (`runForMultipleInputsOutputs`,
  outputs map index → tensor, do not reorder).
- **Contract:**
  - Input `(1, 200, 6)` float32, channels `[linAcc xyz, gyro yaw/pitch/roll]`,
    normalized with `scaler.json`, 200 samples @100 Hz = 2 s, oldest-first.
  - Outputs: `0:v_pred (1,1)`, `1:log_sig_v (1,1)`, `2:att_pred (1,3)`,
    `3:log_sig_att (1,3)`, `4:h (1,64)`.
  - Gate: `reports/tflite_diff.txt` max abs diff <1e-2 FP16 (ADR-005).
- **Change protocol:** re-export → update diff log → `copyModelAssets` pulls it
  into the APK at build → AI crew posts new `val MSE + drift_2d.json`.

## 2. `scaler.json` — normalization stats

- **Producer:** `TrainOnlyScaler` via `python/preprocess.py` (train-only means),
  spec-stamped by `python/core/spec.py::attach_spec`.
- **Consumer:** `android/.../engine/Scaler.kt` (`mean[i]`, `std[i]`, i in 0..5),
  guarded by `WindowSpecGuard.kt` (startup fingerprint check).
- **Contract:** keys `mean[6], std[6], hz, window, stride, train_files,
  spec_version, spec_sha256, gravity_alpha`; channel order identical to §1.
  `hz/window/stride` must match the model's training window or inference
  silently degrades. `spec_version`+`spec_sha256` must match
  `docs/WINDOW_SPEC.md` — unversioned or stale scalers are REFUSED by
  `export_tflite.py` (exit 2) and by `WindowSpecGuard` (crash, once strict).
- **Change protocol:** ships alongside the model, same PR, same review. A
  window-path change bumps the spec version (full retrain + re-export +
  golden-vector regeneration in the same PR).

## 3. CSV log schema — field-test scoring

- **Producer:** `android/.../io/CsvLogger.kt`.
- **Consumer:** manual scoring + `reports/` summaries.
- **Contract:** row 1 header
  `timestamp_s,x_pred,y_pred,p_gnss_lat,p_gnss_lon,v_ai,sigma_v,phi_rad,p_bike,mode`
  with `mode ∈ {GNSS, INS}` (`FusionMode.displayName`); row 2 is a `# run ...`
  comment with build identity (spec/model/scaler — parsers must skip `#` lines).
  Append-only; never rename/reorder columns — old logs must stay parseable.

## 4. Drift metrics JSON — proposal numbers

- **Producer:** `python/eval_drift.py` (`--mode 1d` → `drift_plot.json`,
  `--mode 2d` → `drift_2d.json`).
- **Consumer:** proposal PPT + screening bundle.
- **Contract:** `total_dist_m, *_final_m, *_drift_pct` always present;
  `mode` field says which pipeline produced them. Screening plot = 1D file.
