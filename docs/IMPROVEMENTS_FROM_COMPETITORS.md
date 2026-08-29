# Improvements from Competitors — Line-by-Line Audit 2026-08-30

Source clones: `ref/competitors/harsh` (8.1M, 42 commits), `agastya` (17M, 11 commits, 181 tests), `sivaraman` (102M, 2 commits)
Our code: `python/` + `AVNetLite 460k` at 2026-08-30 01:42 (val 0.336 MSE, 3.35% drift)

---

## P0 — Blocks 95% / ISRO `<5%` (do next)

### 1. Fix `preprocess.py:49-65` no-op `gravity_align` + `interp1d extrapolate` + random shuffle leak
- **Current:** `gravity_align(window) return window` (`preprocess.py:65`) with comment "handled by augmentation" — contradicts harsh `preprocess/pipeline.py:105-108` `R_wb@a - [0,0,9.80665]` shared by train & live. Also `resample_10_to_100:44` uses `scipy.interp1d(...,fill_value="extrapolate")` fabricating outside overlap vs `sivaraman/prepare_training_data.py:180` `np.interp(...,left=np.nan,right=np.nan)` + `finite_mask` drop. Also `preprocess.py:141-143` `permutation(0)` shuffles **windows** across trajectories → leaks future context (harsh `windowing.py:84-174` `split_by_trajectory` + `agastya/feature_registry.py:26` trajectory disjoint).
- **Fix:** Copy harsh `pipeline.py:30-68` `resample_uniform(t_ns,values,rate)` with `period_ns=round(1e9/rate)` + `np.interp` per channel, error if `<2 samples`. Move gravity removal **before** `scaler.json` (harsh `pipeline.py:105-108`), share same `resample_uniform` + `align_gravity` between `preprocess.py` and Android `lean_estimator`. Replace shuffle with `sorted(rglob)` + `np.random.default_rng(26168).permutation` **by trajectory** (`harsh/loaders.py:287`). Add `find_column` regex `sivaraman/prepare_training_data.py:205-230` `r'accelerometer\s*x'` to handle `(m/s²)` vs `(m/s^2)` encoding `cp1252` vs `latin1`.
- **Files to change:** `python/preprocess.py:28,44,49,141` + `python/datasets/iovnbd_dataset.py:20,41,54`
- **Avoid:** `harsh/pipeline.py:67` `values.shape[1]` assumes `(n,k)` — verify; `sivaraman/data_simulation.py:208` `t/gps_dt` int cast not interp.

### 2. Add ZUPT + ZARU wiring (harsh `fusion/zupt.py:29-94` + `eskf.py:237-297`)
- **Current:** `python/utils/zupt.py:17-50` has detector `accel_var 0.05 / gyro_var 0.01` but missing `is_stationary`/`stationary_duration_s` properties (`harsh/zupt.py:84-93`) + candidate persistence (`harsh:74-77`), and **no ESKF wiring**. Harsh `eskf.py:237` `update_zupt: y=-v_world, H=[1 at dv], R=diag(0.02²)` + `update_zaru: y=-(gyro-bg), H[5]=-1, R=0.005` every stationary window.
- **Fix:** Add `python/utils/zupt.py:84-93` properties + `candidate_start_ns` logic, wire `ine_kf.py:correct_zupt()` and `correct_zaru()` (or 8-state `ekf_8state.py` from `sivaraman/ekf_dead_reckoning.py:96-147` Joseph form). Also vehicle gate `speed<0.5` (`zupt.py:24`) — tune from `0.5` to `0.08` for car (Agastya) vs `0.3 m/s` for bike, add `stationary_duration_s` for demo lamp.
- **Copy:** `harsh/zupt.py:39-41` `window_s 0.5, min_duration 0.3` verbatim.
- **Avoid:** Variance on **norm** is orientation-invariant (correct for phone-in-pocket); don't switch to per-axis without yaw-agnostic test (`harsh` comment `zupt.py:35`).

### 3. Train joint NLL cov + random-yaw augmentation (harsh `models/tcn.py:46-238`)
- **Current:** `python/models/avnet.py:20-60` heads `head_vel 1` + `log_sig 1` but `train_avnet.py:48-50` `loss=mse` ignores NLL, `λ-NLL` unused. No `augment_random_yaw`, no causal padding. Harsh `tcn.py:8` causal only (left-pad + `Chomp1d`), `OUT_DIM=5` `mean2+cholesky3`, NLL `0.5*z^Tz+log l11+log l22` with `min_sigma 1e-3`, and `augment_random_yaw` rotates **horizontal x,y only** (`tcn.py:189-193`).
- **Fix:** Implement `gaussian_nll_loss` (`tcn.py:133-170`) with `_cholesky_output_to_cov` parity (`runtime.py:35-36` `softplus+MIN_SIGMA`), train `mean+cov` jointly, add `augment_random_yaw` per batch in `train_avnet.py:36-56` before MSE (rotate `window[0:2],window[3:5]`). Also causal-ize: replace `padding=4` with `pad=(k-1)*dil` + `Chomp1d` or switch backbone to TCN if `AVNetLite` latency >8ms.
- **Avoid:** `tcn.py:105` `downsample=Conv1d(1)` when `in_ch!=out_ch` needs `kaiming` init; your `avnet.py:47` already does.

### 4. Fix dataset `iovnbd_dataset.py:41-58` split + timebase
- **Current:** `Iovnbd_dataset.py:54` assumes `1000/hz` fixed, `t_ns=i*1e9/hz` synthetic (`harsh/timebase/clock.py:42-68` warns median offset, not mean), and `preprocess.py:141` random shuffle leaks.
- **Fix:** Use real `TIME SINCE START (ms)*1e6` as `t_ns`, validate monotonic `np.diff(time_s)<=0` raise (`sivaraman/ekf_dead_reckoning.py:65-66`), and `split_by_trajectory` (harsh `loaders.py:287-317` seeded `26168`).
- **Copy:** `harsh/timebase/reorder.py:35-78` `ReorderBuffer 300ms` min-heap for GPS vs IMU fusion (even offline, verify `sharp_motion_alignment`).

### 5. Replace synthetic drift in `eval_drift.py:64-96`
- **Current:** `generate_drift_plot:67` `v_naive=v_gt+Normal(0,0.5)+0.3` and `96` `drift_map=drift_ai*0.6` placeholder, 1D `cumsum(v*dt)` not 2D, no `ate/rte/coverage`.
- **Fix:** Copy `harsh/eval/metrics.py:49-177` `resample_to` + `_umeyama_se2` SE2 alignment, `ate(align=True)`, `rte(window 60s)`, `final_error`, `drift_pct=final_error/truth_distance*100`, `calibration_coverage` `mean(|err|≤kσ) ≈0.68@1σ`. Also `windowing.py:136-140` `v_world=interp(t_end)` + `rotate_world_to_dev(v_world,psi_now)` and `t_ns=hi-1`.
- **Avoid:** `metrics.py:104` `copy()` overhead not needed.

---

## P1 — Drift & Demo Robustness (do after P0)

### 6. Device-frame Jacobian `dh/dpsi` in filter
- **Harsh `eskf.py:179-201`:** `h_pred=R(-psi)@v_world/s`, `H[0,4]=(-sn*vx+c*vy)/s` heading term, so every vel update corrects heading (path doesn't bend). Our `IneKF` `lie_group.py:42` uses crude `J≈I+0.5*skew(phi)` not exact left Jacobian, and 21-DOF overkill for 2D. Verify or simplify to 7-state `eskf.py:12-78` `P=diag([1,1,1,1,0.1,0.01,0.01])` + `F[0,2]=F[1,3]=dt`.
- **Also copy:** `ChiSquareGate` + `NisLogger` (`gating.py:32-142`) per channel, `R=est.cov*velocity_cov_scale` + `min_velocity_variance` floor, `S=HPH^T+R`, `accept(y,S)` log even on reject, Joseph `P=(I-KH)P(I-KH)^T+KRK^T` every update (`eskf.py:228-231`).

### 7. Mag triple gate + AHRS
- **Harsh `ahrs/mag_gate.py:39-93`:** `mag_tol 0.20`, `dip_tol 0.175 rad (~10°)`, `chi2 0.95`, checks `| |m|-exp|/exp`, `|m|,|g|>1e-15` fail-loud, `sin_dip=dot(m,down)/|m|`, `dip=asin(sin_dip)`, `|dip-exp|>0.175→REJECT_DIP`. Our `lean_estimator.py:42` cheap `atan2(mean_y,mean_z)` — no Madgwick, no gate.
- **Copy:** At least magnitude gate before `hmm_matcher` heading use; `imufusion.Ahrs` wrapper `filter.py:45` `set_sample_period(1/rate)`, `convention=ENU gain=0.5`.

### 8. Preprocessing parity + calibration
- **Harsh `preprocess/pipeline.py:230-232`:** Single `psi_now=heading_rad(orientations[-1])` then `_rotate_horizontal_by(a_uniform,psi_now)` `R(-psi)` on `0:2` only, causal "ends at now". **Agastya `scaler.py:29-56`:** Train-only scaler, `PASS_THROUGH` for binary flags (`if std<1e-8: mean=0,std=1`), `TargetScaler` for 2 residuals. **Sivaraman `prepare_training_data.py:41`:** `encoding='cp1252'` + `find_column` regex.
- **Fix:** Move gravity removal before scaling, share `resample_uniform`, add `CalibrationResult.for_session` dip `atan(2*tan(lat))` (`harsh/calibrate.py:34`) even if stub.

### 9. Temporal & OOD gating (Agastya `objective6/*`)
- **SafetyGuard** `safety.py:25-30` `max_v 3.0, max_yaw 0.5, max_var_v 1.0` → 3-stage sanitize: `!is_sensor_valid or is_stationary → FALLBACK`, `var>thresh → LOW_CONF`, `|Δ|>bound → CLAMP`.
- **SelectivePolicy** `selective_policy.py:36-231` 6-gate order `Sensor→Stationary(v<0.08)→OOD(3.5)→TemporalJump→Confidence→HardClamp`, resets `temporal_monitor` on fallback.
- **TemporalConsistency** `temporal_consistency.py:17-18` `max_v_jump 0.60, max_yaw 0.25, ema 0.30` → `v_jump=abs(Δv-prev)>0.60 → FALLBACK`.
- **Confidence** `confidence.py:64-75` `uncertainty=0.40*u_ood+0.35*u_temp+0.25*u_mag`, `confidence>=0.45`.
- **DistributionMonitor** `distribution_monitor.py:47-55` `z=(mat-mean)/std, dists=mean(z^2)`, `ood_threshold=max(3.5,p99*1.5)` (actual `10.93` too loose).
- **Copy:** Port `SafetyGuard` + `TemporalConsistencyMonitor` + `TrainingDistributionMonitor` into `python/utils/safety.py`, wire `AVNetLite log_sig_v` to `var_v`, disable yaw until `R2>0` (currently negative).

### 10. Classical physics fixes (Agastya `navigation_engine/*` + Sivaraman)
- **Midpoint ENU** `dead_reckoning.py:326-333` `psi_mid=prev+0.5*Δpsi, dE=v*sin(mid)*dt, dN=v*cos(mid)*dt` — 30% less heading error than forward Euler (your `eval_drift.py` 1D only).
- **ZUPT heading freeze** `yaw.py:48-49` `if is_stationary: return heading,0` — freezes drift when stopped.
- **Complementary baseline B** `dead_reckoning.py:301` `w=0.15` normal, `0.65` on slip — adapt to phone `v_fused=(1-w)*v_avnet + w*(v_prev+ax*dt)`.
- **QualityGate** `quality_gate.py:26-110` `dt 0.005-0.50s`, wheel `0-70`, accel `20`, yaw `3.0`, `wrap_to_pi` — add to `ine_kf.py`.
- **Sivaraman EKF 8-state** `ekf_dead_reckoning.py:96` `Q dt^4` for pos, `acc^2*dt^2` for vel, Joseph, yaw wrap `arctan2(sin,cos)` — consider 8-state vs 21-DOF for phone 2D.

---

## P2 — Paper Parity & Deployment (before finale)

### 11. Latency & real-time
- Agastya `objective7/realtime_engine.py:81-500` 11-stage pipeline with `time.perf_counter()` per stage, `deadline 100ms / budget 25ms`, rolling buffer `W=10` `pop(0)/append`, `Watchdog 25ms` → `FALLBACK_AI_TIMEOUT`, `SensorValidator` `min_dt 0.001, max_dt 1.0, max_speed 100, max_accel 25`. Add `LatencyMonitor` + `Watchdog` in Kotlin.
- Harsh `models/runtime.py:58-120` warmup 5, `INFERENCE_BUDGET_MS 10.0`, `benchmark()` median timing.

### 12. Quantization (Agastya `objective8/quantization.py:18-136`)
- Dynamic INT8 only on `Linear+GRU` `quantize_dynamic({nn.Linear, nn.GRU}, qint8)` → `35.7KB, 69%`, `mae 0.008 m/s` ok but yaw `31.5% exceed`. Your `export_tflite.py:69-100` only ONNX, `onnx2tf` fallback — add `compare_quantization_error` on 200 windows, assert `mae<0.01`.

### 13. Geo & simulation (Sivaraman)
- `latlon_to_enu` `ekf:51-53` `north=deg2rad(lat-lat0)*R, east=deg2rad(lon-lon0)*R*cos(lat0)` vs your haversine only for distance — need ENU for trajectory error.
- `data_simulation.py:90-105` soft turn `0.5s dh*frac` + Gaussian pulse `159-170` for pothole aug → copy to `preprocess.py` synthetic 20%.

---

## Do NOT Copy (bugs / over-engineering)

- Harsh `calibrate.py:97-130` stubs `raise NotImplemented` — not real calibration.
- Harsh `eskf.py:413` `G=I` no-op, `windowing.py:47` `move_datasets.py` scripts.
- Agastya `distribution_monitor.py:48` `ood_threshold 10.93` too permissive, `FallbackQuantizedModel scale 0.005` arbitrary, yaw residual worse than zero `R2 -15.1` — keep `enable_yaw_correction=false`.
- Agastya `quantized_model.py:61` `mean_/scale_` vs `means/stds` attr bug.
- Sivaraman `gps_denied_baseline_v2.py:126` `world_acc` swapped `sin/cos`, hardcoded `C:\Users\sivaraman\...`, `time_vs_time` identity plot.

---

## Checklist for `/home/ark/Projects/sih26168`

**P0 (next commit):**
- [ ] `python/utils/zupt.py:37` align to `harsh/zupt.py:30-42` + `comprehensive copy` + wire `IneKF.correct_zupt/zaru`
- [ ] `python/preprocess.py:49` real `align_gravity` via Madgwick or `pipeline.py:105-108` + shared `resample_uniform` + trajectory split + `find_column` regex
- [ ] `python/train_avnet.py:36` `augment_random_yaw` + joint `gaussian_nll_loss` + scaler post-gravity
- [ ] `python/datasets/iovnbd_dataset.py:49-57` `split_by_trajectory` + `t_ns` from file + `ReorderBuffer 300ms` stub
- [ ] `python/eval_drift.py:67` replace synthetic with `metrics.py:81-177` ATE/RTE/coverage

**P1:**
- [ ] Device-frame `dh/dpsi` + `ChiSquareGate` + `MagGate` triple + causal `prepare_window` + `ReorderBuffer`
- [ ] `CalibrationResult` dip + `VelocityModelRuntime` warmup + `ClockMapper` median

**P2:**
- [ ] `SensorValidator` + `LatencyMonitor` + rolling `W=10` + `quantize_dynamic_int8` + `latlon_to_enu`

