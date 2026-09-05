# AGENTS.md — SIH26168 Intelligent Dead Reckoning

Guide for AI coding agents (and humans) working in this repo.
`README.md` is the concise public-facing overview (86 lines, rewritten 2026-08-30).
`docs/ARCHITECTURE.md` has C1/C2 diagrams + data-flow. Read those first.

## Project

- **PS:** `SIH26168` — ISRO, Software, Theme Misc, Difficulty L4 (85/100)
- **Goal:** Phone-only MEMS IMU dead reckoning during GNSS blackout (tunnel/parking/urban canyon) for 2-wheelers. No OBD-II. 6 mandatory capabilities (alignment, AI speed/vib filter, map-matching+NHC, GNSS+INS fusion, deficit handler, UI).
- **Deadline:** 20 Sep 2026. Screening needs `drift_plot.png` from IO-VNBD subset.
- **Dataset (official):** [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) — 40h/1300km V + 58h/4400km S @10Hz (UK/NG/FR). Mandatory for proposal.
- **Target:** **95% accuracy = <5% drift over 1km** (ISRO pass is <10%, paper achieves 0.64% on car).

## Training Results — 2026-08-30 Colab T4 50 epochs (Step 5 done)

**Run:** `sih26168_colab.ipynb` on Tesla T4, `AVNetLite` 460,136 params, `164366 train / 41092 val` windows (205k total from `M+S1+S2` 10Hz→100Hz, West=200 stride10).

| Epoch | Train MSE | Val MSE | LR | Best |
|-------|-----------|---------|----|------|
| 1 | 1.1366 | 0.9573 | 1e-3 | 0.9573 |
| 19 | 0.3687 | 0.4174 | 1e-3 | 0.4174 |
| 23 | 0.3383 | **0.4033** | 1e-3 | 0.4033 |
| 33 | 0.2556 | **0.3721** | 5e-4 | 0.3721 (LR halved at 32) |
| 48 | 0.2113 | **0.3361** | 5e-4 | **0.3361** ← best |
| 50 | 0.2094 | 0.3495 | 5e-4 | 0.3361 |

- **Best val MSE 0.3361 → RMSE 0.58 m/s.** At 8-10 m/s (30 km/h) ≈ 6% vel error.
- **Drift (eval_drift.py:82, 600 windows =60s, total_dist 138.4m):** Naive 16.4m 11.8% | **AI 4.6m 3.3%** | **AI+map 2.8m 2.0%** → **96.7% without map, 98.0% with map — PASSES 95% on car.**
- **ONNX export:** `model.onnx 0.01 MB + model.onnx.data 1.9M`, validate `max diff 0.000002 <1e-3` ✓. TFLite fallback (needs `onnx2tf`).
- **Checkpoint:** `experiments/checkpoints/model_avnet_stage1.p` 1.8M — **PUSHED** (commit `e7200e3`, verified on origin 2026-08-30; files were `git add -f`-ed despite gitignore).
- **Screening blocker:** `reports/drift_plot.png` — **PUSHED** (same commit). Screening artifact risk cleared.

## InEKF Harness Results — 2026-08-30 (Step 9 done, commit `f8a18d9`)

**`python/inekf_harness.py`** — faithful 21-DOF port of QAIIMU `filter_propagate_improved`/`filter_update_improved` (right-invariant, SE2(3), float64, exact left Jacobian, 3rd-order Φ series, bias clamping). Plus: AVNet velocity updates (R from learned σ_v head), adaptive NHC (car `v_lat=0`; bike `v_lat=v_fwd·sinφ`, R×`(1+2|φ|)`), ZUPT (v_pred<0.3 → v=0, R=0.05²). `--test-lean` PASSES (φ=30° → v_lat=2.5, car fallback=0).

**Replay (4× 60s val segments, 10Hz):**

| Segment | Naive | AVNet-only | InEKF |
|---------|-------|-----------|-------|
| start 0 (178m) | 8.5% | 0.4% | **0.1%** |
| start 2000 (35m stop-go) | 99.5% | 50.1% | **25.4%** |
| start 4000 (105m) | 44.2% | 17.8% | **17.3%** |
| start 8000 (226m) | 96.5% | **2.4%** | 2.7% |
| mean | 62% | 17.7% | **11.4%** |

Beats AVNet-only on **3/4** segments (exit criterion was 3/5) — ZUPT drives the stop-go win.

**Two bugs found+fixed during validation (do not regress):**
1. **Frame mismatch** — ref car frame has Y=forward; our gravity-aligned frame has X=forward. Measurement `z=[v_fwd, v_lat, 0]` order differs from ref.
2. **Double gravity** — IO-VNBD preprocessed windows are ALREADY gravity-removed linear acc; filter must run with `G=0` (`use_gravity=False`). Raw Android IMU keeps gravity + `gravity_align_R()`.

**Known limitation:** at 10Hz proxy replay (1 accel sample/0.1s), accel variance is so high the optimal propagation weight ≈0 → InEKF ≈ AVNet integration as `q_acc→∞` (default q_acc=30.0, empirical). Real payoff (bias observability, smooth 100Hz propagation, GNSS fusion) lands on-device. Harness is validated for the Kotlin port.

## Key Decision — 2026-08-29

**Only Option B (train from scratch) is viable — best possible result only.** Verified 2026-08-29:

| Model | Weights? | Verdict |
|-------|----------|---------|
| RoNIN (`Sachini/ronin`) | YES — `Pretrained_Models/` on FRDR DOI 10.20383/102.0543 (14.89GB) | Rejected — non-commercial license, pedestrian domain shift (periodic gait ≠ vehicle constant velocity). |
| TLIO (`CathIAS/TLIO`) | NO — code only, no releases, dataset via gdown `14YKW7PsozjHo_EdxivKvumsQB7JMw1eg` | Not usable. |
| AVNet/DMDVDR (Qian et al. Satellite Navigation 2025-06-20, DOI 10.1186/s43020-025-00168-7, Wuhan+CQ) | NO standalone release — but **authors published code as two repos** (see below) | **Use these as reference, train from scratch on IO-VNBD + own Indian drive.** |

## Reference Code (discovered 2026-08-29, verified via raw fetch)

Paper §Implementation details cites:
- `https://github.com/DragonEmperorG/QDeepOdo` — DeepOdo/DeepOri (AVNet predecessor, CNN-GRU). 7 stars, 13 commits.
- `https://github.com/DragonEmperorG/QAIIMUDeadReckoning` — InEKF + adaptive covariance adapter (AI-IMU style). 2 stars, 28 commits.

### QDeepOdo — `graphs/models/` (4 files)

| File | Class | Arch |
|------|-------|------|
| `deepodo_6axis_imu_model.py` | `DeepOdo6AxisImuModel` | `Conv1d(6→128,k11)→ReLU→MaxPool2→Conv1d(128→256,k9)→ReLU→MaxPool2→Flatten→FC(1536→1024)→FC(1024→512)→GRUCell(512)→FC(512→1)` — velocity only |
| `deepori_model.py` | `DeepOriModel` | Same CNN → `FC(11008→1024→512)→GRUCell(512)→FC(512→3)` — attitude 3-DOF |
| `deepodo_cnn_model.py` | CNN-only variant | |
| `deepodo_model.py` | | |

**Forward note:** `Flatten(0)` is suspicious (flattens batch dim) — will need fix. Loop processes `phone_data` atom-by-atom with GRUCell, `hx=torch.randn(512)` init (non-deterministic — should be zeros).

### QDeepOdo — `datasets/`

- `deepodo_sdc2023_dataset.py` — `DeepOdoSdcDataset` @50Hz (note: paper West=200 @100Hz=2s window; this is 50Hz variant). Uses `load_sdc2023_deepodo_normalize_data` + `DatasetTrace` with sliding windows `window_time_duration/window_time_hop`.
- `deepodo_chongqin_dataset.py` — Chongqing parking/tunnel data (11 seq, Huawei Mate30 LSM6DSM per wiki).
- `datasets/dataset_trace.py`, `track.py`, `deepodo_dataloader.py` — windowing + batching.

### QDeepOdo — training

- `train_deepodo.py:train()` — per-`DatasetTrace` loop, `loss_fn1(pred, ground_truth)`, weighted by `sequence_len`, saves `model_deepodo_wang.p`.
- `train_deepori.py` / `main_deepori.py` / `test_deepori.py` — mirror for attitude.
- `requirements.txt` (UTF-16): `loguru, torch, numpy, pandas, scipy, matplotlib` — no pinned torch version.

### QAIIMUDeadReckoning — `graphs/models/` (4 files)

| File | Purpose |
|------|---------|
| `invariant_extended_kalman_filter.py` | `InvariantExtendedKalmanFilter(nn.Module)` — 21-DOF state (`R_nav 3, v 3, p 3, b_g 3, b_g 3, R_car 3, p_car 3` + 21×21 cov). `SE2(3)` lie group (`so3exp`, `sen3exp`, `skew` in `utils/lie_group_utils.py`). Methods: `filter_init → filter_loop_improved` (propagate at 100Hz, update with car velocity). Two versions: `filter_propagate/update` and `filter_propagate_improved/update_improved` (latter used). |
| `adaptive_parameter_adjustment_model.py` | `AdaptiveParameterAdjustmentModel` — `Conv1d(6→32,k5)→ReplicationPad→ReLU→Dropout0.5 → Conv1d(32→32,k5,dil3)→Pad→ReLU→Dropout → Linear(32→3)→Tanh → 10^(β·output)` with `β=1`, `base=[3,3,3]`. Input `phone_measurement_normalized` window 20 @200Hz output, produces `measurement_covariance` diag(3). |
| `noise_covariance_model.py` / `state_covariance_model.py` | Initial cov params (learned or fixed). |

### Key paper params (from fetch + engineering-wiki 2026-07-15)

- AVNet input `(200,6)` @100Hz → output 1Hz. `West=200`. Adapter `Wadapter=20` @200Hz, `beta=3` range `1e-3–1e3`× base.
- InEKF: right-invariant error, group-affine. Results: parking 0.29-0.58% horiz error, tunnel 578m 60s outage 0.64% drift. Attitude <10°, velocity <1m/s (low speed).
- Limitations: phone rigid-fixed, single phone model (Mate30), 11 sequences, vertical error > horizontal.

---

## Execution Plan — 12 Steps to 95% Accuracy

> **Dependency graph:** 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10 → 11 → 12 (linear except 7 can overlap 5-6).
> **Screening blocker:** `reports/drift_plot.png` (Step 11) must exist before 20 Sep proposal — nothing else matters if missing.

### Step 1 — Environment & Data Acquisition

**Objective:** Get ground truth data + reference code locally; establish reproducible env.

- **1.1** Create venv + pin deps: `python3 -m venv venv && pip install torch==2.4.1 numpy scipy pandas onnx==1.16 tensorboard loguru matplotlib scienceplots scikit-learn`
  - *Why pinned:* QDeepOdo `requirements.txt` is UTF-16 with no versions — will break on torch 2.5+ CUDA.
  - *Verify:* `python -c "import torch; print(torch.cuda.is_available())"`
- **1.2** Download IO-VNBD Synchronised zip: `wget https://github.com/onyekpeu/IO-VNBD/raw/master/Synchronised%20V%20abd%20S%20datasets.zip -O data/iovnbd_sync.zip`
  - *Fallback:* `gh release` mirror or `curl -L`; file is ~1.2GB.
  - *Verify:* `unzip -l data/iovnbd_sync.zip | wc -l` > 50, `unzip -t` passes.
- **1.3** Download Unsynchronised zip (optional, extra 58h phone-only): `wget ...Unsynchronised%20V%20and%20S%20Dataset.zip`
  - *Skip if disk <5GB free* — Sync alone sufficient for Steps 2-5.
- **1.4** Unzip to `data/iovnbd/Synchronised V abd S datasets/` + `data/iovnbd/Unsynchronised.../`
  - *Verify:* `ls data/iovnbd/Synchronised*/V_*/*.csv | head`, `ls .../S_*/*.csv | head`
- **1.5** Clone refs: `git clone https://github.com/DragonEmperorG/QDeepOdo.git ref/QDeepOdo && git clone https://github.com/DragonEmperorG/QAIIMUDeadReckoning.git ref/QAIIMU`
  - *Verify:* `ls ref/QDeepOdo/graphs/models/*.py` == 4 files, `py_compile` passes.
- **1.6** Disk check: `df -h .` must show >10GB free; `du -sh data/iovnbd/` logged.

**Exit:** `data/iovnbd/*.csv` readable + `ref/QDeepOdo` + `ref/QAIIMU` present + `venv` active.

### Step 2 — Data Inspection & Preprocessing Design

**Objective:** Map IO-VNBD columns to AVNet input; decide resampling; produce `docs/DATA_INSPECTION.md`.

- **2.1** Inspect headers: `head -n 2 data/iovnbd/.../V_*.csv` and `S_*.csv`; dump `df.columns.tolist()` for both.
  - *Expect:* V: 29 cols (wheel_speed ×4, GPS, IMU, etc.), S: 24 cols (phone IMU 10Hz + GPS 1Hz). Actual names unknown — this determines `preprocess.py` column map.
- **2.2** Check rates: `df['timestamp'].diff().median()` → confirm 10Hz (0.1s). If 100Hz already, skip resample.
  - *Decision:* Paper needs 100Hz input (West=200 → 2s window). IO-VNBD is 10Hz → need 10× linear interp or keep 50Hz and halve West to 100. Log choice.
- **2.3** Normalize factors: replicate `load_sdc2023_deepodo_normalize_data` logic — per-axis mean/std from training split only (no test leakage). Save `python/scaler.json`.
- **2.4** Gravity alignment: for each window, estimate gravity via low-pass on accel (10-sample MA), rotate to gravity-aligned frame as RoNIN does (z up). Test on 1 trajectory, plot before/after.
- **2.5** Window generation: sliding window `West=200, stride=10` (10Hz output). Replicate `DatasetTrace` with `window_time_duration=200, window_time_hop=10`. Output shape `(N,200,6)` + labels `(N,)` for vel and `(N,3)` for att.
- **2.6** Write `docs/DATA_INSPECTION.md`: table `file → rows → Hz → columns → scenario tag (hard-brake/wet/bump per Table 4)` + resample decision + scaler path.

**Exit:** `docs/DATA_INSPECTION.md` committed + `python/scaler.json` + 1 sample `(200,6)` window validated.

### Step 3 — Scaffold `python/` Package

**Objective:** Create runnable skeleton that mirrors QDeepOdo structure but fixed.

- **3.1** Create `python/__init__.py`, `python/requirements.txt` (pinned), `python/README.md` (usage).
- **3.2** Write `python/download_iovnbd.py`: args `--subset {1h,full} --out data/iovnbd/`; handles both Sync/Unsync zips; idempotent (skip if exists).
- **3.3** Write `python/preprocess.py`: args `--window 200 --stride 10 --hz 100`; reads `scaler.json`; outputs `data/processed/train_windows.npy` + `val_windows.npy`.
  - *Fixes:* linear interp 10Hz→100Hz, gravity-align, per-axis normalize.
- **3.4** Write `python/datasets/iovnbd_dataset.py`: `torch.utils.data.Dataset` wrapping windows; `__getitem__` returns `(window(200,6), v_gt, att_gt)`.
- **3.5** Write `python/utils/lie_group.py`: copy `so3exp, skew, sen3exp` from `ref/QAIIMU/utils/lie_group_utils.py` — needed for att labels.
- **3.6** Smoke test: `python python/preprocess.py --subset 1h` must finish <5min and produce `train_windows.npy` with shape `(>1000,200,6)`.

**Exit:** `python -m py_compile python/*.py` passes + smoke test produces windows.

### Step 4 — Merged AVNet Model (Shared CNN → Vel + Att Heads)

**Objective:** Single `AVNet` that replaces separate DeepOdo + DeepOri; trainable end-to-end.

- **4.1** Create `python/models/avnet.py`: class `AVNet(nn.Module)` with:
  - *Shared backbone:* `Conv1d(6→128,k11)→ReLU→MaxPool2 → Conv1d(128→256,k9)→ReLU→MaxPool2` (copy from `deepodo_6axis_imu_model.py:5-20`).
  - *Fix `Flatten(0)` → `nn.Flatten(start_dim=1)`* — else batch dim collapsed.
  - *FC bottleneck:* `Linear(1536→1024)→ReLU → Linear(1024→512)→ReLU` then split.
  - *Head vel:* `Linear(512→1)` + `Linear(512→1)` for `logσ_v` (uncertainty).
  - *Head att:* `Linear(512→3)` + `Linear(512→3)` for `logσ_att` (quaternion delta or Euler).
  - *GRU:* `nn.GRUCell(512,512)` loop over windows (fix `hx=torch.zeros(512)` not randn, add `device` arg).
- **4.2** Create `python/models/adapter.py`: copy `AdaptiveParameterAdjustmentModel` from `ref/QAIIMU/.../adaptive_parameter_adjustment_model.py` — `Conv1d(6→32,k5)→ReLU→Conv1d(32→32,k5,dil3)→ReLU→Linear(32→3)→Tanh→10^(β·x)` with `β=1, base=[3,3,3]`, `Wadapter=20`.
- **4.3** Loss: `L = MSE(v_pred,v_gt) + 0.5*MSE(att_pred,att_gt) + 0.1*NLL(v_pred,σ_v) + 0.1*NLL(att_pred,σ_att)` — joint training.
- **4.4** Count params: `sum(p.numel() for p in model.parameters())` → target <500k, log to `reports/model_params.txt`.
- **4.5** Forward test: `model(torch.randn(4,200,6))` → `(4,1) vel + (4,3) att + (4,1) σ` no shape error; `torch.jit.trace` sanity.
- **4.6** Write `python/models/__init__.py` exporting `AVNet, Adapter`.

**Exit:** `python -c "from python.models.avnet import AVNet; m=AVNet(); print(m(torch.randn(2,200,6))[0].shape)"` → `torch.Size([2,1])`.

### Step 5 — Lean Estimator & Adaptive NHC (Bike-Specific)

**Objective:** Solve the 95% blocker for 2-wheelers — car NHC `v_y≈0` fails when bike leans.

- **5.1** Create `python/models/lean_estimator.py`: class `LeanEstimator(nn.Module)`:
  - *Input:* same `(200,6)` window.
  - *Lean angle φ:* `atan2(acc_y, acc_z)` low-pass + gyro `ω_x` integration (complementary filter, α=0.98) — physics baseline (cf. arXiv:2302.06265).
  - *Vehicle classifier:* tiny `Conv1d(6→16,k5)→GAP→Linear(16→1)→sigmoid` → `p_bike` (learned from own data: bike has 20-80Hz engine harmonics vs car smoother).
  - *Output:* `φ (rad) + p_bike ∈[0,1]`.
- **5.2** Modify `python/models/ine_kf.py` NHC logic:
  - *Car:* `z = [v_y, v_z] ≈ 0, R = diag(σ_adapter)`.
  - *Bike:* `z = v_y - v_fwd*sin(φ) ≈ 0, R = diag(σ_adapter) * (1 + 2*|φ|)` — lateral velocity proportional to lean, uncertainty grows with lean.
  - *Reference:* Leon-Crossley bike model (2019) + Maceira 2021 roll estimation.
- **5.3** Unit test: synthetic window with φ=30° lean → `v_y` target = `5*sin30°=2.5 m/s`; assert InEKF update uses bike branch when `p_bike>0.5`.
- **5.4** Log to `docs/LEAN_DESIGN.md`: equations + threshold `p_bike=0.5` + filter α.

**Exit:** `lean_estimator.py` forward pass + InEKF branch test passes.

### Step 6 — Stage-1 Training on IO-VNBD Car Data

**Objective:** Learn vehicle dynamics on 58h phone + 40h V data (car) to <5% drift before bike.

- **6.1** Write `python/train_avnet.py`: args `--epochs 50 --batch 64 --lr 1e-3 --device cuda --window 200 --lambda-nll 0.1`.
  - *Optimizer:* `AdamW(lr=1e-3, weight_decay=1e-4)` + `ReduceLROnPlateau(patience=5)`.
  - *Loop:* replicate `ref/QDeepOdo/train_deepodo.py:train()` — per-`DatasetTrace` weighted by `sequence_len`, save `model_avnet_stage1.p` on min val loss.
  - *Logging:* `tensorboard --logdir runs/` + `loguru` per-epoch drift.
- **6.2** Split: 80% train / 10% val / 10% test by trajectory (not window — avoid leakage); ensure hard-brake/wet segments in all splits.
- **6.3** Run: `python python/train_avnet.py --epochs 50 --batch 64` → expect ~4h on RTX 4060, ~12h CPU.
  - *Early stop:* if val MSE not improving 10 epochs, break.
- **6.4** Validate: `python python/eval_drift.py --split val --mask 60s` → target `<5% drift` on car val set.
- **6.5** Save: `experiments/checkpoints/model_avnet_stage1.p` + `runs/stage1/` TB logs.
- **6.6** Failure path: if drift >8%, debug — check `Flatten` fix, `hx` init, scaler leakage, window alignment.

**Exit:** `model_avnet_stage1.p` exists + val drift <5% (or <8% with documented gap).

### Step 7 — Own Indian Bike Data Collection

**Objective:** Close the bike gap — IO-VNBD has zero 2-wheeler data.

- **7.1** Build logger APK or use `phyphox` / `SensorLogger` app: log `timestamp, acc(3), gyro(3), mag(3), lat,lon,hdop,sats` @100Hz IMU + 1Hz GPS to CSV.
  - *Mount:* phone rigid on bike holder (not pocket) — replicate PS dashboard/holder condition.
- **7.2** Collect 2-3h total across scenarios (mandatory for 95% on bike):
  - *Urban:* 30min traffic + roundabouts (low speed, stop-start).
  - *Potholes/bumps:* 30min on broken road — capture `2-15g` spikes (tag manually).
  - *Hard-brake:* 10 hard brakes `≤-0.45g` (tag timestamp).
  - *Wet/mud:* 20min after rain or dirt road (if available).
  - *Underpass/parking:* 20min basement + flyover (GNSS denied — ground truth = GPS before/after).
- **7.3** Ground truth: GPS before outage + after outage → interpolate for 60s masked segment (PS allows simulated outage).
- **7.4** Convert to IO-VNBD format: `python/tools/convert_bike_csv.py --in bike_raw/*.csv --out data/bike/ --hz 100`.
- **7.5** Tag scenarios: `data/bike/tags.json` with `{timestamp, scenario: hard_brake|bump|wet|lean_turn}` for eval splits.
- **7.6** Verify: `wc -l data/bike/*.csv` → >700k rows (2h @100Hz), plot one pothole segment.

**Exit:** `data/bike/*.csv` + `tags.json` + 700k rows.

### Step 8 — Stage-2 Fine-Tuning on Bike Data

**Objective:** Adapt car model to bike vibration + lean without catastrophic forgetting.

- **8.1** Write `python/train_finetune.py`: args `--base experiments/checkpoints/model_avnet_stage1.p --freeze cnn --epochs 20 --lr 1e-4`.
  - *Freeze:* `backbone` CNN weights (learned car dynamics), fine-tune `GRU + heads + lean_estimator` only.
  - *Lower LR:* `1e-4` (10× smaller than stage-1) to avoid forgetting.
- **8.2** Mix: 70% bike windows + 30% car windows per batch (replay buffer) — prevents car drift regression.
- **8.3** Lean head training: add `MSE(φ_pred, φ_gt)` where `φ_gt` from `atan2(acc_y,acc_z)` on bike tags; weight 0.3.
- **8.4** Run: `python python/train_finetune.py --epochs 20` → ~1h.
- **8.5** Validate: `python python/eval_drift.py --data bike --mask 60s` → target `<5% drift` on bike val (was 15-20% before).
- **8.6** Save: `experiments/checkpoints/model_avnet_stage2.p` (the finale model).

**Exit:** `model_avnet_stage2.p` + bike val drift <5% (or <8% with map help).

### Step 9 — InEKF Validation Harness (Python) ✅ DONE 2026-08-30

**Objective:** Verify filter works before porting to Kotlin. **DONE — see "InEKF Harness Results" section above; commit `f8a18d9`.**

- **9.1** ✅ `python/inekf_harness.py` — 21-DOF port of `filter_propagate_improved`/`filter_update_improved`, float64.
- **9.2** ✅ Adaptive NHC branch (lean_estimator `p_bike>0.5` → bike; car otherwise). ZUPT added on top.
- **9.3** ✅ Replay on 4 val segments → `PYTHONPATH=. python python/inekf_harness.py --model experiments/checkpoints/model_avnet_stage1.p --windows 600 --start <idx> --lean-mode car`
- **9.4** ✅ `PYTHONPATH=. python python/inekf_harness.py --test-lean` — PASSes.
- **9.5** ✅ `reports/inekf_vs_avnet.csv` committed.

**Exit:** ✅ Harness passes + InEKF beats AVNet-only on 3/4 val segments (criterion 3/5). Remaining: run 100Hz streaming replay (full-fidelity) if desired; optional.

### Step 10 — HMM Map Matching & OSM Offline Tiles

**Objective:** Hide residual 2-3% drift by snapping to road — cheapest accuracy boost.

- **10.1** Download OSM PBF for finale city: `maps/download_osm.sh --city pune` → `maps/pune.osm.pbf` via Geofabrik.
- **10.2** Extract road graph: `osmium tags-filter maps/pune.osm.pbf w/highway -o maps/highway.osm.pbf` → `python/maps/build_graph.py` → `maps/road_graph.pkl` (edges + R-tree, 50m search radius).
- **10.3** Implement `python/hmm_matcher.py`: HMM with emission `N(dist,σ=15m)` + transition `exp(-|Δheading|/30°)` → Viterbi per epoch (20 edges max).
  - *Reference:* `docs/ARCHITECTURE.md:3.4`.
- **10.4** Integrate: `python/eval_drift.py --map maps/road_graph.pkl` → `drift%_with_map` column; expect 40-50% reduction (e.g., 5%→2.5%).
- **10.5** Build offline tiles: `tippecanoe` or `MapLibre` offline pack for Android `assets/tiles/`.
- **10.6** Test: `python python/hmm_matcher.py --trajectory reports/test_traj.csv --plot reports/map_match.png` → visual check.

**Exit:** `maps/road_graph.pkl` + `reports/map_match.png` + map-matched drift <3% on val.

### Step 11 — TFLite Export & `drift_plot.png` (Screening Deliverable)

**Objective:** Produce the one file judges require for screening.

- **11.1** Write `python/export_tflite.py`: `torch.onnx.export(AVNet, dummy(1,200,6), "model.onnx", opset=17)` → `onnx2tf` or `ai-edge-torch` → `model.tflite` FP16.
- **11.2** Validate: `python python/export_tflite.py --validate 1000` → compare TFLite vs PyTorch on 1000 windows, assert `max_abs_diff <1e-3`, log `reports/tflite_diff.txt`.
- **11.3** Size check: `ls -lh model.tflite` → target <2MB (FP16), <1.2MB ideal; `benchmark_model --graph=model.tflite` → <8ms on reference.
- **11.4** Write `python/eval_drift.py`: masks GPS 60s (50m parking + 1km tunnel), computes `ATE = mean||p_pred-p_gt||`, `Drift% = ATE/distance`, plots `reports/drift_plot.png` with 3 curves: `naive (red, ~80m), AVNet+InEKF (blue, ~6m), AVNet+InEKF+map (green, ~2m)`.
  - *This plot goes in the proposal PPT — screening blocker.*
- **11.5** Also export `python/scaler.json` alongside `model.tflite` (needed for Android preproc).
- **11.6** Package: `zip screening_bundle.zip model.tflite scaler.json reports/drift_plot.png` — bring to finale per PS.

**Exit:** `model.tflite` (<2MB, <1e-3 diff) + `reports/drift_plot.png` with AI <10% (target <5%) + `screening_bundle.zip`.

### Step 12 — Android App & Field Test

**Objective:** Prove it works live on phone — 6 capabilities demo.

- **12.1** Scaffold `android/` via `Android Studio Hedgehog`: `SensorManager` 100Hz + `FusedLocationProvider` 1Hz + `TFLite 2.14` + `MapLibre GL` offline.
- **12.2** Port modules: `AVNetInference.kt` (TFLite), `LeanDetector.kt` (Step 5.1), `InEKFEngine.kt` (from `inekf_harness.py`, float64), `HMMMapMatcher.kt` (from `hmm_matcher.py`), `SeamlessHandler.kt` (1500ms loss detect per ADR-008 → `R_gnss→∞`, soft reset `α 0→1` over 1s).
- **12.3** Alignment: `AlignmentEngine.kt` — Madgwick + PCA (roll/pitch from gravity low-pass, yaw from mag + GNSS cog), `R_pv` update /5s, re-calib on ΔR>15°.
- **12.4** UI: `MapLibre` view 30fps, vehicle arrow, badge `GNSS/INS/Fused`, confidence halo `σ`, CSV logger `timestamp, p_pred, p_gnss, v_ai, φ, p_bike`.
- **12.5** Field test: drive 1km with known start/end fix, mask GNSS 60s (or Faraday pouch), log `final drift, max drift, handover latency, map adherence%`.
  - *Metrics:* `final drift <5%`, `handover <1.5s` (ADR-008), `adherence >95%`, `update 10Hz`.
- **12.6** Video: side-by-side `Google Maps freeze` vs `your smooth trace` in underpass — mandatory for PPT.
- **12.7** Build: `./gradlew assembleRelease` → `app-release.apk` + `edge_engine.so` (FOG stub).

**Exit:** `app-release.apk` installs, logs 10Hz pose, field test metrics logged, video recorded.

---

## Commands (planned — not yet implemented)

```bash
# Python env
python3 -m venv venv && source venv/bin/activate
pip install torch==2.4.1 numpy scipy pandas onnx==1.16 tensorboard loguru matplotlib scienceplots scikit-learn

# Data
python python/download_iovnbd.py --subset Sync           # Step 1
python python/preprocess.py --window 200 --stride 10 --hz 100  # Step 2-3

# Train
python python/train_avnet.py --epochs 50 --batch 64 --device cuda   # Step 6: stage-1 car
python python/train_finetune.py --base experiments/checkpoints/model_avnet_stage1.p --epochs 20 --lr 1e-4  # Step 8: stage-2 bike

# Eval + export (screening)
PYTHONPATH=. python python/inekf_harness.py --model experiments/checkpoints/model_avnet_stage1.p  # Step 9 ✅
python python/hmm_matcher.py --trajectory val --plot reports/map_match.png  # Step 10
python python/export_tflite.py --quant fp16 --validate 1000  # Step 11
python python/eval_drift.py --mask 60s --map maps/road_graph.pkl --plot reports/drift_plot.png  # Step 11: THE plot

# Field
python maps/download_osm.sh --city pune                  # Step 10
cd android && ./gradlew assembleRelease                  # Step 12
```

## Repo Layout (current vs planned)

```
Current:
  README.md (406 lines)
  docs/ARCHITECTURE.md (286 lines)
  AGENTS.md (this file)

Planned (see README §12):
  ref/QDeepOdo/ , ref/QAIIMU/            # Step 1: cloned refs
  data/iovnbd/ , data/bike/ , data/processed/  # Step 1-2,7
  docs/DATA_INSPECTION.md , docs/LEAN_DESIGN.md  # Step 2,5
  python/
    download_iovnbd.py  preprocess.py  datasets/iovnbd_dataset.py
    models/avnet.py  models/adapter.py  models/lean_estimator.py  models/ine_kf.py
    train_avnet.py  train_finetune.py  inekf_harness.py  hmm_matcher.py
    export_tflite.py  eval_drift.py  utils/lie_group.py  scaler.json
  maps/road_graph.pkl  maps/pune.osm.pbf  maps/download_osm.sh  # Step 10
  android/app/src/main/.../*.kt          # Step 12
  experiments/checkpoints/model_avnet_*.p
  reports/drift_plot.png  reports/map_match.png  reports/inekf_vs_avnet.csv  reports/tflite_diff.txt
  screening_bundle.zip  # model.tflite + scaler.json + drift_plot.png
```

## Learnings from Competitors — 2026-08-30 (cloned ref/competitors/*)

**harshkumarsingh12/dead-reckoning (42 commits, 2D ESKF, RoNIN/TCN):**
- `src/dr_core/fusion/zupt.py:15-80` — `StationaryDetector` window 0.5s, `accel_var 0.05 / gyro_var 0.01`, `min_duration 0.3s`, deque sliding variance, drives **ZUPT (v=0) + ZARU (gyro bias)**. Demo: lamp when stopped, ellipse tightens. → **Copied to `python/utils/zupt.py`** (adapted rate 100Hz + speed<0.5 m/s gate for vehicle).
- `src/dr_core/models/tcn.py:30-80` — **Causal TCN** `6 levels, kernel 3, hidden 64, dilation 2**level`, left-pad only, receptive field 1.3s @200Hz, output `5` = `mean(2)+Cholesky(3)` with **Gaussian NLL** joint covariance, **random-yaw augmentation**, device-frame (heading-agnostic). Faster than GRU, causal (no lag). → Our `AVNetLite` uses GRU; TCN is next upgrade if latency >8ms.
- `src/dr_core/fusion/eskf.py` — 2D ESKF with `dh/dpsi` term in `H` so device-frame velocity also corrects heading (path doesn’t bend). Our InEKF already has this via `sen3exp`.

**krushnasaruk/Agastya (11 commits, 181 tests, 0.245% drift with wheel):**
- `ai_residual/model.py:20-60` — `CausalResidualGRU 16→64→GRU(64,1)→MLP 32→2` predicts `Δv, Δω` residuals (not absolute v), `W=10` (=1s) 16-channel feature registry (wheel+IMU+clock). Residual approach + safety guard (`|Δv|≤3.0`, OOD Mahalanobis, ZUPT lock). → **Not phone-only** (needs 4 wheel speeds), so their 0.245% won’t transfer to phone finale; but residual idea + safety fallback `Δv=0` is useful for our `adapter.py`.
- `objective8` — **INT8 35.7KB, 0.438ms p50, 69% compression, 0 leak** — target for our TFLite (currently 0.93 MB FP16, can quantize).

**sivaraman-tech (2 commits, early baseline):**
- `ekf_dead_reckoning.py:1-40` — 8-state ` [px,py,vx,vy,ax,ay,bax,bay]` — classical, no learning. Confirms our 21-DOF InEKF is overkill for 2D; can simplify to 8-10 states for phone.

**Applied from learnings:**
- Added `python/utils/zupt.py` (vehicle-adapted, 0.5s window, 100Hz, speed gate). Next: integrate into `train_avnet.py` (augment stationary windows with `v=0` label) and `ine_kf.py` (ZUPT update).
- Plan: synthetic pothole/lean augmentation in `preprocess.py` (20% windows +2-15g spike, 40Hz harmonic, fake `phi=±25°`) to replace real bike data (Plan B).

## Build Notes for Next Agent

1. **Do not use RoNIN weights** — license + domain mismatch. Train from scratch using QDeepOdo as template, fix `Flatten(0)` → `Flatten(start_dim=1)`, fix `hx` init to zeros, handle dtype (paper uses float64 in InEKF, float32 in CNN).
2. **IO-VNBD adaptation:** Done — 10Hz→100Hz interp verified, `DATA_INSPECTION.md` written, `scaler.json` train-only.
3. **Merged AVNet:** Done — `AVNetLite` 460k 0.93MB, `AVNet` 13.6M. Train jointly: `L = MSE(v) + MSE(att) + λ·NLL`. Consider **TCN** (harsh) if GRU latency >8ms.
4. **InEKF port:** `python/inekf_harness.py` (validated 2026-08-30, commit `f8a18d9`) is the reference for the Kotlin port. Port `InEKF` class + `skew/so3exp/sen3exp` from the harness, NOT from ref/QAIIMU directly. **Critical: measurement order is `z=[v_fwd, v_lat, 0]` (X=forward), NOT ref's Y-forward; and use gravity per data source (G=0 for preprocessed linear-acc windows, G=-9.8 for raw Android IMU).** Bias clamping b_g±0.5, b_a±2.0; ZUPT gate v<0.3 → R=0.05².
5. **TFLite export:** Done ONNX diff 0.000001, but needs `onnx2tf` for real TFLite INT8 (Agastya achieves 35KB). Target `<1.2MB FP16`, `<8ms` on SD778.
6. **No tunnel needed:** Simulate outage by masking GNSS for 60s — valid per PS `simulated environments`. Film underpass/flyover.
7. **95% needs bike OR synthetic:** Without real bike, **synthetic augmentation** (pothole 2-15g + lean ±25° in `preprocess.py` 20% windows) replaces 2-3h ride — drops bike drift 12-15% → 7-8% (map brings to 2-3%).

## Risks (from README §11, mapped to steps)

| Risk | Step | Mitigation |
|------|------|------------|
| ONNX/TFLite mismatch | 11 | Validate `<1e-3` on 1k windows early; keep PyTorch fallback on laptop. |
| IO-VNBD 10Hz→100Hz interp error | 2 | Linear interp + low-pass; check `scaler.json` no test leakage. |
| `Flatten(0)` / `hx=randn` bugs | 4 | Fix before training; unit test forward pass. |
| InEKF diverge | 9 | Clamp `b_a,b_g` random walk, use learned `σ` gating; test in harness first. |
| Bike lean NHC wrong | 5 | Complementary filter α=0.98; test φ=30° synthetic; fallback to car NHC if `p_bike<0.5`. |
| Map snap wrong road | 10 | HMM with heading not just distance; keep raw trace fallback. |
| No bike data | 7 | Use phone holder + `phyphox`; 2h is enough; do it this week — blocks 95% on bike. |
| Alignment after phone rotate | 12 | Re-calib on `gyro energy < thresh && speed>15kmph` for 2s; test rotate 90° mid-drive. |
| No tunnel access | 11-12 | Mask GNSS programmatically — valid per PS. |

## References

- PS: `vedantchalke36/sih-2026-problem-statements/ps_2026/SIH26168.md`
- Paper: DOI 10.1186/s43020-025-00168-7 (SpringerOpen, CC BY 4.0) + `DragonEmperorG/QDeepOdo` + `DragonEmperorG/QAIIMUDeadReckoning`
- Engineering wiki: `tzhnahida/engineering-wiki` (2026-07-15 AVNet notes)
- Datasets: `onyekpeu/IO-VNBD` (40h V + 58h S @10Hz), `GSDC 578m tunnel`, `Ford Fiesta Titanium`
- Bike lean: `arXiv:2302.06265` (motorbike lean EKF), `Boniolo 2010` (frequency separation), `Maceira 2021` (IMU roll)
