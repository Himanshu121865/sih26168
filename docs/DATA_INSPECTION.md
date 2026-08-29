# DATA_INSPECTION.md — IO-VNBD Synchronised Dataset (Step 1.6)

**Date:** 2026-08-30 · **Source:** `data/iovnbd/Synchronised V abd S datasets.zip` (195 MB zip → 826 MB extracted)
**Inspectors:** `S-Vtb1.csv` (Vtb Driver E), `S-Vta6.csv`, `S-Vw14c.csv` + `V-*.csv` counterparts.

## 1. Layout

```
Synchronised V abd S datasets/
├── Categorised IOVNB Dataset/
│   ├── Vtb (Driver E)/ 12 seqs (Vtb01..12)  — each: V-vtb*.csv + S-Vtb*.csv + JPG
│   ├── Vta (Driver E)/ 30 seqs (Vta06..)    — V-vta*.csv + S-Vta*.csv
│   ├── Vw (Driver E)/  20 seqs (Vw14c..)    — V-Vw*.csv + S-Vw*.csv
│   ├── Vf (Driver E)/   2 seqs              — V-Vfa*.csv (4 V files) + 2 S files
│   ├── S (Driver A)/    6 seqs (S1..S4)    — V-S*.csv + S-S*.csv
│   ├── Y (Driver D)/    1 seq (Y1)
│   └── M (Driver B)/    1 file
└── Uncategorised IOVNB Dataset/
    └── S-Dataset 72/    (S-Vta*.csv etc., not yet inspected)
```

- **Total CSVs:** 288 (verified `find -name *.csv | wc -l`)
- **Drivers:** A,B,D,E (per paper Table 1: E=Aggressive, others Defensive)
- **JPGs:** per-seq dashboard photos (unused for training).

## 2. Vehicle (V) — 29 cols @10Hz (0.1s)

**Header (V-vtb1.csv):**
```
No of GPS Satellites Available, Time Since Start of Day (s), Latitude (deg), Longitude (deg),
Velocity (km/h), Heading (deg), Height (km), Vertical velocity (km/h), Sample period (s),
Steering Angle (deg), Wheel Speed FL/FR/RL/RR (rad/s), Yaw Rate (deg/s),
Indicated Vehicle Speed (km/h), Indicated Long Acc (g), Indicated Lat Acc (g),
Handbrake (0/1), Gear Requested, Gear, Engine Speed (rev/min), Coolant Temp,
Clutch (0/1), Brake Pressure (psi), Brake Position (0/1), Battery Volt, Air Temp, Accelerator (0/1)
```

- **Rate:** `Sample period` unique `[0.1, 0.101, 0.099]` → 10 Hz nominal (paper says 10 Hz, confirmed).
- **Rows:** 1.3k–32k per seq (e.g., Vtb01 32459 rows ≈ 54 min, Vta06 1376 rows ≈ 2.3 min, Vw14c 15826 rows ≈ 26 min).
- **Encoding:** `latin1` (not utf-8 due to `0xb2` byte in header `m/s²`).
- **Key signals:**
  - `Velocity (km/h)` mean 38–67 km/h (Vtb01 46.1, Vta06 67.9) — vehicle CAN speed.
  - `Wheel Speed * (rad/s)` mean 46–62 rad/s — raw wheel odometry for slip labels.
  - `Indicated Longitudinal Acceleration (g)` — hard-brake signal `≤-0.45g`.
  - `Brake Pressure (psi)` max 20–90 psi (Vtb01 90.9).
  - `Steering Angle (deg)` + `Yaw Rate (deg/s)` — cornering.

## 3. Smartphone (S) — 24 cols @10Hz (100ms)

**Header (S-Vtb1.csv):**
```
GPS LATITUDE (deg), GPS LONGITUDE (deg), GPS ALTITUDE (m), GPS SPEED (Kmh),
GPS ACCURACY (m), GPS ORIENTATION (°), GPS SATELLITES IN RANGE, TIME SINCE START (ms),
DATE (YYYY-MO-DD HH-MI-SS_SSS), ACCELEROMETER X/Y/Z (m/s²), GRAVITY X/Y/Z (m/s²),
GYROSCOPE Yaw/Pitch/Roll (rad/s), MAGNETIC FIELD X/Y/Z (μT), ORIENTATION Yaw/Pitch/Roll (°)
```

- **Rate:** `TIME SINCE START (ms)` diff median `100.0` ms → 10 Hz (same as V).
- **Rows:** exactly matched to V per seq (Vtb01 32459/32459, Vta06 1376/1376) — Synchronised means row-aligned, same 10 Hz ticks.
- **Encoding:** `latin1` (same `0xb2`).
- **Key signals (phone-only, ISRO finale inputs):**
  - `ACCELEROMETER X/Y/Z` mean ≈0/0/9.8, std 1.6–2.4 (Vtb01 X std 2.24, Vta06 2.40) — 2-15g pothole spikes present as `min -16, max +16` (Vtb01).
  - `GYROSCOPE Yaw/Pitch/Roll` std 0.09 (Vtb01 yaw) — vehicle yaw.
  - `GPS SPEED` mean 11.5 (Vtb01) vs V `46.1` — phone GPS under-reports vs CAN; `GPS SATELLITES IN RANGE` e.g., `15/18` — use as quality gate.
  - `GRAVITY X/Y/Z` already separated (approx `0,0,9.806`) — but phone orientation still arbitrary (holder vs dash), need alignment.
  - `ORIENTATION Yaw/Pitch/Roll` (°) — Android fused orientation (mag+gyro), noisy.

## 4. Hard Brake & Slip

**Per-file scan (3 samples):**

| Seq | Rows | `acc_lon ≤-0.45g` | % | `Brake max psi` | `|wheel-gps|>5 km/h` (naive r=0.315) | Comment |
|-----|------|------------------|---|----------------|-----------------------------------|---------|
| Vtb01 | 32459 | 22 | 0.07% | 90.9 | 23498 (72%) — **calc suspect** (see §4.1) | Urban UK, moderate |
| Vta06 | 1376 | 0 | 0% | 20.9 | 1376 (100%) — suspect | Short, fast |
| Vw14c | 15826 | 5 | 0.03% | 49.9 | 7484 (47%) — suspect | Mixed |

- **Hard brake is rare** in these 3 seqs (0.03–0.07%). Paper Table 4 says hard-brake is a tagged scenario, but not every seq. Need to scan all 72 Categorised seqs to find hard-brake-rich ones for training.
- **4.1 Slip calc caveat:** `slip = |wheel*r*3.6 - gps_kmh|` with `r=0.315` overestimates — wheel speed may be transmission output not wheel, or tyre radius varies, or GPS speed filtered. Direct diff not reliable for label; better to use `Brake Pressure + acc_lon_g` as hard-brake proxy, and treat wheel vs IMU divergence as learned slip signature rather than analytic threshold. The V wheel data is still valuable as auxiliary label for *wheel odometry correction* (WhONet style), but phone-only model must learn IMU-only slip detection.

## 5. Synchronisation & Resampling Decision

- **Sync:** V and S row counts identical per seq, both 10 Hz, `Time Since Start` diff median 0.1s — already time-aligned. No need to align.
- **Resample for AVNet:** Paper needs `West=200 @100 Hz = 2s window`. IO-VNBD is 10 Hz → need **10× linear interp** (100 Hz) or halve West to `100 @50 Hz` (1s) or keep `20 @10 Hz` (2s). **Decision:** 10× interp to 100 Hz (as per `AGENTS.md` Step 2.2) — use `scipy.signal.resample` or `pandas.DataFrame.resample('10ms').interpolate('linear')`. Keep original 10 Hz also for val (less aliasing).
- **Window:** `West=200, stride=10` → 10 Hz output (paper 1 Hz output, but we want 10 Hz for InEKF). Equivalent to 2s window, 0.1s hop.

## 6. Scenario Coverage (from layout)

- **Categorised:** folder names encode driver/style but not scenario; scenario tags (hard-brake, wet, bump) are per CSV, not folder. Need to derive from signals: `acc_lon_g ≤-0.45`, `Brake Pressure >30`, `Steering Angle >30°`, `Yaw Rate >10°/s`, plus Uncategorised may be unlabeled.
- **Uncategorised:** `S-Dataset 72` under Uncategorised — likely extra phone-only data, not yet inspected.

## 7. Next Steps (Step 2)

- **2.1** Scan all 72 Categorised V files for hard-brake/wet/bump counts → `reports/scenario_counts.csv`.
- **2.2** Implement `python/preprocess.py` with 10 Hz→100 Hz interp, gravity-align, window 200/10, scaler `scaler.json` (per-axis mean/std from train split).
- **2.3** Test on Vtb01 1h subset: `train_windows.npy` shape `(>3000,200,6)`.

## 8. Raw Checks

```bash
# V header
head -n 1 "data/iovnbd/Synchronised V abd S datasets/Categorised IOVNB Dataset/Vtb (Driver E)/Vtb01/V-vtb1.csv"
# S header
head -n 1 ".../S-Vtb1.csv"
# Rates
python -c "import pandas as pd; v=pd.read_csv(..., encoding='latin1'); print(v[' Sample period (seconds)'].unique())"
# Shapes
wc -l .../V-vtb1.csv .../S-Vtb1.csv  # both 32460 inc header = 32459 rows
```

**Verified:** 195 MB zip → 826 MB extracted, 288 CSVs, 10 Hz both, latin1, sync rows.
