# Step-2 Training Plan — after the per-file audit (ADR-010)

Run AFTER `python/eval_per_file.py`. The audit's verdict line selects exactly one
branch below. All commands run on Colab T4 from the repo root. Nothing here needs
local edits — the `--split stratified` code is already committed.

## Step 0 — Audit (prerequisite, ~8 min)

```bash
!PYTHONPATH=. python python/eval_per_file.py --model experiments/checkpoints/model_avnet_stage1.p
```

Paste back: the per-file table + `recombined val MSE` + `top-3 share` verdict.

---

## Branch A — top-3 share >60% (split luck) → stratified re-split

Val is dominated by a few hard trajectories the random split isolated. Fix the
split, not the model.

```bash
# 1. Re-preprocess with 80/20 inside each driver × speed-tercile bucket (~7 min)
!rm -rf data/processed
!PYTHONPATH=. python python/preprocess.py --subset full --window 200 --stride 10 --hz 100 --split stratified
!ls -lh data/processed/ && cat python/scaler.json | head -25

# 2. Smoke train, 5 epochs (~5 min)
!PYTHONPATH=. python python/train_avnet.py --epochs 5 --batch 256 --lr 1e-3 --device cuda --augment-yaw --augment-bike --lambda-nll 0.1

# 3. Re-audit against the NEW split
!PYTHONPATH=. python python/eval_per_file.py --model experiments/checkpoints/model_avnet_stage1.p --split stratified
```

**Accept:** val best <1.4 by epoch 5 AND top-3 share <50%. Then run the full
15-epoch train and continue to eval/export cells.
**If val still ≥1.6:** stop — it wasn't the split. Fall through to Branch B
diagnostics on the new split (speed bins are already in the audit table).

## Branch B — systemic, `>15 m/s` bin 2× worse (label noise)

Phone GPS speed (1 Hz, interpolated) is the noise floor at high speed. Do NOT
retrain yet — quantify first:

```bash
# Fill FILES with the 2-3 worst high-speed files from the audit table, then run:
!PYTHONPATH=. python -c "
import pandas as pd
from python.core.signal import find_column
FILES = ['path/to/worst1_S-*.csv', 'path/to/worst2_S-*.csv']
for f in FILES:
    df = pd.read_csv(f, encoding='cp1252'); df.columns = [c.strip() for c in df.columns]
    v = find_column(df, r'gps speed'); sats = find_column(df, r'satellites'); acc = find_column(df, r'gps accuracy')
    print('==', f)
    print(' speed kmh:', df[v].describe()[['mean','std','min','max']].to_dict())
    if sats: print(' sats:', df[sats].describe()[['mean','min']].to_dict())
    if acc: print(' accuracy m:', df[acc].describe()[['mean','max']].to_dict())
"
```

**Next (needs a decision, no code written yet):** CAN-speed supervision
(`Velocity (km/h)` from V-*.csv as labels, phone IMU stays the input) vs GPS
smoothing (median filter on resampled speed before labeling). Report the dropout
finding and wait for the call — this changes `preprocess.py` labeling.

## Branch C — flat across files AND speeds (capacity)

Data is fine; `AVNetLite` 460k underfits 822k diverse windows. Test the 13 M
`AVNet` (already in `python/models/avnet.py`, no new code):

```bash
# NOTE: train_avnet.py currently builds AVNetLite. Capacity test needs a
# --model {lite,full} flag (not yet implemented — request it, ~15 lines).
# Until then, DO NOT run: full-model training without the flag would silently
# train Lite again. Paste "go capacity" and the flag + 3-epoch command follow.
```

**Accept:** 3-epoch full-model val < Lite val at same epoch → commit to a full
15-epoch run (~3× slower, ~2.5 h). Else the bottleneck is elsewhere (return to B).

## After any branch — close-out (same every time)

```bash
!PYTHONPATH=. python python/eval_drift.py --mode 2d --model experiments/checkpoints/model_avnet_stage1.p --plot reports/drift_2d.png
!PYTHONPATH=. python python/inekf_harness.py --model experiments/checkpoints/model_avnet_stage1.p --windows 600 --start 2000 --lean-mode car --q-acc 30.0
```

**Ship criteria:** 2D drift <10% (target <5%) AND harness segment ≤ AVNet-only.
Then: export cell → screening.zip → field test (ADR-009 map-snap is next).
