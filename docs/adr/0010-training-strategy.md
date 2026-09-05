# ADR-010: Full-72-seq training strategy after the val 1.73 plateau

## Status
Proposed — pending per-file audit (`python/eval_per_file.py`, Option A).

## Context
The full run (822,928 train / 178,369 val) plateaus at val MSE 1.73 (RMSE 1.33)
vs 0.336 on the easy 3-file subset; train falls monotonically (1.79 → 1.06) while
val oscillates. Single-segment eval shows AI drift 77% (worse than naive 16%),
yet the InEKF harness still rescues its segment to 2.2%. Three compounding
suspects: (a) random 57/15 split concentrates hard trajectories in val,
(b) phone-GPS-speed labels are noisy at high speed, (c) `q_acc` is
rate-dependent (30.0 @10 Hz proxy vs 0.5 live @100 Hz) so harness gains may not
transfer. `AVNetLite` capacity (460k) on 822k diverse windows is unproven.

## Decision
Pending audit outcome, in order:
1. If top-3 val files hold >60% of weighted MSE → stratified re-split by
   speed × driver, one 7-min re-preprocess, 5-epoch smoke test.
2. Else if `>15 m/s` bin is 2× worse → label fix (CAN-speed supervision or
   GPS smoothing) before any model change.
3. Else → capacity test: `AVNet` 13 M for 3 epochs on the same split.
Do not change `q_acc` defaults until a 100 Hz streaming replay exists; the
current split (30.0 proxy / 0.5 live) is documented, not unified.

## Consequences

### Positive
- Each step falsifies exactly one hypothesis; no stacked changes.

### Negative
- Up to three Colab cycles (~1 h total) before the field test.

### Neutral
- The easy-subset checkpoint stays the screening fallback regardless.

## Alternatives Considered

**More epochs on the current split**
- Rejected: 15-epoch curve is flat from epoch 7; LR halving did nothing.

**Jump straight to the 13 M model**
- Rejected: 28× params without knowing whether the bottleneck is data.

## References
- `python/eval_per_file.py`, `sih26168_colab.ipynb` run 2026-09-04
- `python/inekf_harness.py` (`--q-acc`), `reports/inekf_vs_avnet.csv`
