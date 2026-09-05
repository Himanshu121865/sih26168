# ADR-006: Train from scratch on IO-VNBD; reject RoNIN/TLIO weights

## Status
Accepted

## Context
Pretrained phone-IMU weights exist (RoNIN, TLIO) and transfer learning is tempting
given the 20 Sep 2026 deadline. But RoNIN weights are non-commercial licensed and
pedestrian-trained (periodic gait ≠ vehicle constant velocity); TLIO ships code
only with no usable weights. The AVNet authors published reference code
(QDeepOdo/QAIIMU) without standalone weights.

## Decision
Option B only: train from scratch on IO-VNBD (40 h V + 58 h S @10 Hz → 100 Hz)
using QDeepOdo/QAIIMU as architecture reference, fixing two upstream bugs before
training (`Flatten(0)` → `Flatten(start_dim=1)`, `hx=randn` → zeros).

## Consequences

### Positive
- Clean commercial/IP footing for an ISRO submission; no domain shift from gait.
- Full control of preprocessing (gravity removal, scaling, windowing).

### Negative
- Pays full training cost (Colab T4, ~15 min/full run) and needs the full 72-seq
  pipeline to be correct — no pretrained fallback.

### Neutral
- Reference repos stay vendored under `ref/` for audit, never imported at runtime.

## Alternatives Considered

**RoNIN pretrained weights**
- Rejected: non-commercial license + pedestrian domain mismatch.

**TLIO transfer**
- Rejected: no released weights; dataset access impractical.

## References
- AGENTS.md "Key Decision — 2026-08-29", `docs/DATA_INSPECTION.md`
