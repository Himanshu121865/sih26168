# ADR-001: Learn forward velocity directly (AVNet) instead of IMU biases

## Status
Accepted

## Context
Dead reckoning needs a velocity signal to bound integration drift. Two families exist:
learn the velocity itself (RoNIN/AVNet style) or learn the IMU biases and integrate
corrected raw readings. The Qian et al. AVNet paper reports 0.64% drift in tunnels on
cars with direct velocity learning. IO-VNBD gives us GPS-speed labels for free, while
per-sample bias ground truth does not exist in any available dataset.

## Decision
`AVNetLite` (CNN+GRU, 460k params) predicts forward speed `v_fwd` + uncertainty
`logσ_v` from 2 s gravity-aligned windows. `train_avnet.py` optimizes joint
`MSE + λ·NLL`. TCN backbone is deferred: switch only if GRU latency exceeds 8 ms.

## Consequences

### Positive
- Proven 0.64% tunnel result on the same problem shape; labels already available.
- Uncertainty head feeds the InEKF measurement covariance directly (ADR-002).

### Negative
- Model is only as good as GPS-speed labels; phone GPS noise sets an MSE floor
  (observed on the full 72-seq run: val 1.73 vs 0.336 on the easy 3-file subset).
- Attitude head is currently untrained (att labels are zeros).

### Neutral
- Speed is predicted at 10 Hz; heading still comes from gyro/GNSS integration.

## Alternatives Considered

**Learn biases `b_a, b_g`**
- Rejected: no bias ground truth in IO-VNBD; harder supervision problem.

**Causal TCN backbone (harshkumarsingh12/dead-reckoning)**
- Deferred: faster and causal, but GRU met the latency budget first. Revisit if
  on-device inference exceeds 8 ms.

## References
- Qian et al., Satellite Navigation 2025, DOI 10.1186/s43020-025-00168-7
- `python/models/avnet.py`, `python/train_avnet.py`
