# ADR-002: Right-invariant EKF (21-DOF) for GNSS/INS fusion

## Status
Accepted

## Context
The filter must fuse 100 Hz IMU with 10 Hz AI velocity, 1 Hz GNSS, NHC
pseudo-measurements and ZUPT, with yaw staying observable through turns so the
path does not bend. The DMDVDR reference implementation (QAIIMUDeadReckoning)
uses a right-invariant EKF on SE2(3) with group-affine dynamics.

## Decision
Port `filter_propagate_improved` / `filter_update_improved` faithfully to
`python/inekf_harness.py` (float64, exact left Jacobian, 3rd-order Φ) and then to
Kotlin `InEKFEngine`. State: R_nav, v, p, b_g, b_a, R_car, p_car (21-DOF).
Two frame fixes over the reference are load-bearing: measurement order is
`z=[v_fwd, v_lat, 0]` (our X=forward vs ref Y-forward) and gravity is disabled
(`G=0`) for preprocessed linear-acc windows.

## Consequences

### Positive
- Best yaw observability of the evaluated options; harness beats AVNet-only on
  3/4 replay segments (mean drift 11.4% vs 17.7%).
- Validated Python reference makes the Kotlin port mechanical.

### Negative
- 21 states is overkill for mostly-2D driving; each propagate/update is O(21³)-ish
  matrix math (~1.2 ms on SD778, acceptable but not free).
- `q_acc` is rate-dependent: 30.0 for the 10 Hz proxy replay vs 0.5 live at
  100 Hz (see ADR-010).

### Neutral
- Bias clamping (b_g ±0.5, b_a ±2.0) is a safety rail, not physics.

## Alternatives Considered

**Classic EKF / 2D ESKF**
- Rejected for now: weaker yaw observability. An 8-state 2D filter
  (sivaraman-tech baseline) remains the fallback if the phone budget tightens.

## References
- Barrau & Bonnabel 2017, arXiv:1709.03549
- `python/inekf_harness.py`, `android/.../engine/InEKFEngine.kt`
