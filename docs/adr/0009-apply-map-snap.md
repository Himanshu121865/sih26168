# ADR-009: Apply HMM snap to the emitted pose (currently display-only gap)

## Status
Proposed — fix before the field test.

## Context
Review found `DrPipeline.emitPose()` returns the raw `(lat, lon)` and only stores
the matcher result in `lastSnappedLat/Lon`: the HMM snap never corrects the
pose the UI logs and the judges score. ADR-003's expected 40–50% drift reduction
is therefore currently unearned on-device, even though `eval_drift.py --mode 1d`
plots a map-improved curve.

## Decision
Pending. Recommendation: blend the snap into the emitted pose with a confidence
weight (`pose = α·snap + (1-α)·filter`, α from matcher likelihood), keep the raw
trace in the CSV for scoring transparency, and add a `mapAdherence%` metric to
the field-test log. Never hard-snap: a wrong-road fix must stay recoverable.

## Consequences

### Positive (if applied)
- On-device numbers match the proposal plot; adherence becomes demonstrable.

### Negative (if applied)
- Blending introduces a tuning parameter and a new failure mode (snap lag in
  fast turns); needs the underpass drive to validate.

### Neutral
- `RoadGraph.load` failure path (no bundled graph) stays: matcher absent → raw
  pose, unchanged behavior.

## Alternatives Considered

**Hard snap to the matched edge**
- Rejected: unrecoverable on wrong-road matches.

**Display-only (status quo)**
- Rejected: proposal claims map-matched accuracy the app does not deliver.

## References
- `android/.../engine/DrPipeline.kt:emitPose`, ADR-003
