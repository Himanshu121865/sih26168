# Architecture Decision Records

| ADR | Title | Status |
|-----|-------|--------|
| [0001](0001-learn-velocity-directly.md) | Learn forward velocity directly (AVNet) | Accepted |
| [0002](0002-invariant-ekf-fusion.md) | Right-invariant EKF (21-DOF) for fusion | Accepted |
| [0003](0003-offline-hmm-map-matching.md) | Offline HMM map matching on OSM | Accepted |
| [0004](0004-native-kotlin-stack.md) | Native Kotlin + SensorManager stack | Accepted |
| [0005](0005-tflite-export-gate.md) | TFLite FP16 export with ONNX validation gate | Accepted |
| [0006](0006-train-from-scratch.md) | Train from scratch; reject RoNIN/TLIO weights | Accepted |
| [0007](0007-sealed-fusion-mode.md) | Sealed FusionMode state | Accepted |
| [0008](0008-loss-window-flapping.md) | Loss-window vs 1 Hz cadence (flapping risk) | Accepted |
| [0009](0009-apply-map-snap.md) | Apply HMM snap to emitted pose | Proposed |
| [0010](0010-training-strategy.md) | Full-72-seq strategy after val 1.73 plateau | Proposed |

Conventions follow the skill template: Status ∈ {Proposed, Accepted, Deprecated,
Superseded by ADR-XXX}. Accepted ADRs record what shipped and why; Proposed ADRs
require a stakeholder call before the field test. Update statuses in place —
never rewrite history.
