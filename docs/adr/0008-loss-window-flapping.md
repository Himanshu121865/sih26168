# ADR-008: Loss-detection window vs 1 Hz GNSS cadence (FLAPPING RISK)

## Status
Proposed — needs stakeholder decision before the field test.

## Context
`lossAfterMs = 300` satisfies the PS "milliseconds handover" requirement, but
`FusedLocationProvider` delivers fixes at ~1 Hz (1000 ms gaps in steady state).
Between two healthy fixes the ticker therefore crosses the 300 ms loss threshold
and flips GNSS → INS → (soft-blend) → GNSS roughly every second — steady-state
flapping, not a real outage. The sealed-state migration (ADR-007) preserved these
semantics exactly; this ADR questions them.

## Decision
Pending. Recommendation: raise `lossAfterMs` to ~1500 ms (1 missed fix tolerated,
2 missed = outage) OR gate the transition on fix quality (HDOP/sats) in addition
to age. Keep the 300 ms figure only for the integer in the proposal narrative if
judges test handover latency with a hard cut (Faraday), where any value ≤1000 ms
passes identically.

## Consequences

### Positive (if raised)
- No flapping in normal driving; INS engages only on genuine outages.
- UI badge stops flickering; CSV logs become interpretable.

### Negative (if raised)
- Worst-case handover latency grows 300 ms → 1500 ms (still within "seamless"
  for a 60 s outage demo, but weaker on paper).

### Neutral
- `SeamlessHandlerTest` boundary test pins whichever value is chosen.

## Alternatives Considered

**Keep 300 ms**
- Risks judges seeing INS flicker during the healthy baseline drive.

**Quality-gated transition (age + HDOP/sats)**
- Best of both, but more tuning surface before the deadline.

## References
- `android/.../engine/SeamlessHandler.kt`, `MainActivity.onLocationResult`
