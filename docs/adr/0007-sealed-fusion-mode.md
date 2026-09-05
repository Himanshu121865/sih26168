# ADR-007: Sealed FusionMode state for the deficit handler

## Status
Accepted

## Context
`SeamlessHandler` exposed `enum Mode {GNSS, INS}` plus a parallel `gnssTrust`
field. Callers had to read two fields consistently, `when` branches needed
`else`, and adding a state (e.g. degraded-GNSS) would silently compile at every
call site.

## Decision
`sealed interface FusionMode { GnssAided(trust): displayName "GNSS";
DeadReckoning: "INS" }`. Trust lives inside the state object; all consumers use
exhaustive `when` with no `else`. Migrated `DrPipeline.mode`, the 10 Hz ticker,
status chip, and CSV logger (`mode.displayName` replaces enum `name`).
`tick()`/`onFix()` semantics are bit-identical (traced, incl. the strict `>`
loss boundary).

## Consequences

### Positive
- State and trust can never disagree; new states break callers at compile time.
- All six `!!` operators in the app removed in the same pass (smart-cast locals).

### Negative
- Touches the handover hot path; requires the unit suite green before release
  (`SeamlessHandlerTest`, 5 cases).

### Neutral
- No behavior change intended — pure modeling improvement.

## Alternatives Considered

**Keep enum + trust field**
- Rejected: the exact two-field-disagreement class of bug this removes.

## References
- `android/.../engine/SeamlessHandler.kt`, commit `ce5ce4f`
