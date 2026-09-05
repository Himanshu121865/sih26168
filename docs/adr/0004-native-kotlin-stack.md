# ADR-004: Native Kotlin + SensorManager over cross-platform

## Status
Accepted

## Context
The engine needs a stable 100 Hz IMU stream plus 1 Hz GNSS with millisecond
handover. Cross-platform frameworks add sensor-timing jitter; the team targets a
single Android APK for the finale.

## Decision
Native Kotlin (`SensorManager` @100 Hz, `FusedLocationProvider` @1 Hz, TFLite
2.14, MapLibre GL). Structured concurrency only (`lifecycleScope`, activity-bound,
cancellable via `delay` points); no `GlobalScope` / `runBlocking` in production.

## Consequences

### Positive
- Stable 100 Hz delivery; NNAPI-accelerated inference at 7–8 ms.
- `lifecycleScope` gives teardown cancellation for free.

### Negative
- Android-only; an iOS/FOG variant would need a second port (the C++/Eigen edge
  box path is stubbed, not built).

### Neutral
- 10 Hz engine loop stays polling-style; no Flow — a fixed-rate sensor loop gains
  nothing from reactive streams (deliberate exception, reviewed 2026-09-05).

## Alternatives Considered

**Flutter / React Native**
- Rejected: sensor-timing jitter at 100 Hz; native gives deterministic sampling.

## References
- `android/app/src/main/java/com/sih26168/dr/MainActivity.kt`
- `docs/ARCHITECTURE.md` §8
