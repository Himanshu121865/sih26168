# ADR-003: Offline HMM map matching on OSM

## Status
Accepted

## Context
Residual 2–3% filter drift is the cheapest accuracy to buy: snapping to the road
network typically halves it. The finale is fully offline (no internet), so any
online router (GraphHopper) is disqualified and tiles/graph must ship in the APK.

## Decision
HMM (emission `N(dist, σ=15 m)`, transition on heading change) with Viterbi over
≤20 candidate edges from an offline OSM road graph (`maps/road_graph.pkl`,
50 m search radius). Raw trace is always kept as fallback.

## Consequences

### Positive
- Expected 40–50% drift reduction (e.g. 5% → 2.5%) for ~4 ms on SD778.
- Zero network dependency at the finale.

### Negative
- Wrong-road snaps are worse than no snap; heading (not just distance) must gate.
- City PBF + tiles add ~40 MB to the bundle.

### Neutral
- Map-matched pose is display-grade; evaluation still scores the raw filter output.

## Alternatives Considered

**Online GraphHopper / Valhalla**
- Rejected: finale has no internet.

**No map matching**
- Rejected: leaves the cheapest accuracy gain on the table.

## References
- `docs/ARCHITECTURE.md` §3.4, `python/hmm_matcher.py`, `maps/download_osm.sh`
