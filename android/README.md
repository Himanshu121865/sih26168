# SIH26168 Android App (Step 12)

Phone-only dead reckoning with GNSS fusion, offline OSM maps, and AI speed prediction.

## Build (Android Studio)

1. Open **this `android/` folder** in Android Studio (Hedgehog or newer, JDK 17).
2. Let Gradle sync. The `copyModelAssets` task pulls `../model.tflite` + `../scaler.json`
   from the repo root into `app/src/main/assets/` automatically.
3. Run on a device (needs real sensors — emulator IMUs are too clean).

CLI: `./gradlew assembleDebug`

## What's inside

```
app/src/main/java/com/sih26168/dr/
├── MainActivity.kt          # sensors @100Hz, GNSS 1Hz, 10Hz fusion ticker, map UI
├── engine/
│   ├── LieGroup.kt          # SO(3)/SE2(3) math (port of inekf_harness.py)
│   ├── InEKFEngine.kt       # 21-DOF filter (port of python InEKF — see AGENTS.md caveats)
│   ├── AVNetInference.kt    # TFLite AVNetLite, 200-sample ring @100Hz, 10Hz inference
│   ├── Scaler.kt            # per-channel normalization (scaler.json)
│   ├── LeanDetector.kt      # bike lean + adaptive NHC (v_lat = v_fwd·sinφ)
│   ├── ZuptDetector.kt      # stationary detect -> v=0 update
│   ├── AlignmentEngine.kt   # roll/pitch from gravity, yaw from GNSS course
│   ├── SeamlessHandler.kt   # GNSS loss 1500ms -> INS; soft re-blend 1s
│   └── DrPipeline.kt        # glue: IMU -> AVNet -> InEKF -> pose
├── map/
│   ├── OfflineRegionManager.kt  # "Download this area" — OSM tiles via MapLibre offline
│   ├── RoadGraph.kt             # bundled OSM road graph (maps/road_graph.json)
│   └── HmmMapMatcher.kt         # Viterbi snap (emission N(15m), transition heading)
└── io/CsvLogger.kt          # timestamp,pred,gnss,v,phi,mode log
```

## Offline maps

Two paths (both per the PS):
- **Runtime download**: the save FAB downloads OSM tiles for the visible bbox
  (MapLibre `OfflineManager`, works fully offline afterwards).
- **Pre-bundled road graph**: `python/maps/build_graph.py` exports OSM ways to
  `assets/maps/road_graph.json` — this powers the on-device HMM map matcher
  (the "offline map database" constraint from the PS).

## Tests

`./gradlew test` — JUnit mirrors of the Python harness validation:
- `LieGroupTest` — so3exp identity/orthogonality, sen3exp vs python
- `LeanDetectorTest` — φ=30° → v_lat = v_fwd·sinφ (mirrors `--test-lean` PASS)
- `InEKFEngineTest` — propagate stability, update convergence, bias clamp
- `SeamlessHandlerTest` — 1500ms loss boundary, single-missed-fix tolerance, 1s blend
- `ZuptDetectorTest` — still latches, moving/speed-gate rejects, partial window

## Field-test procedure (finale)

1. Mount phone rigid (holder), start app, drive with GNSS for 30s (alignment + model warmup).
2. Simulate outage: airplane mode (or Faraday pouch). App flips GNSS→INS in <1.5s.
3. Drive 1km @60km/h. Check drift on the CSV log (`Android/data/com.sih26168.dr/files/Documents/`).
4. Re-enable GNSS — soft blend back over 1s. Target: drift <5%, handover seamless.

## Known gaps (skeleton)

- Bike classifier `p_bike` fixed at 0 (car mode) until Step 8 fine-tuning data exists.
- InEKF propagates at inference tick (10Hz) — production should propagate per-sample @100Hz.
- Offline style URL uses MapLibre demotiles — swap to a full OSM style (e.g., OpenFreeMap) for the finale.
