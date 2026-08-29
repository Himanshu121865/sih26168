# Architecture — SIH26168 Intelligent Dead Reckoning (IDR) + AI GNSS/INS Fusion

> **Source PS:** [SIH26168](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26168.md) · [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) · [Main README](../README.md)  
> **Stack:** Cloud Training ([PyTorch](https://pytorch.org/)) → Edge Inference ([TFLite](https://www.tensorflow.org/lite) on [Android](https://developer.android.com/)) · Offline [OSM](https://www.openstreetmap.org) · No cloud at finale

---

## 1. System Context (C1)

```mermaid
graph LR
    subgraph Phone["Smartphone (Edge)"]
        APP[IDR App<br/>Kotlin + TFLite<br/>10Hz pose]
    end
    subgraph Cloud["Cloud/Desktop (Offline Training)"]
        TRAIN[AVNet Training<br/>PyTorch on IO-VNBD + Own]
        OSMDL[OSM PBF<br/>osmium]
    end
    subgraph Env["Environment"]
        GNSS[GNSS<br/>GPS/Galileo/NavIC<br/>1Hz NMEA]
        IMU[MEMS IMU<br/>Accel/Gyro/Mag/Báro<br/>100Hz]
        ROAD[Road Network<br/>OSM Highway]
        TUNNEL[Tunnel/Underpass<br/>GNSS Denied]
    end
    IMU --> APP
    GNSS --> APP
    ROAD --> APP
    TUNNEL -.->|blocks| GNSS
    TRAIN -- "model.tflite<br/>FP16 <2MB" --> APP
    OSMDL -- "offline tiles<br/>PBF" --> APP
    APP -->|MapLibre UI<br/>smooth icon| USER[Driver]
```

**Actors:** Driver (sees uninterrupted icon), ISRO Judge (simulates outage via Faraday mask), Training Engineer (pre-finale).

**Constraints from PS:** Phone-only, no OBD-II, must work with any external IMU (FOG @200Hz), milliseconds handover, 10Hz phone / 200Hz edge.

---

## 2. Container Diagram (C2)

```mermaid
graph TB
    subgraph PhoneContainer["Smartphone Container — Android 13+ (Snapdragon 778+)"]
        direction TB
        SENSOR[Sensors HAL<br/>SensorManager<br/>Accel/Gyro/Mag/Báro 100Hz<br/>FusedLocationProvider 1Hz]
        ALIGN[Alignment Engine<br/>Madgwick + PCA<br/>R_phone→vehicle]
        AVNET[AI Speed & Vibration Filter<br/>AVNet-lite TFLite<br/>CNN+GRU → v_fwd + σ]
        FUSION[Invariant EKF Fusion<br/>p,v,q,b_a,b_g<br/>10Hz]
        HMM[HMM Map Matcher<br/>Viterbi on OSM edges<br/>NHC pseudo]
        HANDLER[Seamless Handler<br/>GNSS loss <200ms<br/>soft reset]
        MAP[Map Engine<br/>OSM R-tree + MapLibre GL<br/>offline tiles]
        UI[UI Layer<br/>MapLibre view 30fps<br/>badge GNSS/INS/Fused]
        STORE[Local Store<br/>CSV logger + Ring Buffer<br/>200 window]
        SENSOR --> STORE
        STORE --> ALIGN
        ALIGN --> AVNET
        AVNET --> FUSION
        FUSION --> HMM
        FUSION --> HANDLER
        HMM --> MAP
        HANDLER --> UI
        MAP --> UI
    end
    subgraph TrainingContainer["Training Container — Cloud/Desktop (Ubuntu 22.04)"]
        IO[IO-VNBD<br/>40h/1300km V + 58h/4400km S]
        OWN[Own Indian Drive<br/>2-3h bike holder]
        PRE[Preprocess<br/>100Hz resample<br/>gravity-aligned window 200]
        TRAIN2[Train AVNet<br/>PyTorch 2.4<br/>L2 + NLL]
        EXPORT[Export<br/>ONNX opset17 → TFLite FP16<br/>quant + validate]
        IO --> PRE
        OWN --> PRE
        PRE --> TRAIN2
        TRAIN2 --> EXPORT
        EXPORT -.->|model.tflite<br/>scaler.json| PhoneContainer
    end
    subgraph EdgeBox["Edge Box (Optional FOG Variant)"]
        FOGIMU[FOG IMU 200Hz]
        CPP[Inference C++<br/>Eigen InEKF 200Hz<br/>same model]
        FOGIMU --> CPP
    end
```

---

## 3. Component Deep-Dive

### 3.1 Alignment Engine — Phone → Vehicle Frame

```mermaid
flowchart LR
    A["Window 2s @100Hz<br/>acc, gyro, mag<br/>+ GNSS cog"] --> B{"speed>15kmph<br/>+ low vibration?"}
    B -- No --> A
    B -- Yes --> C["Gravity low-pass<br/>→ roll/pitch"]
    C --> D["PCA on acc<br/>max variance = forward"]
    D --> E["Mag + GNSS vector<br/>→ yaw"]
    E --> F["Madgwick/Mahony refine<br/>R_pv"]
    F --> G["R_pv updated /5s<br/>detect ΔR>15° → re-calib"]
```

*   **[Links]:** [Madgwick AHRS](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/) · [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) · [Android SensorManager](https://developer.android.com/reference/android/hardware/SensorManager)
*   Output `R_pv` rotates every IMU sample to vehicle frame before AVNet — matches [RoNIN](https://github.com/Sachini/ronin) gravity-aligned trick.

### 3.2 AVNet-lite — AI Speed & Vibration Filter

```
Input: 200×6 (acc 3 + gyro 3, gravity-aligned, @100Hz = 2s)
  │
  ├─► 1D-CNN: Conv1d 6→32 k9 dilated=1 → BN → ReLU
  │           Conv1d 32→64 k9 dilated=2 → BN → ReLU
  │           → MaxPool 2 → Dropout 0.1
  │
  ├─► GRU: 2 layers ×64 hidden, seq_len 100 → last hidden
  │
  └─► FC: 64→32 → ReLU → Dropout → 32→2 (v_fwd, logσ)
       Loss: L = ||v_pred - v_gt||² + λ·NLL(v_pred, σ)
       Params: ~150k, FP16 <1.2MB, <8ms on SD778 via NNAPI
```

*   Training augmentation: inject `2–15g` pothole spikes @20–80Hz + engine harmonics from own data — model learns to ignore vs classical low-pass.
*   **Links:** [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) · [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) · [NLL](https://en.wikipedia.org/wiki/Likelihood_function) · [TFLite](https://www.tensorflow.org/lite) · [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks)
*   Window stride 10 → 10Hz velocity output, feeds EKF.

### 3.3 Invariant EKF Fusion — Core State

**State (16-dim):** `p(3), v(3), q(4 quaternion), b_a(3), b_g(3)`  
**Refs:** [Invariant EKF Paper — Barrau & Bonnabel 2017](https://arxiv.org/abs/1709.03549) · [EKF Wiki](https://en.wikipedia.org/wiki/Extended_Kalman_filter)

```mermaid
sequenceDiagram
    participant IMU as IMU 100Hz
    participant PROP as Propagate Strapdown
    participant EKF as InEKF State
    participant GNSS as GNSS 1Hz
    participant AVNET as AVNet 10Hz
    participant NHC as NHC Pseudo

    IMU->>PROP: acc, gyro - b_a/b_g
    PROP->>EKF: p,v,q integrate
    GNSS-->>EKF: if HDOP<2 && sats≥6 → z = p_gnss - p_pred, R=HDOP²·σ
    AVNET-->>EKF: always → z = v_ai - v_pred, R=σ_ai²
    NHC-->>EKF: always → z = [v_y, v_z]≈0, R=σ_NN+motion
    EKF->>EKF: Update 10Hz pose
```

*   GNSS quality gate Discards jamming: `HDOP>2` or `sats<6` → ignore, prevents ISRO jam test from poisoning.
*   Outputs pose @10Hz (phone) / 200Hz (FOG) — fulfills PS table.

### 3.4 HMM Map Matching + NHC

*   **OSM preprocessing:** [`osmium`](https://osmcode.org/osmium-tool/) extracts `highway=*` → build [R-tree](https://en.wikipedia.org/wiki/R-tree) (50m search).
*   **HMM:** Emission `N(dist, σ=15m)`, Transition `exp(-|Δheading|/30°)` via [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) — [Map matching Wiki](https://en.wikipedia.org/wiki/Map_matching).
*   **NHC:** `v_y≈0, v_z≈0` in vehicle frame — [Nonholonomic Wiki](https://en.wikipedia.org/wiki/Nonholonomic_system) — with learned σ high in turn, low straight. Concept from [AI-IMU `1904.06064`](https://arxiv.org/abs/1904.06064) / [RINS-W `1904.01120`](https://arxiv.org/abs/1904.01120).

### 3.5 Seamless Handler — State Machine

```mermaid
stateDiagram-v2
    [*] --> GNSS_AIDED: HDOP<2, fix 3D
    GNSS_AIDED --> DR: loss >300ms OR HDOP>5×3
    DR --> GNSS_AIDED: fix regained + soft reset
    GNSS_AIDED: EKF updates GNSS+AVNet+NHC
    DR: Updates AVNet+NHC only,<br/>R_gnss→∞ over 200ms

    note right of DR
        Freeze b_a/b_g last GNSS
        Snap to HMM edge
        <200ms latency
    end note
    note right of GNSS_AIDED
        Soft reset<br/>p=α·p_gnss+(1-α)·p_pred<br/>α 0→1 over 1s<br/>no jump
    end note
```

*   Detection `300ms` no fix → requirement: milliseconds handover.

### 3.6 Offline Training Pipeline

```mermaid
flowchart TB
    A["IO-VNBD<br/>github.com/onyekpeu/IO-VNBD<br/>Synchronised V&S"] --> P
    B["Own Indian Drive<br/>bike holder 2-3h"] --> P
    C["RoNIN 42.7h<br/>ronin.cs.sfu.ca<br/>+ GSDC 578m tunnel"] --> P
    P["Preprocess<br/>100Hz resample<br/>gravity-align<br/>200 window, stride10<br/>bias augment"] --> T
    T["Train AVNet<br/>PyTorch + Lightning<br/>L2+NLL λ=0.1<br/>AdamW 1e-3, 50 epochs"] --> E
    E["Export<br/>ONNX opset17 → TFLite FP16<br/>validate <1e-3 diff"] --> M["model.tflite<br/>scaler.json"]
    M --> PHONE["Phone assets/"]
    D["OSM PBF<br/>osmcode.org/osmium<br/>geofabrik.de"] --> TILE["MapLibre tiles"]
    TILE --> PHONE
```

*   **Mandatory for proposal:** Include `drift_plot.png` inferred from IO-VNBD subset — judges screening use *more hidden datasets*.

---

## 4. Data Flow & Interfaces

| Interface | Producer → Consumer | Format | Freq |
|-----------|---------------------|--------|------|
| `imu_raw` | `SensorManager` → `RingBuffer` | `float[6] @100Hz (acc, gyro)` | 100Hz |
| `gnss_nmea` | `FusedLocationProvider` → `InEKF` | `lat,lon,hdop,sats,speedAcc` [NMEA 0183](https://en.wikipedia.org/wiki/NMEA_0183) | 1Hz |
| `R_pv` | `Alignment` → `AVNet` | `3×3 rotation` | /5s |
| `v_ai + σ` | `AVNet TFLite` → `InEKF` | `float v_fwd, float logσ` | 10Hz |
| `pose` | `InEKF` → `HMM/UI` | `p(3) lat/lon, v(3), yaw, σ` | 10Hz |
| `edge_match` | `HMM` → `UI` | `edge_id, snap_point` | 10Hz |
| `model.tflite` | `Cloud` → `Phone assets/` | `FP16 1.2MB` | once |

---

## 5. Deployment Architecture

```
Developer Laptop (Training)
  ├─ Python 3.10 + PyTorch 2.4 + Lightning
  ├─ IO-VNBD (UK/NG/FR) + Own CSV (50MB)
  └─ outputs: model.tflite ─┐
                            │
Smartphone (Finale — Offline)
  ├─ Android 13+, SD778, NNAPI
  ├─ APK (Kotlin, TFLite 2.14, MapLibre GL)
  ├─ model.tflite + scaler.json in assets/
  ├─ OSM offline tiles (city PBF, ~40MB)
  └─ CSV logger → drift evaluation

Edge Box (FOG demo, optional)
  ├─ FOG IMU 200Hz (e.g., Xsens MTi-680)
  └─ C++ Eigen InEKF 200Hz + same model (ONNX Runtime)
```

*   No internet at finale — all inference on-device. Training is a priori (ISRO allows).
*   Links: [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) · [ONNX Runtime](https://onnxruntime.ai/) · [MapLibre](https://maplibre.org/) · [OSMDroid](https://github.com/osmdroid/osmdroid) · [Eigen](https://eigen.tuxfamily.org/)

---

## 6. Security & Privacy

*   No camera, no PII beyond trajectory — unlike [VIO](https://en.wikipedia.org/wiki/Visual_inertial_odometry) (TLIO paper notes privacy advantage of IMU-only).
*   Local store encrypted, GNSS not uploaded.
*   FOG edge variant for fleet — on-prem, no cloud.

---

## 7. Performance Budget

| Component | Latency Budget | On SD778 |
|-----------|---------------|----------|
| IMU read 100Hz | <2ms | 0.5ms |
| Alignment | /5s | 3ms |
| AVNet TFLite infer | <20ms | 7–8ms FP16 |
| InEKF update | <5ms | 1.2ms |
| HMM Viterbi (20 edges) | <10ms | 4ms |
| MapLibre render | 30fps | 16ms frame |
| **Total loop 10Hz** | **<100ms** | **~32ms** |

Memory: Model 1.2MB + OSM tiles 40MB + ring buffer 2MB.

---

## 8. Alternatives Considered

| Decision | Option A (Chosen) | Option B (Rejected) | Why |
|----------|-------------------|---------------------|-----|
| Velocity | Learn `v_fwd` directly ([RoNIN](https://arxiv.org/abs/1905.12853)/[AVNet](https://doi.org/10.1186/s43020-025-00168-7)) | Learn bias `b_a,b_g` | Needs bias GT labels, harder; velocity proven 0.64% tunnel |
| Filter | [InEKF](https://arxiv.org/abs/1709.03549) | Classic EKF | InEKF better yaw observability, used in DMDVDR |
| Map | [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model) offline OSM | Online GraphHopper | Finale offline, no internet |
| Stack | Native [Kotlin](https://kotlinlang.org/) + [SensorManager](https://developer.android.com/reference/android/hardware/SensorManager) | Flutter/React Native | Native gives stable 100Hz, jitter in cross-platform |
| Export | [TFLite FP16](https://www.tensorflow.org/lite) | ONNX Runtime Mobile | TFLite smaller, NNAPI accelerated |

---

## 9. Validation

*   **Offline:** Mask IO-VNBD GPS 60s → `ATE`, `Drift%` — target `<10%` pass, `<3%` impress. Cross-validate [RoNIN](https://ronin.cs.sfu.ca/) + [GSDC](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) 578m.
*   **Live:** ISRO will apply [Faraday](https://en.wikipedia.org/wiki/Faraday_cage) / tunnel sim → log final drift, latency, adherence.
*   Always plot `naive double integration` vs AI — demonstrates learning (see [PMC Sensors 19:1618 Fig.11](https://pmc.ncbi.nlm.nih.gov/articles/PMC6480342/figure/sensors-19-01618-g011/)).

---

## 10. References (Architecture-Specific)

*   [RoNIN Code](https://github.com/Sachini/ronin) · [RoNIN Dataset](https://ronin.cs.sfu.ca/) · [TLIO Project](https://cathias.github.io/TLIO/) · [AVNet Paper Satellite Navigation 2025](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-025-00168-7) · [MGTR/TAMS MDPI 2026](https://www.mdpi.com/2227-7390/14/13/2423) · [AirIMU 2310.04874](https://arxiv.org/abs/2310.04874)
*   Tooling: [Osmium](https://osmcode.org/osmium-tool/) · [OSM PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) · [Osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) · [Geofabrik](https://download.geofabrik.de/) · [ISRO Bharatiya Antariksh Hackathon](https://www.isro.gov.in/BharatiyaAntarikshHackathon.html) · [NavIC](https://en.wikipedia.org/wiki/Indian_Regional_Navigation_Satellite_System)

---

*Next: See [../README.md](../README.md) for roadmap, quickstart, and full reference list.*
