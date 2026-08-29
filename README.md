# SIH26168 — AI-ML based Intelligent Dead Reckoning System for Seamless Navigation

**Organisation:** [Indian Space Research Organisation (ISRO)](https://www.isro.gov.in/) — Department of Space  
**Category:** Software · **Theme:** Miscellaneous · **PS Number:** `SIH26168`  
**Deadline:** 20 September 2026 · **Submitted Ideas:** 0/500 (as of 21 Aug 2026 snapshot)  
**Dataset:** [IO-VNBD: Inertial and Odometry Benchmark Dataset for Ground Vehicle Positioning](https://github.com/onyekpeu/IO-VNBD) (official PS link)  
**Difficulty:** **Level 4 / Very Hard — 85/100** (`technical_complexity:5, algorithmic_depth:5, data_dependency:5, integration:5, mvp_feasibility:2`)  
**Source:** [sih.gov.in/sih2026PS](https://sih.gov.in/sih2026PS) · Snapshot 21 Aug 2026 · Verified 28 Aug 2026 (live portal shows 175 Software + 54 Hardware = 229 total)

> **One-line:** Transform a *standalone smartphone* ([MEMS IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit) only, no [OBD-II](https://en.wikipedia.org/wiki/On-board_diagnostics)/wheel odometry) into an **Intelligent Dead Reckoning (IDR) + AI [GNSS](https://en.wikipedia.org/wiki/Satellite_navigation)/[INS](https://en.wikipedia.org/wiki/Inertial_navigation_system) Fusion engine** that keeps navigation alive in tunnels, underpasses, parking lots and [urban canyons](https://en.wikipedia.org/wiki/Urban_canyon) and fuses seamlessly back to GNSS within milliseconds.

---

## Table of Contents

1. [Complete Official Problem Statement](#1-complete-official-problem-statement)
2. [Why This Is "Impossible" — Core Physics Challenge](#2-why-this-is-impossible--core-physics-challenge)
3. [Research Landscape — What SOTA Has (and Has Not) Solved](#3-research-landscape--what-sota-has-and-has-not-solved)
4. [Official Dataset Deep-Dive — IO-VNBD](#4-official-dataset-deep-dive--io-vnbd)
5. [Performance Benchmarks (ISRO Judging Criteria)](#5-performance-benchmarks-isro-judging-criteria)
6. [Proposed Architecture — Cloud Training / On-Device Inference Hybrid](#6-proposed-architecture--cloud-training--on-device-inference-hybrid)
7. [Module Specifications](#7-module-specifications)
8. [Tech Stack](#8-tech-stack)
9. [Implementation Roadmap — From Zero to SIH Finale](#9-implementation-roadmap--from-zero-to-sih-finale)
10. [Evaluation & Validation Protocol](#10-evaluation--validation-protocol)
11. [Risk Register & Mitigations](#11-risk-register--mitigations)
12. [Repository Layout](#12-repository-layout)
13. [Quickstart — Reproduce Baseline in 15 Minutes](#13-quickstart--reproduce-baseline-in-15-minutes)
14. [References](#14-references)

---

## 1. Complete Official Problem Statement

### Background (verbatim, source: [`ps_2026/SIH26168.md:10-22`](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26168.md))

> Vehicle logistics, ride-hailing, quick commerce and emergency responders rely on smartphone navigation apps ([Google Maps](https://www.google.com/maps) / [MapmyIndia Mappls](https://www.mapmyindia.com/mappls)) powered by GNSS ([GPS](https://en.wikipedia.org/wiki/Global_Positioning_System)/[Galileo](https://en.wikipedia.org/wiki/Galileo_(satellite_navigation))/[NavIC](https://en.wikipedia.org/wiki/Indian_Regional_Navigation_Satellite_System)). When a vehicle enters a long underground tunnel/underpass, multi-level parking, dense forested highway or deep urban canyon, GNSS drops entirely. GNSS is weak and vulnerable to structural blockage and unintentional EM jamming. Apps freeze, jump erratically, or miss exits.
>
> Systems must then rely on self-contained Inertial Navigation Systems (INS) from IMUs (accelerometer + gyroscope) via dead reckoning and switch back to GNSS-aided INS after blackout. Low-cost tactical/MEMS IMUs suffer inherent biases, deterministic errors, thermo-mechanical noise. High-end cars have factory wheel-connected INS; the vast majority of Indian vehicles — commercial trucks, older cars, **millions of 2-wheelers (motorcycles/scooters)** — rely solely on the driver's smartphone mounted on dashboard/holder.
>
> Using smartphone MEMS IMU for dead reckoning during GNSS blackout is highly challenging. The phone is subjected to severe chassis vibrations, engine harmonics, sudden braking, potholes. Without OBD-II speedometer feed, calculating distance/velocity from consumer-grade sensors causes **exponential error accumulation — drift within seconds**.

### Description — Goal (verbatim)

> Develop a **lightweight, edge-deployable software engine and mobile application** that transforms a standalone smartphone into an **Intelligent Dead Reckoning (IDR) system with GNSS Fusion**.

**Hybrid workflow mandated by ISRO:**

1.  **Model Training (Cloud/Desktop, a priori):** Train AI-ML on smartphone-mounted vehicle datasets **or** open datasets (e.g. [`IO-VNBD`](https://github.com/onyekpeu/IO-VNBD)) and bring trained models (+ offline [OSM](https://www.openstreetmap.org) map DB) to SIH finale. Must include **preliminary AI model + position plot inferred from IO-VNBD subset in the proposal** — screening will use *more datasets* for evaluation.
2.  **On-Device Execution (Smartphone, finale):** Exported lightweight model receives **live** `accelerometer, gyroscope, magnetometer/compass + GNSS if available`, removes noise/bias, predicts corrections, performs [map-matching](https://en.wikipedia.org/wiki/Map_matching), outputs continuous position via AI for both dead reckoning and GNSS+INS fusion.

### Expected Solution — Six Mandatory Capabilities

| # | Capability | What Judges Will Test |
|---|------------|----------------------|
| 1 | **In-Vehicle Alignment & Calibration Engine** | Auto-estimate phone `pitch, roll, yaw` relative to vehicle driving direction, whether dashboard-mounted or holder — without manual calibration. |
| 2 | **AI Speed & Vibration Filter** | Deep/statistical model running **locally** that filters high-frequency pothole/road noise and **directly estimates forward velocity** from IMU. |
| 3 | **Advanced Map-Matching & Kinematic Constraints** | Framework (AI or [`UKF`](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) + [HMM Map Matching](https://en.wikipedia.org/wiki/Hidden_Markov_model)) that **binds position to [OSM](https://www.openstreetmap.org) road network + applies [Non-Holonomic Constraints (NHC)](https://en.wikipedia.org/wiki/Nonholonomic_system)** — car cannot slide sideways/fly — to snap drifting path onto road grid. |
| 4 | **GNSS+INS Fusion Engine** | Innovative **AI-based sensor fusion** (not just [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter)) that combines GNSS & IMU, eliminates drift, provides accurate position + velocity at higher rate. |
| 5 | **Seamless GNSS Deficit Handler** | Instant transition `GNSS-aided INS <-> Dead Reckoning` **within milliseconds** of blackout and vice-versa, no jump. |
| 6 | **Real-time Navigation Interface** | Mobile app UI with **smooth, uninterrupted vehicle icon** — not freezing. |

> **Generality requirement:** Algorithms/models must work with **any external IMU**, not just smartphone — edge-deployable software engine must support [FOG](https://en.wikipedia.org/wiki/Fibre-optic_gyroscope)-based IMU at 200Hz as well.

---

## 2. Why This Is "Impossible" — Core Physics Challenge

| Factor | Reality on Low-Cost MEMS (e.g. [BMI160](https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi160/), [ICM-42605](https://invensense.tdk.com/products/motion-tracking/6-axis/icm-42605/) in phones) |
|--------|-------------------------------------------------------------|
| **Bias instability** | `10–30 °/hr` gyro bias drift; `0.05–0.1 m/s²` accel bias. Double integration: `error ∝ t³`. See [MEMS bias explained — VectorNav](https://www.vectornav.com/resources/inertial-navigation-primer/theory-of-operation/theory-gyrocompassing) |
| **Noise density** | `0.005 °/s/√Hz` gyro, `100 µg/√Hz` accel — dominates at 100–200Hz sampling. |
| **Vibration** | 2-wheeler chassis: `2–15 g` spikes on potholes, `20–80 Hz` engine harmonics — indistinguishable from maneuver acceleration without AI. |
| **Without velocity reference** | Traditional INS needs wheel tick or OBD-II speed. Phone-only: no absolute velocity. Naive integration diverges `>100m in 30s` (see [El-Sheimy et al. 2022 — Smartphone GNSS outage analysis](https://www.mdpi.com/1424-8220/22/19/7548) — 8s outage → 10m, 45s → 1200m drift). |
| **Why high-end cars survive** | Have wheel encoders + tactical IMU + yaw-rate sensor + map. 2-wheeler/truck with phone has **none** of that. |

**Classical solution fails:**  Strapdown mechanization `v = ∫(R·a - g) dt`, `p = ∫v dt` accumulates `~98.9m RMSE / 60s` pure inertial on MEMS ([PMC `Sensors 19:1618` Fig.11](https://pmc.ncbi.nlm.nih.gov/articles/PMC6480342/figure/sensors-19-01618-g011/)). ISRO asks to *learn* to cancel this with AI.

---

## 3. Research Landscape — What SOTA Has (and Has Not) Solved

### 3.1 Timeline of Neural Inertial Navigation (2017 → 2025)

| Year | Method | Key Idea | Result | Limitation for SIH26168 |
|------|--------|----------|--------|-------------------------|
| 2017 | **[RIDI](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_ECCV_2018_paper.pdf)** (Yan et al., [ECCV 2018](https://eccv2018.org/)) | Learn velocity correction + linear least squares, use support human gait patterns | Pedestrian DR baseline | Needs pedestrian periodic motion, fails in vehicle straight-driving |
| 2018 | **[IONet](https://arxiv.org/abs/1711.06305)** (Chen et al., [AAAI 2018](https://aaai.org/conference/aaai/aaai-18/)) | [Bi-LSTM](https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks) regress velocity magnitude + heading rate | First end-to-end DR | 2D only, no 3D attitude |
| 2019 | **[RoNIN](https://arxiv.org/abs/1905.12853)** (Herath et al., [ICRA 2020](https://www.icra2020.org/), [SFU](https://www.sfu.ca/)) | **ResNet + LSTM/TCN** regress velocity vector from 200-sample IMU window, gravity-aligned frame. 42.7h, 100 subjects, 3 architectures. | SOTA pedestrian — controls drift to `<5%` distance. [`Code: github.com/Sachini/ronin`](https://github.com/Sachini/ronin) · [`Dataset: ronin.cs.sfu.ca`](https://ronin.cs.sfu.ca/) | Pedestrian-focused periodic acceleration; vehicle constant velocity violates assumption |
| 2020 | **[TLIO](https://arxiv.org/abs/2007.01867)** (Liu et al., [RA-L 2020](https://www.ieee-ras.org/publications/ra-l), [Facebook Reality Labs](https://tech.facebook.com/reality-labs/)/[UPenn](https://www.upenn.edu/)) | **Tightly-coupled [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter)**: Network regresses 3D displacement + uncertainty, fused as EKF measurement. | **27% yaw, 33% position drift reduction vs RoNIN** on pedestrian. Full 3D. [`Project: cathias.github.io/TLIO/`](https://cathias.github.io/TLIO/) | Computationally heavy, unsuitable for mobile directly — inspires hybrid fusion. |
| 2019 | **[AI-IMU Dead Reckoning](https://arxiv.org/abs/1904.06064)** (Brossard et al., [T-IV 2020](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6702522)) | [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) predicts lateral/vertical velocity ≈0 ([NHC](https://en.wikipedia.org/wiki/Nonholonomic_system)) + uncertainty, EKF with 2 pseudo-measurements | Vehicle NHC constraint; [`RINS-W` wheeled variant](https://arxiv.org/abs/1904.01120) | Assumes car NHC — 2-wheeler tilts/banks violates it |
| 2021 | **[RIO](https://arxiv.org/abs/2103.12375)** / **[CTIN](https://arxiv.org/abs/2109.05423)** (Cao et al., [CVPR 2022](https://cvpr2022.thecvf.com/); Rao et al.) | Rotation-equivariance + [Transformer](https://arxiv.org/abs/1706.03762) contextual attention | Improves orientation robustness | Pedestrian again |
| 2024 | **[LLIO](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.16408383.v1)** ([TechRxiv 2024](https://www.techrxiv.org/)) | Lightweight TLIO variant for mobile | -64% params vs TLIO | Still pedestrian |
| 2025 | **[AVNet + DMDVDR](https://doi.org/10.1186/s43020-025-00168-7)** (Qian et al., [*Satellite Navigation*, Jun 2025](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-025-00168-7), [Wuhan University](https://en.whu.edu.cn/)/[Chongqing University](https://www.cqu.edu.cn/)) | **CNN+GRU AVNet** estimates **attitude + velocity** from phone IMU → **[Invariant EKF (InEKF)](https://arxiv.org/abs/1709.03549)** with data-driven NHC/ODO/ATT. Tested on **parking lot + 578m real tunnel ([Google Smartphone Decimeter Challenge (GSDC)](https://www.kaggle.com/c/google-smartphone-decimeter-challenge))** | **0.4% horizontal error in parking, 0.64% drift after 578m tunnel (60s)** — **directly smartphone vehicle, same as ISRO ask!** Closest SOTA to target. See [EurekAlert summary 2025-06-30](https://www.eurekalert.org/news-releases/1089377) | Needs InEKF + adapter tuning |
| 2025 | **[MoE Cycling Odometry](https://arxiv.org/abs/2510.17604)** ([Arxiv 2510.17604](https://arxiv.org/abs/2510.17604)) | [Mixture-of-Experts (MoE)](https://arxiv.org/abs/1701.06538) for 3D DR on bicycle | -64.7% params, -81.8% flops vs LLIO | Cycling-specific, validates MoE efficiency for 2-wheelers — relevant for bike DR |

### 3.2 Recent GNSS/INS Fusion Enhancements (2024–2026)

*   **[Neural-Kalman GNSS/INS](https://doi.org/10.1109/JSEN.2024.3383721)** (Du et al., [IEEE Sensors J. 2024](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7361)): [DNN](https://en.wikipedia.org/wiki/Deep_learning) predicts pseudo-GNSS increments from IMU during outage, fed as [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter) update — similar to required *AI Speed Filter* but needs quality check.
*   **[MGTR + TAMS](https://www.mdpi.com/2227-7390/14/13/2423)** ([MDPI Mathematics 2026, Vol 14](https://www.mdpi.com/journal/mathematics)): Transformer `MGTR (Motion-Guided)` + `TAMS` for **short GNSS outage compensation** on low-cost vehicular MEMS. Key insight: **First ~10s is input-bound, after ~10s bias integration dominates — need tail-aware readout.** Proposes lightweight fusion for `10–30s` urban outages — exactly ISRO's 1km @ 60kmph = 60s tunnel scenario.
*   **[Denoising + Uncertainty Propagation (AirIMU)](https://arxiv.org/abs/2310.04874)** (Qiu et al., [Arxiv 2310.04874](https://arxiv.org/abs/2310.04874), [CMU RI](https://www.ri.cmu.edu/)): Learns high-rate uncertainty propagation via differentiable covariance — improves [TLIO](https://cathias.github.io/TLIO/)/[EuRoC](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)+[KITTI](https://www.cvlibs.net/datasets/kitti/).

### 3.3 What Still Makes SIH26168 Hard

*   **No method achieves <10% drift over 1km pure inertial on MEMS phones without map.** Best pedestrian [RoNIN](https://ronin.cs.sfu.ca/) ~3–8% in periodic motion; vehicle constant speed lacks periodic cues; [AVNet/DMDVDR](https://doi.org/10.1186/s43020-025-00168-7) reached 0.64% *with InEKF + NHC + map constraints* in taxi, not yet on vibrating 2-wheeler at 60kmph with potholes.
*   **Phone orientation arbitrary:** Dashboard vs holder vs pocket — needs online alignment; [RoNIN](https://github.com/Sachini/ronin)'s gravity-aligned trick helps but vehicle pitch/roll dynamic.
*   **Vibration vs motion separation:** Need AI to distinguish pothole spike vs braking — classical low-pass fails.

**Conclusion:** Feasible to **beat 10% benchmark** with **[AVNet](https://doi.org/10.1186/s43020-025-00168-7)/[RoNIN](https://arxiv.org/abs/1905.12853)-style velocity regression + [InEKF](https://arxiv.org/abs/1709.03549) + [NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) + [HMM map matching](https://en.wikipedia.org/wiki/Hidden_Markov_model)** — but requires careful fusion, not naive double integration.

---

## 4. Official Dataset Deep-Dive — IO-VNBD

**Source:** [onyekpeu/IO-VNBD](https://github.com/onyekpeu/IO-VNBD) — *Inertial Odometry Vehicle Navigation Benchmark Dataset* — first large-scale public vehicle inertial benchmark.

| Property | Value |
|----------|-------|
| **Capture** | Research vehicle on public roads **UK, Nigeria, France** — ego-motion sensors + smartphone IMU @ 10Hz |
| **Sensors** | [GPS](https://en.wikipedia.org/wiki/Global_Positioning_System) receiver + research-grade [INS](https://en.wikipedia.org/wiki/Inertial_navigation_system) + wheel-speed sensors + smartphone GPS+IMU ([Android](https://www.android.com/), 10Hz) — *exactly the phone vs reference setup ISRO expects* |
| **Coverage** | **Vehicle:** ~40h / 1,300km · **Smartphone:** ~58h / 4,400km |
| **Scenarios** | Traffic, roundabouts, hard-braking, country roads, motorways, varying driving patterns |
| **Files** | `Synchronised V and S datasets` + `Unsynchronised V and S Dataset` [zips](https://github.com/onyekpeu/IO-VNBD) |
| **Use for SIH** | **Mandatory subset for proposal:** Train AI, infer position plot on IO-VNBD subset, include in proposal. Finale will use *more hidden datasets* for evaluation. |
| **License** | Public for benchmarking — satisfies ISRO's "bring trained models + [OSM](https://www.openstreetmap.org) map" rule |

**Complementary datasets for pre-training:**

*   **[RoNIN](https://ronin.cs.sfu.ca/)** (42.7h, 100 subjects, 3 devices, umbrella IMU) — `ronin.cs.sfu.ca` — pedestrian but good for backbone pre-training.
*   **[RIDI](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_ECCV_2018_paper.pdf)**, **[OxIOD](https://arxiv.org/abs/1803.03502)** ([Dataset for Deep Inertial Odometry](https://arxiv.org/abs/1803.03502), Chen et al. 2018), [TLIO pedestrian](https://cathias.github.io/TLIO/) — for inertial augmentation.
*   **[EuRoC MAV](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)**, **[KITTI](https://www.cvlibs.net/datasets/kitti/)**, [GSDC (Google Smartphone Decimeter Challenge)](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) — vehicle + tunnel real data; [AVNet](https://doi.org/10.1186/s43020-025-00168-7) validated on GSDC tunnel `578m`.
*   **Own collection (MANDATORY):** 2–3h Indian road with phone on bike holder + dashboard, underpass, parking. Ground truth = GPS before/after outage. This is your differentiator — Nigerian/UK roads ≠ Indian potholes.

---

## 5. Performance Benchmarks (ISRO Judging Criteria)

| Mode | Requirement | Example Test |
|------|-------------|--------------|
| **Dead Reckoning (GNSS denied)** | **Drift <10% of distance** | `<5m drift over 50m in <1 min` OR `<100m drift over 1km @ 60kmph in tunnel/underground metro/simulated` — **smartphone IMU.** Pure inertial baseline diverges `~100–400m / 60s` — must beat by 10×. |
| **GNSS+INS Fusion** | **Position update 10Hz on phone, ~200Hz on [FOG](https://en.wikipedia.org/wiki/Fibre-optic_gyroscope) edge engine** | Continuous output, no jump at handover, milliseconds latency. |
| **Functional** | All 6 capabilities must work live at finale with exported [TFLite](https://www.tensorflow.org/lite)/[ONNX](https://onnx.ai/) model | No cloud inference at finale — edge only. |

**Judging extras (inferred from ISRO [Bharatiya Antariksh Hackathon](https://www.isro.gov.in/BharatiyaAntarikshHackathon.html)):** Seamlessness video, map-matching adherence, yaw stability, comparison plot (your AI vs naive INS vs GPS ground truth), alignment robustness (rotate phone 90° during drive).

---

## 6. Proposed Architecture — Cloud Training / On-Device Inference Hybrid

> **Full C1/C2 + Component diagrams + Data-flow + Deployment:** See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — Mermaid graphs, interfaces, budgets, and alternatives.

```
┌──────────────────────── OFFLINE (Cloud/Desktop) — Training ──────────────────────┐
│                                                                                 │
│  [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) (UK/NG/FR) + [RoNIN](https://ronin.cs.sfu.ca/) + [GSDC](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) + Own Indian Drive (phone) │
│       │                                                                       │
│       ▼                                                                       │
│  Preprocess: 100Hz resample, gravity-aligned frame, bias augmentation         │
│       │                                                                       │
│       ▼                                                                       │
│  [AVNet](https://doi.org/10.1186/s43020-025-00168-7) / [RoNIN-ResNet](https://github.com/Sachini/ronin) ([CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)+[GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit)) ──► velocity + attitude + uncertainty     │
│  Trained with L2 + [NLL](https://en.wikipedia.org/wiki/Likelihood_function) uncertainty loss, window 200 (2s @100Hz)              │
│       │  export [TFLite](https://www.tensorflow.org/lite) FP16 / [ONNX](https://onnx.ai/) opset 17                                    │
│       └──────────────┬────────────────────────────────────────────────────────┘
│                      │ [OSM](https://www.openstreetmap.org) [PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) download (city of finale)
┌──────────────────────▼────────────── ON-DEVICE (Smartphone / Edge Box) ─────────┐
│                                                                                 │
│  Sensors: [Accel](https://en.wikipedia.org/wiki/Accelerometer) 100Hz + [Gyro](https://en.wikipedia.org/wiki/Gyroscope) 100Hz + [Mag](https://en.wikipedia.org/wiki/Magnetometer) 50Hz + [Baro](https://en.wikipedia.org/wiki/Barometer) 10Hz + GNSS 1Hz ([NMEA](https://en.wikipedia.org/wiki/NMEA_0183))   │
│       │                                                                       │
│       ▼                                                                       │
│  ┌─────────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   │
│  │ Alignment Engine    │──►│ AI Speed & Vib Filter│──►│ [InEKF](https://arxiv.org/abs/1709.03549) / [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter)      │   │
│  │ Est. R_phone2vehicle│   │ [AVNet](https://doi.org/10.1186/s43020-025-00168-7) [TFLite](https://www.tensorflow.org/lite) infer   │   │ State: p,v,q,    │   │
│  │ ([Madgwick](https://en.wikipedia.org/wiki/Madgwick_filter) + [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis))    │   │ v_fwd + σ + [NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) pseudo│   │ bias_a, bias_g  │   │
│  └─────────────────────┘   └──────────────────────┘   │ Fusion 10Hz      │   │
│       ▲                         │                   │ + [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model) [MapMatch](https://en.wikipedia.org/wiki/Map_matching)   │   │
│       │                     velocity                 │ + GNSS quality   │   │
│       │                         │                   │   gate ([HDOP](https://en.wikipedia.org/wiki/Dilution_of_precision))    │   │
│  Mag + Gravity             bias correction           └────────┬─────────┘   │
│                                                              │            │
│  ┌─────────────────────┐   ┌──────────────────────┐   ┌──────▼───────┐   │
│  │ Seamless Handler    │◄──│ [OSM](https://www.openstreetmap.org) road graph       │◄──│ Pose @10Hz   │   │
│  │ GNSS loss <200ms    │   │ Snap to edge,        │   │ p,v, yaw     │   │
│  │ switch logic        │   │ [NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) snap             │   │              │   │
│  └─────────────────────┘   └──────────────────────┘   └──────┬───────┘   │
│                                                              │            │
│                                                     ┌────────▼────────┐  │
│                                                     │ UI: [MapLibre](https://maplibre.org/)/  │  │
│                                                     │ [Leaflet](https://leafletjs.com/) smooth  │  │
│                                                     │ vehicle icon    │  │
│                                                     └─────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key design choice — Pure Learned Velocity ([RoNIN](https://ronin.cs.sfu.ca/)/[AVNet](https://doi.org/10.1186/s43020-025-00168-7)) vs Hybrid [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter)+AI Bias:**
*   **For 36h hackathon: Pure Learned Velocity + [InEKF](https://arxiv.org/abs/1709.03549) fusion (Recommended)** — regress `v_fwd` directly, integrate with gyro-derived orientation inside [InEKF](https://arxiv.org/abs/1709.03549); [NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) as pseudo-measurement (`v_lateral≈0, v_vertical≈0` with learned σ). Proven by [AVNet/DMDVDR](https://doi.org/10.1186/s43020-025-00168-7) to hit 0.4–0.64% error.
*   Alternative: Learn bias `b_a, b_g` and correct mechanization — more physics-exact but needs precise ground truth bias labels (harder).

---

## 7. Module Specifications

### 7.1 In-Vehicle Alignment & Calibration Engine
*   **Input:** 2s window of `acc, gyro, mag, GNSS course-over-ground` when `speed>15kmph` and low vibration.
*   **Method:** Gravity vector from accelerometer low-pass → `roll,pitch`. Yaw from `mag + GNSS velocity vector` via [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) on acceleration (forward axis = max variance during acceleration). [Madgwick](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)/[Mahony filter](https://arxiv.org/abs/1711.02508) for online refinement.
*   **Output:** `R_pv` rotation phone→vehicle, updated every 5s. Must handle holder reorientation (detect sudden `R` change >15° → re-calibrate).
*   **Test:** Rotate phone 90° mid-drive — trajectory must not jump.

### 7.2 AI Speed & Vibration Filter (AVNet-lite)
*   **Architecture:** `Input 200×6 (acc 3 + gyro 3, gravity-aligned) → 1D-[CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network) (6→32→64, kernel 9, dilated) → [GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) 2×64 → FC → v_x (forward) + σ`  — 150k params, FP16 [TFLite](https://www.tensorflow.org/lite) <1.2MB, <8ms on Snapdragon 778.
*   **Loss:** `L = ||v_pred - v_gt||² + λ·[NLL](https://en.wikipedia.org/wiki/Likelihood_function)(v_pred, σ)` — learns uncertainty for fusion gate.
*   **Vib filter:** Training augmentation — inject pothole spikes from own dataset (2–15g, 20–80Hz) + engine harmonics; model learns to ignore.
*   **Inference:** Sliding window stride 10 (10Hz output). No [OBD-II](https://en.wikipedia.org/wiki/On-board_diagnostics) — pure [IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit).

### 7.3 Map-Matching & Kinematic Constraints
*   **[OSM](https://www.openstreetmap.org) preprocessing:** [`osmium`](https://osmcode.org/osmium-tool/) → extract `highway=*` edges, build [R-tree](https://en.wikipedia.org/wiki/R-tree) for snap.
*   **[HMM Map Matching](https://en.wikipedia.org/wiki/Hidden_Markov_model):** States = nearby road edges within 50m, emission `p = N(dist, σ=15m)`, transition `p = exp(-|heading_diff|/30°)` — [Viterbi](https://en.wikipedia.org/wiki/Viterbi_algorithm) per epoch.
*   **[NHC](https://en.wikipedia.org/wiki/Nonholonomic_system):** Pseudo-measurement `v_y≈0, v_z≈0` in vehicle frame with `σ = σ_NN + motion-dependent` (high during turn, low straight). Applied as [EKF](https://en.wikipedia.org/wiki/Extended_Kalman_filter) update.
*   **Snap:** Clamp fused position to [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model)-matched edge when GNSS denied >5s.

### 7.4 GNSS+INS Fusion Engine ([InEKF](https://arxiv.org/abs/1709.03549))
*   **State:** `p(3), v(3), q(4), b_a(3), b_g(3)` — 16-dim.
*   **Propagation:** Strapdown with bias-corrected IMU at 100Hz.
*   **Update 1 (when GNSS good):** `z = p_gnss - p_pred` with `R = [HDOP](https://en.wikipedia.org/wiki/Dilution_of_precision)²·σ_gnss`. GNSS quality gate: `HDOP<2.0 && satellites≥6 && speed accuracy <1.5m/s` else discard.
*   **Update 2 (always):** AI velocity `v_ai` from [AVNet](https://doi.org/10.1186/s43020-025-00168-7) + uncertainty `σ_ai` → `z = v_ai - v_pred`, `R = σ_ai²`.
*   **Update 3 ([NHC](https://en.wikipedia.org/wiki/Nonholonomic_system)):** `v_y, v_z` constraints.
*   **Output:** 10Hz pose. On [FOG](https://en.wikipedia.org/wiki/Fibre-optic_gyroscope) edge, run at 200Hz with same model.

### 7.5 Seamless GNSS Deficit Handler
*   **Detection:** `GNSS loss` if no valid fix for `300ms` or `HDOP>5` for 3 consecutive epochs.
*   **Transition:** Freeze last GNSS bias estimate, switch to AI+[NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) updates, ramp `R_gnss → ∞` over `200ms` — prevents jump. On reacquisition, do **soft reset**: `p = α·p_gnss + (1-α)·p_pred` with `α` ramp `0→1` over 1s.
*   **Latency:** <200ms.

### 7.6 Real-time Navigation Interface
*   **Stack:** [Kotlin](https://kotlinlang.org/) + [`SensorManager`](https://developer.android.com/reference/android/hardware/SensorManager) (100Hz) + [`FusedLocationProvider`](https://developers.google.com/android/reference/com/google/android/gms/location/FusedLocationProviderClient) (1Hz) + [MapLibre GL](https://maplibre.org/) offline [OSM](https://www.openstreetmap.org) tiles.
*   **Display:** Vehicle arrow interpolated at 30fps, confidence halo (`σ`), mode badge `GNSS / INS / Fused`, seamless pan.

---

## 8. Tech Stack

| Layer | Choice | Why | Links |
|-------|--------|-----|-------|
| **Training** | [Python 3.10](https://www.python.org/), [PyTorch 2.4](https://pytorch.org/), [`pytorch-lightning`](https://lightning.ai/docs/pytorch/stable/), [`onnx==1.16`](https://onnx.ai/) | Standard for [RoNIN](https://github.com/Sachini/ronin)/[AVNet](https://doi.org/10.1186/s43020-025-00168-7) replication | [PyTorch](https://pytorch.org/) · [ONNX](https://onnx.ai/) |
| **Data** | [`numpy`](https://numpy.org/), [`scipy`](https://scipy.org/), [`pandas`](https://pandas.pydata.org/), [`xarray`](https://xarray.pydata.org/), [`osmium`](https://osmcode.org/osmium-tool/), [`geopandas`](https://geopandas.org/) | [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) CSV + [OSM](https://www.openstreetmap.org) [PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) parsing | [OSM Wiki PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) |
| **Model export** | [`ONNX` opset 17](https://github.com/onnx/onnx) → [TFLite FP16](https://www.tensorflow.org/lite) | ISRO requires edge deployable; [TFLite](https://www.tensorflow.org/lite) runs on phone [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks) | [TensorFlow Lite](https://www.tensorflow.org/lite) |
| **Mobile** | [Kotlin](https://kotlinlang.org/), [Android Studio Hedgehog](https://developer.android.com/studio), [`SensorManager`](https://developer.android.com/reference/android/hardware/SensorManager), [TFLite 2.14](https://www.tensorflow.org/lite/guide), [MapLibre](https://maplibre.org/), [OSMDroid](https://github.com/osmdroid/osmdroid) offline | Direct high-rate [IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit), <20ms inference | [Android Sensors](https://developer.android.com/develop/sensors-and-location/sensors-overview) |
| **Fusion** | Custom [InEKF](https://arxiv.org/abs/1709.03549) in [Kotlin](https://kotlinlang.org/) (100Hz) + [`Eigen`](https://eigen.tuxfamily.org/) via [NDK](https://developer.android.com/ndk) optional | No cloud | [Eigen Docs](https://eigen.tuxfamily.org/dox/) |
| **Dashboard** | (Optional) [FastAPI](https://fastapi.tiangolo.com/) + [Leaflet](https://leafletjs.com/) for demo replay | Side-by-side plot for judges | [FastAPI](https://fastapi.tiangolo.com/) |
| **Hardware test** | 1× [Android](https://www.android.com/) phone ([BMI160](https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi160/)/[LSM6DSM](https://www.st.com/en/mems-and-sensors/lsm6dsm.html)), phone holder, bike/car, power bank | Real Indian road vibration | [LSM6DSM Datasheet](https://www.st.com/en/mems-and-sensors/lsm6dsm.html) |

---

## 9. Implementation Roadmap — From Zero to SIH Finale

### Pre-Hackathon (Do NOW — 2 weeks before 20 Sep)

| Day | Milestone | Deliverable |
|-----|-----------|-------------|
| **D0–3** | Toolchain live + [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) download | `python/download_iovnbd.py` works; visualize `Synchronised V and S` trajectories on [OSM](https://www.openstreetmap.org) |
| **D4–7** | [Android](https://developer.android.com/) logger app v0.1 | Logs `timestamp, acc(3), gyro(3), mag(3), baro, gnss(lat,lon,hdop,sats)` to CSV @100Hz; test 15 min drive |
| **D7–10** | Own Indian dataset collection | 2–3h total: urban, underpass, basement parking, potholes. Ground truth = [GNSS](https://en.wikipedia.org/wiki/Satellite_navigation) before/after outage. **Critical differentiator.** |
| **D10–14** | Baseline [AVNet](https://doi.org/10.1186/s43020-025-00168-7)/[RoNIN](https://github.com/Sachini/ronin) training | Train ResNet velocity regressor on [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) + own, export [TFLite](https://www.tensorflow.org/lite), evaluate <10% drift on 50m/1km simulated outage |

### 36-Hour Hackathon (Finale)

| Hour | Task |
|------|------|
| 0–4 | Preprocess + window generation (200 window, 10 stride), gravity alignment |
| 4–10 | Fine-tune [AVNet-lite](https://doi.org/10.1186/s43020-025-00168-7) on full data, quantize [TFLite](https://www.tensorflow.org/lite), validate [TFLite](https://www.tensorflow.org/lite) vs [PyTorch](https://pytorch.org/) error <2% |
| 10–16 | Implement [InEKF](https://arxiv.org/abs/1709.03549) + alignment engine + [NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) in [Kotlin](https://kotlinlang.org/), unit test with [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) replay |
| 16–22 | [HMM map matching](https://en.wikipedia.org/wiki/Hidden_Markov_model) + [OSM](https://www.openstreetmap.org) offline tiles + handover logic |
| 22–28 | Field test: real underpass/tunnel drive, log metrics, tune `σ` adapter |
| 28–34 | UI polish + seamless video + error plot (AI vs naive vs ground truth) |
| 34–36 | PPT ([SIH template](https://sih.gov.in/sih2026PS)) + architecture diagram + demo rehearsal |

### Deliverables for SIH Submission

*   `app-release.apk` (IDR app) + `edge_engine.so` ([FOG](https://en.wikipedia.org/wiki/Fibre-optic_gyroscope) 200Hz variant stub for demo)
*   `model.tflite` (FP16, <2MB) + `scaler.json`
*   Offline [OSM PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) + tiles for finale city
*   Video: side-by-side `Google Maps freeze` vs `your smooth trace` in tunnel
*   PPT: architecture, [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) preliminary results plot, field test metrics
*   GitHub README with one-click setup

---

## 10. Evaluation & Validation Protocol

### Offline (Before Finale)

*   **[IO-VNBD](https://github.com/onyekpeu/IO-VNBD) replay:** Simulate GNSS outage by masking GPS for 60s segments (50m and 1km). Metric: `ATE = mean ||p_pred - p_gt||`, `Drift% = ATE / distance`. Target: `<10%` (ISRO pass), stretch `<3%` to impress.
*   **Cross-dataset:** Validate on [RoNIN](https://ronin.cs.sfu.ca/) + [GSDC](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) tunnel 578m — expect `~1–2%` before map.

### Live (At Finale)

*   **Controlled outage:** Drive `~1km` with known start/end GNSS fix; ISRO will cut GNSS ([faraday](https://en.wikipedia.org/wiki/Faraday_cage)/tunnel sim). Log `p_pred` @10Hz.
*   **Metrics:** `Final drift (m)`, `max drift`, `update rate`, `handover latency`, `map adherence %`.
*   **Comparison:** Always show `naive double integration` divergence vs `AI IDR` — proves learning.

---

## 11. Risk Register & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| [ONNX](https://onnx.ai/)/[TFLite](https://www.tensorflow.org/lite) export mismatch | High | Validate `[TFLite](https://www.tensorflow.org/lite) vs [PyTorch](https://pytorch.org/)` output diff `<1e-3` on 1k windows week 2 day 1; keep [PyTorch](https://pytorch.org/) fallback on laptop. |
| Phone holder vibration too harsh | High | Use dashboard mount + holder with dampening; train with own pothole spikes; add `AI vibration filter` layer. |
| No tunnel access for test | High | **Simulate outage programmatically** — mask GNSS, still valid per PS ("simulated environments"). Film underpass/flyover. |
| Alignment fails after phone rotate | Medium | Re-calibrate on `gyro energy < threshold && speed >15kmph` for 2s. |
| [InEKF](https://arxiv.org/abs/1709.03549) diverges without GNSS | Medium | Clamp `b_a,b_g` random walk, use learned `σ_ai` to down-weight bad AI velocity at high speed. |
| [Map matching](https://en.wikipedia.org/wiki/Map_matching) snaps to wrong road | Medium | [HMM](https://en.wikipedia.org/wiki/Hidden_Markov_model) with heading, not just distance; keep raw trace as fallback. |
| Team no [Android](https://developer.android.com/) dev | High | Pair 1 member to [`SensorManager` tutorial](https://developer.android.com/develop/sensors-and-location/sensors-overview) week 1; fallback: [Python](https://www.python.org/) laptop with phone USB tethering [IMU](https://en.wikipedia.org/wiki/Inertial_measurement_unit). |

---

## 12. Repository Layout

```
sih26168/
├── README.md                  # this file
├── docs/
│   ├── architecture.md        # detailed diagrams
│   └── stateflow.md           # handover logic
├── python/
│   ├── download_iovnbd.py     # fetch + unzip [IO-VNBD](https://github.com/onyekpeu/IO-VNBD)
│   ├── preprocess.py          # 100Hz resample, gravity-aligned windows
│   ├── train_avnet.py         # [CNN](https://en.wikipedia.org/wiki/Convolutional_neural_network)+[GRU](https://en.wikipedia.org/wiki/Gated_recurrent_unit) training, [NLL](https://en.wikipedia.org/wiki/Likelihood_function) loss
│   ├── export_tflite.py       # [ONNX](https://onnx.ai/) → [TFLite](https://www.tensorflow.org/lite) FP16 + validation
│   └── eval_drift.py          # ATE/drift% on 50m/1km masks
├── android/
│   ├── app/src/main/java/.../IDRService.kt  # Sensor 100Hz, [InEKF](https://arxiv.org/abs/1709.03549), [TFLite](https://www.tensorflow.org/lite)
│   ├── app/src/main/assets/model.tflite
│   └── build.gradle
├── edge_engine/               # C++ [FOG](https://en.wikipedia.org/wiki/Fibre-optic_gyroscope) 200Hz variant (optional)
│   └── infer.cpp
├── maps/
│   └── download_osm.sh        # [osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) [PBF](https://wiki.openstreetmap.org/wiki/PBF_Format) for city
├── scenarios/
│   ├── tunnel_1km.csv         # simulated outage trace
│   └── parking_lot.csv
└── reports/
    ├── technical_report.pdf
    └── demo_video.mp4
```

**Reproducibility rule:** `./python/train_avnet.py && ./python/export_tflite.py && ./android/gradlew assembleRelease` rebuilds everything. No hand-edited model.

---

## 13. Quickstart — Reproduce Baseline in 15 Minutes

```bash
# 1. [Python](https://www.python.org/) env
python3 -m venv venv && source venv/bin/activate
pip install torch numpy scipy pandas onnx tflite-runtime

# 2. Fetch [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) (small subset for quick test)
python python/download_iovnbd.py --subset 1h

# 3. Train tiny [AVNet](https://doi.org/10.1186/s43020-025-00168-7) (5 epochs, CPU-friendly)
python python/train_avnet.py --epochs 5 --window 200 --batch 64

# 4. Export & validate
python python/export_tflite.py --quant fp16
python python/eval_drift.py --mask 60s --plot reports/drift_plot.png
# expect: naive drift ~80m, AI drift ~6m over 50m/60s → <12%
```

[Android](https://developer.android.com/): `cd android && ./gradlew assembleDebug` → install APK, grant `BODY_SENSORS + LOCATION`, mount phone, drive.

---

## 14. References

*   **Official PS:** [SIH26168 `ps_2026/SIH26168.md`](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26168.md) · [sih.gov.in/sih2026PS](https://sih.gov.in/sih2026PS) · [IO-VNBD Dataset](https://github.com/onyekpeu/IO-VNBD) · [SIH 2026 All PS PDF (22 Aug 2026, p.220-222 for SIH26168)](https://sih-2026-problem-statements.shaikrohit187.workers.dev/public/pdfs/SIH_2026_All_PS.pdf) · [SIH 2026 Intelligence Landscape PDF — local copy](/home/ark/Downloads/SIH_2026_Intelligence_Landscape.pdf) — p.5: ISRO 11 PS, judges expect real satellite/geo data pipelines.
*   **[RoNIN](https://ronin.cs.sfu.ca/):** Herath et al., *RoNIN: Robust Neural Inertial Navigation in the Wild*, [ICRA 2020](https://www.icra2020.org/). [`arxiv 1905.12853`](https://arxiv.org/abs/1905.12853) · [`github.com/Sachini/ronin`](https://github.com/Sachini/ronin) · [`ronin.cs.sfu.ca`](https://ronin.cs.sfu.ca/)
*   **[TLIO](https://cathias.github.io/TLIO/):** Liu et al., *TLIO: Tight Learned Inertial Odometry*, [RA-L 2020](https://www.ieee-ras.org/publications/ra-l). [`arxiv 2007.01867`](https://arxiv.org/abs/2007.01867) · [`cathias.github.io/TLIO/`](https://cathias.github.io/TLIO/) · [IEEE Xplore 9134860](https://ieeexplore.ieee.org/document/9134860)
*   **[RIDI](https://openaccess.thecvf.com/content_ECCV_2018/papers/Hang_Yan_RIDI_ECCV_2018_paper.pdf):** Yan et al., *RIDI: Robust IMU Double Integration*, [ECCV 2018](https://eccv2018.org/) · [`arXiv 1712.04150`](https://arxiv.org/abs/1712.04150)
*   **[IONet](https://arxiv.org/abs/1711.06305):** Chen et al., *IONet: Learning to Cure the Curse of Drift in Inertial Odometry*, [AAAI 2018](https://aaai.org/conference/aaai/aaai-18/) · [`arxiv 1711.06305`](https://arxiv.org/abs/1711.06305)
*   **[OxIOD](https://arxiv.org/abs/1803.03502):** Chen et al., *OxIOD: The Dataset for Deep Inertial Odometry*, 2018 · [`arxiv 1803.03502`](https://arxiv.org/abs/1803.03502)
*   **[AVNet / DMDVDR](https://doi.org/10.1186/s43020-025-00168-7):** Qian et al., *AVNet: learning attitude and velocity for vehicular dead reckoning using smartphone by adapting an invariant EKF*, [*Satellite Navigation* 2025](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-025-00168-7) · [`DOI 10.1186/s43020-025-00168-7`](https://doi.org/10.1186/s43020-025-00168-7) — **0.4% parking, 0.64% 578m tunnel on phone only** — closest prior to ISRO ask. See [EurekAlert summary 2025-06-30](https://www.eurekalert.org/news-releases/1089377) · [PDF SpringerOpen](https://satellite-navigation.springeropen.com/articles/10.1186/s43020-025-00168-7)
*   **[Neural-Kalman GNSS/INS](https://doi.org/10.1109/JSEN.2024.3383721):** Du et al., *Neural-Kalman GNSS/INS Navigation for Precision Agriculture*, [IEEE Sensors J. 2024](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7361) · [`DOI 10.1109/JSEN.2024.3383721`](https://doi.org/10.1109/JSEN.2024.3383721)
*   **[MGTR/TAMS Transformer for Short Outage](https://www.mdpi.com/2227-7390/14/13/2423):** *IMU-Sequence-Based GNSS Short Outage Compensation and Hybrid Positioning Strategy*, [Mathematics 2026, Vol 14](https://www.mdpi.com/journal/mathematics) · [`MDPI 2227-7390/14/13/2423`](https://www.mdpi.com/2227-7390/14/13/2423)
*   **[AirIMU](https://arxiv.org/abs/2310.04874):** Qiu et al., *AirIMU: Learning Uncertainty Propagation for Inertial Odometry*, [Arxiv 2310.04874](https://arxiv.org/abs/2310.04874) · [CMU RI](https://www.ri.cmu.edu/)
*   **El-Sheimy Smartphone GNSS Outage Baseline:** [“Smartphone MEMS drift 8s→10m, 45s→1200m” — Sensors 2022](https://www.mdpi.com/1424-8220/22/19/7548) — quantifies baseline to beat.
*   **PMC Sensors 19:1618:** [Steering Angle Assisted Vehicular Navigation Using Portable Devices in GNSS-Denied Environments — Fig.11 pure inertial 98.97m RMSE/60s](https://pmc.ncbi.nlm.nih.gov/articles/PMC6480342/figure/sensors-19-01618-g011/) · [Full PMC6480342](https://pmc.ncbi.nlm.nih.gov/articles/PMC6480342/)
*   **[RINS-W / AI-IMU](https://arxiv.org/abs/1904.06064):** Brossard et al., *AI-IMU Dead-Reckoning*, [T-IV 2020](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6702522) — [NHC](https://en.wikipedia.org/wiki/Nonholonomic_system) pseudo-measurements · [`arXiv 1904.06064`](https://arxiv.org/abs/1904.06064) · [`RINS-W arxiv 1904.01120`](https://arxiv.org/abs/1904.01120)
*   **[RIO](https://arxiv.org/abs/2103.12375)/[CTIN](https://arxiv.org/abs/2109.05423):** [RIO is rotation-equivariance — CVPR 2022](https://arxiv.org/abs/2103.12375) · [CTIN — Transformer](https://arxiv.org/abs/2109.05423)
*   **[LLIO Lightweight](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.16408383.v1):** [TechRxiv 16408383](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.16408383.v1)
*   **[MoE Cycling](https://arxiv.org/abs/2510.17604):** *Learned Inertial Odometry for Cycling Based on MoE*, [Arxiv 2510.17604](https://arxiv.org/abs/2510.17604) — 64.7% params, 81.8% flops reduction, validates efficiency for 2-wheelers.
*   **Additional Tooling Links:** [MapLibre GL](https://maplibre.org/) · [Leaflet](https://leafletjs.com/) · [OSMDroid](https://github.com/osmdroid/osmdroid) · [Osmium Tool](https://osmcode.org/osmium-tool/) · [OSM PBF Format](https://wiki.openstreetmap.org/wiki/PBF_Format) · [Osmosis](https://wiki.openstreetmap.org/wiki/Osmosis) · [TensorFlow Lite](https://www.tensorflow.org/lite) · [ONNX](https://onnx.ai/) · [PyTorch](https://pytorch.org/) · [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) · [Eigen](https://eigen.tuxfamily.org/) · [Android SensorManager](https://developer.android.com/reference/android/hardware/SensorManager) · [FusedLocationProvider](https://developers.google.com/android/reference/com/google/android/gms/location/FusedLocationProviderClient) · [Madgwick AHRS](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/) · [Invariant EKF Paper — Barrau & Bonnabel 2017](https://arxiv.org/abs/1709.03549) · [Kotlin](https://kotlinlang.org/) · [FastAPI](https://fastapi.tiangolo.com/) · [NMEA 0183](https://en.wikipedia.org/wiki/NMEA_0183) · [NavIC](https://en.wikipedia.org/wiki/Indian_Regional_Navigation_Satellite_System) · [GSDC Kaggle](https://www.kaggle.com/c/google-smartphone-decimeter-challenge) · [KITTI Dataset](https://www.cvlibs.net/datasets/kitti/) · [EuRoC MAV Dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) · [BMI160 Datasheet](https://www.bosch-sensortec.com/products/motion-sensors/imus/bmi160/) · [LSM6DSM Datasheet](https://www.st.com/en/mems-and-sensors/lsm6dsm.html) · [ISRO Bharatiya Antariksh Hackathon](https://www.isro.gov.in/BharatiyaAntarikshHackathon.html)

---

**Why this team should pick SIH26168 (over the other 4 impossible PS you listed):**

| PS | Data access | Hardware | 36h feasibility | Wow |
|----|-------------|----------|-----------------|-----|
| **SIH26168 IDR** | **Public [IO-VNBD](https://github.com/onyekpeu/IO-VNBD) + own phone drive = controllable** | Phone only — every team has it | **Medium** — [AVNet](https://doi.org/10.1186/s43020-025-00168-7) pipeline exists, you fine-tune | **High** — fixes daily Maps freeze for 2-wheelers; ISRO [NavIC](https://en.wikipedia.org/wiki/Indian_Regional_Navigation_Satellite_System) story |
| [SIH26054 DRDO Digital Twin](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26054.md) | Confidential engine maps, no public | Needs piston engine rig | Very Low | High but unverifiable |
| [SIH26157 NTRO SOC](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26157.md) | Classified [NCIIPC](https://www.nciipc.gov.in/) logs | None | Low | Medium — synthetic data |
| [SIH26067 MoES 3D Ocean](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26067.md) | Public but TB [NetCDF](https://www.unidata.ucar.edu/software/netcdf/) volumetric | None | Medium — heavy viz | Medium |
| [SIH26153 NTRO Attack Forecast](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26153.md) | Public [CIC-IDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) etc | None | Medium-High | High — but crowded AI |

**Next step:** Create `sih26168` repo, run quickstart, collect 30 min Indian drive today, and add `drift_plot.png` to proposal before 20 Sep. Judges require that plot for screening.
