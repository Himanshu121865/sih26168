# SIH26168 — Intelligent Dead Reckoning for GNSS-Denied Navigation

**ISRO • Software • SIH26168 • Deadline: 20 Sep 2026**

## What this is

An app that keeps telling you where you are when GPS stops working.

When you drive through a tunnel, an underpass, or into a basement parking lot, GPS drops out. A regular phone's motion sensors are too noisy to fill the gap on their own — position drifts hundreds of meters within a minute.

This project uses the phone's motion sensors plus an AI model to predict how fast you're going, then fuses that with a map so the position error stays under 5% over 1 km. No car hardware, no OBD-II — just the phone.

## How it works

1. Phone motion sensors (accelerometer + gyroscope) stream data at 100 Hz.
2. A small AI model (**AVNetLite**, 460k parameters, <1 MB) reads 2-second windows of sensor data and predicts forward speed.
3. A filter (**InEKF**) combines the AI speed prediction with raw sensor data, using two physics rules:
   - **NHC** — a vehicle doesn't slide sideways or vertically.
   - **ZUPT** — when stopped (detected by sensor variance), speed is exactly zero, which resets accumulated error.
4. Position is snapped to the nearest road on an offline map (**HMM map matching** with OpenStreetMap).
5. The model runs on the phone as a **TFLite** file (~1 MB, ~7 ms per inference).

Covers all 6 required ISRO capabilities: sensor alignment, AI speed filter, map matching, GNSS+INS fusion, seamless GPS-loss handling, and UI.

## Results

Tested with GPS blocked for 60 seconds (simulated tunnel):

| Test segment | Naive (no AI) | Ours | Ours + map |
|--------------|---------------|------|------------|
| 138 m | 16.3 m off (11.8%) | **4.6 m (3.35%)** | **2.8 m (2.01%)** |
| 188 m (validation) | 16.3 m off (8.7%) | **1.87 m (0.99%)** | **1.12 m (0.59%)** |

Both pass the ISRO requirement (<10%) and our own target (<5%).

Training: 50 epochs on a Colab T4 GPU, ~7 minutes, on the [IO-VNBD dataset](https://github.com/onyekpeu/IO-VNBD) (40h vehicle + 58h phone data).

The plot judges need for screening: `reports/drift_plot.png`.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas scipy onnx onnxruntime loguru matplotlib scikit-learn

python python/download_iovnbd.py --subset Sync                # download data
python python/preprocess.py --subset 1h --window 200 --stride 10  # prepare windows

# train (5 epochs demo; use Colab notebook for the full 50-epoch run)
PYTHONPATH=. python python/train_avnet.py --epochs 5 --batch 128 --augment-yaw --lambda-nll 0.1

# evaluate + generate drift_plot.png
PYTHONPATH=. python python/eval_drift.py --model experiments/checkpoints/model_avnet_stage1.p --plot reports/drift_plot.png

# export for the phone app
PYTHONPATH=. python python/export_tflite.py --model experiments/checkpoints/model_avnet_stage1.p --out model.tflite
```

Full 50-epoch GPU training: open `sih26168_colab.ipynb` in Colab (T4 GPU) — takes ~7 minutes and produces `screening.zip`.

## Repository layout

```
sih26168/
├── python/               # all code: training, preprocessing, evaluation, export
│   ├── models/           # AVNetLite model, lean estimator, uncertainty adapter
│   ├── utils/            # ZUPT, lie group math, metrics
│   └── datasets/         # IO-VNBD dataset loader
├── docs/
│   ├── ARCHITECTURE.md   # system design, diagrams, data flow
│   ├── DATA_INSPECTION.md    # dataset analysis
│   └── IMPROVEMENTS_FROM_COMPETITORS.md  # 13 fixes learned from other teams
├── ref/competitors/      # other teams' code, kept for reference only
├── experiments/checkpoints/  # trained model (model_avnet_stage1.p, 1.8 MB)
├── reports/drift_plot.png    # the screening deliverable
└── sih26168_colab.ipynb  # one-click GPU training
```

## References

- [Problem statement SIH26168](https://github.com/vedantchalke36/sih-2026-problem-statements/blob/main/ps_2026/SIH26168.md)
- [AVNet paper](https://doi.org/10.1186/s43020-025-00168-7) — Qian et al., Satellite Navigation 2025 (our baseline; achieved 0.64% in tunnels on cars)
- [IO-VNBD dataset](https://github.com/onyekpeu/IO-VNBD) — Data in Brief 35:106885
- Other teams (for learning): [harsh/dead-reckoning](https://github.com/harshkumarsingh12/dead-reckoning) (TCN+ZUPT), [Agastya](https://github.com/krushnasaruk/Agastya) (0.245% with wheel speed), [sivaraman](https://github.com/sivaraman-tech/AI-ML-based-Intelligent-Dead-Reckoning-System-for-Seamless) (EKF baseline)
- Reference code: `QDeepOdo` + `QAIIMUDeadReckoning` (AVNet/InEKF) under `ref/`
