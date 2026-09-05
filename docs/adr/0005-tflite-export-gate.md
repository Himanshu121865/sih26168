# ADR-005: TFLite FP16 export with ONNX validation gate

## Status
Accepted

## Context
The model trains in PyTorch but must run on-device at <8 ms and <2 MB. Export
pipelines (ONNX → TFLite) are a classic silent-mismatch risk: a graph that
converts but computes differently ships a broken model with green logs.

## Decision
`torch.onnx.export` (opset 17) → `ai-edge-torch`/FP16 → `model.tflite`, with a
hard gate: max abs diff vs PyTorch on val windows must be <1e-3 (FP32) / <1e-2
(FP16), recorded in `reports/tflite_diff.txt`. `export_tflite.py` refuses to
bless a bundle over the threshold.

## Consequences

### Positive
- Mismatches fail loudly before they reach the APK (observed diff 1e-6).
- `scaler.json` ships alongside the model so preprocessing parity is versioned.

### Negative
- Converter toolchain (`ai-edge-torch`) is version-fragile; Colab pins it per run.
- INT8 (Agastya achieved 35 KB) is deferred; FP16 1.7 MB is fine but not minimal.

### Neutral
- `screening_bundle.zip` (model + scaler + drift plot) is the single handoff artifact.

## Alternatives Considered

**ONNX Runtime Mobile**
- Rejected: larger binary, no NNAPI acceleration path as mature as TFLite.

## References
- `python/export_tflite.py`, `reports/tflite_diff.txt`
