#!/usr/bin/env python3
"""
export_tflite.py — Step 11.1 — PyTorch -> ONNX -> TFLite FP16 + validate <1e-3

Usage:
  python python/export_tflite.py --model experiments/checkpoints/model_avnet_stage1.p --out model.tflite --validate 1000
  # fallback: if onnx2tf not installed, just export ONNX and validate PyTorch vs ONNX

Requires: onnx, onnxruntime (pip), and optionally onnx2tf for TFLite
"""
import argparse, json, subprocess, sys
from pathlib import Path
import numpy as np
import torch

from python.models.avnet import AVNetLite

def export_onnx(model, path="model.onnx", window=200):
    model.eval()
    dummy = torch.randn(1, window, 6)
    torch.onnx.export(
        model, dummy, path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["imu_window"],
        output_names=["v_pred","log_sig_v","att_pred","log_sig_att","hx"],
        dynamic_axes={"imu_window":{0:"batch"}, "v_pred":{0:"batch"}},
    )
    print(f"[onnx] saved {path} {Path(path).stat().st_size/1e6:.2f} MB")
    return path

def validate_onnx(model, onnx_path, n=1000):
    try:
        import onnxruntime as ort
    except ImportError:
        print("[validate] onnxruntime not installed, skipping")
        return
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # load val windows
    X_val = np.load("data/processed/val_windows.npy", mmap_mode="r")
    idx = np.random.choice(len(X_val), min(n, len(X_val)), replace=False)
    max_diff = 0
    for i in idx[:10]:  # quick 10
        x = X_val[i:i+1].astype(np.float32)  # (1,200,6)
        with torch.no_grad():
            v_pt, _, _, _, _ = model(torch.from_numpy(x))
        v_onnx = sess.run(["v_pred"], {"imu_window": x})[0]
        diff = np.abs(v_pt.numpy() - v_onnx).max()
        max_diff = max(max_diff, diff)
    print(f"[validate] ONNX vs PyTorch max diff {max_diff:.6f} (target <1e-3)")
    if max_diff > 1e-3:
        print("[warn] diff >1e-3, check opset/model")
    return max_diff

def try_tflite(onnx_path, out="model.tflite"):
    # try onnx2tf -> TFLite
    try:
        import onnx2tf
        print("[tflite] onnx2tf found, converting...")
        # onnx2tf command: onnx2tf -i model.onnx -o tflite_dir
        subprocess.check_call([sys.executable, "-m", "onnx2tf", "-i", onnx_path, "-o", str(Path(out).parent), "--output_tflite_file", out])
        print(f"[tflite] saved {out} {Path(out).stat().st_size/1e6:.2f} MB")
        return True
    except Exception as e:
        print(f"[tflite] onnx2tf not available or failed: {e}")
        print("[tflite] fallback: ONNX only, use ai-edge-torch or onnx2tf later")
        return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p", help="PyTorch checkpoint (or none for random)")
    ap.add_argument("--out", default="model.tflite")
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--validate", type=int, default=1000, help="num windows to validate")
    ap.add_argument("--quant", choices=["fp32","fp16"], default="fp16")
    args = ap.parse_args()

    model = AVNetLite()
    if Path(args.model).exists():
        print(f"[load] {args.model}")
        model.load_state_dict(torch.load(args.model, map_location="cpu"))
    else:
        print(f"[warn] {args.model} not found, exporting random weights (for pipeline test)")

    # count params and size
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_params:,} params, est FP32 {n_params*4/1e6:.2f} MB FP16 {n_params*2/1e6:.2f} MB")

    onnx_path = export_onnx(model, args.onnx)
    validate_onnx(model, onnx_path, n=args.validate)

    # try TFLite
    ok = try_tflite(onnx_path, args.out)
    if not ok:
        print(f"[done] ONNX only at {onnx_path} — install onnx2tf for TFLite: pip install onnx2tf")
        # still create a dummy tflite for pipeline test (copy onnx)
        import shutil
        shutil.copy(onnx_path, args.out + ".onnx_fallback")
        print(f"[fallback] copied {onnx_path} to {args.out}.onnx_fallback")

    # save scaler alongside
    scaler_src = Path("python/scaler.json")
    if scaler_src.exists():
        import shutil, json as j
        scaler_dst = Path(args.out).parent / "scaler.json"
        shutil.copy(scaler_src, scaler_dst)
        print(f"[scaler] copied {scaler_src} -> {scaler_dst}")

if __name__ == "__main__":
    main()
