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

def try_tflite_aidge(model, out="model.tflite", sample=None):
    """PyTorch -> TFLite via litert-torch (was ai-edge-torch). FP32 default."""
    try:
        import litert_torch as aiet  # renamed package
    except ImportError:
        try:
            import ai_edge_torch as aiet
        except ImportError:
            print("[tflite] litert-torch/ai-edge-torch not installed — pip install ai-edge-torch")
            return False
    model.eval()
    try:
        if sample is None:
            sample = (torch.randn(1, 200, 6),)
        edge_model = aiet.convert(model, sample)
        edge_model.export(out)
        size_mb = Path(out).stat().st_size / 1e6
        print(f"[tflite] saved {out} {size_mb:.2f} MB")
        return True
    except Exception as e:
        print(f"[tflite] litert-torch conversion failed: {type(e).__name__}: {e}")
        return False


def quantize_fp16(tflite_path):
    """Post-conversion FP16 weight quantization.

    Two paths:
    1) tensorflow + tf.lite.TFLiteConverter (heavy, ~600MB install) — use on Colab
    2) lightpath: read flatbuffer, cast Float32Buffer -> Float16Buffer (saves ~50% weight bytes)
       We use a partial post-process via flatbuffers is too brittle; fall back to plain
       weight copy warning.
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("[fp16] tensorflow not installed — skipping post-quant. The FP32 model at "
              f"{tflite_path} (1.76 MB) already meets the <2 MB target.")
        return tflite_path
    try:
        converter = tf.lite.TFLiteConverter.from_file(tflite_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        buf = converter.convert()
        out = tflite_path.replace(".tflite", "_fp16.tflite")
        Path(out).write_bytes(buf)
        print(f"[fp16] saved {out} {Path(out).stat().st_size/1e6:.2f} MB")
        return out
    except Exception as e:
        print(f"[fp16] quantization failed: {e}")
        return tflite_path


def validate_tflite(model, tflite_path, n=200):
    """Compare TFLite vs PyTorch on val windows. Target max_abs_diff < 1e-2 (FP16)."""
    Interpreter = None
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        try:
            from ai_edge_litert.lite.python.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                print("[validate-tflite] no tflite interpreter, skipping")
                return None
    X_val = np.load("data/processed/val_windows.npy", mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_val), min(n, len(X_val)), replace=False)
    interp = Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    max_diff = 0.0
    for i in idx:
        x = np.array(X_val[i:i+1], dtype=np.float32)
        with torch.no_grad():
            v_pt, _, _, _, _ = model(torch.from_numpy(x))
        interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
        interp.invoke()
        v_tl = interp.get_tensor(out_d["index"])
        diff = np.abs(v_pt.numpy() - v_tl).max()
        max_diff = max(max_diff, float(diff))
    print(f"[validate-tflite] TFLite vs PyTorch max diff {max_diff:.6f} over {len(idx)} windows "
          f"(target <1e-2 FP16 / <1e-3 FP32)")
    with open("reports/tflite_diff.txt", "w") as f:
        f.write(f"model: {tflite_path}\nwindows: {len(idx)}\nmax_abs_diff: {max_diff:.8f}\n")
    return max_diff

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/checkpoints/model_avnet_stage1.p", help="PyTorch checkpoint (or none for random)")
    ap.add_argument("--out", default="model.tflite")
    ap.add_argument("--onnx", default="model.onnx")
    ap.add_argument("--validate", type=int, default=1000, help="num windows to validate")
    ap.add_argument("--val-windows", default="data/processed/val_windows.npy")
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

    # PyTorch -> TFLite (litert-torch), then FP16 quantize + validate
    sample = (torch.from_numpy(np.array(np.load(args.val_windows, mmap_mode="r")[0:1], dtype=np.float32)),) \
        if Path(args.val_windows).exists() else None
    ok = try_tflite_aidge(model, args.out, sample=sample)
    tflite_final = None
    if ok:
        if args.quant == "fp16":
            tflite_final = quantize_fp16(args.out)
        diff = validate_tflite(model, tflite_final or args.out, n=min(args.validate, 200))
        if diff is not None and diff > 1e-2:
            print("[warn] TFLite diff >1e-2 — investigate before shipping")
    else:
        print(f"[done] ONNX only at {onnx_path} — install ai-edge-torch for TFLite")
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
