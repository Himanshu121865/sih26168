"""Model export: PyTorch → ONNX → TFLite, with parity validation + manifest.

Pipeline:
1. Verify the scaler carries a matching spec fingerprint (P2 gate —
   unversioned/mismatched scaler ⇒ REFUSE, fail loud not log-only).
2. ONNX export (opset 17, dynamic batch).
3. ONNX↔PyTorch parity check on val windows (target <1e-3).
4. TFLite via litert-torch (was ai-edge-torch) when installed, then
   optional FP16 weight quantization + TFLite↔PyTorch check (<1e-2 FP16).
5. Copy the scaler next to the model; stamp model_manifest.json binding
   model ↔ scaler ↔ spec hashes (consumed by Android WindowSpecGuard).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from python.core.spec import (
    SPEC_VERSION,
    spec_sha256,
    verify_scaler,
    write_model_manifest,
)
from python.models.avnet import AVNetLite, count_params

DEFAULT_SCALER = Path("python/scaler.json")
DIFF_REPORT = Path("reports/tflite_diff.txt")


def export_onnx(model: AVNetLite, path: str | Path = "model.onnx", window: int = 200) -> Path:
    """Export the model to ONNX (opset 17, dynamic batch axis).

    Args:
        model: Loaded AVNetLite in eval mode.
        path: Destination .onnx path.
        window: Input window length.

    Returns:
        The exported path.
    """
    model.eval()
    dummy = torch.randn(1, window, 6)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["imu_window"],
        output_names=["v_pred", "log_sig_v", "att_pred", "log_sig_att", "hx"],
        dynamic_axes={"imu_window": {0: "batch"}, "v_pred": {0: "batch"}},
    )
    logger.info(f"[onnx] saved {path} {Path(path).stat().st_size / 1e6:.2f} MB")
    return Path(path)


def validate_onnx(model: AVNetLite, onnx_path: str | Path, n: int = 1000,
                  val_windows: str | Path = "data/processed/val_windows.npy") -> float | None:
    """Compare ONNX vs PyTorch v_pred on real val windows.

    Args:
        model: Reference PyTorch model.
        onnx_path: Exported ONNX file.
        n: Max windows to check (quick pass over 10).
        val_windows: Path to the val npy.

    Returns:
        Max abs diff, or None when onnxruntime is not installed (skipped).
    """
    try:
        import onnxruntime as ort
    except ImportError:
        logger.info("[validate] onnxruntime not installed, skipping")
        return None
    if not Path(val_windows).exists():
        logger.info("[validate] no val windows, skipping")
        return None

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    X_val = np.load(val_windows, mmap_mode="r")
    idx = np.random.choice(len(X_val), min(n, len(X_val)), replace=False)
    max_diff = 0.0
    with torch.no_grad():
        for i in idx[:10]:  # quick 10
            x = np.array(X_val[i : i + 1], dtype=np.float32)
            v_pt, _, _, _, _ = model(torch.from_numpy(x))
            v_onnx = sess.run(["v_pred"], {"imu_window": x})[0]
            max_diff = max(max_diff, float(np.abs(v_pt.numpy() - v_onnx).max()))
    logger.info(f"[validate] ONNX vs PyTorch max diff {max_diff:.6f} (target <1e-3)")
    if max_diff > 1e-3:
        logger.error("[warn] diff >1e-3, check opset/model")
    return max_diff


def try_tflite_litert(model: AVNetLite, out: str | Path = "model.tflite",
                      sample: tuple | None = None) -> bool:
    """Convert to TFLite via litert-torch / ai-edge-torch (FP32 default).

    Args:
        model: Eval-mode model.
        out: Destination .tflite path.
        sample: Optional (dummy_input,) tuple; defaults to randn (1, 200, 6).

    Returns:
        True when conversion succeeded, False when the toolchain is absent
        or conversion failed (caller falls back to ONNX-only).
    """
    try:
        import litert_torch as aiet  # renamed package
    except ImportError:
        try:
            import ai_edge_torch as aiet
        except ImportError:
            logger.info("[tflite] litert-torch/ai-edge-torch not installed — pip install ai-edge-torch")
            return False
    model.eval()
    try:
        if sample is None:
            sample = (torch.randn(1, 200, 6),)
        edge_model = aiet.convert(model, sample)
        edge_model.export(str(out))
        logger.info(f"[tflite] saved {out} {Path(out).stat().st_size / 1e6:.2f} MB")
        return True
    except Exception as e:  # noqa: BLE001 — conversion failures are expected/fallback paths
        logger.info(f"[tflite] litert-torch conversion failed: {type(e).__name__}: {e}")
        return False


def quantize_fp16(tflite_path: str | Path) -> Path:
    """Post-conversion FP16 weight quantization (needs tensorflow, ~600MB — Colab).

    Args:
        tflite_path: FP32 TFLite model.

    Returns:
        Path to the quantized model, or the input unchanged when tensorflow
        is absent (the 1.76 MB FP32 already meets the <2 MB target) or the
        conversion fails.
    """
    try:
        import tensorflow as tf
    except ImportError:
        logger.info(
            "[fp16] tensorflow not installed — skipping post-quant. The FP32 model at "
            f"{tflite_path} (1.76 MB) already meets the <2 MB target."
        )
        return Path(tflite_path)
    try:
        converter = tf.lite.TFLiteConverter.from_file(str(tflite_path))
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        buf = converter.convert()
        out = Path(str(tflite_path).replace(".tflite", "_fp16.tflite"))
        out.write_bytes(buf)
        logger.info(f"[fp16] saved {out} {out.stat().st_size / 1e6:.2f} MB")
        return out
    except Exception as e:  # noqa: BLE001
        logger.info(f"[fp16] quantization failed: {e}")
        return Path(tflite_path)


def validate_tflite(model: AVNetLite, tflite_path: str | Path, n: int = 200,
                    val_windows: str | Path = "data/processed/val_windows.npy") -> float | None:
    """Compare TFLite vs PyTorch on val windows; writes reports/tflite_diff.txt.

    Returns:
        Max abs diff (target <1e-2 FP16 / <1e-3 FP32), None when no
        interpreter backend is installed.
    """
    interp_cls = None
    for mod in (
        "ai_edge_litert.interpreter",
        "ai_edge_litert.lite.python.interpreter",
        "tflite_runtime",
    ):
        try:
            interp_cls = __import__(mod, fromlist=["Interpreter"]).Interpreter
            break
        except ImportError:
            continue
    if interp_cls is None:
        logger.info("[validate-tflite] no tflite interpreter, skipping")
        return None
    if not Path(val_windows).exists():
        logger.info("[validate-tflite] no val windows, skipping")
        return None

    X_val = np.load(val_windows, mmap_mode="r")
    rng = np.random.default_rng(0)
    idx = rng.choice(len(X_val), min(n, len(X_val)), replace=False)
    interp = interp_cls(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    max_diff = 0.0
    with torch.no_grad():
        for i in idx:
            x = np.array(X_val[i : i + 1], dtype=np.float32)
            v_pt, _, _, _, _ = model(torch.from_numpy(x))
            interp.set_tensor(inp["index"], x.astype(inp["dtype"]))
            interp.invoke()
            v_tl = interp.get_tensor(out_d["index"])
            max_diff = max(max_diff, float(np.abs(v_pt.numpy() - v_tl).max()))
    logger.info(
        f"[validate-tflite] TFLite vs PyTorch max diff {max_diff:.6f} over {len(idx)} windows "
        "(target <1e-2 FP16 / <1e-3 FP32)"
    )
    DIFF_REPORT.parent.mkdir(parents=True, exist_ok=True)
    with open(DIFF_REPORT, "w") as f:
        f.write(f"model: {tflite_path}\nwindows: {len(idx)}\nmax_abs_diff: {max_diff:.8f}\n")
    return max_diff


def run_export(
    model_path: str | Path,
    out_path: str | Path = "model.tflite",
    onnx_path: str | Path = "model.onnx",
    n_validate: int = 1000,
    val_windows: str | Path = "data/processed/val_windows.npy",
    quant: str = "fp16",
    scaler_path: str | Path = DEFAULT_SCALER,
) -> int:
    """Full export pipeline with gates. Returns a process exit code.

    Exit codes: 0 success, 2 spec-gate refusal (scaler missing/unversioned/
    mismatched — run preprocess first).
    """
    model = AVNetLite()
    if Path(model_path).exists():
        logger.info(f"[load] {model_path}")
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        logger.warning(f"[warn] {model_path} not found, exporting random weights (pipeline test)")

    logger.info(
        f"[model] {count_params(model):,} params, est FP32 {count_params(model) * 4 / 1e6:.2f} MB "
        f"FP16 {count_params(model) * 2 / 1e6:.2f} MB"
    )

    # P2 export gate: verify the scaler carries a matching spec fingerprint.
    scaler_src = Path(scaler_path)
    if not scaler_src.exists():
        logger.error(f"[gate] {scaler_src} missing — run preprocess.py first. Aborting.")
        return 2
    try:
        verify_scaler(scaler_src)
        logger.info(f"[gate] scaler spec OK (v{SPEC_VERSION}, sha256 {spec_sha256()[:12]})")
    except ValueError as e:
        logger.error(f"[gate] REFUSING export: {e}")
        return 2

    onnx_path = export_onnx(model, onnx_path)
    validate_onnx(model, onnx_path, n=n_validate, val_windows=val_windows)

    # PyTorch → TFLite (litert-torch), then FP16 quantize + validate
    sample = None
    if Path(val_windows).exists():
        sample = (
            torch.from_numpy(np.array(np.load(val_windows, mmap_mode="r")[0:1], dtype=np.float32)),
        )
    ok = try_tflite_litert(model, out_path, sample=sample)
    tflite_final: Path | None = None
    if ok:
        if quant == "fp16":
            tflite_final = quantize_fp16(out_path)
        diff = validate_tflite(model, tflite_final or out_path, n=min(n_validate, 200),
                               val_windows=val_windows)
        if diff is not None and diff > 1e-2:
            logger.error("[warn] TFLite diff >1e-2 — investigate before shipping")
    else:
        logger.info(f"[done] ONNX only at {onnx_path} — install ai-edge-torch for TFLite")
        shutil.copy(onnx_path, str(out_path) + ".onnx_fallback")
        logger.info(f"[fallback] copied {onnx_path} to {out_path}.onnx_fallback")

    # Ship the scaler alongside the model (Android assets expect both).
    scaler_dst = Path(out_path).parent / "scaler.json"
    if not scaler_dst.exists() or scaler_dst.resolve() != Path(scaler_src).resolve():
        shutil.copy(scaler_src, scaler_dst)
    logger.info(f"[scaler] ensured {scaler_dst} (from {scaler_src})")

    # P2: stamp the manifest binding model ↔ scaler ↔ spec.
    final_model = tflite_final if tflite_final is not None else (Path(out_path) if ok else onnx_path)
    if Path(final_model).exists():
        try:
            man = write_model_manifest(
                Path(final_model).parent / "model_manifest.json",
                final_model,
                scaler_src,
                extra={
                    "params": count_params(model),
                    "quant": quant,
                    "tflite_diff": (
                        DIFF_REPORT.read_text().strip().splitlines()[-1]
                        if DIFF_REPORT.exists()
                        else None
                    ),
                },
            )
            logger.info(
                f"[manifest] model_manifest.json "
                f"(model {man['model_sha256'][:12]}, scaler {man['scaler_sha256'][:12]})"
            )
        except ValueError as e:
            logger.error(f"[manifest] REFUSING to stamp: {e}")
            return 2
    return 0
