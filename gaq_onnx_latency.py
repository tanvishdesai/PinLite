"""
====================================================================================
GAQ — Honest edge latency on a NAMED backend (ONNX Runtime, INT8 kernels)
====================================================================================
The old "Combined" pipeline reported 171 ms because dynamic-INT8 had no optimized
kernel on the test CPU. This script reports latency on backends that *do* have INT8
kernels, so the speedup claim is real:

  1. PyTorch FP32 student            (CPU)
  2. PyTorch dynamic-INT8 GAQ        (CPU, fbgemm/qnnpack — reaches attention linears)
  3. ONNX Runtime FP32               (CPU)
  4. ONNX Runtime INT8 (dynamic)     (CPU)

Because GAQ decomposes attention into explicit Linear ops, the INT8 export covers
the attention projections — not just the CNN/GRU.

Run:
    python gaq_onnx_latency.py
Outputs: gaq_latency.csv
====================================================================================
"""

import os
import time

import numpy as np
import torch
import torch.nn as nn

import gaq_config as C
import gaq_core as G


class OnnxWrapper(nn.Module):
    """ONNX needs a single tensor output."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, video, audio):
        logits, _, _ = self.model(video, audio)
        return logits


def _torch_latency(model, v, a, repeats, warmup):
    return G.measure_cpu_latency(model, v, a, repeats, warmup)


def _onnx_latency(onnx_path, v, a, repeats, warmup):
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    vi = v[:1].cpu().numpy().astype(np.float32)
    ai = a[:1].cpu().numpy().astype(np.float32)
    feed = {"video": vi, "audio": ai}
    for _ in range(warmup):
        sess.run(None, feed)
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def main():
    C.banner(f"GAQ — NAMED-HARDWARE LATENCY @ {C.TARGET_CPU_NAME}")
    G.set_seed(C.SEED)
    rows = []

    print("\n[1/5] Loading student + a real LAV-DF sample ...")
    student, s_cfg = G.load_student(device="cpu")
    loaders = G.make_loaders(s_cfg, splits=("test",), batch_size=2)
    batch = G.get_one_batch(loaders["test"])
    v, a = batch["video"], batch["audio"]

    print("\n[2/5] PyTorch FP32 student (CPU) ...")
    fp32_lat = _torch_latency(student, v, a, C.LATENCY_REPEATS, C.LATENCY_WARMUP)
    rows.append(("PyTorch FP32 student", fp32_lat))
    print(f"  {fp32_lat:.1f} ms")

    print("\n[3/5] PyTorch dynamic-INT8 GAQ (CPU) ...")
    explicit = G.build_quantizable_student(student)
    real_int8 = G.to_dynamic_int8(explicit)
    int8_lat = _torch_latency(real_int8, v, a, C.LATENCY_REPEATS, C.LATENCY_WARMUP)
    rows.append(("PyTorch dynamic-INT8 GAQ", int8_lat))
    print(f"  {int8_lat:.1f} ms  (speedup vs FP32 = {fp32_lat / max(int8_lat,1e-6):.2f}x)")

    # --- ONNX paths -----------------------------------------------------------
    # Separate the two failure modes: (a) packages missing -> tell the user to pip
    # install; (b) export/quantize blows up (GRU/5D-video/attention can be finicky
    # for the ONNX exporter) -> report the REAL error so it can be fixed, and still
    # keep the PyTorch rows we already measured instead of losing the whole run.
    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
        from onnxruntime.quantization import quantize_dynamic, QuantType
        have_onnx = True
    except ImportError:
        have_onnx = False
        print("\n[ONNX skipped] install: pip install onnx onnxruntime")

    if have_onnx:
        try:
            print("\n[4/5] Export + ONNX Runtime FP32 ...")
            wrapper = OnnxWrapper(student).eval()
            onnx_fp32 = C.out("gaq_student_fp32.onnx")
            torch.onnx.export(
                wrapper, (v[:1], a[:1]), onnx_fp32,
                input_names=["video", "audio"], output_names=["logits"],
                dynamic_axes={"video": {0: "b"}, "audio": {0: "b", 1: "t"}, "logits": {0: "b"}},
                opset_version=17, do_constant_folding=True)
            onnx_fp32_lat = _onnx_latency(onnx_fp32, v, a, C.LATENCY_REPEATS, C.LATENCY_WARMUP)
            rows.append(("ONNX Runtime FP32", onnx_fp32_lat))
            print(f"  {onnx_fp32_lat:.1f} ms")

            print("\n[5/5] ONNX Runtime INT8 (dynamic) ...")
            onnx_int8 = C.out("gaq_student_int8.onnx")
            quantize_dynamic(onnx_fp32, onnx_int8, weight_type=QuantType.QInt8)
            onnx_int8_lat = _onnx_latency(onnx_int8, v, a, C.LATENCY_REPEATS, C.LATENCY_WARMUP)
            rows.append(("ONNX Runtime INT8 (GAQ)", onnx_int8_lat))
            print(f"  {onnx_int8_lat:.1f} ms  (speedup vs FP32 student = "
                  f"{fp32_lat / max(onnx_int8_lat,1e-6):.2f}x)")
            fp32_size = os.path.getsize(onnx_fp32) / 1e6
            int8_size = os.path.getsize(onnx_int8) / 1e6
            print(f"  ONNX size: FP32={fp32_size:.2f}MB -> INT8={int8_size:.2f}MB")
        except Exception as e:  # export/quantize/runtime failure, NOT a missing package
            print(f"\n[ONNX export FAILED] {type(e).__name__}: {e}")
            print("  Keeping the PyTorch latency rows above. Common fixes: bump")
            print("  opset_version, or set dynamo=False / try torch.onnx.dynamo_export.")

    import csv
    with open(C.out("gaq_latency.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["backend", "latency_ms", "hardware"])
        for name, lat in rows:
            w.writerow([name, round(lat, 2), C.TARGET_CPU_NAME])
    print(f"\nSaved -> {C.out('gaq_latency.csv')}")


if __name__ == "__main__":
    main()
