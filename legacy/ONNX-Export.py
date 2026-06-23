"""
====================================================================================
PIN-Lite: ONNX Export & Edge Deployment Benchmarking (Phase F)
====================================================================================
Exports models to ONNX format, applies ONNX Runtime optimizations, and 
benchmarks inference latency vs PyTorch.

Usage (on Kaggle):
    python ONNX-Export.py

Outputs:
    - pinlite_distilled.onnx
    - pinlite_pruned.onnx (if available)
    - pinlite_combined.onnx (if available)
    - onnx_benchmark_results.csv
====================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
import os
import sys
import time
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# =================================================================================
# 1. MODULE IMPORTS
# =================================================================================
try:
    from PinPoint import Config as TeacherConfig, LAVDFDataset, collate_fn
    print("✅ Loaded PinPoint module")
except ImportError:
    print("FATAL: PinPoint module not found.")
    sys.exit(1)

try:
    from Distill_PinPoint import ConfigLite, PinpointTransformerLite
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Distill_PinPoint",
            os.path.join(os.path.dirname(__file__), "Distill-student.py"))
        DP = importlib.util.module_from_spec(spec)
        sys.modules["Distill_PinPoint"] = DP
        spec.loader.exec_module(DP)
        ConfigLite = DP.ConfigLite
        PinpointTransformerLite = DP.PinpointTransformerLite
    except Exception as e:
        print(f"Warning: Could not load Distill_PinPoint: {e}")


# =================================================================================
# 2. ONNX WRAPPER (Single output for ONNX compatibility)
# =================================================================================

class OnnxExportWrapper(nn.Module):
    """Wraps model to return only classification logits for ONNX export."""
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, video, audio):
        logits, _, _ = self.model(video, audio)
        return logits


# =================================================================================
# 3. CONFIGURATION
# =================================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Model paths — update these for your environment
MODELS_TO_EXPORT = {
    "Distilled": {
        "path": "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_LITE_model.pth",
        "onnx_name": "pinlite_distilled.onnx",
        "config_class": "lite",
    },
    "Pruned": {
        "path": "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_PRUNED_model.pth",
        "onnx_name": "pinlite_pruned.onnx",
        "config_class": "lite",
    },
    # Uncomment these as models become available:
    # "Combined": {
    #     "path": "best_pinpoint_COMBINED_pruned.pth",
    #     "onnx_name": "pinlite_combined.onnx",
    #     "config_class": "lite",
    # },
}

BENCHMARK_RUNS = 100  # Number of inference runs for benchmarking
OPSET_VERSION = 17


# =================================================================================
# 4. EXPORT FUNCTIONS
# =================================================================================

def export_to_onnx(model, config, onnx_path, model_name):
    """Export a PyTorch model to ONNX format."""
    print(f"\n--- Exporting {model_name} to ONNX ---")
    
    model.eval()
    model.cpu()
    wrapped = OnnxExportWrapper(model)
    wrapped.eval()
    
    # Create dummy inputs
    video_input = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1])
    audio_input = torch.randn(1, 400, config.NUM_MFCC)
    
    # Export
    try:
        torch.onnx.export(
            wrapped,
            (video_input, audio_input),
            onnx_path,
            input_names=['video', 'audio'],
            output_names=['logits'],
            dynamic_axes={
                'video': {0: 'batch_size'},
                'audio': {0: 'batch_size', 1: 'audio_length'},
                'logits': {0: 'batch_size'}
            },
            opset_version=OPSET_VERSION,
            do_constant_folding=True,
        )
        
        onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  Exported to: {onnx_path}")
        print(f"  ONNX Size: {onnx_size:.2f} MB")
        return True, onnx_size
    except Exception as e:
        print(f"  ERROR exporting {model_name}: {e}")
        return False, 0


def optimize_onnx(onnx_path, optimized_path=None):
    """Apply ONNX Runtime graph optimizations."""
    try:
        import onnxruntime as ort
        
        if optimized_path is None:
            base, ext = os.path.splitext(onnx_path)
            optimized_path = f"{base}_optimized{ext}"
        
        # Create optimized session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.optimized_model_filepath = optimized_path
        
        # This creates the optimized model file
        _ = ort.InferenceSession(onnx_path, sess_options, providers=['CPUExecutionProvider'])
        
        if os.path.exists(optimized_path):
            opt_size = os.path.getsize(optimized_path) / (1024 * 1024)
            print(f"  Optimized ONNX: {optimized_path} ({opt_size:.2f} MB)")
            return optimized_path, opt_size
        
        return onnx_path, os.path.getsize(onnx_path) / (1024 * 1024)
    except ImportError:
        print("  onnxruntime not installed, skipping optimization.")
        return onnx_path, os.path.getsize(onnx_path) / (1024 * 1024)


def quantize_onnx(onnx_path, quantized_path=None):
    """Apply ONNX Runtime dynamic INT8 quantization."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        
        if quantized_path is None:
            base, ext = os.path.splitext(onnx_path)
            quantized_path = f"{base}_int8{ext}"
        
        quantize_dynamic(
            onnx_path,
            quantized_path,
            weight_type=QuantType.QInt8
        )
        
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        print(f"  Quantized ONNX: {quantized_path} ({quant_size:.2f} MB)")
        return quantized_path, quant_size
    except ImportError:
        print("  onnxruntime.quantization not available.")
        return None, 0
    except Exception as e:
        print(f"  ONNX quantization error: {e}")
        return None, 0


# =================================================================================
# 5. BENCHMARKING
# =================================================================================

def benchmark_pytorch(model, config, device, num_runs=100):
    """Benchmark PyTorch inference latency."""
    model.eval()
    model.to(device)
    
    latencies = []
    for i in range(num_runs):
        video = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).to(device)
        audio = torch.randn(1, 400, config.NUM_MFCC).to(device)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        start = time.time()
        with torch.no_grad():
            _ = model(video, audio)
        
        if device == "cuda":
            torch.cuda.synchronize()
        
        latencies.append((time.time() - start) * 1000)
    
    # Skip warmup
    return np.mean(latencies[10:])


def benchmark_onnx(onnx_path, config, num_runs=100):
    """Benchmark ONNX Runtime inference latency."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed.")
        return 0.0
    
    providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(onnx_path, providers=providers)
    
    latencies = []
    for i in range(num_runs):
        video = np.random.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).astype(np.float32)
        audio = np.random.randn(1, 400, config.NUM_MFCC).astype(np.float32)
        
        start = time.time()
        _ = sess.run(None, {'video': video, 'audio': audio})
        latencies.append((time.time() - start) * 1000)
    
    return np.mean(latencies[10:])


def benchmark_onnx_accuracy(onnx_path, config, test_loader):
    """Evaluate ONNX model accuracy on test set."""
    try:
        import onnxruntime as ort
    except ImportError:
        return {}
    
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    all_preds = []
    all_labels = []
    
    for batch in test_loader:
        if batch is None:
            continue
        
        video = batch['video'].numpy()
        audio = batch['audio'].numpy()
        labels = batch['label'].numpy()
        
        outputs = sess.run(None, {'video': video, 'audio': audio})
        logits = outputs[0]
        preds = (1.0 / (1.0 + np.exp(-logits)) > 0.5).astype(float).flatten()
        
        all_preds.extend(preds)
        all_labels.extend(labels)
    
    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    return {"accuracy": acc, "f1": f1}


# =================================================================================
# 6. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIN-LITE: ONNX EXPORT & EDGE DEPLOYMENT BENCHMARKING")
    print("="*60)
    
    results = []
    
    for model_name, model_info in MODELS_TO_EXPORT.items():
        model_path = model_info["path"]
        onnx_name = model_info["onnx_name"]
        
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: {model_path} not found.")
            continue
        
        print(f"\n\n{'='*60}")
        print(f"Processing: {model_name}")
        print(f"{'='*60}")
        
        # Load model
        config = ConfigLite()
        model = PinpointTransformerLite(config)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        
        pytorch_size = os.path.getsize(model_path) / (1024 * 1024)
        
        # 1. Benchmark PyTorch
        print("\n[1] PyTorch Benchmark...")
        pytorch_latency = benchmark_pytorch(model, config, DEVICE, BENCHMARK_RUNS)
        print(f"  PyTorch Latency: {pytorch_latency:.2f} ms (on {DEVICE})")
        
        # 2. Export to ONNX
        print("\n[2] ONNX Export...")
        success, onnx_size = export_to_onnx(model, config, onnx_name, model_name)
        if not success:
            continue
        
        # 3. Optimize ONNX
        print("\n[3] ONNX Optimization...")
        opt_path, opt_size = optimize_onnx(onnx_name)
        
        # 4. Quantize ONNX
        print("\n[4] ONNX Quantization...")
        quant_path, quant_size = quantize_onnx(onnx_name)
        
        # 5. Benchmark ONNX
        print("\n[5] ONNX Benchmarks...")
        onnx_latency = benchmark_onnx(onnx_name, config, BENCHMARK_RUNS)
        print(f"  ONNX Float Latency: {onnx_latency:.2f} ms (CPU)")
        
        onnx_opt_latency = benchmark_onnx(opt_path, config, BENCHMARK_RUNS) if opt_path else 0
        print(f"  ONNX Optimized Latency: {onnx_opt_latency:.2f} ms (CPU)")
        
        onnx_quant_latency = 0
        if quant_path and os.path.exists(quant_path):
            onnx_quant_latency = benchmark_onnx(quant_path, config, BENCHMARK_RUNS)
            print(f"  ONNX INT8 Latency: {onnx_quant_latency:.2f} ms (CPU)")
        
        # 6. Accuracy check on ONNX
        print("\n[6] ONNX Accuracy Check...")
        try:
            test_dataset = LAVDFDataset(config, split='test')
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
            onnx_metrics = benchmark_onnx_accuracy(onnx_name, config, test_loader)
            print(f"  ONNX Accuracy: {onnx_metrics.get('accuracy', 'N/A')}")
        except Exception as e:
            print(f"  Could not evaluate accuracy: {e}")
            onnx_metrics = {}
        
        # Compile results
        row = {
            "Model": model_name,
            "PyTorch_Size_MB": round(pytorch_size, 2),
            "ONNX_Size_MB": round(onnx_size, 2),
            "ONNX_Optimized_MB": round(opt_size, 2),
            "ONNX_INT8_MB": round(quant_size, 2) if quant_size else "N/A",
            "PyTorch_Latency_ms": round(pytorch_latency, 2),
            "ONNX_Latency_ms": round(onnx_latency, 2),
            "ONNX_Opt_Latency_ms": round(onnx_opt_latency, 2),
            "ONNX_INT8_Latency_ms": round(onnx_quant_latency, 2) if onnx_quant_latency else "N/A",
            "Speedup_vs_PyTorch": round(pytorch_latency / onnx_latency, 2) if onnx_latency > 0 else "N/A",
            "ONNX_Accuracy": round(onnx_metrics.get('accuracy', 0), 4) if onnx_metrics else "N/A",
        }
        results.append(row)
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    if results:
        df = pd.DataFrame(results)
        df.to_csv("onnx_benchmark_results.csv", index=False)
        print(f"\n\n{'='*60}")
        print("ONNX BENCHMARK SUMMARY")
        print(f"{'='*60}")
        print(df.to_string(index=False))
        print("\nResults saved to onnx_benchmark_results.csv")
    
    print("\n--- ONNX Export & Benchmarking Complete ---")
