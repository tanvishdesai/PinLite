"""
====================================================================================
PIN-Lite: Student Model FP16 Mixed-Precision Inference (Phase A)
====================================================================================
Applies FP16 (half-precision) inference to the distilled student model.
This halves model size and improves GPU inference speed with minimal accuracy loss.

Usage (on Kaggle):
    python Quantized-StudentFP16.py
====================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import time
import numpy as np
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

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
    from Distill import ConfigLite, PinpointTransformerLite
    print("✅ Loaded Distill_PinPoint module")
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Distill_PinPoint",
            os.path.join(os.path.dirname(__file__), "Distill-student.py"))
        Distill_PinPoint = importlib.util.module_from_spec(spec)
        sys.modules["Distill_PinPoint"] = Distill_PinPoint
        spec.loader.exec_module(Distill_PinPoint)
        ConfigLite = Distill_PinPoint.ConfigLite
        PinpointTransformerLite = Distill_PinPoint.PinpointTransformerLite
        print("✅ Loaded Distill-student module via file path")
    except Exception as e:
        print(f"FATAL: Could not load Distill-student module: {e}")
        sys.exit(1)


# =================================================================================
# 2. CONFIGURATION
# =================================================================================
STUDENT_MODEL_PATH = "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_LITE_model.pth"
OUTPUT_PATH = "best_pinpoint_STUDENT_FP16.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =================================================================================
# 3. FP16 CONVERSION
# =================================================================================

def convert_to_fp16(model_path, output_path):
    """
    Convert student model to FP16 (half precision).
    Saves FP16 weights for size comparison, but at inference time we use
    torch.cuda.amp.autocast which keeps LayerNorm/BatchNorm in FP32
    automatically — avoiding dtype mismatch errors.
    """
    print("\n" + "="*60)
    print("FP16 Mixed-Precision Conversion")
    print("="*60)
    
    config = ConfigLite()
    
    # Load FP32 model
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Record FP32 size
    torch.save(model.state_dict(), "_temp_fp32.pth")
    fp32_size = os.path.getsize("_temp_fp32.pth") / (1024 * 1024)
    os.remove("_temp_fp32.pth")
    
    # Save FP16 weights (for size measurement)
    fp16_state = {k: v.half() for k, v in model.state_dict().items()}
    torch.save(fp16_state, output_path)
    fp16_size = os.path.getsize(output_path) / (1024 * 1024)
    
    print(f"FP32 Size: {fp32_size:.2f} MB")
    print(f"FP16 Size: {fp16_size:.2f} MB")
    print(f"Compression: {fp32_size / fp16_size:.2f}x")
    print(f"Saved to: {output_path}")
    
    # Return the FP32 model — autocast handles FP16 at inference time
    return model, config


# =================================================================================
# 4. EVALUATION WITH FP16
# =================================================================================

def evaluate_fp16(model, config, model_name="FP16-Student"):
    """
    Evaluate model using torch.cuda.amp.autocast for mixed-precision.
    autocast automatically keeps LayerNorm/BatchNorm in FP32 while running
    Linear/Conv/MatMul in FP16 — avoiding the dtype mismatch error.
    """
    print(f"\n--- Evaluating {model_name} ---")
    
    use_amp = (DEVICE == "cuda")
    if not use_amp:
        print("WARNING: FP16 inference on CPU is not recommended (slow, no benefit).")
        print("Evaluating in FP32 on CPU instead.")
    
    model = model.float().to(DEVICE)
    
    try:
        test_dataset = LAVDFDataset(config, split='test')
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    except Exception as e:
        print(f"Could not load test dataset: {e}")
        return {}
    
    model.eval()
    correct = 0
    total = 0
    latencies = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            if batch is None:
                continue
            
            video = batch['video'].to(DEVICE)
            audio = batch['audio'].to(DEVICE)
            labels = batch['label']
            
            start = time.time()
            # autocast handles FP16 per-operator; LayerNorm stays FP32
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits, _, _ = model(video, audio)
            latencies.append((time.time() - start) * 1000 / video.size(0))
            
            preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1).float()
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total > 0:
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_preds)
        except:
            auc = 0.0
        avg_latency = np.mean(latencies[2:]) if len(latencies) > 2 else np.mean(latencies)
        
        results = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "AUC": auc,
            "Avg Latency (ms)": avg_latency
        }
        
        print(f"\n{model_name} Results:")
        for k, v in results.items():
            print(f"  {k}: {v:.4f}")
        return results
    
    return {}


# =================================================================================
# 5. FP32 vs FP16 LATENCY COMPARISON
# =================================================================================

def compare_fp32_fp16(model_path):
    """Side-by-side latency comparison of FP32 vs FP16 (autocast) inference."""
    print("\n" + "="*60)
    print("FP32 vs FP16 LATENCY COMPARISON")
    print("="*60)
    
    if DEVICE != "cuda":
        print("Skipping comparison — FP16 requires CUDA GPU.")
        return
    
    config = ConfigLite()
    
    # Load model (stays FP32, autocast handles FP16 per-operator)
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    
    num_runs = 100
    
    # FP32 benchmark (no autocast)
    fp32_times = []
    for i in range(num_runs):
        video = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).to(DEVICE)
        audio = torch.randn(1, 400, config.NUM_MFCC).to(DEVICE)
        
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            _ = model(video, audio)
        torch.cuda.synchronize()
        fp32_times.append((time.time() - start) * 1000)
    
    # FP16 benchmark (with autocast)
    fp16_times = []
    for i in range(num_runs):
        video = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).to(DEVICE)
        audio = torch.randn(1, 400, config.NUM_MFCC).to(DEVICE)
        
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad(), torch.cuda.amp.autocast():
            _ = model(video, audio)
        torch.cuda.synchronize()
        fp16_times.append((time.time() - start) * 1000)
    
    # Skip warmup
    fp32_avg = np.mean(fp32_times[10:])
    fp16_avg = np.mean(fp16_times[10:])
    
    print(f"\nFP32 Avg Latency: {fp32_avg:.2f} ms")
    print(f"FP16 Avg Latency: {fp16_avg:.2f} ms")
    print(f"Speedup:          {fp32_avg / fp16_avg:.2f}x")
    
    del model
    torch.cuda.empty_cache()


# =================================================================================
# 6. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIN-LITE: FP16 MIXED-PRECISION CONVERSION")
    print("="*60)
    
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Student model not found at {STUDENT_MODEL_PATH}")
        sys.exit(1)
    
    # 1. Convert to FP16 (saves FP16 weights for size comparison)
    model, config = convert_to_fp16(STUDENT_MODEL_PATH, OUTPUT_PATH)
    
    # 2. Evaluate using autocast mixed-precision
    results = evaluate_fp16(model, config, "FP16-Student")
    
    # 3. Compare latencies
    compare_fp32_fp16(STUDENT_MODEL_PATH)
    
    print("\n--- FP16 Conversion Complete ---")
