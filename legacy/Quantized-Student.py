"""
====================================================================================
PIN-Lite: Student Model Quantization (Phase A — Fix Quantization)
====================================================================================
This script quantizes the DISTILLED STUDENT model (not the teacher), which is the
correct approach for model compression. Three methods are provided:

1. Post-Training Dynamic Quantization (INT8) — simplest, no calibration needed
2. Post-Training Static Quantization (INT8) — requires calibration data
3. ONNX Runtime Quantization — export to ONNX and quantize

Usage (on Kaggle):
    python Quantized-Student.py

Requires:
    - best_pinpoint_LITE_model.pth (distilled student model)
    - PinPoint-main.py (base module)
    - Distill-student.py (student architecture)
====================================================================================
"""

import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
from torch.utils.data import DataLoader
import os
import sys
import time
import copy
import types
import warnings
import numpy as np
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =================================================================================
# 1. MODULE IMPORTS
# =================================================================================
try:
    from PinPoint import (
        Config as TeacherConfig,
        LAVDFDataset,
        collate_fn,
        get_sinusoidal_embeddings,
        AudioFeatureExtractor,
        GatedCrossAttentionBlock,
    )
    print("✅ Loaded PinPoint module")
except ImportError:
    print("FATAL: PinPoint module not found. Ensure PinPoint-main.py is accessible.")
    sys.exit(1)

try:
    from Distill import (
        ConfigLite,
        PinpointTransformerLite,
        VideoFeatureExtractorLite,
    )
    print("✅ Loaded Distill_PinPoint module")
except ImportError:
    # Try alternative import path (direct file)
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Distill_PinPoint", 
            os.path.join(os.path.dirname(__file__), "Distill-student.py"))
        Distill_PinPoint = importlib.util.module_from_spec(spec)
        sys.modules["Distill_PinPoint"] = Distill_PinPoint
        spec.loader.exec_module(Distill_PinPoint)
        ConfigLite = Distill_PinPoint.ConfigLite
        PinpointTransformerLite = Distill_PinPoint.PinpointTransformerLite
        VideoFeatureExtractorLite = Distill_PinPoint.VideoFeatureExtractorLite
        print("✅ Loaded Distill-student module via file path")
    except Exception as e:
        print(f"FATAL: Could not load Distill-student module: {e}")
        sys.exit(1)


# =================================================================================
# 2. CONFIGURATION
# =================================================================================

# UPDATE THESE PATHS for your environment (Kaggle / local)
STUDENT_MODEL_PATH = "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_LITE_model.pth"
OUTPUT_DIR = "."  # Output directory for quantized models

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CALIBRATION_SAMPLES = 200  # Number of samples for static quantization calibration
CALIBRATION_STEPS = 100    # Calibration forward passes


# =================================================================================
# 3. METHOD 1: POST-TRAINING DYNAMIC QUANTIZATION
# =================================================================================

def quantize_dynamic(model_path, output_path="best_pinpoint_STUDENT_DYNAMIC_INT8.pth"):
    """
    Post-Training Dynamic Quantization.
    Quantizes Linear and LSTM/GRU layers to INT8 dynamically at inference time.
    This is the simplest approach — no calibration data needed.
    Works on CPU only.
    """
    print("\n" + "="*60)
    print("METHOD 1: Post-Training Dynamic Quantization (INT8)")
    print("="*60)
    
    # 1. Load the float student model
    config = ConfigLite()
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.cpu()
    
    # Record original size
    torch.save(model.state_dict(), "_temp_float_model.pth")
    float_size = os.path.getsize("_temp_float_model.pth") / (1024 * 1024)
    os.remove("_temp_float_model.pth")
    
    # 2. Apply dynamic quantization
    # Target: all Linear layers and GRU layers
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},  # Layers to quantize
        dtype=torch.qint8
    )
    
    # 3. Save
    output_full_path = os.path.join(OUTPUT_DIR, output_path)
    torch.save(quantized_model.state_dict(), output_full_path)
    quant_size = os.path.getsize(output_full_path) / (1024 * 1024)
    
    print(f"\n--- Dynamic Quantization Results ---")
    print(f"Original Size:   {float_size:.2f} MB")
    print(f"Quantized Size:  {quant_size:.2f} MB")
    print(f"Compression:     {float_size / quant_size:.2f}x")
    print(f"Saved to: {output_full_path}")
    
    return quantized_model, config


# =================================================================================
# 4. METHOD 2: POST-TRAINING STATIC QUANTIZATION
# =================================================================================

class QuantizedStudentWrapper(nn.Module):
    """
    Wraps the PinpointTransformerLite with QuantStub/DeQuantStub for static quantization.
    Only quantizes the video CNN path (most compute-heavy and quantization-friendly).
    Audio GRU and Transformer attention remain in FP32 for stability.
    """
    def __init__(self, student_model):
        super().__init__()
        self.model = student_model
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, video, audio, video_mask=None):
        # We quantize at the input level for the video path
        # The GRU and attention layers stay in FP32 for stability
        b, t, c, h, w = video.shape
        video_flat = video.view(b * t, c, h, w)
        
        # Quantize video input
        video_q = self.quant(video_flat)
        
        # Pass through MobileNetV3 feature extractor (quantized)
        features = self.model.video_extractor.feature_extractor(video_q)
        pooled = self.model.video_extractor.pool(features).view(b * t, -1)
        
        # Dequantize before linear projection
        pooled = self.dequant(pooled)
        projected = self.model.video_extractor.projection(pooled)
        video_feat = projected.view(b, t, -1)
        
        # Audio path (FP32)
        audio_feat = self.model.audio_extractor(audio)
        
        # Positional encoding
        video_feat = video_feat + self.model.video_pos_encoder[:, :video_feat.size(1), :]
        audio_len = audio_feat.size(1)
        audio_pos_encoding = get_sinusoidal_embeddings(audio_len, self.model.config.EMBED_DIM).to(audio_feat.device)
        audio_feat = audio_feat + audio_pos_encoding
        
        # Attention (FP32)
        last_attention_map = None
        for layer in self.model.gated_attention_layers:
            audio_feat, attention_map = layer(audio_feat, video_feat, video_mask)
            last_attention_map = attention_map
        
        pooled_output = audio_feat.mean(dim=1)
        classification_logits = self.model.classification_head(pooled_output)
        offset_logits = self.model.offset_head(pooled_output)
        
        return classification_logits, offset_logits, last_attention_map


def quantize_static(model_path, output_path="best_pinpoint_STUDENT_STATIC_INT8.pth"):
    """
    Post-Training Static Quantization.
    Requires a calibration dataset to determine optimal quantization parameters.
    More accurate than dynamic quantization but requires data.
    """
    print("\n" + "="*60)
    print("METHOD 2: Post-Training Static Quantization (INT8)")
    print("="*60)
    
    # 1. Load student model
    config = ConfigLite()
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.cpu()
    
    # 2. Wrap with quant/dequant stubs
    wrapped_model = QuantizedStudentWrapper(model)
    
    # 3. Configure quantization
    backend = 'qnnpack'
    wrapped_model.qconfig = torch.quantization.get_default_qconfig(backend)
    
    # Disable quantization for sensitive layers
    wrapped_model.model.audio_extractor.qconfig = None
    wrapped_model.model.gated_attention_layers.qconfig = None  
    wrapped_model.model.classification_head.qconfig = None
    wrapped_model.model.offset_head.qconfig = None
    wrapped_model.model.video_extractor.projection.qconfig = None
    
    # 4. Prepare
    wrapped_model.eval()
    torch.quantization.prepare(wrapped_model, inplace=True)
    
    # 5. Calibrate with data
    print(f"Calibrating with {CALIBRATION_STEPS} forward passes...")
    try:
        train_dataset = LAVDFDataset(config, split='train')
        if len(train_dataset) > CALIBRATION_SAMPLES:
            indices = list(range(CALIBRATION_SAMPLES))
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
        calib_loader = DataLoader(train_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(calib_loader, desc="Calibrating", total=CALIBRATION_STEPS)):
                if i >= CALIBRATION_STEPS:
                    break
                if batch is None:
                    continue
                video = batch['video'].cpu()
                audio = batch['audio'].cpu()
                wrapped_model(video, audio)
                
    except Exception as e:
        print(f"Warning: Calibration data loading failed ({e}). Using random calibration.")
        # Fallback: random calibration
        with torch.no_grad():
            for _ in tqdm(range(CALIBRATION_STEPS), desc="Random Calibrating"):
                video = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1])
                audio = torch.randn(1, 400, config.NUM_MFCC)
                wrapped_model(video, audio)
    
    # 6. Convert to quantized model
    quantized_model = torch.quantization.convert(wrapped_model, inplace=False)
    
    # 7. Save
    output_full_path = os.path.join(OUTPUT_DIR, output_path)
    torch.save(quantized_model.state_dict(), output_full_path)
    quant_size = os.path.getsize(output_full_path) / (1024 * 1024)
    
    print(f"\n--- Static Quantization Results ---")
    print(f"Quantized Size: {quant_size:.2f} MB")
    print(f"Saved to: {output_full_path}")
    
    return quantized_model, config


# =================================================================================
# 5. METHOD 3: ONNX RUNTIME QUANTIZATION
# =================================================================================

def quantize_onnx(model_path, output_path="best_pinpoint_STUDENT_ONNX_INT8.onnx"):
    """
    ONNX Runtime Quantization.
    Export the student model to ONNX, then quantize using onnxruntime.quantization.
    This is the industry-standard approach for deployment.
    """
    print("\n" + "="*60)
    print("METHOD 3: ONNX Runtime Quantization")
    print("="*60)
    
    try:
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic as ort_quantize_dynamic, QuantType
    except ImportError:
        print("onnx/onnxruntime not installed. Install with:")
        print("  pip install onnx onnxruntime onnxruntime-quantization")
        print("Skipping ONNX quantization.")
        return None, None
    
    # 1. Load student model
    config = ConfigLite()
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    model.cpu()
    
    # 2. Create dummy inputs for tracing
    video_input = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1])
    audio_input = torch.randn(1, 400, config.NUM_MFCC)
    
    # 3. Export to ONNX
    onnx_float_path = os.path.join(OUTPUT_DIR, "student_float.onnx")
    print("Exporting to ONNX...")
    
    # We need a wrapper that only returns the classification logits (ONNX doesn't support multiple returns easily)
    class OnnxWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        def forward(self, video, audio):
            logits, _, _ = self.model(video, audio)
            return logits
    
    wrapped = OnnxWrapper(model)
    
    torch.onnx.export(
        wrapped,
        (video_input, audio_input),
        onnx_float_path,
        input_names=['video', 'audio'],
        output_names=['logits'],
        dynamic_axes={
            'video': {0: 'batch_size'},
            'audio': {0: 'batch_size', 1: 'audio_length'},
            'logits': {0: 'batch_size'}
        },
        opset_version=17,
        do_constant_folding=True
    )
    
    float_size = os.path.getsize(onnx_float_path) / (1024 * 1024)
    print(f"ONNX Float model size: {float_size:.2f} MB")
    
    # 4. Quantize ONNX model
    onnx_quant_path = os.path.join(OUTPUT_DIR, output_path)
    print("Quantizing ONNX model (Dynamic INT8)...")
    
    ort_quantize_dynamic(
        onnx_float_path,
        onnx_quant_path,
        weight_type=QuantType.QInt8
    )
    
    quant_size = os.path.getsize(onnx_quant_path) / (1024 * 1024)
    
    print(f"\n--- ONNX Quantization Results ---")
    print(f"ONNX Float Size:     {float_size:.2f} MB")
    print(f"ONNX Quantized Size: {quant_size:.2f} MB")
    print(f"Compression:         {float_size / quant_size:.2f}x")
    print(f"Saved to: {onnx_quant_path}")
    
    # 5. Quick benchmark
    print("\nBenchmarking ONNX inference...")
    sess = ort.InferenceSession(onnx_quant_path, providers=['CPUExecutionProvider'])
    
    latencies = []
    for _ in range(50):
        v = np.random.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).astype(np.float32)
        a = np.random.randn(1, 400, config.NUM_MFCC).astype(np.float32)
        
        start = time.time()
        _ = sess.run(None, {'video': v, 'audio': a})
        latencies.append((time.time() - start) * 1000)
    
    avg_latency = np.mean(latencies[5:])  # Skip warmup
    print(f"ONNX INT8 Avg Latency: {avg_latency:.2f} ms/sample")
    
    # Cleanup float ONNX
    if os.path.exists(onnx_float_path):
        os.remove(onnx_float_path)
    
    return onnx_quant_path, config


# =================================================================================
# 6. EVALUATION 
# =================================================================================

def evaluate_quantized_model(model, config, model_name="Quantized-Student"):
    """Evaluate a quantized model on the test set."""
    print(f"\n--- Evaluating {model_name} ---")
    
    try:
        test_dataset = LAVDFDataset(config, split='test')
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)
    except Exception as e:
        print(f"Could not load test dataset: {e}")
        return {}
    
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc=f"Evaluating {model_name}")):
            if batch is None:
                continue
            
            video = batch['video'].cpu()
            audio = batch['audio'].cpu()
            labels = batch['label']
            
            start = time.time()
            logits, _, _ = model(video, audio)
            latencies.append((time.time() - start) * 1000 / video.size(0))
            
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).float()
            correct += (preds == labels).sum().item()
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
    
    print("No samples evaluated.")
    return {}


# =================================================================================
# 7. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIN-LITE: STUDENT MODEL QUANTIZATION")
    print("="*60)
    
    # Check model exists
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Student model not found at {STUDENT_MODEL_PATH}")
        print("Please update STUDENT_MODEL_PATH to point to your distilled student model.")
        sys.exit(1)
    
    results_summary = {}
    
    # ---- METHOD 1: Dynamic Quantization ----
    try:
        dyn_model, dyn_config = quantize_dynamic(STUDENT_MODEL_PATH)
        dyn_results = evaluate_quantized_model(dyn_model, dyn_config, "Dynamic-INT8-Student")
        results_summary["Dynamic-INT8"] = dyn_results
    except Exception as e:
        print(f"Dynamic quantization failed: {e}")
    
    # ---- METHOD 2: Static Quantization ----
    try:
        stat_model, stat_config = quantize_static(STUDENT_MODEL_PATH)
        stat_results = evaluate_quantized_model(stat_model, stat_config, "Static-INT8-Student")
        results_summary["Static-INT8"] = stat_results
    except Exception as e:
        print(f"Static quantization failed: {e}")
    
    # ---- METHOD 3: ONNX Quantization ----
    try:
        onnx_path, onnx_config = quantize_onnx(STUDENT_MODEL_PATH)
        if onnx_path:
            results_summary["ONNX-INT8"] = {"Note": f"See ONNX benchmark above. Model at {onnx_path}"}
    except Exception as e:
        print(f"ONNX quantization failed: {e}")
    
    # ---- Final Summary ----
    print("\n" + "="*60)
    print("QUANTIZATION SUMMARY")
    print("="*60)
    for method, res in results_summary.items():
        print(f"\n{method}:")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
    
    print("\n--- Student Quantization Complete ---")
