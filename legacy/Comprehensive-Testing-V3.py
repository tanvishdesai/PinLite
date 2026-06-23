"""
====================================================================================
PIN-Lite: Comprehensive Testing V3 — All Model Variants
====================================================================================
Evaluates ALL model variants from a single unified Kaggle dataset:
  - Base (Teacher)
  - Distilled (PIN-Lite)
  - Pruned
  - FP16 (Mixed-Precision)
  - Linear Attention variant
  - MQA variant
  - LowRank Attention variant
  - Combined Pipeline (Distill+Prune+Quantize)

Uses Attention-Map-Based EPS (consistent with EPS-Enhanced.py V2).

Usage (on Kaggle):
    python Comprehensive-Testing-V3.py
    
Outputs:
    - comprehensive_benchmark_v3.csv
    - Pareto_All_Models_v3.png
====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import importlib.util
import time
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

try:
    from thop import profile
except ImportError:
    print("Warning: 'thop' not found. FLOPs will be skipped.")
    profile = None

# =================================================================================
# 1. MODULE LOADING — ALL FROM SINGLE DATASET
# =================================================================================

# --- Unified dataset path (models only) ---
DATASET_DIR = "/kaggle/input/datasets/shivamansari/pinlite-models-v2-2002"

# --- Module imports ---
# Modules are written to /kaggle/working/ via %%writefile magic:
#   PinPoint.py, Distill.py, Attentionvariants.py

print("--- Loading Core Modules ---")

try:
    import PinPoint
    print("✅ Loaded PinPoint")

    import Distill as Distill_PinPoint
    print("✅ Loaded Distill_PinPoint")
except ImportError as e:
    print(f"❌ Error loading core modules: {e}")
    sys.exit(1)

# Try loading attention variants
try:
    import Attentionvariants as AV
    print("✅ Loaded Attention_Variants")
    HAS_ATTENTION_VARIANTS = True
except ImportError as e:
    print(f"⚠️ Attention variants not available: {e}")
    HAS_ATTENTION_VARIANTS = False


# =================================================================================
# 2. CONFIGURATION
# =================================================================================

# All models from the single dataset directory
MODEL_REGISTRY = {
    "Base": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_model_antisocial.pth"),
        "type": "base",
        "device": "auto",
    },
    "Distilled": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_LITE_model.pth"),
        "type": "lite",
        "device": "auto",
    },
    "Pruned": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_PRUNED_model.pth"),
        "type": "lite",
        "device": "auto",
    },
    "FP16": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_LITE_model.pth"),
        "type": "fp16",
        "device": "cuda",
    },
    "Linear-Attn": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_LINEAR_model.pth"),
        "type": "linear",
        "device": "auto",
    },
    "MQA": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_MQA_model.pth"),
        "type": "mqa",
        "device": "auto",
    },
    "LowRank": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_LOWRANK_model.pth"),
        "type": "lowrank",
        "device": "auto",
    },
    "Combined": {
        "path": os.path.join(DATASET_DIR, "best_pinpoint_COMBINED_model.pth"),
        "type": "dynamic_quantized",
        "device": "cpu",
    },
}

# Test data (4-part LAV-DF dataset)
TEST_DATA_DIRECTORIES = [
    "/kaggle/input/datasets/shivamansari/la-df-testrin-1",
    "/kaggle/input/datasets/shivamansari/lav-df-testing-part-2",
    "/kaggle/input/datasets/shivamansari/lav-df-testing-part-3",
    "/kaggle/input/datasets/shivamansari/lavdf-testing-part-4"
]
ORIGINAL_METADATA_PATH = "/kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPS_SAMPLES = 300
DEBUG_MODE = False

# Models whose evaluation is already done — skip them in the main loop
SKIP_MODELS = {"Base", "Distilled", "Pruned", "FP16", "Linear-Attn", "MQA", "LowRank"}

# Pre-populated results from previous run (these will be included in final CSV/plots)
PREVIOUS_RESULTS = [
    {"Model": "Base",        "Size (MB)": 57.32, "Params (M)": 15.0,  "FLOPs (G)": 0.0, "Inference (ms)": 98.62, "Peak VRAM (MB)": 886.1,  "Accuracy": 0.9737, "Precision": 0.9843, "Recall": 0.9798, "F1-Score": 0.9821, "AUC": 0.9683, "EPS": 1.0},
    {"Model": "Distilled",   "Size (MB)": 6.62,  "Params (M)": 1.69,  "FLOPs (G)": 0.0, "Inference (ms)": 45.93, "Peak VRAM (MB)": 555.3,  "Accuracy": 0.9753, "Precision": 0.9727, "Recall": 0.9944, "F1-Score": 0.9834, "AUC": 0.9584, "EPS": 0.6091},
    {"Model": "Pruned",      "Size (MB)": 6.62,  "Params (M)": 1.69,  "FLOPs (G)": 0.0, "Inference (ms)": 44.18, "Peak VRAM (MB)": 555.3,  "Accuracy": 0.9738, "Precision": 0.971,  "Recall": 0.994,  "F1-Score": 0.9824, "AUC": 0.9558, "EPS": 0.5887},
    {"Model": "FP16",        "Size (MB)": 3.31,  "Params (M)": 1.69,  "FLOPs (G)": 0.0, "Inference (ms)": 37.93, "Peak VRAM (MB)": 569.3,  "Accuracy": 0.9752, "Precision": 0.9726, "Recall": 0.9943, "F1-Score": 0.9833, "AUC": 0.9582, "EPS": 0.6091},
    {"Model": "Linear-Attn", "Size (MB)": 6.64,  "Params (M)": 1.69,  "FLOPs (G)": 0.0, "Inference (ms)": 38.56, "Peak VRAM (MB)": 710.42, "Accuracy": 0.6082, "Precision": 0.7859, "Recall": 0.6421, "F1-Score": 0.7068, "AUC": 0.578,  "EPS": 0.0333},
    {"Model": "MQA",         "Size (MB)": 6.25,  "Params (M)": 1.6,   "FLOPs (G)": 0.0, "Inference (ms)": 36.35, "Peak VRAM (MB)": 695.21, "Accuracy": 0.98,   "Precision": 0.9805, "Recall": 0.9926, "F1-Score": 0.9865, "AUC": 0.9689, "EPS": 0.6055},
    {"Model": "LowRank",     "Size (MB)": 6.26,  "Params (M)": 1.6,   "FLOPs (G)": 0.0, "Inference (ms)": 38.28, "Peak VRAM (MB)": 703.82, "Accuracy": 0.9625, "Precision": 0.9714, "Recall": 0.9778, "F1-Score": 0.9746, "AUC": 0.9489, "EPS": 0.5719},
]


# =================================================================================
# 3. MODEL LOADING 
# =================================================================================

def load_model(name, info):
    """Universal model loader."""
    path = info["path"]
    model_type = info["type"]
    target_device = info["device"]
    
    if target_device == "auto":
        target_device = DEVICE
    
    if not os.path.exists(path):
        return None, None, target_device
    
    print(f"Loading {name} from {path}...")
    
    try:
        if model_type == "base":
            config = PinPoint.Config()
            model = PinPoint.PinpointTransformer(config)
            model.load_state_dict(torch.load(path, map_location=target_device, weights_only=False))
            model.to(target_device)
            model.eval()
            return model, config, target_device
        
        elif model_type == "lite":
            config = Distill_PinPoint.ConfigLite()
            model = Distill_PinPoint.PinpointTransformerLite(config)
            model.load_state_dict(torch.load(path, map_location=target_device, weights_only=False))
            model.to(target_device)
            model.eval()
            return model, config, target_device
        
        elif model_type == "dynamic_quantized":
            config = Distill_PinPoint.ConfigLite()
            model = Distill_PinPoint.PinpointTransformerLite(config)
            model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
            model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=False)
            model.eval()
            return model, config, 'cpu'
        
        elif model_type == "fp16":
            if DEVICE != "cuda":
                print(f"  Skipping {name}: FP16 requires CUDA.")
                return None, None, target_device
            config = Distill_PinPoint.ConfigLite()
            model = Distill_PinPoint.PinpointTransformerLite(config)
            state = torch.load(path, map_location='cpu', weights_only=False)
            # Keep model in FP32 — we use torch.cuda.amp.autocast during inference
            model.load_state_dict({k: v.float() for k, v in state.items()})
            model.to('cuda')
            model.eval()
            return model, config, 'cuda'
        
        elif model_type in ["linear", "mqa", "lowrank"] and HAS_ATTENTION_VARIANTS:
            config_map = {
                "linear": AV.ConfigLinear,
                "mqa": AV.ConfigMQA,
                "lowrank": AV.ConfigLowRank,
            }
            config = config_map[model_type]()
            model = AV.PinpointTransformerVariant(config)
            model.load_state_dict(torch.load(path, map_location=target_device, weights_only=False), strict=False)
            model.to(target_device)
            model.eval()
            return model, config, target_device
        
        else:
            print(f"  Unknown model type: {model_type}")
            return None, None, target_device
    
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        import traceback
        traceback.print_exc()
        return None, None, target_device


# =================================================================================
# 4. DATASET (Multi-part LAV-DF test set)
# =================================================================================

class MultiPartLAVDFDataset(torch.utils.data.Dataset):
    def __init__(self, directories, metadata_path, config):
        self.config = config
        self.samples = []
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, 'r') as f:
            original_metadata = json.load(f)
        
        expected = {}
        for item in original_metadata:
            if item.get('split') == 'test':
                base = os.path.splitext(os.path.basename(item['file']))[0]
                expected[base] = 'fake' if item.get('n_fakes', 0) > 0 else 'real'
        
        for data_dir in directories:
            if not os.path.isdir(data_dir):
                continue
            for root, _, files in os.walk(data_dir):
                for f in files:
                    if f.endswith("_video.pt"):
                        base = f.replace("_video.pt", "")
                        if base in expected:
                            video_path = os.path.join(root, f)
                            audio_path = os.path.join(root, f"{base}_audio.pt")
                            if os.path.exists(audio_path):
                                self.samples.append({
                                    "video_path": video_path,
                                    "audio_path": audio_path,
                                    "label": 1.0 if expected[base] == 'fake' else 0.0,
                                })
        
        print(f"Loaded {len(self.samples)} test samples.")
        if DEBUG_MODE:
            self.samples = self.samples[:10]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            video = torch.load(s['video_path']).float() / 255.0
            audio = torch.load(s['audio_path'])
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Truncate or uniformly sample to NUM_FRAMES (default 30)
            num_frames = getattr(self.config, 'NUM_FRAMES', 30)
            t = video.shape[0]
            if t > num_frames:
                # Uniformly sample num_frames from the video
                indices = torch.linspace(0, t - 1, num_frames).long()
                video = video[indices]
            elif t < num_frames:
                # Pad with zeros if fewer frames
                pad = torch.zeros(num_frames - t, *video.shape[1:])
                video = torch.cat([video, pad], dim=0)
            
            mask = torch.ones(video.shape[0])
            return {
                "video": video, "audio": audio, "video_mask": mask,
                "label": torch.tensor(s['label'], dtype=torch.float),
            }
        except Exception as e:
            return None


def custom_collate_fn(batch):
    """Custom collate that filters None samples and pads video/audio to max length."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    
    # Pad videos to max temporal length in batch
    max_t = max(b['video'].shape[0] for b in batch)
    videos, audios, masks, labels = [], [], [], []
    
    for b in batch:
        v = b['video']
        t = v.shape[0]
        if t < max_t:
            pad = torch.zeros(max_t - t, *v.shape[1:])
            v = torch.cat([v, pad], dim=0)
        mask = torch.zeros(max_t)
        mask[:t] = 1.0
        videos.append(v)
        audios.append(b['audio'])
        masks.append(mask)
        labels.append(b['label'])
    
    # Pad audio to max length
    max_a = max(a.shape[0] for a in audios)
    padded_audios = []
    for a in audios:
        if a.shape[0] < max_a:
            pad = torch.zeros(max_a - a.shape[0], *a.shape[1:])
            a = torch.cat([a, pad], dim=0)
        padded_audios.append(a)
    
    return {
        'video': torch.stack(videos),
        'audio': torch.stack(padded_audios),
        'video_mask': torch.stack(masks),
        'label': torch.stack(labels),
    }


# =================================================================================
# 5. EVALUATION METRICS
# =================================================================================

def evaluate_model(model, dataloader, device, model_name, is_fp16=False):
    """Evaluate model accuracy and classification metrics."""
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {model_name}", leave=False):
            if batch is None:
                continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            labels = batch['label']
            with torch.cuda.amp.autocast(enabled=is_fp16):
                logits, _, _ = model(video, audio)
            preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if not all_labels:
        return {}
    
    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds, zero_division=0),
        "F1-Score": f1_score(all_labels, all_preds, zero_division=0),
        "AUC": roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0,
    }


def measure_inference_time(model, dataloader, device, is_fp16=False, num_batches=10):
    """Measure average inference time per sample."""
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches + 1:
                break
            if batch is None:
                continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            
            if i == 0:  # Warmup
                with torch.cuda.amp.autocast(enabled=is_fp16):
                    _ = model(video, audio)
                continue
            
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            with torch.cuda.amp.autocast(enabled=is_fp16):
                _ = model(video, audio)
            if device == "cuda":
                torch.cuda.synchronize()
            
            latencies.append((time.time() - start) * 1000 / video.size(0))
    
    return np.mean(latencies) if latencies else 0.0


def measure_flops(model, config, device, is_fp16=False):
    """Calculate FLOPs using thop."""
    if profile is None:
        return 0.0, 0.0
    
    model.eval()
    video = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).to(device)
    audio = torch.randn(1, 400, config.NUM_MFCC).to(device)
    mask = torch.ones(1, config.NUM_FRAMES).to(device)
    
    try:
        # FLOPs profiling always runs in FP32 (thop doesn't support autocast)
        flops, params = profile(model, inputs=(video, audio, mask), verbose=False)
        return flops / 1e9, params / 1e6
    except:
        return 0.0, 0.0


# =================================================================================
# 6. ATTENTION-MAP BASED EPS (consistent with EPS-Enhanced.py V2)
# =================================================================================

def calculate_eps(teacher_model, student_model, dataloader, device, model_name, 
                  num_samples=200, is_fp16=False):
    """
    Attention-map-based EPS: compares cross-attention maps between teacher and student.
    Returns the mean EPS score (0.5 * Spearman + 0.5 * IoU).
    """
    teacher_model.eval()
    student_model.eval()
    eps_scores = []
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples:
                break
            if batch is None:
                continue
            
            # Determine each model's device independently
            try:
                t_dev = next(teacher_model.parameters()).device
            except:
                t_dev = DEVICE
            try:
                s_dev = next(student_model.parameters()).device
            except:
                s_dev = 'cpu'
            
            # Teacher attention map (on teacher's device)
            t_video = batch['video'].to(t_dev)
            t_audio = batch['audio'].to(t_dev)
            _, _, attn_T = teacher_model(t_video, t_audio)
            
            # Student attention map (on student's device)
            s_video = batch['video'].to(s_dev)
            s_audio = batch['audio'].to(s_dev)
            
            with torch.cuda.amp.autocast(enabled=is_fp16):
                _, _, attn_S = student_model(s_video, s_audio)
            
            if attn_T is None or attn_S is None:
                continue
            
            attn_T_np = attn_T.cpu().float().numpy()
            attn_S_np = attn_S.cpu().float().numpy()
            
            for i in range(t_video.size(0)):
                if count >= num_samples:
                    break
                m_T = attn_T_np[i].flatten()
                m_S = attn_S_np[i].flatten()
                
                # Ensure same length
                min_len = min(len(m_T), len(m_S))
                m_T, m_S = m_T[:min_len], m_S[:min_len]
                
                # Spearman correlation
                corr = 0.0
                if np.std(m_T) > 1e-9 and np.std(m_S) > 1e-9:
                    corr = spearmanr(m_T, m_S)[0]
                    if np.isnan(corr):
                        corr = 0.0
                
                # IoU of top-20% attended regions
                thresh_T = np.percentile(m_T, 80)
                thresh_S = np.percentile(m_S, 80)
                inter = np.logical_and(m_T > thresh_T, m_S > thresh_S).sum()
                union = np.logical_or(m_T > thresh_T, m_S > thresh_S).sum()
                iou = inter / union if union > 0 else 0.0
                
                # Cosine similarity
                norm_T = np.linalg.norm(m_T)
                norm_S = np.linalg.norm(m_S)
                cosine = np.dot(m_T, m_S) / (norm_T * norm_S) if norm_T > 1e-9 and norm_S > 1e-9 else 0.0
                
                # Composite EPS: 0.5*Spearman + 0.5*IoU
                eps_scores.append(0.5 * corr + 0.5 * iou)
                count += 1
    
    return np.mean(eps_scores) if eps_scores else 0.0


# =================================================================================
# 7. PLOTTING
# =================================================================================

def plot_pareto_curves(df):
    """Generate Pareto frontier plots."""
    colors = plt.cm.tab10(np.linspace(0, 1, len(df)))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Accuracy vs Latency
    for i, row in df.iterrows():
        ax1.scatter(row['Inference (ms)'], row['Accuracy'], c=[colors[i]], s=100, zorder=5)
        ax1.annotate(row['Model'], (row['Inference (ms)'], row['Accuracy']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax1.set_xlabel('Inference Latency (ms)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Pareto: Accuracy vs. Latency')
    ax1.grid(True, alpha=0.3)
    
    # EPS vs Latency
    for i, row in df.iterrows():
        ax2.scatter(row['Inference (ms)'], row['EPS'], c=[colors[i]], s=100, zorder=5)
        ax2.annotate(row['Model'], (row['Inference (ms)'], row['EPS']),
                    textcoords="offset points", xytext=(5, 5), fontsize=8)
    ax2.set_xlabel('Inference Latency (ms)')
    ax2.set_ylabel('EPS')
    ax2.set_title('Pareto: EPS vs. Latency')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Pareto_All_Models_v3.png', dpi=150)
    print("Saved Pareto_All_Models_v3.png")


# =================================================================================
# 8. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE MODEL TESTING V3 — ALL VARIANTS")
    print("="*60)
    print(f"Dataset: {DATASET_DIR}")
    print(f"Device: {DEVICE}")
    print()
    
    # Setup dataset
    config = PinPoint.Config()
    config.DEBUG_MODE = DEBUG_MODE
    
    try:
        test_dataset = MultiPartLAVDFDataset(TEST_DATA_DIRECTORIES, ORIGINAL_METADATA_PATH, config)
        if len(test_dataset) == 0:
            print("CRITICAL: No samples found.")
            sys.exit(1)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,
                                 collate_fn=custom_collate_fn)
    except Exception as e:
        print(f"FATAL: Dataset error: {e}")
        sys.exit(1)
    
    # Load teacher first (for EPS)
    teacher_model = None
    teacher_info = MODEL_REGISTRY.get("Base")
    if teacher_info and os.path.exists(teacher_info["path"]):
        teacher_model, _, _ = load_model("Base", teacher_info)
    
    # Start with pre-populated results from previous runs
    results = list(PREVIOUS_RESULTS)
    print(f"\nLoaded {len(results)} pre-computed results: {[r['Model'] for r in results]}")
    
    for name, info in MODEL_REGISTRY.items():
        if name in SKIP_MODELS:
            print(f"\n\n>>> Skipping: {name} (already evaluated) <<<")
            continue
        print(f"\n\n>>> Processing: {name} <<<")
        
        model, model_config, device = load_model(name, info)
        if model is None:
            print(f"  Skipping {name} (not available)")
            continue
        
        is_fp16 = (info["type"] == "fp16")
        
        # File size
        file_size = os.path.getsize(info["path"]) / (1024 * 1024) if os.path.exists(info["path"]) else 0
        
        # For FP16, report the half-precision size (approximately half)
        if is_fp16:
            file_size = file_size / 2.0
        
        # Metrics  
        metrics = evaluate_model(model, test_loader, device, name, is_fp16=is_fp16)
        if not metrics:
            continue
        
        # Inference time
        inf_time = measure_inference_time(model, test_loader, device, is_fp16=is_fp16)
        
        # Parameters
        try:
            params = sum(p.numel() for p in model.parameters()) / 1e6
        except:
            params = 0
        
        # FLOPs
        flops = 0
        if device != 'cpu' and model_config:
            flops, _ = measure_flops(model, model_config, device, is_fp16=is_fp16)
        
        # Peak memory
        peak_mem = 0
        if torch.cuda.is_available() and device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            try:
                _ = measure_inference_time(model, test_loader, device, is_fp16=is_fp16, num_batches=1)
                peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except:
                pass
        
        # EPS (attention-map-based)
        eps = 0.0
        if name == "Base":
            eps = 1.0
        elif teacher_model is not None:
            eps = calculate_eps(teacher_model, model, test_loader, device, name, 
                              num_samples=EPS_SAMPLES, is_fp16=is_fp16)
        
        row = {
            "Model": name,
            "Size (MB)": round(file_size, 2),
            "Params (M)": round(params, 2),
            "FLOPs (G)": round(flops, 2),
            "Inference (ms)": round(inf_time, 2),
            "Peak VRAM (MB)": round(peak_mem, 2),
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
            "AUC": round(metrics["AUC"], 4),
            "EPS": round(eps, 4),
        }
        results.append(row)
        
        if name != "Base":
            del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Incremental save
        pd.DataFrame(results).to_csv("comprehensive_benchmark_v3_partial.csv", index=False)
        print(f"  Saved partial results ({len(results)} models)")
    
    # Final summary
    print("\n\n" + "="*60)
    print("FINAL BENCHMARK RESULTS V3")
    print("="*60)
    
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        df.to_csv("comprehensive_benchmark_v3.csv", index=False)
        print("\nSaved to comprehensive_benchmark_v3.csv")
        plot_pareto_curves(df)
    
    print("\nDone.")
