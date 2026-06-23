"""
====================================================================================
PIN-Lite: Enhanced EPS (Explainability Preservation Score) — Phase D (V2)
====================================================================================
Attention-Map-Based EPS — compares cross-attention maps between teacher and
student models. This is the correct methodology because:
  1. All models (teacher + students) output attention maps from cross-attention
  2. Attention maps ARE the explainability mechanism (where the model looks)
  3. The distillation loss already trains for attention map alignment (β weight)
  4. Architecture-agnostic: works regardless of backbone (ResNet vs MobileNet)

Metrics computed:
  - Spearman rank correlation of flattened attention maps
  - IoU of top-20% attended regions
  - Cosine similarity of attention maps
  - Composite EPS with weight ablation (w1 ∈ {0.3, 0.5, 0.7})
  - Bootstrap 95% confidence intervals on all metrics

Usage (on Kaggle):
    python EPS-Enhanced.py

Outputs:
    - eps_enhanced_results.csv (per-model EPS with all variants)
====================================================================================
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import os
import sys
import time
from tqdm import tqdm
from scipy.stats import spearmanr
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# =================================================================================
# 1. MODULE IMPORTS
# =================================================================================

# --- Unified dataset path (models only) ---
DATASET_DIR = "/kaggle/input/datasets/shivamansari/pinlite-models-v2-2002"

# --- Module imports ---
# Modules are written to /kaggle/working/ via %%writefile magic:
#   PinPoint.py, Distill.py, Attentionvariants.py

try:
    from PinPoint import (
        Config as TeacherConfig,
        PinpointTransformer as TeacherPinpointTransformer,
        LAVDFDataset, collate_fn,
    )
    print("✅ Loaded PinPoint module")
except ImportError as e:
    print(f"FATAL: Could not load PinPoint: {e}")
    sys.exit(1)

try:
    from Distill import ConfigLite, PinpointTransformerLite
    print("✅ Loaded Distill module")
except ImportError as e:
    print(f"FATAL: Could not load Distill: {e}")
    sys.exit(1)

try:
    from Attentionvariants import (
        PinpointTransformerVariant,
        ConfigLinear, ConfigMQA, ConfigLowRank,
    )
    print("✅ Loaded Attentionvariants module")
    HAS_ATTENTION_VARIANTS = True
except ImportError as e:
    print(f"⚠️ Attention variants not available: {e}")
    HAS_ATTENTION_VARIANTS = False


# =================================================================================
# 2. CONFIGURATION
# =================================================================================
TEACHER_MODEL_PATH = os.path.join(DATASET_DIR, "best_pinpoint_model_antisocial.pth")
STUDENT_MODEL_PATHS = {
    "Distilled":  os.path.join(DATASET_DIR, "best_pinpoint_LITE_model.pth"),
    "Pruned":     os.path.join(DATASET_DIR, "best_pinpoint_PRUNED_model.pth"),
    "LinearAttn": os.path.join(DATASET_DIR, "best_pinpoint_LINEAR_model.pth"),
    "MQA":        os.path.join(DATASET_DIR, "best_pinpoint_MQA_model.pth"),
    "LowRank":    os.path.join(DATASET_DIR, "best_pinpoint_LOWRANK_model.pth"),
    "Combined":   os.path.join(DATASET_DIR, "best_pinpoint_COMBINED_model.pth"),
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS_SAMPLES = 500       # Number of samples for EPS computation
BOOTSTRAP_ITERATIONS = 1000
WEIGHT_ABLATIONS = [0.3, 0.5, 0.7]  # w1 values (w2 = 1 - w1)
OUTPUT_CSV = "eps_enhanced_results.csv"


# =================================================================================
# 3. ATTENTION-MAP SALIENCY EXTRACTION
# =================================================================================

def get_attention_map(model, video, audio, device):
    """
    Extract the cross-attention map from a model's forward pass.
    All PinPoint models (teacher + all student variants) return 
    (logits, offset_logits, last_attention_map) from their forward() method.
    
    The attention map represents WHERE the model attends — this is the core
    explainability signal in PinPoint's architecture.
    """
    model.eval()
    with torch.no_grad():
        try:
            model_device = next(model.parameters()).device
        except StopIteration:
            model_device = 'cpu'
        
        v = video.to(model_device)
        a = audio.to(model_device)
        _, _, attn_map = model(v, a)
        
        if attn_map is None:
            return None
        
        return attn_map.detach().cpu().float().numpy()


# =================================================================================
# 4. METRIC FUNCTIONS
# =================================================================================

def compute_spearman_correlation(map_T, map_S):
    """Spearman rank correlation between two flattened maps."""
    if np.std(map_T) > 1e-9 and np.std(map_S) > 1e-9:
        corr, _ = spearmanr(map_T, map_S)
        if np.isnan(corr):
            return 0.0
        return corr
    return 0.0


def compute_iou_top_k(map_T, map_S, percentile=80):
    """IoU of top-k% salient regions."""
    threshold_T = np.percentile(map_T, percentile)
    threshold_S = np.percentile(map_S, percentile)
    
    mask_T = map_T > threshold_T
    mask_S = map_S > threshold_S
    
    intersection = np.logical_and(mask_T, mask_S).sum()
    union = np.logical_or(mask_T, mask_S).sum()
    
    return intersection / union if union > 0 else 0.0


def compute_cosine_similarity(map_T, map_S):
    """Cosine similarity between attention maps."""
    norm_T = np.linalg.norm(map_T)
    norm_S = np.linalg.norm(map_S)
    
    if norm_T < 1e-9 or norm_S < 1e-9:
        return 0.0
    
    return np.dot(map_T, map_S) / (norm_T * norm_S)


def bootstrap_confidence_interval(scores, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for the mean of scores."""
    if len(scores) == 0:
        return 0.0, 0.0, 0.0
    
    scores = np.array(scores)
    bootstrap_means = []
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.sort(bootstrap_means)
    alpha = 1 - ci
    lower_idx = int(alpha / 2 * n_bootstrap)
    upper_idx = int((1 - alpha / 2) * n_bootstrap)
    
    return np.mean(scores), bootstrap_means[lower_idx], bootstrap_means[upper_idx]


# =================================================================================
# 5. ENHANCED EPS CALCULATION (ATTENTION-MAP BASED)
# =================================================================================

def calculate_enhanced_eps(teacher_model, student_model, dataloader, device, 
                           model_name, num_samples=500):
    """
    Enhanced EPS calculation using ATTENTION MAPS.
    
    Compares the cross-attention maps between teacher and student models.
    This measures whether the compressed model preserves WHERE the original
    model attends — the core explainability signal.
    
    Returns a dict with:
    - EPS for each weight combination (w1=0.3, 0.5, 0.7)
    - Spearman correlation (mean + 95% CI)
    - IoU (mean + 95% CI)  
    - Cosine similarity (mean + 95% CI)
    - Combined EPS (mean + 95% CI)
    """
    print(f"\n  Computing Attention-Based EPS for {model_name} ({num_samples} samples)...")
    
    teacher_model.eval()
    student_model.eval()
    
    # Collect per-sample metrics
    spearman_scores = []
    iou_scores = []
    cosine_scores = []
    count = 0
    
    for batch in tqdm(dataloader, desc=f"  EPS-{model_name}", leave=False):
        if count >= num_samples:
            break
        if batch is None:
            continue
        
        video = batch['video']
        audio = batch['audio']
        
        # Get teacher attention map
        attn_T = get_attention_map(teacher_model, video, audio, device)
        
        # Get student attention map
        attn_S = get_attention_map(student_model, video, audio, device)
        
        if attn_T is None or attn_S is None:
            continue
        
        batch_size = video.size(0)
        for i in range(batch_size):
            if count >= num_samples:
                break
            
            # Flatten maps
            map_T = attn_T[i].flatten()
            map_S = attn_S[i].flatten()
            
            # Ensure same length (teacher may have different attention dimensions)
            min_len = min(len(map_T), len(map_S))
            map_T = map_T[:min_len]
            map_S = map_S[:min_len]
            
            # Compute metrics
            spearman = compute_spearman_correlation(map_T, map_S)
            iou = compute_iou_top_k(map_T, map_S)
            cosine = compute_cosine_similarity(map_T, map_S)
            
            spearman_scores.append(spearman)
            iou_scores.append(iou)
            cosine_scores.append(cosine)
            
            count += 1
    
    if count == 0:
        print(f"  WARNING: No valid samples for {model_name}")
        return {}
    
    # Bootstrap confidence intervals
    spearman_mean, spearman_lo, spearman_hi = bootstrap_confidence_interval(
        spearman_scores, BOOTSTRAP_ITERATIONS)
    iou_mean, iou_lo, iou_hi = bootstrap_confidence_interval(
        iou_scores, BOOTSTRAP_ITERATIONS)
    cosine_mean, cosine_lo, cosine_hi = bootstrap_confidence_interval(
        cosine_scores, BOOTSTRAP_ITERATIONS)
    
    # EPS for different weight combinations (Spearman + IoU)
    eps_variants = {}
    for w1 in WEIGHT_ABLATIONS:
        w2 = 1 - w1
        eps_values = [w1 * s + w2 * i for s, i in zip(spearman_scores, iou_scores)]
        eps_mean, eps_lo, eps_hi = bootstrap_confidence_interval(eps_values, BOOTSTRAP_ITERATIONS)
        eps_variants[f"EPS_w{w1}"] = eps_mean
        eps_variants[f"EPS_w{w1}_CI_lower"] = eps_lo
        eps_variants[f"EPS_w{w1}_CI_upper"] = eps_hi
    
    # Cosine-based EPS alternative (w1*cosine + w2*iou)
    cosine_eps_values = [0.5 * c + 0.5 * i for c, i in zip(cosine_scores, iou_scores)]
    cosine_eps_mean, cosine_eps_lo, cosine_eps_hi = bootstrap_confidence_interval(
        cosine_eps_values, BOOTSTRAP_ITERATIONS)
    
    results = {
        "Model": model_name,
        "Num_Samples": count,
        
        # Raw metrics with CIs
        "Spearman_Mean": spearman_mean,
        "Spearman_CI_Lower": spearman_lo,
        "Spearman_CI_Upper": spearman_hi,
        
        "IoU_Mean": iou_mean,
        "IoU_CI_Lower": iou_lo,
        "IoU_CI_Upper": iou_hi,
        
        "Cosine_Mean": cosine_mean,
        "Cosine_CI_Lower": cosine_lo,
        "Cosine_CI_Upper": cosine_hi,
        
        # EPS variants
        **eps_variants,
        
        # Cosine-based alternative
        "CosineEPS_Mean": cosine_eps_mean,
        "CosineEPS_CI_Lower": cosine_eps_lo,
        "CosineEPS_CI_Upper": cosine_eps_hi,
    }
    
    # Print summary
    print(f"\n  {model_name} Attention-Based EPS Results:")
    print(f"    Spearman: {spearman_mean:.4f} [{spearman_lo:.4f}, {spearman_hi:.4f}]")
    print(f"    IoU:      {iou_mean:.4f} [{iou_lo:.4f}, {iou_hi:.4f}]")
    print(f"    Cosine:   {cosine_mean:.4f} [{cosine_lo:.4f}, {cosine_hi:.4f}]")
    for w1 in WEIGHT_ABLATIONS:
        k = f"EPS_w{w1}"
        print(f"    EPS(w1={w1}): {results[k]:.4f} [{results[k+'_CI_lower']:.4f}, {results[k+'_CI_upper']:.4f}]")
    print(f"    CosineEPS: {cosine_eps_mean:.4f} [{cosine_eps_lo:.4f}, {cosine_eps_hi:.4f}]")
    
    return results


# =================================================================================
# 6. MODEL LOADING HELPERS
# =================================================================================

def load_model(model_name, model_path, device):
    """Load a model by name. Returns (model, use_cpu_flag)."""
    
    # --- Standard distilled / pruned student ---
    if model_name in ["Distilled", "Pruned"]:
        config = ConfigLite()
        model = PinpointTransformerLite(config)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False), strict=False)
        model.to(device)
        model.eval()
        return model
    
    # --- Attention variants ---
    if model_name in ["LinearAttn", "MQA", "LowRank"] and HAS_ATTENTION_VARIANTS:
        variant_configs = {
            "LinearAttn": ConfigLinear,
            "MQA": ConfigMQA, 
            "LowRank": ConfigLowRank,
        }
        config = variant_configs[model_name]()
        model = PinpointTransformerVariant(config)
        model.load_state_dict(
            torch.load(model_path, map_location=device, weights_only=False), strict=False)
        model.to(device)
        model.eval()
        return model
    
    # --- Combined pipeline (dynamically quantized) ---
    if model_name in ["Combined"]:
        config = ConfigLite()
        model = PinpointTransformerLite(config)
        # Apply dynamic quantization to match saved state dict structure
        model = torch.quantization.quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.GRU},
            dtype=torch.qint8
        )
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint, strict=False)
        model.to('cpu')  # Dynamically quantized layers must stay on CPU
        model.eval()
        return model
    
    # --- Fallback ---
    print(f"WARNING: Unknown model type '{model_name}', attempting standard loading...")
    config = ConfigLite()
    model = PinpointTransformerLite(config)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=False), strict=False)
    model.to(device)
    model.eval()
    return model


# =================================================================================
# 7. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIN-LITE: ATTENTION-BASED ENHANCED EPS ANALYSIS (V2)")
    print("="*60)
    print(f"Method: Cross-attention map comparison (architecture-agnostic)")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Device: {DEVICE}")
    print(f"Samples per model: {EPS_SAMPLES}")
    print()
    
    # Load teacher
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"Teacher model not found: {TEACHER_MODEL_PATH}")
        sys.exit(1)
    
    teacher_config = TeacherConfig()
    teacher = TeacherPinpointTransformer(teacher_config).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE, weights_only=False))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher model loaded.")
    
    # Load test data
    config = ConfigLite()
    test_dataset = LAVDFDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, 
                             collate_fn=collate_fn, num_workers=2)
    
    # Evaluate each student model
    all_results = []
    
    # Teacher self-similarity (EPS = 1.0 by definition)
    teacher_result = {
        "Model": "Base (Teacher)",
        "Num_Samples": EPS_SAMPLES,
        "Spearman_Mean": 1.0, "Spearman_CI_Lower": 1.0, "Spearman_CI_Upper": 1.0,
        "IoU_Mean": 1.0, "IoU_CI_Lower": 1.0, "IoU_CI_Upper": 1.0,
        "Cosine_Mean": 1.0, "Cosine_CI_Lower": 1.0, "Cosine_CI_Upper": 1.0,
    }
    for w1 in WEIGHT_ABLATIONS:
        teacher_result[f"EPS_w{w1}"] = 1.0
        teacher_result[f"EPS_w{w1}_CI_lower"] = 1.0
        teacher_result[f"EPS_w{w1}_CI_upper"] = 1.0
    teacher_result["CosineEPS_Mean"] = 1.0
    teacher_result["CosineEPS_CI_Lower"] = 1.0
    teacher_result["CosineEPS_CI_Upper"] = 1.0
    all_results.append(teacher_result)
    
    for model_name, model_path in STUDENT_MODEL_PATHS.items():
        if not os.path.exists(model_path):
            print(f"\nSkipping {model_name}: {model_path} not found.")
            continue
        
        try:
            model = load_model(model_name, model_path, DEVICE)
            result = calculate_enhanced_eps(
                teacher, model, test_loader, DEVICE,
                model_name, num_samples=EPS_SAMPLES,
            )
            if result:
                all_results.append(result)
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            import traceback
            print(f"\nERROR processing {model_name}: {e}")
            print(traceback.format_exc())
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n\nResults saved to {OUTPUT_CSV}")
        print("\n" + df.to_string(index=False))
    
    print("\n--- Attention-Based EPS Analysis Complete ---")
