"""
====================================================================================
PIN-Lite: Enhanced EPS (Explainability Preservation Score) — Phase D
====================================================================================
Strengthened EPS calculation with:
1. Bootstrap confidence intervals (95% CI via 1000 resamples)
2. Weight ablation (w1 ∈ {0.3, 0.5, 0.7} for Spearman vs IoU)
3. Alternative metrics: Cosine similarity of saliency maps (inspired by e²KD)
4. Support for quantized/attention-map-based EPS (gradient-free fallback)
5. Larger sample size support (500+)

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
try:
    from PinPoint import (
        Config as TeacherConfig,
        PinpointTransformer as TeacherPinpointTransformer,
        LAVDFDataset, collate_fn,
    )
    print("✅ Loaded PinPoint module")
except ImportError:
    print("FATAL: PinPoint module not found.")
    sys.exit(1)

try:
    from Distill import ConfigLite, PinpointTransformerLite
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
        print(f"Warning: Distill_PinPoint not available: {e}")


# =================================================================================
# 2. CONFIGURATION
# =================================================================================
TEACHER_MODEL_PATH = "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_model_antisocial.pth"
STUDENT_MODEL_PATHS = {
    "Distilled": "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_LITE_model.pth",
    "Pruned": "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_PRUNED_model.pth",
    # Add more model paths here as they become available:
    # "LinearAttn": "best_pinpoint_LINEAR_model.pth",
    # "MQA": "best_pinpoint_MQA_model.pth",
    # "LowRank": "best_pinpoint_LOWRANK_model.pth",
    # "Combined": "best_pinpoint_COMBINED_model.pth",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS_SAMPLES = 500  # Increased from 200 for better statistical reliability
BOOTSTRAP_ITERATIONS = 1000
WEIGHT_ABLATIONS = [0.3, 0.5, 0.7]  # w1 values (w2 = 1 - w1)
OUTPUT_CSV = "eps_enhanced_results.csv"


# =================================================================================
# 3. SALIENCY MAP EXTRACTION
# =================================================================================

def get_gradient_saliency(model, video, audio, device):
    """
    Compute gradient-based saliency map: |input × gradient|.
    Uses the video input as the primary attribution target.
    """
    old_cudnn = torch.backends.cudnn.enabled
    torch.backends.cudnn.enabled = False
    
    try:
        model_device = next(model.parameters()).device
        v = video.to(model_device).detach().requires_grad_(True)
        a = audio.to(model_device).detach().requires_grad_(True)
        
        model.zero_grad()
        logits, _, _ = model(v, a)
        target_score = logits.max()
        target_score.backward()
        
        if v.grad is None:
            return None
        
        # Saliency = |input × gradient|
        saliency = (v * v.grad).abs()
        return saliency.detach().cpu().numpy()
    except Exception as e:
        return None
    finally:
        torch.backends.cudnn.enabled = old_cudnn


def get_attention_map_saliency(model, video, audio, device):
    """
    Fallback for quantized models: use attention maps directly as saliency.
    This doesn't require gradients.
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
        
        return attn_map.detach().cpu().numpy()


# =================================================================================
# 4. ENHANCED EPS CALCULATION
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
    """Cosine similarity between saliency maps (alternative to Spearman)."""
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


def calculate_enhanced_eps(teacher_model, student_model, dataloader, device, 
                           model_name, num_samples=500, use_attention_fallback=False):
    """
    Enhanced EPS calculation.
    
    Returns a dict with:
    - EPS for each weight combination (w1=0.3, 0.5, 0.7)
    - Spearman correlation (mean + 95% CI)
    - IoU (mean + 95% CI)  
    - Cosine similarity (mean + 95% CI)
    - Combined EPS (mean + 95% CI)
    """
    print(f"\n  Computing Enhanced EPS for {model_name} ({num_samples} samples)...")
    
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
        
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        
        # Get teacher saliency (always gradient-based)
        saliency_T = get_gradient_saliency(teacher_model, video, audio, device)
        
        # Get student saliency
        if use_attention_fallback:
            # For quantized models, use attention maps
            saliency_S = get_attention_map_saliency(student_model, video, audio, device)
            # Also get teacher attention for fair comparison
            saliency_T_attn = get_attention_map_saliency(teacher_model, video, audio, device)
            if saliency_T_attn is not None:
                saliency_T = saliency_T_attn
        else:
            saliency_S = get_gradient_saliency(student_model, video, audio, device)
        
        if saliency_T is None or saliency_S is None:
            continue
        
        batch_size = video.size(0)
        for i in range(batch_size):
            if count >= num_samples:
                break
            
            # Flatten maps
            map_T = saliency_T[i].flatten()
            map_S = saliency_S[i].flatten()
            
            # Ensure same length (in case of attention map fallback)
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
    
    # EPS for different weight combinations
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
    print(f"\n  {model_name} Enhanced EPS Results:")
    print(f"    Spearman: {spearman_mean:.4f} [{spearman_lo:.4f}, {spearman_hi:.4f}]")
    print(f"    IoU:      {iou_mean:.4f} [{iou_lo:.4f}, {iou_hi:.4f}]")
    print(f"    Cosine:   {cosine_mean:.4f} [{cosine_lo:.4f}, {cosine_hi:.4f}]")
    for w1 in WEIGHT_ABLATIONS:
        k = f"EPS_w{w1}"
        print(f"    EPS(w1={w1}): {results[k]:.4f} [{results[k+'_CI_lower']:.4f}, {results[k+'_CI_upper']:.4f}]")
    print(f"    CosineEPS: {cosine_eps_mean:.4f} [{cosine_eps_lo:.4f}, {cosine_eps_hi:.4f}]")
    
    return results


# =================================================================================
# 5. MODEL LOADING HELPERS
# =================================================================================

def load_model(model_name, model_path, device):
    """Load a model by name."""
    if model_name in ["Distilled", "Pruned"]:
        config = ConfigLite()
        model = PinpointTransformerLite(config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        return model, False  # use_attention_fallback = False
    
    # For attention variants
    try:
        from Attention_Variants import PinpointTransformerVariant, ConfigLinear, ConfigMQA, ConfigLowRank
        variant_configs = {
            "LinearAttn": ConfigLinear,
            "MQA": ConfigMQA, 
            "LowRank": ConfigLowRank,
        }
        if model_name in variant_configs:
            config = variant_configs[model_name]()
            model = PinpointTransformerVariant(config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, False
    except ImportError:
        pass
    
    # For combined/quantized models (try dynamic quantized loading)
    if model_name in ["Combined", "Quantized-Student"]:
        config = ConfigLite()
        model = PinpointTransformerLite(config)
        # Try loading as dynamic quantized
        try:
            model = torch.quantization.quantize_dynamic(model, {nn.Linear, nn.GRU}, dtype=torch.qint8)
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            return model, True  # use_attention_fallback = True
        except:
            # Try as regular model
            model = PinpointTransformerLite(config)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model, True
    
    print(f"WARNING: Unknown model type '{model_name}', attempting standard loading...")
    config = ConfigLite()
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, False


# =================================================================================
# 6. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIN-LITE: ENHANCED EPS ANALYSIS")
    print("="*60)
    
    # Load teacher
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"Teacher model not found: {TEACHER_MODEL_PATH}")
        sys.exit(1)
    
    teacher_config = TeacherConfig()
    teacher = TeacherPinpointTransformer(teacher_config).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    print("Teacher model loaded.")
    
    # Load test data
    config = ConfigLite()
    test_dataset = LAVDFDataset(config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn, num_workers=2)
    
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
            model, use_fallback = load_model(model_name, model_path, DEVICE)
            result = calculate_enhanced_eps(
                teacher, model, test_loader, DEVICE,
                model_name, num_samples=EPS_SAMPLES,
                use_attention_fallback=use_fallback
            )
            if result:
                all_results.append(result)
            
            # Cleanup
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"\nERROR processing {model_name}: {e}")
    
    # Save results
    if all_results:
        df = pd.DataFrame(all_results)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n\nResults saved to {OUTPUT_CSV}")
        print("\n" + df.to_string(index=False))
    
    print("\n--- Enhanced EPS Analysis Complete ---")
