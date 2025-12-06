"""
EPS-Only Testing Script
========================
This script calculates the Explainability Preservation Score (EPS) by comparing
the ATTENTION MAPS between the Teacher (Base) and Student models.

EPS = 0.5 * Spearman_Correlation(Attn_T, Attn_S) + 0.5 * IoU(Top20%_T, Top20%_S)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import importlib.util
import json
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

# Path to existing benchmark results (for Pareto graphs)
EXISTING_BENCHMARK_CSV = "/kaggle/input/pinlite-all-models-v2-011225/comprehensive_benchmark_results.csv"

# =============================================================================
# 1. CONFIGURATION
# =============================================================================

# Define the source directories (Update for Kaggle if needed)
MODULE_SOURCE_DIR = "/kaggle/input/pinlite-all-models-v2-011225"
MODEL_SOURCE_DIR = "/kaggle/input/pinlite-all-models-v2-011225"

MODEL_PATHS = {
    "Base": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_model_antisocial.pth"),
    "Distilled": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_LITE_model.pth"),
    "Pruned": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_PRUNED_model.pth"),
}

# Data directories
TEST_DATA_DIRECTORIES = [
    "/kaggle/input/la-df-testrin-1",
    "/kaggle/input/lav-df-testing-part-2",
    "/kaggle/input/lav-df-testing-part-3",
    "/kaggle/input/lavdf-testing-part-4"
]
ORIGINAL_METADATA_PATH = "/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
EPS_SAMPLES = 200  # Number of samples for EPS calculation

# =============================================================================
# 2. DYNAMIC MODULE LOADING
# =============================================================================

def import_module_from_path(module_name, file_path):
    """Imports a module from a file path."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

print("--- Loading Modules ---")
if MODULE_SOURCE_DIR not in sys.path:
    sys.path.append(os.path.abspath(MODULE_SOURCE_DIR))

try:
    pinpoint_path = os.path.join(MODULE_SOURCE_DIR, "PinPoint-main.py")
    PinPoint = import_module_from_path("PinPoint", pinpoint_path)
    print("✅ Loaded PinPoint-main.py")

    distill_path = os.path.join(MODULE_SOURCE_DIR, "Distill-student.py")
    Distill_PinPoint = import_module_from_path("Distill_PinPoint", distill_path)
    print("✅ Loaded Distill-student.py")
except Exception as e:
    print(f"❌ Error loading modules: {e}")
    sys.exit(1)

# =============================================================================
# 3. MODEL LOADING
# =============================================================================

def load_base_model(path):
    print(f"Loading Base Model from {path}...")
    config = PinPoint.Config()
    model = PinPoint.PinpointTransformer(config)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def load_lite_model(path):
    print(f"Loading Lite Model from {path}...")
    config = Distill_PinPoint.ConfigLite()
    model = Distill_PinPoint.PinpointTransformerLite(config)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# =============================================================================
# 4. DATASET (Simplified)
# =============================================================================

class MultiPartLAVDFDataset(torch.utils.data.Dataset):
    def __init__(self, directories, metadata_path):
        self.samples = []
        print(f"--- Building test dataset ---")

        with open(metadata_path, 'r') as f:
            original_metadata = json.load(f)

        expected_test_files = {}
        for item in original_metadata:
            if item.get('split') == 'test':
                base_name = os.path.splitext(os.path.basename(item['file']))[0]
                expected_test_files[base_name] = 'fake' if item.get('n_fakes', 0) > 0 else 'real'

        print(f"Found {len(expected_test_files)} expected test files.")

        for data_dir in directories:
            if not os.path.isdir(data_dir):
                continue
            for root, _, files in os.walk(data_dir):
                for file in files:
                    if file.endswith("_video.pt"):
                        base_name = file.replace("_video.pt", "")
                        if base_name in expected_test_files:
                            video_path = os.path.join(root, file)
                            audio_path = os.path.join(root, f"{base_name}_audio.pt")
                            if os.path.exists(audio_path):
                                self.samples.append({
                                    "video_path": video_path,
                                    "audio_path": audio_path,
                                })

        print(f"Loaded {len(self.samples)} test samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            video = torch.load(sample['video_path']).float() / 255.0
            audio = torch.load(sample['audio_path'])
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            video_mask = torch.ones(video.shape[0])
            return {
                "video": video,
                "audio": audio,
                "video_mask": video_mask,
                "label": torch.tensor(0.0, dtype=torch.float),  # Dummy label (not used for EPS)
                "offset_label": 0,  # Dummy offset (not used for EPS)
                "is_fake": False,  # Dummy value (not used for EPS)
                "video_path": sample['video_path']
            }
        except Exception as e:
            return None

# =============================================================================
# 5. CORRECTED EPS CALCULATION (Using Attention Maps)
# =============================================================================

def calculate_eps_attention(teacher_model, student_model, dataloader, device, num_samples=200):
    """
    Calculates EPS by comparing the ATTENTION MAPS from both models.
    This is the correct approach as per the Implementation Plan.
    
    EPS = 0.5 * Spearman_Correlation + 0.5 * IoU(Top20%)
    """
    print(f"  Calculating EPS using Attention Maps on {num_samples} samples...")
    
    teacher_model.eval()
    student_model.eval()
    
    eps_scores = []
    correlations = []
    ious = []
    count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Calculating EPS"):
            if count >= num_samples:
                break
            if batch is None:
                continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            
            # Get attention maps from both models
            _, _, teacher_attn = teacher_model(video, audio, video_mask)
            _, _, student_attn = student_model(video, audio, video_mask)
            
            if teacher_attn is None or student_attn is None:
                continue
            
            # Resize student attention to match teacher if different
            # Attention maps shape: (batch, audio_len, video_len)
            if teacher_attn.shape != student_attn.shape:
                # Interpolate student to teacher's shape
                student_attn = F.interpolate(
                    student_attn.unsqueeze(1),  # Add channel dim
                    size=teacher_attn.shape[1:],
                    mode='bilinear',
                    align_corners=False
                ).squeeze(1)
            
            batch_size = video.size(0)
            for i in range(batch_size):
                if count >= num_samples:
                    break
                
                map_T = teacher_attn[i].cpu().numpy().flatten()
                map_S = student_attn[i].cpu().numpy().flatten()
                
                # A. Spearman Correlation
                if np.std(map_T) > 1e-9 and np.std(map_S) > 1e-9:
                    corr, _ = spearmanr(map_T, map_S)
                    if np.isnan(corr):
                        corr = 0.0
                else:
                    corr = 0.0
                
                # B. IoU of Top 20%
                threshold_T = np.percentile(map_T, 80)
                threshold_S = np.percentile(map_S, 80)
                
                mask_T = map_T > threshold_T
                mask_S = map_S > threshold_S
                
                intersection = np.logical_and(mask_T, mask_S).sum()
                union = np.logical_or(mask_T, mask_S).sum()
                iou = intersection / union if union > 0 else 0.0
                
                # Combined EPS
                eps = 0.5 * corr + 0.5 * iou
                eps_scores.append(eps)
                correlations.append(corr)
                ious.append(iou)
                count += 1
    
    return {
        "EPS": np.mean(eps_scores) if eps_scores else 0.0,
        "Correlation": np.mean(correlations) if correlations else 0.0,
        "IoU": np.mean(ious) if ious else 0.0,
        "Samples": count
    }

# =============================================================================
# 6. PARETO GRAPH GENERATION
# =============================================================================

def generate_pareto_graphs(eps_results_df):
    """
    Generates Pareto curves by merging new EPS results with existing benchmark data.
    Creates: Accuracy vs Latency and EPS vs Latency plots.
    """
    print("\n--- Generating Pareto Graphs ---")
    
    # Load existing benchmark data
    if not os.path.exists(EXISTING_BENCHMARK_CSV):
        print(f"⚠️ Existing benchmark CSV not found: {EXISTING_BENCHMARK_CSV}")
        print("   Skipping Pareto graph generation.")
        return
    
    benchmark_df = pd.read_csv(EXISTING_BENCHMARK_CSV)
    print(f"Loaded existing benchmark data with {len(benchmark_df)} models.")
    
    # Create a mapping of new EPS values (model name -> EPS)
    eps_mapping = {}
    for _, row in eps_results_df.iterrows():
        model_name = row['Model'].replace(" (Teacher)", "")  # Normalize name
        eps_mapping[model_name] = row['EPS']
    
    # Update the benchmark DataFrame with corrected EPS values
    def get_corrected_eps(model_name):
        if model_name in eps_mapping:
            return eps_mapping[model_name]
        return 0.0  # Default for models not in EPS results (e.g., Quantized)
    
    benchmark_df['EPS_Corrected'] = benchmark_df['Model'].apply(get_corrected_eps)
    
    # Filter out Quantized model for cleaner Pareto (optional, since it underperforms)
    plot_df = benchmark_df[benchmark_df['Model'] != 'Quantized'].copy()
    
    # Color scheme
    colors = {'Base': '#2ecc71', 'Distilled': '#3498db', 'Pruned': '#e74c3c'}
    
    # ===========================================
    # Pareto 1: Accuracy vs Latency
    # ===========================================
    plt.figure(figsize=(10, 7))
    
    for _, row in plot_df.iterrows():
        model = row['Model']
        color = colors.get(model, '#95a5a6')
        plt.scatter(row['Inference (ms/sample)'], row['Accuracy'], 
                   s=200, c=color, label=model, edgecolors='black', linewidth=1.5, zorder=5)
        plt.annotate(f"{model}\n({row['Accuracy']:.2%})", 
                    (row['Inference (ms/sample)'], row['Accuracy']),
                    textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')
    
    plt.xlabel('Inference Latency (ms/sample)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Pareto Frontier: Accuracy vs. Latency', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    
    # Add a shaded "better" region
    plt.fill_between([0, 100], [1.0, 1.0], [0.95, 0.95], alpha=0.1, color='green')
    
    plt.tight_layout()
    plt.savefig('Pareto_Accuracy_vs_Latency_v2.png', dpi=150)
    print("✅ Saved 'Pareto_Accuracy_vs_Latency_v2.png'")
    plt.close()
    
    # ===========================================
    # Pareto 2: EPS vs Latency
    # ===========================================
    plt.figure(figsize=(10, 7))
    
    for _, row in plot_df.iterrows():
        model = row['Model']
        color = colors.get(model, '#95a5a6')
        eps_val = row['EPS_Corrected']
        plt.scatter(row['Inference (ms/sample)'], eps_val, 
                   s=200, c=color, label=model, edgecolors='black', linewidth=1.5, zorder=5)
        plt.annotate(f"{model}\n(EPS={eps_val:.2f})", 
                    (row['Inference (ms/sample)'], eps_val),
                    textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')
    
    plt.xlabel('Inference Latency (ms/sample)', fontsize=12)
    plt.ylabel('Explainability Preservation Score (EPS)', fontsize=12)
    plt.title('Pareto Frontier: EPS vs. Latency', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='lower right', fontsize=10)
    plt.ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig('Pareto_EPS_vs_Latency_v2.png', dpi=150)
    print("✅ Saved 'Pareto_EPS_vs_Latency_v2.png'")
    plt.close()
    
    # ===========================================
    # Save Updated Benchmark CSV with Corrected EPS
    # ===========================================
    benchmark_df.to_csv('comprehensive_benchmark_results_v2.csv', index=False)
    print("✅ Saved 'comprehensive_benchmark_results_v2.csv' with corrected EPS values")

# =============================================================================
# 7. MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("EPS-ONLY TESTING (Attention Map Comparison)")
    print("=" * 60)
    
    # Load Dataset
    test_dataset = MultiPartLAVDFDataset(TEST_DATA_DIRECTORIES, ORIGINAL_METADATA_PATH)
    if len(test_dataset) == 0:
        print("CRITICAL: No samples found.")
        sys.exit(1)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=PinPoint.collate_fn
    )
    
    # Load Teacher Model
    print("\n--- Loading Teacher (Base) Model ---")
    teacher_model = load_base_model(MODEL_PATHS["Base"])
    
    # Results storage
    results = []
    
    # Base model EPS = 1.0 by definition (self-similarity)
    results.append({
        "Model": "Base",
        "EPS": 1.0,
        "Correlation": 1.0,
        "IoU": 1.0,
        "Samples": EPS_SAMPLES
    })
    
    # Calculate EPS for each student model
    for name in ["Distilled", "Pruned"]:
        print(f"\n--- Processing: {name} ---")
        path = MODEL_PATHS[name]
        
        if not os.path.exists(path):
            print(f"  ⚠️ Model not found: {path}")
            continue
        
        student_model = load_lite_model(path)
        
        eps_result = calculate_eps_attention(
            teacher_model, student_model, test_loader, DEVICE, EPS_SAMPLES
        )
        
        results.append({
            "Model": name,
            "EPS": round(eps_result["EPS"], 4),
            "Correlation": round(eps_result["Correlation"], 4),
            "IoU": round(eps_result["IoU"], 4),
            "Samples": eps_result["Samples"]
        })
        
        del student_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Print Final Results
    print("\n" + "=" * 60)
    print("EPS RESULTS (Attention Map Comparison)")
    print("=" * 60)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    # Save to CSV
    df.to_csv("eps_results.csv", index=False)
    print("\n✅ Results saved to 'eps_results.csv'")
    
    # Generate Pareto Graphs
    generate_pareto_graphs(df)
    
    print("\n" + "=" * 60)
    print("ALL DONE!")
    print("=" * 60)

