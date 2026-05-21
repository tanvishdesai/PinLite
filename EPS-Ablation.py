"""
====================================================================================
EPS Weight Sensitivity Ablation
====================================================================================
Evaluates the Explainability Preservation Score (EPS) with different hyperparameter 
settings to demonstrate metric robustness.

Variables ablated:
- w1: Weight for Spearman Correlation
- w2: Weight for Top-k% IoU
- k: Threshold percentage for active attention regions

Usage (on Kaggle):
    python EPS-Ablation.py
    
Outputs:
    - eps_ablation_results.csv
    - EPS_Ablation_Plot.png
====================================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# =================================================================================
# 1. MODULE LOADING AND CONFIGURATION
# =================================================================================

# --- Unified dataset path (models only) ---
DATASET_DIR = "/kaggle/input/datasets/shivamansari/pinlite-models-v2-2002"

# Test data (4-part LAV-DF dataset as used in Comprehensive-Testing)
TEST_DATA_DIRECTORIES = [
    "/kaggle/input/datasets/shivamansari/la-df-testrin-1",
    "/kaggle/input/datasets/shivamansari/lav-df-testing-part-2",
    "/kaggle/input/datasets/shivamansari/lav-df-testing-part-3",
    "/kaggle/input/datasets/shivamansari/lavdf-testing-part-4"
]
ORIGINAL_METADATA_PATH = "/kaggle/input/datasets/elin75/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json"

print("--- Loading Core Modules ---")
try:
    import PinPoint
    import Distill as Distill_PinPoint
except ImportError as e:
    print(f"❌ Error loading core modules: {e}")
    sys.exit(1)

try:
    import Attentionvariants as AV
    HAS_ATTENTION_VARIANTS = True
except ImportError as e:
    HAS_ATTENTION_VARIANTS = False

MODEL_REGISTRY = {
    "Base": {"path": os.path.join(DATASET_DIR, "best_pinpoint_model_antisocial.pth"), "type": "base", "device": "auto"},
    "Distilled": {"path": os.path.join(DATASET_DIR, "best_pinpoint_LITE_model.pth"), "type": "lite", "device": "auto"},
    "Pruned": {"path": os.path.join(DATASET_DIR, "best_pinpoint_PRUNED_model.pth"), "type": "lite", "device": "auto"},
    "MQA": {"path": os.path.join(DATASET_DIR, "best_pinpoint_MQA_model.pth"), "type": "mqa", "device": "auto"},
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPS_SAMPLES = 300

# =================================================================================
# 2. DATASET
# =================================================================================
import json

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
            if not os.path.isdir(data_dir): continue
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
    
    def __len__(self): return len(self.samples)
    
    def __getitem__(self, idx):
        s = self.samples[idx]
        try:
            video = torch.load(s['video_path']).float() / 255.0
            audio = torch.load(s['audio_path'])
            if audio.dim() == 1: audio = audio.unsqueeze(0)
            
            num_frames = getattr(self.config, 'NUM_FRAMES', 30)
            t = video.shape[0]
            if t > num_frames:
                indices = torch.linspace(0, t - 1, num_frames).long()
                video = video[indices]
            elif t < num_frames:
                pad = torch.zeros(num_frames - t, *video.shape[1:])
                video = torch.cat([video, pad], dim=0)
            
            mask = torch.ones(video.shape[0])
            return {"video": video, "audio": audio, "video_mask": mask, "label": torch.tensor(s['label'], dtype=torch.float)}
        except Exception as e: return None

def custom_collate_fn(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    
    max_t = max(b['video'].shape[0] for b in batch)
    videos, audios, masks, labels = [], [], [], []
    for b in batch:
        v = b['video']
        t = v.shape[0]
        if t < max_t:
            pad = torch.zeros(max_t - t, *v.shape[1:])
            v = torch.cat([v, pad], dim=0)
        mask = torch.zeros(max_t); mask[:t] = 1.0
        videos.append(v); audios.append(b['audio']); masks.append(mask); labels.append(b['label'])
    
    max_a = max(a.shape[0] for a in audios)
    padded_audios = []
    for a in audios:
        if a.shape[0] < max_a:
            pad = torch.zeros(max_a - a.shape[0], *a.shape[1:])
            a = torch.cat([a, pad], dim=0)
        padded_audios.append(a)
    
    return {'video': torch.stack(videos), 'audio': torch.stack(padded_audios), 'video_mask': torch.stack(masks), 'label': torch.stack(labels)}

# =================================================================================
# 3. MODEL LOADING AND ABLATED EPS CALCULATION
# =================================================================================

def load_model(name, info):
    path = info["path"]
    model_type = info["type"]
    target_device = info["device"] if info["device"] != "auto" else DEVICE
    if not os.path.exists(path): return None, None, target_device
    print(f"Loading {name} from {path}...")
    try:
        if model_type == "base": model, config = PinPoint.PinpointTransformer(PinPoint.Config()), PinPoint.Config()
        elif model_type == "lite": model, config = Distill_PinPoint.PinpointTransformerLite(Distill_PinPoint.ConfigLite()), Distill_PinPoint.ConfigLite()
        elif model_type == "mqa" and HAS_ATTENTION_VARIANTS: model, config = AV.PinpointTransformerVariant(AV.ConfigMQA()), AV.ConfigMQA()
        else: return None, None, target_device
        
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=False), strict=False)
        model.to(target_device)
        model.eval()
        return model, config, target_device
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return None, None, target_device

def calculate_ablated_eps(teacher_model, student_model, dataloader, device, 
                          w1=0.5, w2=0.5, k=20, num_samples=200):
    teacher_model.eval()
    student_model.eval()
    eps_scores = []
    count = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if count >= num_samples: break
            if batch is None: continue
            
            t_vid, t_aud = batch['video'].to(device), batch['audio'].to(device)
            _, _, attn_T = teacher_model(t_vid, t_aud)
            _, _, attn_S = student_model(t_vid, t_aud)
            
            if attn_T is None or attn_S is None: continue
            
            attn_T_np = attn_T.cpu().float().numpy()
            attn_S_np = attn_S.cpu().float().numpy()
            
            for i in range(t_vid.size(0)):
                if count >= num_samples: break
                m_T, m_S = attn_T_np[i].flatten(), attn_S_np[i].flatten()
                min_len = min(len(m_T), len(m_S))
                m_T, m_S = m_T[:min_len], m_S[:min_len]
                
                # Spearman
                corr = spearmanr(m_T, m_S)[0] if np.std(m_T) > 1e-9 and np.std(m_S) > 1e-9 else 0.0
                if np.isnan(corr): corr = 0.0
                
                # Top-k% IoU
                thresh_T = np.percentile(m_T, 100 - k)
                thresh_S = np.percentile(m_S, 100 - k)
                inter = np.logical_and(m_T > thresh_T, m_S > thresh_S).sum()
                union = np.logical_or(m_T > thresh_T, m_S > thresh_S).sum()
                iou = inter / union if union > 0 else 0.0
                
                eps_scores.append(w1 * corr + w2 * iou)
                count += 1
                
    return np.mean(eps_scores) if eps_scores else 0.0

# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("EPS HYPERPARAMETER SENSITIVITY ABLATION")
    print("="*60)
    
    config = PinPoint.Config()
    try:
        test_dataset = MultiPartLAVDFDataset(TEST_DATA_DIRECTORIES, ORIGINAL_METADATA_PATH, config)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    except Exception as e:
        print(f"FATAL: Dataset error: {e}")
        sys.exit(1)
        
    teacher_model, _, _ = load_model("Base", MODEL_REGISTRY["Base"])
    if teacher_model is None:
        print("CRITICAL: Base (Teacher) model required for EPS computation not found.")
        sys.exit(1)
        
    student_models = {}
    for name in ["Distilled", "Pruned", "MQA"]:
        if name in MODEL_REGISTRY:
            m, _, _ = load_model(name, MODEL_REGISTRY[name])
            if m is not None:
                student_models[name] = m
                
    print(f"\nLoaded {len(student_models)} student models for ablation.")
    
    ablation_settings = [
        {"w1": 0.5, "w2": 0.5, "k": 20}, # Baseline
        {"w1": 0.8, "w2": 0.2, "k": 20}, # Spearman heavy
        {"w1": 0.2, "w2": 0.8, "k": 20}, # IoU heavy
        {"w1": 0.6, "w2": 0.4, "k": 20},
        {"w1": 0.4, "w2": 0.6, "k": 20},
        {"w1": 0.5, "w2": 0.5, "k": 10}, # Strict top-10%
        {"w1": 0.5, "w2": 0.5, "k": 30}, # Relaxed top-30%
    ]
    
    results = []
    
    for settings in ablation_settings:
        w1, w2, k = settings["w1"], settings["w2"], settings["k"]
        print(f"\nEvaluating EPS with w1={w1}, w2={w2}, k={k}%...")
        
        for name, student in student_models.items():
            score = calculate_ablated_eps(teacher_model, student, test_loader, DEVICE, 
                                          w1=w1, w2=w2, k=k, num_samples=EPS_SAMPLES)
            print(f"  {name} EPS: {score:.4f}")
            results.append({
                "Model": name,
                "w1 (Spearman)": w1,
                "w2 (IoU)": w2,
                "k (Top-%)": k,
                "EPS": round(score, 4)
            })
            
    df = pd.DataFrame(results)
    print("\nFINAL EPS ABLATION RESULTS")
    print(df.to_string(index=False))
    df.to_csv("eps_ablation_results.csv", index=False)
    print("Saved to eps_ablation_results.csv")
    
    # Plotting EPS Stability
    plt.figure(figsize=(12, 6))
    groups = df.groupby('Model')
    
    x_labels = [f"w1:{w1}\nw2:{w2}\nk:{k}%" for w1, w2, k in zip(df[df['Model']=='Distilled']['w1 (Spearman)'], 
                                                               df[df['Model']=='Distilled']['w2 (IoU)'], 
                                                               df[df['Model']=='Distilled']['k (Top-%)'])]
    x_positions = np.arange(len(x_labels))
    width = 0.25
    
    for i, (name, group) in enumerate(groups):
        plt.bar(x_positions + i*width, group['EPS'].values, width=width, label=name)
        
    plt.title('EPS Stability Across Hyperparameters ($w_1, w_2, k$)')
    plt.xlabel('Hyperparameter Settings')
    plt.ylabel('Explainability Preservation Score (EPS)')
    plt.xticks(x_positions + width, x_labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('EPS_Ablation_Plot.png', dpi=150)
    print("Saved EPS_Ablation_Plot.png")
    
    print("\nDone.")
