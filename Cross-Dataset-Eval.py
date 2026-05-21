"""
====================================================================================
Cross-Dataset Evaluation on FakeAVCeleb
====================================================================================
Evaluates the PinPoint models on the FakeAVCeleb dataset.
This uses the unified metadata preprocessing format.

Usage (on Kaggle):
    python Cross-Dataset-Eval.py
    
Outputs:
    - cross_dataset_benchmark.csv
    - Cross_Dataset_Pareto.png
====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T
import numpy as np
import os
import sys
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
# 1. MODULE LOADING AND CONFIGURATION
# =================================================================================

# --- Unified dataset path (models only) ---
DATASET_DIR = "/kaggle/input/datasets/shivamansari/pinlite-models-v2-2002"

# --- FakeAVCeleb Data paths (MUST BE UPDATED BY USER) ---
FAV_DATA_DIRECTORY = "/kaggle/input/fav-pre-process/preprocessed_data"
FAV_METADATA_PATH = "/kaggle/input/fav-pre-process/preprocessed_data/preprocessed_metadata_test_only_20250908_142055.json"

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

# All models from the single dataset directory
MODEL_REGISTRY = {
    "Base": {"path": os.path.join(DATASET_DIR, "best_pinpoint_model_antisocial.pth"), "type": "base", "device": "auto"},
    "Distilled": {"path": os.path.join(DATASET_DIR, "best_pinpoint_LITE_model.pth"), "type": "lite", "device": "auto"},
    "Pruned": {"path": os.path.join(DATASET_DIR, "best_pinpoint_PRUNED_model.pth"), "type": "lite", "device": "auto"},
    "FP16": {"path": os.path.join(DATASET_DIR, "best_pinpoint_LITE_model.pth"), "type": "fp16", "device": "cuda"},
    "Linear-Attn": {"path": os.path.join(DATASET_DIR, "best_pinpoint_LINEAR_model.pth"), "type": "linear", "device": "auto"},
    "MQA": {"path": os.path.join(DATASET_DIR, "best_pinpoint_MQA_model.pth"), "type": "mqa", "device": "auto"},
    "LowRank": {"path": os.path.join(DATASET_DIR, "best_pinpoint_LOWRANK_model.pth"), "type": "lowrank", "device": "auto"},
    # Exclude combined for this eval to save time, or keep it if desired
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8

# =================================================================================
# 2. DATASET (FakeAVCeleb)
# =================================================================================

class FakeAVCelebDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_path, data_dir, config, split='test'):
        self.config = config
        self.split = split
        self.samples = []
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Unified metadata file not found at: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            all_metadata = json.load(f)
            
        split_metadata = [item for item in all_metadata if item.get('split') == self.split]
        print(f"Found {len(split_metadata)} total entries for split '{self.split}' in FakeAVCeleb metadata.")
        
        for item in split_metadata:
            video_path = os.path.join(data_dir, item['preprocessed_video_path'])
            audio_path = os.path.join(data_dir, item['preprocessed_audio_path'])

            if os.path.exists(video_path) and os.path.exists(audio_path):
                is_fake = (item['label'] == 'fake')
                label_value = 1.0 if is_fake else 0.0
                self.samples.append({
                    "video_path": video_path,
                    "audio_path": audio_path,
                    "label": label_value,
                })

        print(f"Successfully loaded {len(self.samples)} FakeAVCeleb samples.")
        self.normalize_transform = T.Normalize(mean=self.config.NORM_MEAN, std=self.config.NORM_STD)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        try:
            sample = self.samples[idx]
            video = torch.load(sample['video_path']).float() / 255.0
            audio = torch.load(sample['audio_path'])
            
            # Robust audio handling
            if audio.ndim == 3:
                if audio.shape[0] == self.config.NUM_MFCC:
                    audio = audio.mean(dim=1).transpose(0, 1)
                elif audio.shape[1] == self.config.NUM_MFCC:
                    audio = audio.mean(dim=0).transpose(0, 1)
                elif audio.shape[2] == self.config.NUM_MFCC:
                    audio = audio.mean(dim=1)
                else: return None
            elif audio.ndim == 2:
                if audio.shape[0] == self.config.NUM_MFCC:
                    audio = audio.transpose(0, 1)
                elif audio.shape[1] != self.config.NUM_MFCC:
                    return None
            else: return None
            
            video = self.normalize_transform(video)
            
            num_frames = getattr(self.config, 'NUM_FRAMES', 30)
            t = video.shape[0]
            if t > num_frames:
                indices = torch.linspace(0, t - 1, num_frames).long()
                video = video[indices]
            elif t < num_frames:
                pad = torch.zeros(num_frames - t, *video.shape[1:])
                video = torch.cat([video, pad], dim=0)

            mask = torch.ones(video.shape[0])
            return {
                "video": video, "audio": audio, "video_mask": mask,
                "label": torch.tensor(sample['label'], dtype=torch.float),
            }
        except Exception as e:
            return None

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
        mask = torch.zeros(max_t)
        mask[:t] = 1.0
        videos.append(v)
        audios.append(b['audio'])
        masks.append(mask)
        labels.append(b['label'])
    
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
# 3. MODEL LOADING AND EVALUATION
# =================================================================================

def load_model(name, info):
    path = info["path"]
    model_type = info["type"]
    target_device = info["device"] if info["device"] != "auto" else DEVICE
    
    if not os.path.exists(path): return None, None, target_device
    print(f"Loading {name} from {path}...")
    try:
        if model_type == "base":
            config = PinPoint.Config()
            model = PinPoint.PinpointTransformer(config)
        elif model_type in ["lite", "fp16"]:
            config = Distill_PinPoint.ConfigLite()
            model = Distill_PinPoint.PinpointTransformerLite(config)
        elif model_type in ["linear", "mqa", "lowrank"] and HAS_ATTENTION_VARIANTS:
            config_map = {"linear": AV.ConfigLinear, "mqa": AV.ConfigMQA, "lowrank": AV.ConfigLowRank}
            config = config_map[model_type]()
            model = AV.PinpointTransformerVariant(config)
        else: return None, None, target_device
        
        state = torch.load(path, map_location='cpu', weights_only=False)
        if model_type == "fp16":
            model.load_state_dict({k: v.float() for k, v in state.items()})
        else:
            model.load_state_dict(state, strict=False)
            
        model.to(target_device)
        model.eval()
        return model, config, target_device
    except Exception as e:
        print(f"  Error loading {name}: {e}")
        return None, None, target_device

def evaluate_model(model, dataloader, device, model_name, is_fp16=False):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Eval {model_name}", leave=False):
            if batch is None: continue
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label']
            with torch.cuda.amp.autocast(enabled=is_fp16):
                logits, _, _ = model(video, audio)
            preds = (torch.sigmoid(logits.float()) > 0.5).squeeze(1).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    if not all_labels: return None
    return {
        "Accuracy": accuracy_score(all_labels, all_preds),
        "Precision": precision_score(all_labels, all_preds, zero_division=0),
        "Recall": recall_score(all_labels, all_preds, zero_division=0),
        "F1-Score": f1_score(all_labels, all_preds, zero_division=0),
        "AUC": roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else 0.0,
    }

# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("CROSS-DATASET EVALUATION ON FAKEAVCELEB")
    print("="*60)
    
    config = PinPoint.Config()
    try:
        test_dataset = FakeAVCelebDataset(FAV_METADATA_PATH, FAV_DATA_DIRECTORY, config)
        if len(test_dataset) == 0:
            print("CRITICAL: No samples found. Please check FakeAVCeleb paths.")
            sys.exit(1)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    except Exception as e:
        print(f"FATAL: Dataset error: {e}")
        sys.exit(1)
    
    results = []
    
    for name, info in MODEL_REGISTRY.items():
        print(f"\n>>> Processing: {name} <<<")
        model, model_config, device = load_model(name, info)
        if model is None: continue
        
        is_fp16 = (info["type"] == "fp16")
        metrics = evaluate_model(model, test_loader, device, name, is_fp16=is_fp16)
        if metrics is None: continue
        
        row = {
            "Model": name,
            "FakeAVCeleb_Acc": round(metrics["Accuracy"], 4),
            "FakeAVCeleb_F1": round(metrics["F1-Score"], 4),
            "FakeAVCeleb_AUC": round(metrics["AUC"], 4),
        }
        results.append(row)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    if not df.empty:
        print("\nFINAL CROSS-DATASET BENCHMARK RESULTS")
        print(df.to_string(index=False))
        df.to_csv("cross_dataset_benchmark.csv", index=False)
        print("Saved to cross_dataset_benchmark.csv")
        
        # Plot Simple Bar Chart
        plt.figure(figsize=(10, 6))
        bars = plt.bar(df['Model'], df['FakeAVCeleb_Acc'], color='skyblue')
        plt.title('Cross-Dataset Accuracy on FakeAVCeleb')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 3), ha='center', va='bottom', fontsize=9)
        plt.tight_layout()
        plt.savefig('Cross_Dataset_Accuracy.png', dpi=150)
        print("Saved Cross_Dataset_Accuracy.png")

    print("\nDone.")
