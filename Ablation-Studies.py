"""
====================================================================================
PIN-Lite: Ablation Studies (Phase E)
====================================================================================
Configurable ablation script for systematic hyperparameter analysis.
Trains student models with varied settings and logs results.

Supported sweeps:
    --ablation_type temperature  → T ∈ {1.0, 2.0, 4.0, 8.0}
    --ablation_type loss_weights → α ∈ {0.3, 0.5, 0.7}, β ∈ {1.0, 3.0, 5.0}
    --ablation_type layers       → NUM_LAYERS ∈ {1, 2, 3}
    --ablation_type pruning_rate → 0% to 80% in 10% steps

Usage (on Kaggle):
    python Ablation-Studies.py --ablation_type temperature
    python Ablation-Studies.py --ablation_type loss_weights
    python Ablation-Studies.py --ablation_type layers
    python Ablation-Studies.py --ablation_type pruning_rate
====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
import sys
import argparse
import pandas as pd
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# =================================================================================
# 1. MODULE IMPORTS
# =================================================================================
try:
    from PinPoint import (
        Config as TeacherConfig,
        PinpointTransformer as TeacherPinpointTransformer,
        AudioFeatureExtractor, GatedCrossAttentionBlock,
        get_sinusoidal_embeddings,
        LAVDFDataset, collate_fn,
    )
    print("✅ Loaded PinPoint module")
except ImportError:
    print("FATAL: PinPoint module not found.")
    sys.exit(1)

try:
    from Distill import ConfigLite, PinpointTransformerLite, VideoFeatureExtractorLite
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
        VideoFeatureExtractorLite = DP.VideoFeatureExtractorLite
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)


# =================================================================================
# 2. CONFIGURATION
# =================================================================================
TEACHER_MODEL_PATH = "/kaggle/input/pp-xai-full-model-v1/best_pinpoint_model_antisocial.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
STUDENT_MODEL_PATH = "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_LITE_model.pth"


# =================================================================================
# 3. DYNAMIC CONFIG FOR ABLATION
# =================================================================================

class AblationConfig(TeacherConfig):
    """Dynamic config that can be modified at runtime for ablation sweeps."""
    EMBED_DIM = 128
    NUM_HEADS = 4
    NUM_LAYERS = 2
    DROPOUT = 0.15
    EPOCHS = 10  # Shorter for ablation (10 instead of 20)
    LEARNING_RATE = 2e-4
    BATCH_SIZE = 8
    KD_ALPHA = 0.5
    KD_BETA = 5.0
    KD_TEMPERATURE = 2.0
    
    def __init__(self, **overrides):
        super().__init__()
        for k, v in overrides.items():
            setattr(self, k, v)


# =================================================================================
# 4. TRAINING AND EVALUATION FUNCTIONS
# =================================================================================

def train_one_epoch(teacher, student, dataloader, optimizer, scheduler, device, config, scaler):
    """Single training epoch with distillation."""
    teacher.eval()
    student.train()
    total_loss = 0
    correct, total = 0, 0
    
    for batch in dataloader:
        if batch is None:
            continue
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        video_mask = batch['video_mask'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.no_grad():
            t_logits, _, t_attn = teacher(video, audio, video_mask)
        
        with autocast():
            s_logits, _, s_attn = student(video, audio, video_mask)
            
            hard_loss = nn.BCEWithLogitsLoss()(s_logits.squeeze(1), labels)
            T = config.KD_TEMPERATURE
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(s_logits / T, dim=1),
                F.softmax(t_logits / T, dim=1)
            ) * (T * T)
            
            attn_loss = torch.tensor(0.0, device=device)
            if s_attn is not None and t_attn is not None:
                attn_loss = F.mse_loss(s_attn, t_attn)
            
            loss = config.KD_ALPHA * hard_loss + (1 - config.KD_ALPHA) * soft_loss + config.KD_BETA * attn_loss
        
        if torch.isfinite(loss):
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        
        total_loss += loss.item()
        preds = (torch.sigmoid(s_logits) > 0.5).squeeze(1)
        correct += (preds == labels.bool()).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total if total > 0 else 0


def evaluate(model, dataloader, device):
    """Quick evaluation — returns accuracy."""
    model.eval()
    correct, total = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            if batch is None:
                continue
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            labels = batch['label']
            
            logits, _, _ = model(video, audio, video_mask)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).float()
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    acc = accuracy_score(all_labels, all_preds) if total > 0 else 0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except:
        auc = 0.0
    
    return {"accuracy": acc, "f1": f1, "auc": auc}


def run_training(config, teacher, train_loader, dev_loader, test_loader, device, label=""):
    """Full training run with given config, returns test results."""
    print(f"\n--- Training: {label} ---")
    print(f"    T={config.KD_TEMPERATURE}, α={config.KD_ALPHA}, β={config.KD_BETA}, "
          f"Layers={config.NUM_LAYERS}, Epochs={config.EPOCHS}")
    
    student = PinpointTransformerLite(config).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=config.LEARNING_RATE, 
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS * len(train_loader))
    scaler = GradScaler()
    
    best_val_acc = 0
    best_state = None
    
    for epoch in range(config.EPOCHS):
        loss, train_acc = train_one_epoch(
            teacher, student, train_loader, optimizer, scheduler, device, config, scaler)
        val_results = evaluate(student, dev_loader, device)
        
        if val_results["accuracy"] > best_val_acc:
            best_val_acc = val_results["accuracy"]
            best_state = {k: v.clone() for k, v in student.state_dict().items()}
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}: Loss={loss:.4f}, Train Acc={train_acc:.4f}, "
                  f"Val Acc={val_results['accuracy']:.4f}")
    
    # Load best and evaluate on test
    if best_state:
        student.load_state_dict(best_state)
    test_results = evaluate(student, test_loader, device)
    
    params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f"    Final Test: Acc={test_results['accuracy']:.4f}, F1={test_results['f1']:.4f}, "
          f"AUC={test_results['auc']:.4f}, Params={params:.2f}M")
    
    del student
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return {**test_results, "params_M": params, "best_val_acc": best_val_acc}


# =================================================================================
# 5. ABLATION SWEEPS
# =================================================================================

def ablation_temperature(teacher, train_loader, dev_loader, test_loader, device):
    """Sweep distillation temperature T."""
    print("\n" + "="*60)
    print("ABLATION: Distillation Temperature")
    print("="*60)
    
    temperatures = [1.0, 2.0, 4.0, 8.0]
    results = []
    
    for T in temperatures:
        config = AblationConfig(KD_TEMPERATURE=T)
        res = run_training(config, teacher, train_loader, dev_loader, test_loader, device,
                          label=f"T={T}")
        results.append({"Temperature": T, **res})
    
    df = pd.DataFrame(results)
    df.to_csv("ablation_temperature.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    print("Saved to ablation_temperature.csv")
    return df


def ablation_loss_weights(teacher, train_loader, dev_loader, test_loader, device):
    """Sweep α (hard/soft balance) and β (attention loss weight)."""
    print("\n" + "="*60)
    print("ABLATION: Loss Weights (α, β)")
    print("="*60)
    
    alphas = [0.3, 0.5, 0.7]
    betas = [1.0, 3.0, 5.0]
    results = []
    
    for alpha in alphas:
        for beta in betas:
            config = AblationConfig(KD_ALPHA=alpha, KD_BETA=beta)
            res = run_training(config, teacher, train_loader, dev_loader, test_loader, device,
                              label=f"α={alpha}, β={beta}")
            results.append({"Alpha": alpha, "Beta": beta, **res})
    
    df = pd.DataFrame(results)
    df.to_csv("ablation_loss_weights.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    print("Saved to ablation_loss_weights.csv")
    return df


def ablation_layers(teacher, train_loader, dev_loader, test_loader, device):
    """Sweep number of transformer layers."""
    print("\n" + "="*60)
    print("ABLATION: Number of Layers")
    print("="*60)
    
    layer_counts = [1, 2, 3]
    results = []
    
    for n_layers in layer_counts:
        config = AblationConfig(NUM_LAYERS=n_layers)
        res = run_training(config, teacher, train_loader, dev_loader, test_loader, device,
                          label=f"Layers={n_layers}")
        results.append({"Num_Layers": n_layers, **res})
    
    df = pd.DataFrame(results)
    df.to_csv("ablation_layers.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    print("Saved to ablation_layers.csv")
    return df


def ablation_pruning_rate(teacher, train_loader, dev_loader, test_loader, device):
    """Sweep pruning rate from 0% to 80%."""
    print("\n" + "="*60)
    print("ABLATION: Pruning Rate")
    print("="*60)
    
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Student model not found: {STUDENT_MODEL_PATH}")
        return None
    
    pruning_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    results = []
    
    for rate in pruning_rates:
        print(f"\n--- Pruning Rate: {rate*100:.0f}% ---")
        
        config = ConfigLite()
        model = PinpointTransformerLite(config)
        model.load_state_dict(torch.load(STUDENT_MODEL_PATH, map_location=device))
        model.to(device)
        
        if rate > 0:
            # Apply pruning (local layer-wise to ensure correct sparsity)
            prunable = []
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, nn.Conv2d)):
                    prunable.append((module, 'weight'))
            
            if prunable:
                for module, name in prunable:
                    prune.l1_unstructured(module, name, amount=rate)
                
                # Make permanent
                for module, name in prunable:
                    try:
                        prune.remove(module, 'weight')
                    except:
                        pass
            
            # Short finetune (3 epochs)
            model.train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = GradScaler()
            criterion = nn.BCEWithLogitsLoss()
            
            for epoch in range(3):
                for batch in train_loader:
                    if batch is None:
                        continue
                    video = batch['video'].to(device)
                    audio = batch['audio'].to(device)
                    video_mask = batch['video_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    optimizer.zero_grad()
                    with autocast():
                        logits, _, _ = model(video, audio, video_mask)
                        loss = criterion(logits.squeeze(1), labels)
                    if torch.isfinite(loss):
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
        
        # Evaluate
        test_results = evaluate(model, test_loader, device)
        
        # Count non-zero params
        total_params = sum(p.numel() for p in model.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in model.parameters())
        actual_sparsity = 1.0 - nonzero_params / total_params
        
        print(f"    Results: Acc={test_results['accuracy']:.4f}, "
              f"Sparsity={actual_sparsity*100:.1f}%")
        
        results.append({
            "Pruning_Rate": rate,
            "Actual_Sparsity": actual_sparsity,
            "Nonzero_Params_M": nonzero_params / 1e6,
            **test_results
        })
        
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    df = pd.DataFrame(results)
    df.to_csv("ablation_pruning_rate.csv", index=False)
    print(f"\n{df.to_string(index=False)}")
    print("Saved to ablation_pruning_rate.csv")
    return df


# =================================================================================
# 6. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PinLite Ablation Studies")
    parser.add_argument('--ablation_type', type=str, required=True,
                        choices=['temperature', 'loss_weights', 'layers', 'pruning_rate', 'all'],
                        help='Type of ablation study to run')
    args = parser.parse_args()
    
    print("="*60)
    print(f"PIN-LITE ABLATION STUDIES: {args.ablation_type.upper()}")
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
    
    # Setup data
    config = AblationConfig()
    train_dataset = LAVDFDataset(config, split='train')
    dev_dataset = LAVDFDataset(config, split='dev')
    test_dataset = LAVDFDataset(config, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    
    # Run selected ablation
    if args.ablation_type == 'temperature' or args.ablation_type == 'all':
        ablation_temperature(teacher, train_loader, dev_loader, test_loader, DEVICE)
    
    if args.ablation_type == 'loss_weights' or args.ablation_type == 'all':
        ablation_loss_weights(teacher, train_loader, dev_loader, test_loader, DEVICE)
    
    if args.ablation_type == 'layers' or args.ablation_type == 'all':
        ablation_layers(teacher, train_loader, dev_loader, test_loader, DEVICE)
    
    if args.ablation_type == 'pruning_rate' or args.ablation_type == 'all':
        ablation_pruning_rate(teacher, train_loader, dev_loader, test_loader, DEVICE)
    
    print("\n--- Ablation Studies Complete ---")
