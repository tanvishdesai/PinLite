"""
====================================================================================
PIN-Lite: Combined Compression Pipeline (Phase C)
====================================================================================
Implements the full compression pipeline: Distill → Prune → Quantize
This gives the smallest and fastest deployment-ready model.

Usage (on Kaggle):
    python Combined-Pipeline.py

Requires:
    - best_pinpoint_LITE_model.pth (pre-trained distilled student)
    - PinPoint-main.py, Distill-student.py modules
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
        LAVDFDataset, collate_fn, get_sinusoidal_embeddings,
        AudioFeatureExtractor, GatedCrossAttentionBlock,
    )
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
        DP = importlib.util.module_from_spec(spec)
        sys.modules["Distill_PinPoint"] = DP
        spec.loader.exec_module(DP)
        ConfigLite = DP.ConfigLite
        PinpointTransformerLite = DP.PinpointTransformerLite
        print("✅ Loaded Distill-student via file path")
    except Exception as e:
        print(f"FATAL: {e}")
        sys.exit(1)


# =================================================================================
# 2. CONFIGURATION
# =================================================================================
STUDENT_MODEL_PATH = "/kaggle/input/datasets/shivamansari/pinlite-models-v2-2002/best_pinpoint_LITE_model.pth"
TEACHER_MODEL_PATH = "/kaggle/input/datasets/shivamansari/pinlite-models-v2-2002/best_pinpoint_model_antisocial.pth"
OUTPUT_PATH = "best_pinpoint_COMBINED_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Pruning config
PRUNING_ITERATIONS = 3        # Number of prune-finetune cycles
PRUNING_AMOUNT_PER_ITER = 0.2 # Fraction of weights to prune per iteration
FINETUNE_EPOCHS = 3           # Epochs to finetune after each pruning step


# =================================================================================
# 3. STEP 1: LOAD DISTILLED MODEL
# =================================================================================

def load_distilled_model(model_path):
    """Load the pre-trained distilled student model."""
    print("\n" + "="*60)
    print("STEP 1: Loading Distilled Student Model")
    print("="*60)
    
    config = ConfigLite()
    model = PinpointTransformerLite(config)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded student model: {total_params:,} parameters")
    
    return model, config


# =================================================================================
# 4. STEP 2: STRUCTURED PRUNING
# =================================================================================

def get_prunable_layers(model):
    """Get all prunable (Linear, Conv2d) layers from the model."""
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            prunable.append((module, 'weight'))
    return prunable


def apply_pruning(model, amount):
    """Apply global unstructured L1 pruning across all prunable layers."""
    prunable = get_prunable_layers(model)
    if not prunable:
        print("WARNING: No prunable layers found.")
        return model
    
    prune.global_unstructured(
        prunable,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Calculate sparsity
    total_zeros = 0
    total_elements = 0
    for module, name in prunable:
        total_zeros += torch.sum(getattr(module, name) == 0).item()
        total_elements += getattr(module, name).nelement()
    
    sparsity = 100.0 * total_zeros / total_elements
    print(f"  Global sparsity: {sparsity:.1f}%")
    
    return model


def make_pruning_permanent(model):
    """Remove pruning reparameterization and make zeros permanent."""
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass  # Not pruned
    return model


def finetune_after_pruning(model, teacher_model, train_loader, device, config, epochs=3):
    """
    Short finetuning after pruning to recover accuracy.
    
    Uses only hard label loss + KL soft loss (NO attention distillation).
    Attention distillation causes 'backward through graph a second time' because
    nn.MultiheadAttention's returned attention weights are part of the computation
    graph, and MSE loss on them creates a second backward path through the same
    attention nodes — which conflicts with pruning's forward_pre_hooks.
    """
    print(f"  Finetuning for {epochs} epochs to recover accuracy...")
    
    model.train()
    teacher_model.eval()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE * 0.1,  # Lower LR for finetuning
        weight_decay=config.WEIGHT_DECAY
    )
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f"  Finetune E{epoch+1}", leave=False):
            if batch is None:
                continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad(set_to_none=True)
            
            # Teacher forward — only need logits for KL loss
            with torch.no_grad():
                teacher_logits, _, _ = teacher_model(video, audio, video_mask)
                teacher_logits = teacher_logits.detach()
            
            # Student forward — ignore attention map to avoid graph issues
            student_logits, _, _ = model(video, audio, video_mask)
            
            # Loss: hard labels + KL soft distillation (no attention loss)
            hard_loss = criterion(student_logits.squeeze(1), labels)
            T = config.KD_TEMPERATURE
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits / T, dim=1),
                F.softmax(teacher_logits / T, dim=1)
            ) * (T * T)
            
            loss = 0.5 * hard_loss + 0.5 * soft_loss
            
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(student_logits) > 0.5).squeeze(1)
                correct += (preds == labels.bool()).sum().item()
            total += labels.size(0)
        
        acc = correct / total if total > 0 else 0
        print(f"  Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.4f}")
    
    model.eval()
    return model


def prune_model(model, teacher_model, train_loader, device, config):
    """
    Iterative pruning with finetuning.
    
    KEY FIX: PyTorch pruning hooks (forward_pre_hooks) dynamically compute
    weight = weight_orig * weight_mask, creating computation graph nodes that
    conflict with loss.backward(). The fix is to:
      1. Apply pruning (creates hooks + masks)
      2. Make pruning permanent immediately (removes hooks, bakes zeros into weights)
      3. Save a mask of where zeros are
      4. Finetune freely (no hooks = no graph conflicts)
      5. Re-zero any weights that drifted back using the saved mask
    """
    print("\n" + "="*60)
    print("STEP 2: Iterative Structured Pruning")
    print("="*60)
    
    for iteration in range(PRUNING_ITERATIONS):
        print(f"\n--- Pruning Iteration {iteration+1}/{PRUNING_ITERATIONS} ---")
        
        # Step A: Apply pruning (creates hooks + masks)
        model = apply_pruning(model, PRUNING_AMOUNT_PER_ITER)
        
        # Step B: Save the pruning masks before removing hooks
        masks = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                if hasattr(module, 'weight_mask'):
                    masks[name] = module.weight_mask.clone()
        
        # Step C: Make pruning permanent (removes hooks, bakes zeros into weights)
        model = make_pruning_permanent(model)
        
        # Step D: Finetune freely (no hooks = no graph conflicts)
        model = finetune_after_pruning(model, teacher_model, train_loader, device, config, FINETUNE_EPOCHS)
        
        # Step E: Re-apply saved masks to zero out any weights that drifted back
        with torch.no_grad():
            for name, module in model.named_modules():
                if name in masks:
                    module.weight.data *= masks[name].to(module.weight.device)
    
    # Final sparsity report
    total_zeros = 0
    total_elements = 0
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            total_zeros += torch.sum(module.weight == 0).item()
            total_elements += module.weight.nelement()
    print(f"\nFinal global sparsity: {100.0 * total_zeros / total_elements:.1f}%")
    print("Pruning complete.")
    
    return model


# =================================================================================
# 5. STEP 3: QUANTIZATION (Post-Training Dynamic INT8)
# =================================================================================

def quantize_model(model):
    """Apply post-training dynamic quantization to the pruned model."""
    print("\n" + "="*60)
    print("STEP 3: Post-Training Dynamic Quantization (INT8)")
    print("="*60)
    
    model.cpu()
    model.eval()
    
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.GRU},
        dtype=torch.qint8
    )
    
    print("Dynamic quantization applied to Linear and GRU layers.")
    return quantized_model


# =================================================================================
# 6. EVALUATION
# =================================================================================

def evaluate_pipeline_model(model, test_loader, device, model_name):
    """Evaluate a model and return metrics."""
    print(f"\n--- Evaluating {model_name} ---")
    
    model.eval()
    all_preds = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Testing {model_name}"):
            if batch is None:
                continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            labels = batch['label']
            
            start = time.time()
            logits, _, _ = model(video, audio, video_mask)
            latencies.append((time.time() - start) * 1000 / video.size(0))
            
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
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
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC:       {auc:.4f}")
    print(f"  Latency:   {avg_latency:.2f} ms/sample")
    
    return {"Accuracy": acc, "Precision": prec, "Recall": rec, 
            "F1-Score": f1, "AUC": auc, "Latency_ms": avg_latency}


# =================================================================================
# 7. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("PIN-LITE: COMBINED COMPRESSION PIPELINE")
    print("Distill → Prune → Quantize")
    print("="*60)
    
    # Check paths
    if not os.path.exists(STUDENT_MODEL_PATH):
        print(f"Student model not found: {STUDENT_MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(TEACHER_MODEL_PATH):
        print(f"Teacher model not found: {TEACHER_MODEL_PATH}")
        sys.exit(1)
    
    # Step 0: Setup data
    config = ConfigLite()
    train_dataset = LAVDFDataset(config, split='train')
    test_dataset = LAVDFDataset(config, split='test')
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    
    # Load teacher for pruning finetuning
    teacher_config = TeacherConfig()
    teacher = TeacherPinpointTransformer(teacher_config).to(DEVICE)
    teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=DEVICE))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # Step 1: Load distilled model
    model, config = load_distilled_model(STUDENT_MODEL_PATH)
    
    # Evaluate baseline (distilled)
    print("\n--- Baseline (Distilled Only) ---")
    baseline_results = evaluate_pipeline_model(model, test_loader, DEVICE, "Distilled-Baseline")
    
    # Step 2: Prune
    model = prune_model(model, teacher, train_loader, DEVICE, config)
    
    # Evaluate after pruning
    print("\n--- After Pruning ---")
    pruned_results = evaluate_pipeline_model(model, test_loader, DEVICE, "Distilled+Pruned")
    
    # Save pruned model before quantization
    torch.save(model.state_dict(), "best_pinpoint_COMBINED_pruned.pth")
    pruned_size = os.path.getsize("best_pinpoint_COMBINED_pruned.pth") / (1024 * 1024)
    print(f"Pruned model size: {pruned_size:.2f} MB")
    
    # Step 3: Quantize
    quantized_model = quantize_model(model)
    
    # Save quantized model
    torch.save(quantized_model.state_dict(), OUTPUT_PATH)
    final_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    
    # Evaluate final combined model (on CPU since quantized)
    print("\n--- Final Combined Model (Distill+Prune+Quantize) ---")
    final_results = evaluate_pipeline_model(quantized_model, test_loader, 'cpu', "Combined")
    
    # Summary
    print("\n" + "="*60)
    print("PIPELINE RESULTS SUMMARY")
    print("="*60)
    print(f"{'Stage':<25} {'Accuracy':>10} {'Latency (ms)':>15} {'Size (MB)':>12}")
    print("-"*65)
    
    orig_size = os.path.getsize(STUDENT_MODEL_PATH) / (1024 * 1024)
    print(f"{'Distilled (input)':25} {baseline_results['Accuracy']:>10.4f} {baseline_results['Latency_ms']:>15.2f} {orig_size:>12.2f}")
    print(f"{'+ Pruned':25} {pruned_results['Accuracy']:>10.4f} {pruned_results['Latency_ms']:>15.2f} {pruned_size:>12.2f}")
    print(f"{'+ Quantized (final)':25} {final_results['Accuracy']:>10.4f} {final_results['Latency_ms']:>15.2f} {final_size:>12.2f}")
    
    print(f"\nFinal model saved to: {OUTPUT_PATH}")
    print("--- Combined Pipeline Complete ---")
