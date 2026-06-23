"""
====================================================================================
PIN-Lite: Distillation Training for Attention Variants (Phase B)
====================================================================================
Trains each attention variant student model using knowledge distillation from
the teacher (PinPoint). Same KD setup as the original Distill-student.py but
with configurable attention mechanism.

Usage (on Kaggle):
    python Distill-Attention-Variants.py --variant linear
    python Distill-Attention-Variants.py --variant mqa
    python Distill-Attention-Variants.py --variant lowrank

Or set TEST_ONLY_MODE = True and provide a checkpoint path to skip training.
====================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import time
import os
import sys
import argparse
from tqdm import tqdm

# =================================================================================
# 1. MODULE IMPORTS
# =================================================================================
try:
    from PinPoint import (
        Config as TeacherConfig,
        PinpointTransformer as TeacherPinpointTransformer,
        LAVDFDataset,
        collate_fn,
        evaluate_model,
        test_and_evaluate,
    )
    print("✅ Loaded PinPoint module")
except ImportError:
    print("FATAL: PinPoint module not found.")
    sys.exit(1)

try:
    from Attentionvariants import (
        ConfigLinear, ConfigMQA, ConfigLowRank,
        PinpointTransformerVariant,
    )
    print("✅ Loaded Attention_Variants module")
except ImportError:
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("Attention_Variants",
            os.path.join(os.path.dirname(__file__), "Attention-Variants.py"))
        AV = importlib.util.module_from_spec(spec)
        sys.modules["Attention_Variants"] = AV
        spec.loader.exec_module(AV)
        ConfigLinear = AV.ConfigLinear
        ConfigMQA = AV.ConfigMQA
        ConfigLowRank = AV.ConfigLowRank
        PinpointTransformerVariant = AV.PinpointTransformerVariant
        print("✅ Loaded Attention-Variants via file path")
    except Exception as e:
        print(f"FATAL: Could not load Attention-Variants module: {e}")
        sys.exit(1)


# =================================================================================
# 2. CONFIGURATION
# =================================================================================

TEACHER_MODEL_PATH = "/kaggle/input/pp-xai-full-model-v1/best_pinpoint_model_antisocial.pth"

# Maps variant name to (config class, output filename)
VARIANT_MAP = {
    "linear": (ConfigLinear, "best_pinpoint_LINEAR_model.pth"),
    "mqa": (ConfigMQA, "best_pinpoint_MQA_model.pth"),
    "lowrank": (ConfigLowRank, "best_pinpoint_LOWRANK_model.pth"),
}

# Set to True to skip training and only evaluate
TEST_ONLY_MODE = False

# If TEST_ONLY_MODE, specify the model checkpoint path here
VARIANT_MODEL_PATH_FOR_TESTING = ""


# =================================================================================
# 3. DISTILLATION TRAINING LOOP (same as original, works with any variant)
# =================================================================================

def train_distillation_epoch(teacher_model, student_model, dataloader, optimizer, 
                             scheduler, loss_fns, device, config, epoch, scaler):
    """Knowledge distillation training epoch — identical to Distill-student.py."""
    teacher_model.eval()
    student_model.train()
    total_loss, total_hard, total_soft, total_attn = 0, 0, 0, 0
    correct, total = 0, 0
    progress_bar = tqdm(dataloader, desc=f"Distilling E{epoch+1}", leave=False)

    for batch in progress_bar:
        if batch is None:
            continue
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        video_mask = batch['video_mask'].to(device)
        cls_labels = batch['label'].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            teacher_cls_logits, _, teacher_attn_map = teacher_model(video, audio, video_mask)

        with autocast():
            student_cls_logits, _, student_attn_map = student_model(video, audio, video_mask)

            # Hard loss
            hard_loss = loss_fns['classification'](student_cls_logits.squeeze(1), cls_labels)

            # Soft loss (KL divergence on softened logits)
            T = config.KD_TEMPERATURE
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_cls_logits / T, dim=1),
                F.softmax(teacher_cls_logits / T, dim=1)
            ) * (T * T)

            # Attention distillation loss
            if student_attn_map is not None and teacher_attn_map is not None:
                attn_loss = F.mse_loss(student_attn_map, teacher_attn_map)
            else:
                attn_loss = torch.tensor(0.0, device=device)

            combined_loss = (config.KD_ALPHA * hard_loss + 
                           (1 - config.KD_ALPHA) * soft_loss + 
                           config.KD_BETA * attn_loss)

        if not torch.isfinite(combined_loss):
            print("WARNING: Non-finite loss, skipping batch.")
            continue

        scaler.scale(combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += combined_loss.item()
        total_hard += hard_loss.item()
        total_soft += soft_loss.item()
        total_attn += attn_loss.item()
        preds = (torch.sigmoid(student_cls_logits) > 0.5).squeeze(1)
        correct += (preds == cls_labels.bool()).sum().item()
        total += cls_labels.size(0)

        progress_bar.set_postfix({
            "Loss": f"{combined_loss.item():.4f}",
            "Acc": f"{correct/total:.2f}"
        })

    n = len(dataloader)
    return total_loss/n, total_hard/n, total_soft/n, total_attn/n, correct/total

def test_and_evaluate_variant(model_path, test_loader, config, variant_class):
    from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
    from tqdm import tqdm
    
    print("\n" + "="*50)
    print("--- Starting Final Model Evaluation on Test Set ---")
    print(f"Loading variant model from: {model_path}")

    device = config.DEVICE
    model = variant_class(config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_labels, all_preds = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            if batch is None: continue
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            labels = batch['label']

            with autocast():
                cls_logits, _, _ = model(video, audio, video_mask)

            preds = (torch.sigmoid(cls_logits) > 0.5).squeeze(1).cpu().numpy().astype(int)
            all_labels.extend(labels.numpy().astype(int))
            all_preds.extend(preds)

    if not all_labels: return

    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=['Real', 'Fake'], zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print("\n--- Final Test Results ---")
    print(f"Total Samples Tested: {len(all_labels)}")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall:    {recall:.4f}")
    print(f"  - F1-Score:  {f1:.4f}")
    print("\n--- Classification Report ---")
    print(report)
    print("\n--- Confusion Matrix ---")
    print(cm)
    print("="*50 + "\n")


# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train attention variant via distillation")
    parser.add_argument('--variant', type=str, default='linear',
                        choices=['linear', 'mqa', 'lowrank'],
                        help='Attention variant to train')
    parser.add_argument('--test_only', action='store_true', default=TEST_ONLY_MODE,
                        help='Skip training, only evaluate')
    parser.add_argument('--model_path', type=str, default=VARIANT_MODEL_PATH_FOR_TESTING,
                        help='Path to pre-trained variant model (for test_only mode)')
    args = parser.parse_args()

    variant = args.variant
    ConfigClass, output_filename = VARIANT_MAP[variant]
    config = ConfigClass()
    teacher_config = TeacherConfig()
    device = config.DEVICE

    print("=" * 60)
    print(f"KNOWLEDGE DISTILLATION — {variant.upper()} ATTENTION VARIANT")
    print("=" * 60)
    print(f"Attention Type: {variant}")
    print(f"Student: EMBED_DIM={config.EMBED_DIM}, HEADS={config.NUM_HEADS}, LAYERS={config.NUM_LAYERS}")
    print(f"Device: {device}")

    # Setup data
    print("\n--- [1/5] Setting up datasets ---")
    train_dataset = LAVDFDataset(config, split='train')
    dev_dataset = LAVDFDataset(config, split='dev')
    test_dataset = LAVDFDataset(config, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
                             collate_fn=collate_fn, num_workers=2)
    print(f"Loaded {len(train_dataset)} train, {len(dev_dataset)} dev, {len(test_dataset)} test samples.")

    if args.test_only:
        # ---- TEST ONLY MODE ----
        print(f"\n--- TEST ONLY MODE ---")
        model_path = args.model_path if args.model_path else os.path.join(".", output_filename)
        if not os.path.exists(model_path):
            print(f"FATAL: Model not found at {model_path}")
            sys.exit(1)
        
        student = PinpointTransformerVariant(config)
        student.load_state_dict(torch.load(model_path, map_location=device))
        student.to(device)
        student.eval()
        
        test_and_evaluate_variant(
            model_path=model_path,
            test_loader=test_loader,
            config=config,
            variant_class=PinpointTransformerVariant
        )
    else:
        # ---- FULL TRAINING ----
        print("\n--- [2/5] Loading Teacher Model ---")
        teacher = TeacherPinpointTransformer(teacher_config).to(device)
        if not os.path.exists(TEACHER_MODEL_PATH):
            print(f"FATAL: Teacher model not found at {TEACHER_MODEL_PATH}")
            sys.exit(1)
        teacher.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device))
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad = False
        print("Teacher loaded and frozen.")

        print(f"\n--- [3/5] Initializing {variant.upper()} Student ---")
        student = PinpointTransformerVariant(config).to(device)
        
        # Print param count
        total_params = sum(p.numel() for p in student.parameters())
        trainable_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

        loss_fns = {'classification': nn.BCEWithLogitsLoss()}
        optimizer = torch.optim.AdamW(student.parameters(), lr=config.LEARNING_RATE, 
                                       weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.EPOCHS * len(train_loader)
        )
        scaler = GradScaler()

        print(f"\n--- [4/5] Training ({config.EPOCHS} epochs) ---")
        start_time = time.time()
        best_val_loss = float('inf')

        for epoch in range(config.EPOCHS):
            print(f"\n===== Epoch {epoch+1}/{config.EPOCHS} =====")
            train_loss, h, s, a, train_acc = train_distillation_epoch(
                teacher, student, train_loader, optimizer, scheduler,
                loss_fns, device, config, epoch, scaler
            )
            print(f"Train -> Loss: {train_loss:.4f} (H:{h:.2f}, S:{s:.2f}, A:{a:.2f}), Acc: {train_acc:.4f}")

            val_loss, val_acc = evaluate_model(student, dev_loader, loss_fns, device, config)
            print(f"Val   -> Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(student.state_dict(), output_filename)
                print(f"  -> New best model saved: {output_filename}")

        elapsed = (time.time() - start_time) / 60
        print(f"\n--- Training complete in {elapsed:.2f} minutes ---")

        # Final test
        print(f"\n--- [5/5] Testing best {variant.upper()} model ---")
        if os.path.exists(output_filename):
            test_and_evaluate_variant(
                model_path=output_filename,
                test_loader=test_loader,
                config=config,
                variant_class=PinpointTransformerVariant
            )

    print(f"\n--- {variant.upper()} variant script complete ---")
