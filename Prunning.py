%%writefile Prune_PinPoint.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import time
import os
from tqdm import tqdm

# --- IMPORT FROM YOUR PREVIOUS SCRIPTS ---
# This script relies on the architectures and data loaders you've already defined.
# Make sure PinPoint.py and Distill_PinPoint.py are in the same directory.
try:
    from PinPoint import (
        LAVDFDataset,
        collate_fn,
        test_and_evaluate  # We reuse the final testing function
    )
    from Distill_PinPoint import (
        ConfigLite,
        PinpointTransformerLite  # We are pruning the LITE model
    )
    print("Successfully imported components from PinPoint.py and Distill_PinPoint.py")
except ImportError as e:
    print(f"FATAL ERROR: Could not import necessary components. {e}")
    print("Please ensure Prune_PinPoint.py is in the same directory as PinPoint.py and Distill_PinPoint.py")
    exit()

# =================================================================================
# 1. PRUNING CONFIGURATION
# =================================================================================
class ConfigPrune(ConfigLite):
    """Inherits from the Lite config and adds pruning-specific parameters."""
    # --- Pruning Strategy ---
    # We will prune one attention head per iteration.
    PRUNING_ITERATIONS = 2 # Prune a total of 2 heads (e.g., from 4 heads down to 2)
    FINETUNE_EPOCHS = 3    # Epochs to fine-tune after each pruning step

    # --- Fine-Tuning Hyperparameters ---
    FINETUNE_LR = 1e-5     # Use a smaller learning rate for fine-tuning

# =================================================================================
# 2. PRUNING HELPER FUNCTIONS (CORRECTED)
# =================================================================================

def find_least_important_head(model: PinpointTransformerLite, config: ConfigPrune):
    """
    Identifies the attention head with the lowest L1-norm across all layers.
    This head is considered the least important and is the best candidate for pruning.
    
    --- FIX ---
    This function now correctly handles the fused `in_proj_weight` parameter
    in PyTorch's MultiheadAttention module.
    """
    head_l1_norms = torch.zeros(config.NUM_HEADS, device=config.DEVICE)
    head_dim = config.EMBED_DIM // config.NUM_HEADS

    for layer in model.gated_attention_layers:
        attn_module = layer.audio_to_video_attn
        
        # Access the fused weight matrix for Q, K, V projections
        if not hasattr(attn_module, 'in_proj_weight'):
             continue # Skip if the weight doesn't exist

        # The shape is (3 * embed_dim, embed_dim). We chunk it into Q, K, V parts.
        q_weights, k_weights, v_weights = attn_module.in_proj_weight.chunk(3)

        for i in range(config.NUM_HEADS):
            start_row = i * head_dim
            end_row = start_row + head_dim

            # A head's importance is the sum of L1 norms of the weight matrix *rows*
            # that produce its output in Q, K, and V projections.
            q_head_weights = q_weights[start_row:end_row, :]
            k_head_weights = k_weights[start_row:end_row, :]
            v_head_weights = v_weights[start_row:end_row, :]

            head_l1_norms[i] += torch.norm(q_head_weights, p=1) + \
                                torch.norm(k_head_weights, p=1) + \
                                torch.norm(v_head_weights, p=1)

    least_important_head_idx = torch.argmin(head_l1_norms).item()
    print(f"Head L1 Norms: {[f'{n:.2f}' for n in head_l1_norms.detach().cpu().numpy()]}")
    print(f"Identified head {least_important_head_idx} as least important.")
    return least_important_head_idx


def apply_structured_head_pruning(model: PinpointTransformerLite, head_index: int, config: ConfigPrune):
    """
    Applies structured pruning to a specific attention head across all transformer layers.
    It creates a custom mask for the fused `in_proj_weight` parameter.
    
    --- FIX ---
    This function now correctly creates a mask for the `in_proj_weight` and
    applies it directly to the MultiheadAttention module.
    """
    head_dim = config.EMBED_DIM // config.NUM_HEADS
    start_row = head_index * head_dim
    end_row = start_row + head_dim

    print(f"Applying structured pruning to head {head_index} (rows {start_row} to {end_row} in each Q,K,V block)...")

    for layer_idx, layer in enumerate(model.gated_attention_layers):
        attn_module = layer.audio_to_video_attn

        # Create a mask of all ones for the fused weight matrix
        mask = torch.ones_like(attn_module.in_proj_weight)

        # Zero out the rows corresponding to the target head in each section (Q, K, V)
        # Q section rows
        mask[start_row:end_row, :] = 0.0
        # K section rows
        mask[config.EMBED_DIM + start_row : config.EMBED_DIM + end_row, :] = 0.0
        # V section rows
        mask[2 * config.EMBED_DIM + start_row : 2 * config.EMBED_DIM + end_row, :] = 0.0

        # Apply the custom mask directly to the 'in_proj_weight' parameter of the module
        prune.custom_from_mask(attn_module, name='in_proj_weight', mask=mask)
        print(f"  - Pruned head {head_index} in layer {layer_idx}")

def calculate_sparsity(model: nn.Module):
    """Calculates the global sparsity of a model."""
    total_params = 0
    zero_params = 0
    # Include all relevant layers for sparsity calculation
    for module in model.modules():
        if hasattr(module, "weight") and module.weight is not None:
            total_params += module.weight.nelement()
            zero_params += torch.sum(module.weight == 0)
        # Handle the fused weight in MultiheadAttention
        if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
            total_params += module.in_proj_weight.nelement()
            zero_params += torch.sum(module.in_proj_weight == 0)

    if total_params == 0:
        return 0.0
    sparsity = 100. * float(zero_params) / float(total_params)
    return sparsity

def make_pruning_permanent(model: nn.Module):
    """
    Physically removes the pruned weights and masks from the model.
    
    --- FIX ---
    Now correctly handles the `in_proj_weight` of `MultiheadAttention` modules.
    """
    print("\n--- Making pruning permanent ---")
    for module in model.modules():
        # Check for standard layers
        if isinstance(module, (nn.Linear, nn.Conv1d)) and prune.is_pruned(module):
            prune.remove(module, 'weight')
            print(f"Pruning made permanent for module: {type(module).__name__}")
        # Check specifically for the attention module
        elif isinstance(module, nn.MultiheadAttention) and prune.is_pruned(module):
            prune.remove(module, 'in_proj_weight')
            print(f"Pruning made permanent for module: {type(module).__name__}")
    return model

# =================================================================================
# 3. FINE-TUNING LOOP (Unchanged)
# =================================================================================

def finetune_one_epoch(model, dataloader, optimizer, loss_fn, device, scaler):
    """A simplified training loop for fine-tuning the pruned model."""
    model.train()
    total_loss, correct_preds, total_samples = 0, 0, 0
    progress_bar = tqdm(dataloader, desc="Fine-tuning", leave=False)

    for batch in progress_bar:
        if batch is None: continue
        video, audio, video_mask, cls_labels = (
            batch['video'].to(device), batch['audio'].to(device),
            batch['video_mask'].to(device), batch['label'].to(device)
        )

        optimizer.zero_grad(set_to_none=True)
        with autocast():
            cls_logits, _, _ = model(video, audio, video_mask)
            loss = loss_fn(cls_logits.squeeze(1), cls_labels)

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = (torch.sigmoid(cls_logits) > 0.5).squeeze(1)
        correct_preds += (preds == cls_labels.bool()).sum().item()
        total_samples += cls_labels.size(0)
        progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{correct_preds/total_samples:.2f}"})

    return total_loss / len(dataloader), correct_preds / total_samples

# =================================================================================
# 4. MAIN EXECUTION (Unchanged)
# =================================================================================

if __name__ == '__main__':
    config = ConfigPrune()
    device = config.DEVICE

    DISTILLED_MODEL_PATH = "/kaggle/input/pin-lite-distill-student/pytorch/default/1/best_pinpoint_LITE_model.pth"
    PRUNED_MODEL_PATH = "best_pinpoint_PRUNED_model.pth"

    print("=" * 60)
    print("--- STRUCTURED PRUNING FOR PINPOINT-LITE MODEL ---")
    print(f"Loading distilled student model from: {DISTILLED_MODEL_PATH}")
    print(f"Pruning iterations: {config.PRUNING_ITERATIONS}")
    print(f"Fine-tuning epochs per iteration: {config.FINETUNE_EPOCHS}")
    print(f"Final pruned model will be saved to: {PRUNED_MODEL_PATH}")
    print("=" * 60)

    if not os.path.exists(DISTILLED_MODEL_PATH):
        print(f"FATAL ERROR: Distilled model not found at '{DISTILLED_MODEL_PATH}'.")
        print("Please run the distillation script (Distill_PinPoint.py) first.")
        exit()

    print("\n--- [1/4] Setting up datasets and model ---")
    train_dataset = LAVDFDataset(config, split='train')
    dev_dataset = LAVDFDataset(config, split='dev')
    test_dataset = LAVDFDataset(config, split='test')

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)

    model = PinpointTransformerLite(config).to(device)
    model.load_state_dict(torch.load(DISTILLED_MODEL_PATH, map_location=device))
    print("Distilled student model loaded successfully.")

    print("\n--- [2/4] Starting Iterative Pruning and Fine-tuning ---")
    start_time = time.time()
    
    for i in range(config.PRUNING_ITERATIONS):
        print(f"\n===== Pruning Iteration {i + 1}/{config.PRUNING_ITERATIONS} =====")
        
        head_to_prune = find_least_important_head(model, config)
        apply_structured_head_pruning(model, head_to_prune, config)
        
        sparsity = calculate_sparsity(model)
        print(f"Model sparsity after pruning: {sparsity:.2f}%")

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.FINETUNE_LR)
        loss_fn = nn.BCEWithLogitsLoss()
        scaler = GradScaler()
        best_val_loss = float('inf')

        for epoch in range(config.FINETUNE_EPOCHS):
            print(f"--- Fine-tuning Epoch {epoch + 1}/{config.FINETUNE_EPOCHS} ---")
            train_loss, train_acc = finetune_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler)
            
            model.eval()
            val_loss, val_correct, val_total = 0, 0, 0
            with torch.no_grad():
                for batch in dev_loader:
                    if batch is None: continue
                    video, audio, _, labels = batch['video'].to(device), batch['audio'].to(device), batch['video_mask'].to(device), batch['label'].to(device)
                    logits, _, _ = model(video, audio)
                    loss = loss_fn(logits.squeeze(1), labels)
                    val_loss += loss.item()
                    preds = (torch.sigmoid(logits) > 0.5).squeeze(1)
                    val_correct += (preds == labels.bool()).sum().item()
                    val_total += labels.size(0)

            avg_val_loss = val_loss / len(dev_loader) if len(dev_loader) > 0 else 0
            val_acc = val_correct / val_total if val_total > 0 else 0
            print(f"Fine-tuning Summary: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), "temp_pruned_model.pth")
                print("  -> Saved temporary best model for this iteration.")

        model.load_state_dict(torch.load("temp_pruned_model.pth"))

    print(f"\n--- Pruning and fine-tuning finished in {(time.time() - start_time)/60:.2f} minutes ---")
    
    model = make_pruning_permanent(model)
    torch.save(model.state_dict(), PRUNED_MODEL_PATH)
    print(f"Final permanently pruned model saved to: {PRUNED_MODEL_PATH}")

    print("\n--- [4/4] Loading final pruned model and running on Test Set ---")
    if os.path.exists(PRUNED_MODEL_PATH):
        test_and_evaluate(
            model_path=PRUNED_MODEL_PATH,
            test_loader=test_loader,
            config=config,
            model_class=PinpointTransformerLite
        )
    else:
        print("Final pruned model was not found. Skipping evaluation.")

    print("\n--- Pruning script execution complete ---")