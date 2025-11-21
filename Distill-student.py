%%writefile Distill_PinPoint.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
from torch.cuda.amp import GradScaler, autocast
import time
import os
from tqdm import tqdm
import random

# --- IMPORT FROM YOUR ORIGINAL PinPoint.py FILE ---
# We reuse the data loading, configuration, and teacher model architecture directly
try:
    from PinPoint import (
        Config as TeacherConfig,
        PinpointTransformer as TeacherPinpointTransformer,
        AudioFeatureExtractor,
        GatedCrossAttentionBlock,
        get_sinusoidal_embeddings,
        LAVDFDataset,
        collate_fn,
        evaluate_model, # We can reuse the evaluation function for the student
        test_and_evaluate # We can reuse the final testing function
    )
    print("Successfully imported components from PinPoint.py")
except ImportError:
    print("FATAL ERROR: Make sure Distill_PinPoint.py is in the same directory as PinPoint.py")
    exit()

# =================================================================================
# 1. LITE CONFIGURATION (Defining the "Student")
# =================================================================================
class ConfigLite(TeacherConfig):
    """Inherits from the teacher's config and overrides parameters for a smaller model."""
    # --- Student Model Architecture ---
    EMBED_DIM = 128      # Reduced from 256
    NUM_HEADS = 4        # Reduced from 8
    NUM_LAYERS = 2       # Reduced from 3
    DROPOUT = 0.15       # Slightly increased dropout for smaller model regularization

    # --- Student Training ---
    EPOCHS = 20          # Student might need more epochs to learn from the teacher
    LEARNING_RATE = 2e-4 # Student can sometimes use a slightly higher learning rate
    BATCH_SIZE = 8       # Can often increase batch size with a smaller model

    # --- Knowledge Distillation Parameters ---
    KD_ALPHA = 0.5       # Balances hard vs. soft labels. 0.5 gives them equal weight.
    KD_BETA = 0.3        # Weight for the attention distillation loss.
    KD_TEMPERATURE = 2.0 # Softens probabilities to provide more information.

# =================================================================================
# 2. LITE MODEL ARCHITECTURE (The "Student" Model)
# =================================================================================

class VideoFeatureExtractorLite(nn.Module):
    """A lightweight video feature extractor using MobileNetV3-Small."""
    def __init__(self, embed_dim):
        super().__init__()
        mobilenet = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.feature_extractor = mobilenet.features
        mobilenet_out_features = 576
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projection = nn.Linear(mobilenet_out_features, embed_dim)
        for param in self.feature_extractor[:3].parameters():
            param.requires_grad = False
        print("Initialized VideoFeatureExtractorLite with a pretrained MobileNetV3-Small backbone.")

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.feature_extractor(x)
        pooled_features = self.pool(features).view(b * t, -1)
        projected_features = self.projection(pooled_features)
        output = projected_features.view(b, t, -1)
        return output

class PinpointTransformerLite(nn.Module):
    """The 'Student' model. Structurally similar to the teacher but with smaller dimensions."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.video_extractor = VideoFeatureExtractorLite(config.EMBED_DIM)
        self.audio_extractor = AudioFeatureExtractor(config.NUM_MFCC, config.EMBED_DIM)
        self.video_pos_encoder = nn.Parameter(torch.randn(1, config.NUM_FRAMES, config.EMBED_DIM))
        self.gated_attention_layers = nn.ModuleList([
            GatedCrossAttentionBlock(config.EMBED_DIM, config.NUM_HEADS, config.DROPOUT)
            for _ in range(config.NUM_LAYERS)
        ])
        self.classification_head = nn.Linear(config.EMBED_DIM, 1)
        num_offset_classes = 2 * config.MAX_OFFSET + 1
        self.offset_head = nn.Linear(config.EMBED_DIM, num_offset_classes)
        print("PinpointTransformerLite (Student Model) initialized.")

    def forward(self, video, audio, video_mask=None):
        video_feat = self.video_extractor(video)
        audio_feat = self.audio_extractor(audio)
        video_feat = video_feat + self.video_pos_encoder[:, :video_feat.size(1), :]
        audio_len = audio_feat.size(1)
        audio_pos_encoding = get_sinusoidal_embeddings(audio_len, self.config.EMBED_DIM).to(audio_feat.device)
        audio_feat = audio_feat + audio_pos_encoding
        last_attention_map = None
        for layer in self.gated_attention_layers:
            audio_feat, attention_map = layer(audio_feat, video_feat, video_mask)
            last_attention_map = attention_map
        pooled_output = audio_feat.mean(dim=1)
        classification_logits = self.classification_head(pooled_output)
        offset_logits = self.offset_head(pooled_output)
        return classification_logits, offset_logits, last_attention_map

# =================================================================================
# 3. DISTILLATION TRAINING LOOP
# =================================================================================

def train_distillation_epoch(teacher_model, student_model, dataloader, optimizer, scheduler, loss_fns, device, config, epoch, scaler):
    teacher_model.eval()
    student_model.train()
    total_loss, total_hard_loss, total_soft_loss, total_attn_loss = 0, 0, 0, 0
    correct_preds, total_samples = 0, 0
    progress_bar = tqdm(dataloader, desc=f"Distilling E{epoch+1}", leave=False)

    for batch in progress_bar:
        if batch is None: continue
        video, audio, video_mask, cls_labels = (
            batch['video'].to(device), batch['audio'].to(device),
            batch['video_mask'].to(device), batch['label'].to(device)
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_cls_logits, _, teacher_attention_map = teacher_model(video, audio, video_mask)
        with autocast():
            student_cls_logits, _, student_attention_map = student_model(video, audio, video_mask)
            hard_loss = loss_fns['classification'](student_cls_logits.squeeze(1), cls_labels)
            T = config.KD_TEMPERATURE
            soft_loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_cls_logits / T, dim=1),
                F.softmax(teacher_cls_logits / T, dim=1)
            ) * (T * T)
            attn_loss = F.mse_loss(student_attention_map, teacher_attention_map) if student_attention_map is not None and teacher_attention_map is not None else torch.tensor(0.0, device=device)
            combined_loss = (config.KD_ALPHA * hard_loss) + ((1 - config.KD_ALPHA) * soft_loss) + (config.KD_BETA * attn_loss)
        if not torch.isfinite(combined_loss):
            print("WARNING: Encountered non-finite loss. Skipping batch.")
            continue
        scaler.scale(combined_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total_loss += combined_loss.item()
        total_hard_loss += hard_loss.item()
        total_soft_loss += soft_loss.item()
        total_attn_loss += attn_loss.item()
        preds = (torch.sigmoid(student_cls_logits) > 0.5).squeeze(1)
        correct_preds += (preds == cls_labels.bool()).sum().item()
        total_samples += cls_labels.size(0)
        progress_bar.set_postfix({
            "Loss": f"{combined_loss.item():.4f}", "HardL": f"{hard_loss.item():.3f}",
            "SoftL": f"{soft_loss.item():.3f}", "AttnL": f"{attn_loss.item():.3f}",
            "Acc": f"{correct_preds/total_samples:.2f}"
        })
    avg_loss = total_loss / len(dataloader)
    avg_hard = total_hard_loss / len(dataloader)
    avg_soft = total_soft_loss / len(dataloader)
    avg_attn = total_attn_loss / len(dataloader)
    accuracy = correct_preds / total_samples
    return avg_loss, avg_hard, avg_soft, avg_attn, accuracy

# =================================================================================
# 4. MAIN EXECUTION
# =================================================================================

if __name__ == '__main__':
    # <<< MODIFICATION: Hardcoded settings for execution control >>>
    # ------------------------------------------------------------------------------
    # Set to True to skip training and only run evaluation on the test set.
    # Set to False to run the full training and distillation process.
    TEST_ONLY_MODE = True

    # Path to the pre-trained student model (used only if TEST_ONLY_MODE is True).
    STUDENT_MODEL_PATH_FOR_TESTING = "/kaggle/input/pin-lite-distill-student/pytorch/default/1/best_pinpoint_LITE_model.pth"

    # Path to the pre-trained, full-size teacher model (used only for training).
    TEACHER_MODEL_PATH = "/kaggle/input/pp-xai-full-model-v1/best_pinpoint_model_antisocial.pth"
    # ------------------------------------------------------------------------------
    # <<< END MODIFICATION >>>

    teacher_config = TeacherConfig()
    student_config = ConfigLite()
    device = student_config.DEVICE

    print("=" * 60)
    print("--- KNOWLEDGE DISTILLATION FOR PINPOINT MODEL ---")
    print(f"Teacher Model Path: {TEACHER_MODEL_PATH}")
    print(f"Student Device: {device}")

    if TEST_ONLY_MODE:
        print(f"Test-Only Mode: ON")
        print(f"Student Model Path: {STUDENT_MODEL_PATH_FOR_TESTING}")
    else:
        print(f"Test-Only Mode: OFF (Full Training)")
        print(f"Student Epochs: {student_config.EPOCHS}, Batch Size: {student_config.BATCH_SIZE}")
        print(f"Distillation Params: Alpha={student_config.KD_ALPHA}, Beta={student_config.KD_BETA}, Temp={student_config.KD_TEMPERATURE}")
    print("=" * 60)

    print("\n--- [1/5] Setting up datasets and dataloaders ---")
    train_dataset = LAVDFDataset(student_config, split='train')
    dev_dataset = LAVDFDataset(student_config, split='dev')
    test_dataset = LAVDFDataset(student_config, split='test')
    train_loader = DataLoader(train_dataset, batch_size=student_config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=student_config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=student_config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=2)
    print(f"Loaded {len(train_dataset)} train, {len(dev_dataset)} dev, and {len(test_dataset)} test samples.")

    # Logic to decide whether to train or test directly based on the hardcoded flag
    if TEST_ONLY_MODE:
        print("\n--- [TESTING ONLY] ---")
        if not os.path.exists(STUDENT_MODEL_PATH_FOR_TESTING):
            print(f"FATAL ERROR: Student model not found at '{STUDENT_MODEL_PATH_FOR_TESTING}'.")
            exit()
            
        print(f"\n--- Loading student model and running on Test Set ---")
        test_and_evaluate(
            model_path=STUDENT_MODEL_PATH_FOR_TESTING,
            test_loader=test_loader,
            config=student_config,
            model_class=PinpointTransformerLite
        )
    else:
        # Proceed with the full training and evaluation pipeline
        print("\n--- [STARTING FULL TRAINING] ---")
        print("\n--- [2/5] Initializing Teacher and Student models ---")
        teacher_model = TeacherPinpointTransformer(teacher_config).to(device)
        if not os.path.exists(TEACHER_MODEL_PATH):
            print(f"FATAL ERROR: Teacher model not found at '{TEACHER_MODEL_PATH}'. Please train the main model first.")
            exit()
        teacher_model.load_state_dict(torch.load(TEACHER_MODEL_PATH, map_location=device))
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        print("Teacher model loaded and frozen.")

        student_model = PinpointTransformerLite(student_config).to(device)

        print("\n--- [3/5] Setting up optimizer and loss functions ---")
        loss_fns = {'classification': nn.BCEWithLogitsLoss()}
        optimizer = torch.optim.AdamW(student_model.parameters(), lr=student_config.LEARNING_RATE, weight_decay=student_config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=student_config.EPOCHS * len(train_loader))
        scaler = GradScaler()

        print("\n--- [4/5] Starting Distillation Training ---")
        start_time = time.time()
        best_val_loss = float('inf')
        best_student_model_path = "best_pinpoint_LITE_model.pth"

        for epoch in range(student_config.EPOCHS):
            print(f"\n===== Epoch {epoch + 1}/{student_config.EPOCHS} =====")
            train_loss, h, s, a, train_acc = train_distillation_epoch(
                teacher_model, student_model, train_loader, optimizer, scheduler, loss_fns, device, student_config, epoch, scaler
            )
            print(f"Epoch {epoch + 1} Distill -> Avg Loss: {train_loss:.4f} (H:{h:.2f}, S:{s:.2f}, A:{a:.2f}), Acc: {train_acc:.4f}")
            val_loss, val_acc = evaluate_model(student_model, dev_loader, loss_fns, device, student_config)
            print(f"Epoch {epoch + 1} Validation -> Avg Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(student_model.state_dict(), best_student_model_path)
                print(f"  -> New best student model saved to {best_student_model_path} (Val Loss: {val_loss:.4f})")

        print(f"\n--- Distillation Finished in {(time.time() - start_time)/60:.2f} minutes ---")

        print(f"\n--- [5/5] Loading best student model and running on Test Set ---")
        if os.path.exists(best_student_model_path):
            test_and_evaluate(
                model_path=best_student_model_path,
                test_loader=test_loader,
                config=student_config,
                model_class=PinpointTransformerLite
            )
        else:
            print("No best student model was saved. Skipping final test evaluation.")
    
    print("\n--- Script execution complete ---")