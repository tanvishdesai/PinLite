import torch
import torch.nn as nn
import torch.quantization
from torch.quantization import QuantStub, DeQuantStub
import copy
import os
import types
import warnings
from tqdm import tqdm
from torchvision.models.resnet import BasicBlock

# Import your model definition
from PinPoint import PinpointTransformer, Config, LAVDFDataset, collate_fn

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# 1. QUANTIZATION-FRIENDLY MONKEY PATCHES
# =============================================================================

class QuantizedBasicBlockWrapper(nn.Module):
    """
    Wraps a ResNet BasicBlock to make it quantization compatible.
    Fixes:
    1. Replaces '+' with FloatFunctional for INT8 addition.
    2. Splits 'self.relu' into 'relu1' and 'relu2' so fusion doesn't delete the final activation.
    """
    def __init__(self, original_block):
        super().__init__()
        self.conv1 = original_block.conv1
        self.bn1 = original_block.bn1
        
        # CRITICAL FIX: Deepcopy the ReLU. 
        # Original ResNet uses the same 'self.relu' object twice.
        # If we fuse 'conv1+bn1+relu', that object becomes Identity.
        # Then the final activation 'out = self.relu(out)' ALSO becomes Identity.
        # We split them so we can fuse the first one safely.
        import copy
        self.relu1 = copy.deepcopy(original_block.relu)
        self.relu2 = copy.deepcopy(original_block.relu)
        
        self.conv2 = original_block.conv2
        self.bn2 = original_block.bn2
        self.downsample = original_block.downsample
        self.stride = original_block.stride
        
        # THE MAGIC FIX: Use FloatFunctional for the residual addition
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out) # Use relu1 (will be fused)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        # Replace 'out += identity' with FloatFunctional
        out = self.skip_add.add(out, identity)
        out = self.relu2(out) # Use relu2 (remains as activation)

        return out

def prepare_model_architecture_for_quantization(model):
    """
    Modifies the model structure in-place to be compatible with QAT.
    """
    print("[Setup] patching ResNet BasicBlocks for Quantization compatibility...")
    
    # 1. Fix ResNet Backbone in VideoFeatureExtractor
    resnet_layers = model.video_extractor.feature_extractor
    
    # Helper to replace blocks inside the Sequential container
    def replace_blocks(sequential_module):
        for i, module in enumerate(sequential_module):
            if isinstance(module, BasicBlock):
                # Replace with our safe wrapper
                sequential_module[i] = QuantizedBasicBlockWrapper(module)
            elif isinstance(module, nn.Sequential):
                replace_blocks(module)

    replace_blocks(resnet_layers)

    # 2. Insert QuantStubs (Entry points) and DeQuantStubs (Exit points)
    model.quant = QuantStub()
    model.dequant = DeQuantStub()
    
    return model

# =============================================================================
# 2. CUSTOM FORWARD PASS FOR QAT
# =============================================================================

def quantized_forward(self, video, audio, video_mask=None):
    """
    A patched forward method for PinPointTransformer that handles 
    quantization/dequantization of inputs.
    """
    # === Video Flow ===
    b, t, c, h, w = video.shape
    # Flatten for ResNet
    video_flat = video.view(b * t, c, h, w)
    
    # Quantize the flattened video input
    video_q = self.quant(video_flat)
    
    # Pass through quantized feature extractor (now using QuantizedBasicBlockWrapper)
    features = self.video_extractor.feature_extractor(video_q)
    
    # Pool and Project
    pooled = self.video_extractor.pool(features).view(b * t, -1)
    
    # Dequantize before Linear projection to be safe
    pooled = self.dequant(pooled) 
    
    projected = self.video_extractor.projection(pooled)
    video_feat = projected.view(b, t, -1)

    # === Audio Flow ===
    # We manually quantize the input to the Conv1d layers
    # But since we can't fuse F.relu easily, we just run convs on quantized data
    # and dequantize before the GRU.
    
    audio_t = audio.transpose(1, 2) # [B, C, T]
    audio_q = self.quant(audio_t) # Quantize input
    
    # Manual forward of audio extractor
    # Note: model.audio_extractor.conv1 is quantized automatically by prepare_qat
    x = self.audio_extractor.conv1(audio_q)
    # x is now Quantized Tensor. F.relu works on Quantized Tensors in PyTorch.
    x = torch.relu(x) 
    x = self.audio_extractor.conv2(x)
    x = torch.relu(x)
    
    # Dequantize before LayerNorm/GRU (RNNs unstable in INT8)
    x = self.dequant(x)
    
    x = x.transpose(1, 2)
    x = self.audio_extractor.ln(x)
    audio_feat, _ = self.audio_extractor.gru(x)

    # === Transformer / Fusion ===
    # Proceed with standard FP32 flow for attention
    
    if self.training and self.config.MODALITY_DROPOUT_PROB > 0:
         # Simplified dropout for QAT stability (masking entire modality)
        if torch.rand(1).item() < self.config.MODALITY_DROPOUT_PROB:
            if torch.rand(1).item() < 0.5:
                video_feat = video_feat * 0
            else:
                audio_feat = audio_feat * 0

    # Positional encodings
    video_feat = video_feat + self.video_pos_encoder[:, :video_feat.size(1), :]
    # We assume get_sinusoidal_embeddings handles device correctly
    from PinPoint import get_sinusoidal_embeddings # Ensure import inside function if needed
    audio_len = audio_feat.size(1)
    audio_pos_encoding = get_sinusoidal_embeddings(audio_len, self.config.EMBED_DIM).to(audio_feat.device)
    audio_feat = audio_feat + audio_pos_encoding
    
    # Run Attention Layers (FP32)
    last_attention_map = None
    for layer in self.gated_attention_layers:
        audio_feat, attention_map = layer(audio_feat, video_feat, video_mask)
        last_attention_map = attention_map

    pooled_output = audio_feat.mean(dim=1)
    classification_logits = self.classification_head(pooled_output)
    offset_logits = self.offset_head(pooled_output)
    
    return classification_logits, offset_logits, last_attention_map

# =============================================================================
# 3. MAIN QAT ROUTINE
# =============================================================================

def main_quantization(model_path, output_path="quantized_pinpoint.pth"):
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Float Model
    print("Loading float model...")
    model = PinpointTransformer(config)
    # Check if weights exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
        
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.to(device)
    # REMOVED: model.eval() - This was causing the error!

    # 2. Prepare Architecture
    model = prepare_model_architecture_for_quantization(model)
    
    # Patch the forward method
    model.forward = types.MethodType(quantized_forward, model)

    # 3. Configure QConfig
    backend = 'qnnpack' 
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)

    # --- CRITICAL: DISABLE QUANTIZATION FOR PROBLEMATIC LAYERS ---
    print("Disabling quantization for sensitive layers (GRU, LayerNorm, Attention, Projections)...")
    
    # Audio GRU and LayerNorm
    model.audio_extractor.gru.qconfig = None
    model.audio_extractor.ln.qconfig = None
    
    # Video projection layer (Linear after dequant)
    model.video_extractor.projection.qconfig = None
    
    # Classification and offset heads
    model.classification_head.qconfig = None
    model.offset_head.qconfig = None
    
    # Transformer layers
    for layer in model.gated_attention_layers:
        layer.qconfig = None
        
    # 4. Fuse Modules (requires eval mode)
    print("Fusing layers...")
    model.eval()  # Fusion requires eval mode
    
    # --- FIX: REMOVED AUDIO FUSION ---
    # model.audio_extractor has functional ReLUs, so we CANNOT use fuse_modules on it.
    # We skip it. The layers will still be quantized individually.
    
    # Fuse ResNet Layers (using our fixed wrapper)
    for m in model.modules():
        if isinstance(m, QuantizedBasicBlockWrapper):
            # Fuse conv1+bn1+relu1 (Safe now because we split the relu)
            torch.quantization.fuse_modules(m, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)

    # 5. Prepare QAT (model must be in training mode)
    print("Preparing for QAT...")
    model.train()  # CRITICAL: Switch to training mode for prepare_qat
    torch.quantization.prepare_qat(model, inplace=True)

    # 6. QAT Training Loop
    print("Starting QAT Fine-tuning...")
    
    # Load small subset for calibration
    config.BATCH_SIZE = 4 
    # Ensure you have the preprocessed data available
    try:
        train_dataset = LAVDFDataset(config, split='train')
        if len(train_dataset) > 100:
            # Create a small subset to speed up QAT
            indices = list(range(100))
            train_dataset = torch.utils.data.Subset(train_dataset, indices)
            
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)
    except Exception as e:
        print(f"Warning: Could not load dataset ({e}). QAT requires data.")
        print("Using dummy data for demonstration (Model accuracy will degrade without real data).")
        # Minimal dummy data fallback just to prove script runs
        train_loader = [] 

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    # Already in training mode from prepare_qat
    steps = 50 
    
    if len(train_loader) > 0:
        with tqdm(total=steps, desc="QAT Calibrating") as pbar:
            for i, batch in enumerate(train_loader):
                if i >= steps: break
                if batch is None: continue

                video = batch['video'].to(device)
                audio = batch['audio'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                
                # QAT Forward
                logits, _, _ = model(video, audio)
                loss = criterion(logits.squeeze(1), labels)
                
                loss.backward()
                optimizer.step()
                
                pbar.update(1)
                pbar.set_postfix({'loss': loss.item()})
    else:
        print("Skipping QAT loop due to data issues.")

    # 7. Convert
    print("Converting to Quantized Model...")
    model.cpu() 
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)

    print(f"Saving quantized model to {output_path}...")
    torch.save(quantized_model.state_dict(), output_path)
    
    # 8. Size Stats
    float_size = os.path.getsize(model_path) / (1024 * 1024)
    quant_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\nStats:")
    print(f"Original Size: {float_size:.2f} MB")
    print(f"Quantized Size: {quant_size:.2f} MB")
    print(f"Reduction: {float_size / quant_size:.2f}x")

    return quantized_model

# =============================================================================
# 4. EVALUATION
# =============================================================================

def test_quantized_model(quantized_model, config):
    print("\nTesting Quantized Model...")
    try:
        test_dataset = LAVDFDataset(config, split='test')
        # Use subset for speed if testing flag set
        if config.TESTING and len(test_dataset) > 50:
             test_dataset = torch.utils.data.Subset(test_dataset, list(range(50)))
             
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, collate_fn=collate_fn)
    except:
        print("Could not load test dataset. Skipping eval.")
        return

    device = 'cpu'
    quantized_model.to(device)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            if batch is None: continue
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            labels = batch['label'].to(device)
            
            logits, _, _ = quantized_model(video, audio)
            preds = (torch.sigmoid(logits) > 0.5).squeeze(1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    if total > 0:
        print(f"Quantized Accuracy: {correct/total:.4f}")
    else:
        print("No samples evaluated.")

if __name__ == "__main__":
    MODEL_PATH = "/kaggle/input/pinpoint-xai/pytorch/default/1/best_pinpoint_model_antisocial.pth" 
    
    if not os.path.exists(MODEL_PATH):
        print(f"Please train the model first and save it to {MODEL_PATH}")
    else:
        q_model = main_quantization(MODEL_PATH)
        
        # Configuration for testing
        config = Config()
        config.TESTING = False 
        test_quantized_model(q_model, config)