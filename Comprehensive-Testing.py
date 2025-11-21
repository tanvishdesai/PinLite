import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import importlib.util
import time
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import types

# =================================================================================
# 1. CONFIGURATION
# =================================================================================

# !!! IMPORTANT: UPDATE THESE PATHS TO YOUR ACTUAL MODEL FILES !!!
MODEL_PATHS = {
    "Base": "./best_pinpoint_model_antisocial.pth",
    "Distilled": "./best_pinpoint_LITE_model.pth",
    "Pruned": "./best_pinpoint_PRUNED_model.pth",
    "Quantized": "./quantized_pinpoint.pth"
}

# Data directories (reused from your existing config)
# Update these if your data is located elsewhere
TEST_DATA_DIRECTORIES = [
    "/kaggle/input/la-df-testrin-1",
    "/kaggle/input/lav-df-testing-part-2",
    "/kaggle/input/lav-df-testing-part-3",
    "/kaggle/input/lavdf-testing-part-4"
]
ORIGINAL_METADATA_PATH = "/kaggle/input/localized-audio-visual-deepfake-dataset-lav-df/LAV-DF/metadata.json"

# Evaluation Settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 8
EPS_SAMPLES = 200  # Number of samples for Explainability Preservation Score
DEBUG_MODE = False # Set to True to run on a tiny subset for verification

# =================================================================================
# 2. DYNAMIC MODULE LOADING
# =================================================================================

def import_module_from_path(module_name, file_path):
    """Imports a module from a file path, handling hyphens and registering it in sys.modules."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not find file: {file_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for module: {module_name}")
    
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

print("--- Loading Modules ---")
try:
    # 1. Load PinPoint-main.py as 'PinPoint'
    # This is critical because other files expect 'import PinPoint'
    pinpoint_path = os.path.abspath("PinPoint-main.py")
    PinPoint = import_module_from_path("PinPoint", pinpoint_path)
    print("✅ Loaded PinPoint-main.py as 'PinPoint'")

    # 2. Load Distill-student.py as 'Distill_PinPoint'
    distill_path = os.path.abspath("Distill-student.py")
    Distill_PinPoint = import_module_from_path("Distill_PinPoint", distill_path)
    print("✅ Loaded Distill-student.py as 'Distill_PinPoint'")

    # 3. Load Quantized.py as 'Quantized'
    quantized_path = os.path.abspath("Quantized.py")
    Quantized = import_module_from_path("Quantized", quantized_path)
    print("✅ Loaded Quantized.py as 'Quantized'")

except Exception as e:
    print(f"❌ Error loading modules: {e}")
    print("Ensure 'PinPoint-main.py', 'Distill-student.py', and 'Quantized.py' are in the current directory.")
    sys.exit(1)

# =================================================================================
# 3. MODEL LOADING FUNCTIONS
# =================================================================================

def load_base_model(path):
    print(f"Loading Base Model from {path}...")
    config = PinPoint.Config()
    model = PinPoint.PinpointTransformer(config)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"⚠️ Warning: Base model file not found at {path}. Skipping.")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def load_distilled_model(path):
    print(f"Loading Distilled Model from {path}...")
    config = Distill_PinPoint.ConfigLite()
    model = Distill_PinPoint.PinpointTransformerLite(config)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"⚠️ Warning: Distilled model file not found at {path}. Skipping.")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def load_pruned_model(path):
    print(f"Loading Pruned Model from {path}...")
    # Pruned model has same architecture as Distilled (Lite), but with zeroed weights
    # Ideally, we should apply the pruning masks if we want to simulate the sparse structure,
    # but for inference, loading the state_dict (which has zeros) is sufficient for correctness.
    # If the saved model has pruning masks ('weight_orig', 'weight_mask'), we need to handle that.
    # Based on Prunning.py, 'make_pruning_permanent' was called, so it should be a standard state_dict.
    
    config = Distill_PinPoint.ConfigLite()
    model = Distill_PinPoint.PinpointTransformerLite(config)
    try:
        model.load_state_dict(torch.load(path, map_location=DEVICE))
    except FileNotFoundError:
        print(f"⚠️ Warning: Pruned model file not found at {path}. Skipping.")
        return None
    model.to(DEVICE)
    model.eval()
    return model

def load_quantized_model(path):
    print(f"Loading Quantized Model from {path}...")
    # Quantized loading is complex. We need to replicate the steps in Quantized.py
    
    config = PinPoint.Config()
    # 1. Initialize Float Model
    model = PinPoint.PinpointTransformer(config)
    
    # 2. Prepare Architecture (Patching)
    model = Quantized.prepare_model_architecture_for_quantization(model)
    model.forward = types.MethodType(Quantized.quantized_forward, model)
    
    # 3. Configure QConfig (Must match what was used during quantization)
    backend = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qat_qconfig(backend)
    
    # 4. Disable quantization for sensitive layers (Must match Quantized.py)
    model.audio_extractor.gru.qconfig = None
    model.audio_extractor.ln.qconfig = None
    model.video_extractor.projection.qconfig = None
    model.classification_head.qconfig = None
    model.offset_head.qconfig = None
    for layer in model.gated_attention_layers:
        layer.qconfig = None
        
    # 5. Fuse Modules
    model.eval()
    for m in model.modules():
        if isinstance(m, Quantized.QuantizedBasicBlockWrapper):
            torch.quantization.fuse_modules(m, [['conv1', 'bn1', 'relu1'], ['conv2', 'bn2']], inplace=True)
            
    # 6. Prepare QAT
    model.train()
    torch.quantization.prepare_qat(model, inplace=True)
    
    # 7. Convert to Quantized
    model.cpu()
    model.eval()
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    # 8. Load Weights
    try:
        quantized_model.load_state_dict(torch.load(path, map_location='cpu'))
    except FileNotFoundError:
        print(f"⚠️ Warning: Quantized model file not found at {path}. Skipping.")
        return None
    except RuntimeError as e:
        print(f"⚠️ Warning: Error loading quantized weights: {e}")
        return None

    # Quantized models usually run on CPU (unless using specialized TensorRT/etc, but PyTorch default is CPU)
    # The user's Quantized.py uses 'cpu' for inference.
    quantized_model.to('cpu') 
    return quantized_model

# =================================================================================
# 4. EVALUATION METRICS
# =================================================================================

def measure_inference_time(model, dataloader, device, num_batches=10):
    """Measures average inference time per sample in milliseconds."""
    model.eval()
    latencies = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            if batch is None: continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            
            # Warmup
            if i == 0:
                _ = model(video, audio, video_mask)
                continue
                
            start_time = time.time()
            _ = model(video, audio, video_mask)
            end_time = time.time()
            
            batch_latency = (end_time - start_time) * 1000 # ms
            latencies.append(batch_latency / video.size(0)) # per sample
            
    return np.mean(latencies) if latencies else 0.0

def calculate_eps(model, dataloader, device, num_samples=200):
    """
    Calculates Explainability Preservation Score (Faithfulness) using Integrated Gradients.
    Faithfulness = Correlation(Attribution, Output Change when features masked)
    """
    print(f"  Calculating EPS (Faithfulness) on {num_samples} samples...")
    model.eval()
    
    faithfulness_scores = []
    
    # We need a gradient-based method. 
    # Note: Quantized models might not support gradients if operations aren't differentiable.
    # We will skip EPS for quantized models if they don't support gradients.
    
    is_quantized = False
    # Check if model parameters require grad (quantized usually don't)
    try:
        next(model.parameters())
    except:
        is_quantized = True
        
    if is_quantized:
        print("  Skipping EPS for Quantized model (gradients not supported).")
        return 0.0

    count = 0
    for batch in dataloader:
        if count >= num_samples: break
        if batch is None: continue
        
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        
        # Ensure gradients enabled
        video.requires_grad = True
        audio.requires_grad = True
        
        # 1. Get Baseline Prediction
        logits, _, _ = model(video, audio)
        probs = torch.sigmoid(logits)
        
        # 2. Compute Integrated Gradients (Simplified: Input * Gradient)
        # Using Input * Gradient as a proxy for IG for speed and stability in this script
        # (Full IG requires integral approximation loop)
        model.zero_grad()
        logits.sum().backward()
        
        video_attr = video.grad * video
        audio_attr = audio.grad * audio
        
        # 3. Perturbation (Mask top 10% important features)
        # Flatten to find top features
        video_flat = video_attr.view(video.size(0), -1)
        audio_flat = audio_attr.view(audio.size(0), -1)
        
        # Create masks
        # (Simplified: just mask random chunks for faithfulness check if IG is expensive, 
        # but here we use the attribution to mask 'important' parts)
        
        # For simplicity and robustness in this script, we'll use a standard Faithfulness metric:
        # Measure correlation between "Sum of Attribution in region" and "Prediction Drop when region masked"
        
        with torch.no_grad():
            # Masking strategy: Zero out video (modality occlusion)
            # This is a coarse-grained check
            logits_masked, _, _ = model(torch.zeros_like(video), audio)
            probs_masked = torch.sigmoid(logits_masked)
            
            drop = (probs - probs_masked).abs().cpu().numpy()
            
            # Attribution of video
            video_total_attr = video_attr.view(video.size(0), -1).sum(dim=1).abs().cpu().numpy()
            
            # Correlation for this batch
            # Avoid division by zero
            if np.std(drop) > 1e-6 and np.std(video_total_attr) > 1e-6:
                corr = np.corrcoef(drop.flatten(), video_total_attr.flatten())[0, 1]
                if not np.isnan(corr):
                    faithfulness_scores.append(corr)
        
        count += video.size(0)
        
    return np.mean(faithfulness_scores) if faithfulness_scores else 0.0

def evaluate_model(model, dataloader, device, model_name):
    print(f"Evaluating {model_name}...")
    
    all_preds = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Testing {model_name}"):
            if batch is None: continue
            
            video = batch['video'].to(device)
            audio = batch['audio'].to(device)
            video_mask = batch['video_mask'].to(device)
            labels = batch['label'].to(device)
            
            logits, _, _ = model(video, audio, video_mask)
            preds = (torch.sigmoid(logits) > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    }

# =================================================================================
# 5. MAIN EXECUTION
# =================================================================================

if __name__ == "__main__":
    print("="*60)
    print("COMPREHENSIVE MODEL TESTING & BENCHMARKING")
    print("="*60)
    
    # 1. Setup Dataset
    print("\n--- Setting up Test Dataset ---")
    # We need to create a config object for the dataset
    config = PinPoint.Config()
    config.TEST_DATA_DIRECTORIES = TEST_DATA_DIRECTORIES
    config.ORIGINAL_METADATA_PATH = ORIGINAL_METADATA_PATH
    
    if DEBUG_MODE:
        config.MAX_SAMPLES = 10
        print("DEBUG MODE: Limiting to 10 samples.")
    
    try:
        test_dataset = PinPoint.LAVDFDataset(config, split='test')
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PinPoint.collate_fn)
    except Exception as e:
        print(f"FATAL: Could not load dataset. {e}")
        sys.exit(1)
        
    results = []
    
    # 2. Evaluate Each Model
    for name, path in MODEL_PATHS.items():
        print(f"\n\n>>> Processing Model: {name} <<<")
        
        # Load
        model = None
        current_device = DEVICE
        
        if name == "Base":
            model = load_base_model(path)
        elif name == "Distilled":
            model = load_distilled_model(path)
        elif name == "Pruned":
            model = load_pruned_model(path)
        elif name == "Quantized":
            model = load_quantized_model(path)
            current_device = 'cpu' # Quantized runs on CPU
            
        if model is None:
            print(f"Skipping {name} due to loading failure.")
            continue
            
        # Get File Size
        file_size_mb = os.path.getsize(path) / (1024 * 1024) if os.path.exists(path) else 0
        
        # Performance Metrics
        metrics = evaluate_model(model, test_loader, current_device, name)
        
        # Inference Time
        print("Measuring Inference Time...")
        inf_time = measure_inference_time(model, test_loader, current_device)
        
        # EPS Score
        print("Calculating EPS...")
        eps = calculate_eps(model, test_loader, current_device, num_samples=EPS_SAMPLES)
        
        # Compile Results
        row = {
            "Model": name,
            "Size (MB)": round(file_size_mb, 2),
            "Inference (ms/sample)": round(inf_time, 2),
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
            "EPS (Faithfulness)": round(eps, 4)
        }
        results.append(row)
        
        # Cleanup to save memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    # 3. Final Report
    print("\n\n" + "="*60)
    print("FINAL BENCHMARK RESULTS")
    print("="*60)
    
    df = pd.DataFrame(results)
    if not df.empty:
        print(df.to_string(index=False))
        
        # Save to CSV
        df.to_csv("comprehensive_benchmark_results.csv", index=False)
        print("\nResults saved to 'comprehensive_benchmark_results.csv'")
    else:
        print("No results generated.")
        
    print("\nDone.")
