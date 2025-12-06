import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
import importlib.util
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
from scipy.stats import spearmanr
import json
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
try:
    from thop import profile
except ImportError:
    print("Warning: 'thop' library not found. FLOPs calculation will be skipped.")
    profile = None

# =================================================================================
# 1. CONFIGURATION
# =================================================================================

# Define the source directories
# For Kaggle, these might be:
# MODULE_SOURCE_DIR = "/kaggle/input/my-code-dataset/"
# MODEL_SOURCE_DIR = "/kaggle/input/my-model-dataset/"
MODULE_SOURCE_DIR = "/kaggle/input/pinlite-all-models-v2-011225" 
MODEL_SOURCE_DIR = "/kaggle/input/pinlite-all-models-v2-011225"

# !!! IMPORTANT: UPDATE THESE PATHS TO YOUR ACTUAL MODEL FILES !!!
MODEL_PATHS = {
    "Base": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_model_antisocial.pth"),
    "Distilled": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_LITE_model.pth"),
    "Pruned": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_PRUNED_model.pth"),
    "Quantized": os.path.join(MODEL_SOURCE_DIR, "best_pinpoint_QUANTIZED_model.pth")
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
# Add module source dir to sys.path to ensure internal imports work
if MODULE_SOURCE_DIR not in sys.path:
    sys.path.append(os.path.abspath(MODULE_SOURCE_DIR))

try:
    # 1. Load PinPoint-main.py as 'PinPoint'
    # This is critical because other files expect 'import PinPoint'
    pinpoint_path = os.path.join(MODULE_SOURCE_DIR, "PinPoint-main.py")
    PinPoint = import_module_from_path("PinPoint", pinpoint_path)
    print("✅ Loaded PinPoint-main.py as 'PinPoint'")

    # 2. Load Distill-student.py as 'Distill_PinPoint'
    distill_path = os.path.join(MODULE_SOURCE_DIR, "Distill-student.py")
    Distill_PinPoint = import_module_from_path("Distill_PinPoint", distill_path)
    print("✅ Loaded Distill-student.py as 'Distill_PinPoint'")

    # 3. Load Quantized.py as 'Quantized'
    quantized_path = os.path.join(MODULE_SOURCE_DIR, "Quantized.py")
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

def measure_flops(model, config, device):
    """Calculates FLOPs and Parameters using thop."""
    if profile is None:
        return 0.0, 0.0
    
    model.eval()
    # Create dummy inputs
    video_input = torch.randn(1, config.NUM_FRAMES, 3, config.VIDEO_SIZE[0], config.VIDEO_SIZE[1]).to(device)
    # Audio input shape depends on the model config, assuming standard here
    audio_input = torch.randn(1, 400, config.NUM_MFCC).to(device) 
    # Mask
    video_mask = torch.ones(1, config.NUM_FRAMES).to(device)

    try:
        # thop.profile expects inputs as a tuple
        flops, params = profile(model, inputs=(video_input, audio_input, video_mask), verbose=False)
        return flops / 1e9, params / 1e6 # GFLOPs, MParams
    except Exception as e:
        print(f"  Error calculating FLOPs: {e}")
        return 0.0, 0.0

def plot_pareto_curves(results_df):
    """Generates and saves Pareto curves."""
    print("Generating Pareto Curves...")
    
    # 1. Accuracy vs Latency
    plt.figure(figsize=(10, 6))
    for i, row in results_df.iterrows():
        plt.scatter(row['Inference (ms/sample)'], row['Accuracy'], label=row['Model'], s=100)
        plt.text(row['Inference (ms/sample)'], row['Accuracy'], f"  {row['Model']}", fontsize=9)
    
    plt.xlabel('Inference Latency (ms)')
    plt.ylabel('Accuracy')
    plt.title('Pareto Frontier: Accuracy vs. Latency')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('Pareto_Accuracy_vs_Latency.png')
    print("  Saved 'Pareto_Accuracy_vs_Latency.png'")
    
    # 2. EPS vs Latency
    if 'EPS' in results_df.columns and results_df['EPS'].sum() > 0:
        plt.figure(figsize=(10, 6))
        for i, row in results_df.iterrows():
            plt.scatter(row['Inference (ms/sample)'], row['EPS'], label=row['Model'], s=100)
            plt.text(row['Inference (ms/sample)'], row['EPS'], f"  {row['Model']}", fontsize=9)
        
        plt.xlabel('Inference Latency (ms)')
        plt.ylabel('Explainability Preservation Score (EPS)')
        plt.title('Pareto Frontier: EPS vs. Latency')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('Pareto_EPS_vs_Latency.png')
        print("  Saved 'Pareto_EPS_vs_Latency.png'")

def calculate_eps(teacher_model, student_model, dataloader, device, model_name, num_samples=200):
    """
    Calculates Explainability Preservation Score (EPS) by comparing Teacher and Student saliency maps.
    EPS = 0.5 * Correlation(S_T, S_S) + 0.5 * IoU(S_T, S_S)
    """
    print(f"  Calculating EPS (Teacher vs Student) on {num_samples} samples...")
    
    # Explicitly skip for Quantized model as it runs on CPU and gradients are problematic/slow
    if model_name == "Quantized":
        print("  Skipping EPS for Quantized model (Slow on CPU / Gradients not supported).")
        return 0.0

    # Check if student supports gradients (Quantized models usually don't)
    is_quantized = False
    try:
        next(student_model.parameters())
    except:
        is_quantized = True
        
    if is_quantized:
        print("  Skipping EPS for Quantized model (gradients not supported for IG).")
        return 0.0

    teacher_model.eval()
    student_model.eval()
    
    # Enable gradients for inputs
    # We need to wrap the forward pass to get gradients w.r.t input
    
    eps_scores = []
    count = 0
    
    for batch in dataloader:
        if count >= num_samples: break
        if batch is None: continue
        
        video = batch['video'].to(device)
        audio = batch['audio'].to(device)
        
        # We need gradients on input
        video.requires_grad = True
        audio.requires_grad = True
        
        # --- Helper to get Saliency Map ---
        def get_saliency(model, v, a):
            # Fix for "cudnn RNN backward can only be called in training mode"
            # We temporarily disable cuDNN to allow backward pass on RNNs in eval mode
            old_cudnn = torch.backends.cudnn.enabled
            torch.backends.cudnn.enabled = False
            
            try:
                # Ensure inputs are on the same device as the model and are leaf nodes
                model_device = next(model.parameters()).device
                v_in = v.to(model_device).detach().requires_grad_(True)
                a_in = a.to(model_device).detach().requires_grad_(True)

                model.zero_grad()
                logits, _, _ = model(v_in, a_in)
                # Use max logit as target for attribution
                target_score = logits.max()
                target_score.backward()
                
                if v_in.grad is None: return None
                
                # Saliency = |Input * Gradient|
                # We focus on Video saliency for now as it's the primary high-dim input
                saliency = (v_in * v_in.grad).abs()
                return saliency.detach().cpu().numpy()
            finally:
                torch.backends.cudnn.enabled = old_cudnn
            
        # 1. Get Teacher Saliency
        saliency_T = get_saliency(teacher_model, video, audio)
        
        # 2. Get Student Saliency
        # Reset gradients
        video.grad = None
        audio.grad = None
        saliency_S = get_saliency(student_model, video, audio)
        
        if saliency_T is None or saliency_S is None:
            continue
            
        # 3. Calculate Metrics per sample
        batch_size = video.size(0)
        for i in range(batch_size):
            # Flatten maps
            map_T = saliency_T[i].flatten()
            map_S = saliency_S[i].flatten()
            
            # A. Spearman Correlation
            # Add small noise to avoid constant input issues
            if np.std(map_T) > 1e-9 and np.std(map_S) > 1e-9:
                corr, _ = spearmanr(map_T, map_S)
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
            
        count += batch_size
        
    return np.mean(eps_scores) if eps_scores else 0.0

class MultiPartLAVDFDataset(torch.utils.data.Dataset):
    def __init__(self, directories, metadata_path, config):
        self.config = config
        self.samples = []
        
        print(f"--- Dynamically building test dataset ---")

        # Step 1: Load original metadata to get a complete list of test files and their labels
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Canonical metadata not found: {metadata_path}")

        print(f"Loading original metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            original_metadata = json.load(f)
        
        expected_test_files = {} # {basename: label_str}
        for item in original_metadata:
            # The original LAV-DF metadata doesn't have a 'label' field, only 'n_fakes'
            if item.get('split') == 'test':
                base_name = os.path.splitext(os.path.basename(item['file']))[0]
                # n_fakes == 0 means real video, n_fakes > 0 means fake video
                n_fakes = item.get('n_fakes', 0)
                label_str = 'fake' if n_fakes > 0 else 'real'
                expected_test_files[base_name] = label_str
        
        print(f"Found {len(expected_test_files)} expected files in 'test' split from original metadata.")
        if not expected_test_files:
             raise ValueError(f"Could not find any files for the 'test' split in the metadata. Check paths and file content.")

        # Step 2: Scan all preprocessed directories to find matching files
        print(f"Scanning {len(directories)} directories for preprocessed files...")
        found_files = 0
        for data_dir in directories:
            if not os.path.isdir(data_dir):
                print(f"  - Warning: Directory not found, skipping: {data_dir}")
                continue
            
            # Use os.walk to search recursively
            for root, _, files in tqdm(os.walk(data_dir), desc=f"Scanning {os.path.basename(data_dir)}"):
                for file in files:
                    if file.endswith("_video.pt"):
                        base_name = file.replace("_video.pt", "")
                        
                        # Check if this file is part of the target split
                        if base_name in expected_test_files:
                            video_path = os.path.join(root, file)
                            audio_path = os.path.join(root, f"{base_name}_audio.pt")
                            
                            if os.path.exists(audio_path):
                                label_str = expected_test_files[base_name]
                                is_fake = (label_str == 'fake')
                                self.samples.append({
                                    "video_path": video_path,
                                    "audio_path": audio_path,
                                    "label": 1.0 if is_fake else 0.0,
                                    "is_fake": is_fake
                                })
                                found_files += 1

        print(f"Successfully located and loaded {len(self.samples)} test samples.")
        
        if config.DEBUG_MODE:
             print("DEBUG MODE: Limiting to 10 samples.")
             self.samples = self.samples[:10]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            video = torch.load(sample['video_path']).float() / 255.0
            audio = torch.load(sample['audio_path'])
            # Fix audio shape if needed (from PinPoint logic)
            if audio.dim() == 1: audio = audio.unsqueeze(0)
            
            # Create mask (dummy)
            video_mask = torch.ones(video.shape[0]) 
            
            return {
                "video": video,
                "audio": audio,
                "video_mask": video_mask,
                "label": torch.tensor(sample['label'], dtype=torch.float),
                "offset_label": 0,
                "is_fake": sample['is_fake'],
                "video_path": sample['video_path']
            }
        except Exception as e:
            print(f"Error loading {sample['video_path']}: {e}")
            return None

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
    
    try:
        auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        auc = 0.0

    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "AUC": auc
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
    config.DEBUG_MODE = DEBUG_MODE
    
    if DEBUG_MODE:
        print("DEBUG MODE: Limiting to 10 samples.")
    
    try:
        # Use our new MultiPart Dataset
        test_dataset = MultiPartLAVDFDataset(TEST_DATA_DIRECTORIES, ORIGINAL_METADATA_PATH, config)
        if len(test_dataset) == 0:
            print("CRITICAL: No samples found. Please check data directories and metadata.")
            sys.exit(1)
            
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=PinPoint.collate_fn)
    except Exception as e:
        print(f"FATAL: Could not load dataset. {e}")
        sys.exit(1)
        
    results = []
    
    # Load Base Model (Teacher) first for EPS
    print("\n--- Loading Base Model (Teacher) for EPS ---")
    base_model_path = MODEL_PATHS["Base"]
    teacher_model = load_base_model(base_model_path)
    if teacher_model is None:
        print("CRITICAL: Base model not found. EPS calculation will be disabled for all models.")
    
    # 2. Evaluate Each Model
    for name, path in MODEL_PATHS.items():
        print(f"\n\n>>> Processing Model: {name} <<<")
        
        # Load
        model = None
        current_device = DEVICE
        
        if name == "Base":
            # We already loaded it, but let's use the one we have or reload if needed.
            # To be safe and consistent with loop, we can just use teacher_model if it exists
            if teacher_model is not None:
                model = teacher_model
            else:
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
        eps = 0.0
        if teacher_model is not None:
            if name == "Base":
                eps = 1.0 # Self-similarity
            else:
                print("Calculating EPS (Teacher vs Student)...")
                eps = calculate_eps(teacher_model, model, test_loader, current_device, name, num_samples=EPS_SAMPLES)
        else:
            print("Skipping EPS (No Teacher Model).")
        
        # FLOPs
        flops = 0.0
        if current_device != 'cpu': # thop often needs CUDA for correct profiling if model is on CUDA
             # We need a config object to know input shapes. 
             # We can try to infer it or use the one from PinPoint/Distill
             if name == "Base":
                 temp_config = PinPoint.Config()
             else:
                 temp_config = Distill_PinPoint.ConfigLite()
             flops, _ = measure_flops(model, temp_config, current_device)

        # Peak Memory
        peak_mem_mb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            # Run a dummy pass to trigger memory usage
            try:
                _ = measure_inference_time(model, test_loader, current_device, num_batches=1)
                peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            except:
                pass

        # Compile Results
        row = {
            "Model": name,
            "Size (MB)": round(file_size_mb, 2),
            "Params (M)": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
            "FLOPs (G)": round(flops, 2),
            "Inference (ms/sample)": round(inf_time, 2),
            "Peak VRAM (MB)": round(peak_mem_mb, 2),
            "Accuracy": round(metrics["Accuracy"], 4),
            "Precision": round(metrics["Precision"], 4),
            "Recall": round(metrics["Recall"], 4),
            "F1-Score": round(metrics["F1-Score"], 4),
            "AUC": round(metrics["AUC"], 4),
            "EPS": round(eps, 4)
        }
        results.append(row)
        
        # Cleanup to save memory
        # Don't delete teacher_model if we need it for next iterations
        if name != "Base":
            del model
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Incremental Saving ---
        # Save results after each model to prevent loss in case of timeout
        df_temp = pd.DataFrame(results)
        df_temp.to_csv("comprehensive_benchmark_results_partial.csv", index=False)
        print(f"Saved partial results to 'comprehensive_benchmark_results_partial.csv' (Models: {[r['Model'] for r in results]})")
            
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
        
        # Generate Pareto Plots
        plot_pareto_curves(df)
    else:
        print("No results generated.")
        
    print("\nDone.")