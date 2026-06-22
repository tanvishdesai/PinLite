"""
====================================================================================
GAQ (Gated-Attention Quantization) — Central Configuration & Module Loader
====================================================================================
This is the ONLY file you need to edit when moving between environments (Kaggle,
local, etc.). Every GAQ script imports its paths from here.

It also provides a robust loader that imports the teacher (`PinPoint`) and student
(`Distill`) modules regardless of whether they live as:
  - importable modules `PinPoint` / `Distill` (the Kaggle %%writefile convention), or
  - the hyphenated source files `PinPoint-main.py` / `Distill-student.py` (this repo).

------------------------------------------------------------------------------------
HOW TO USE ON KAGGLE
------------------------------------------------------------------------------------
1. %%writefile PinPoint.py   <- paste contents of PinPoint-main.py
2. %%writefile Distill.py     <- paste contents of Distill-student.py
3. %%writefile gaq_config.py  <- paste THIS file, then EDIT THE PATHS BELOW
4. %%writefile gaq_core.py    <- paste contents of gaq_core.py
5. %%writefile <phase script>.py
6. python <phase script>.py
====================================================================================
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

# =================================================================================
# 1. EDIT THESE PATHS FOR YOUR ENVIRONMENT
# =================================================================================

# --- Model checkpoints --------------------------------------------------------
# Teacher (full PinPoint, FP32) checkpoint.
TEACHER_CKPT = "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_model_antisocial.pth"
# Distilled student (PinpointTransformerLite, FP32) checkpoint. This is the model
# we quantize with GAQ.
STUDENT_CKPT = "/kaggle/input/pinlite-all-models-v2-011225/best_pinpoint_LITE_model.pth"

# --- In-domain data (LAV-DF) --------------------------------------------------
# These override PinPoint.Config.DATA_DIRECTORY / METADATA_PATH at runtime so you
# don't have to edit PinPoint-main.py.
LAVDF_DATA_DIRECTORY = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data"
LAVDF_METADATA_PATH = "/kaggle/input/new-model-unified-pre-processing/preprocessed_data/unified_metadata.json"

# --- Cross-dataset data (FakeAVCeleb) -----------------------------------------
FAVC_DATA_DIRECTORY = "/kaggle/input/fav-pre-process/preprocessed_data"
FAVC_METADATA_PATH = "/kaggle/input/fav-pre-process/preprocessed_data/preprocessed_metadata_test_only_20250908_142055.json"

# --- Output directory ---------------------------------------------------------
OUTPUT_DIR = "/kaggle/working"

# =================================================================================
# 2. EXPERIMENT KNOBS (sensible defaults; usually no need to edit)
# =================================================================================

SEED = 11

# Number of test samples used for EPS / faithfulness (keep modest for CPU speed).
EPS_SAMPLES = 300
FAITHFULNESS_SAMPLES = 120

# Calibration set size for post-training quantization (the "LAV-DF-200" profile).
CALIBRATION_SAMPLES = 200
CALIBRATION_BATCH_SIZE = 4

# Eval batch size (GPU eval of accuracy/AUC).
EVAL_BATCH_SIZE = 8

# Latency: number of timed single-sample forward passes on CPU (named hardware).
LATENCY_REPEATS = 50
LATENCY_WARMUP = 5

# Locked success bar (from quant_gated_attention_main.md).
ACC_DROP_LIMIT = 0.01        # <= 1.0 accuracy point
EPS_DROP_LIMIT = 0.05        # <= 0.05 EPS
SPEEDUP_MIN = 1.3            # >= 1.3x CPU speedup vs FP32 teacher

# QAT recovery defaults.
QAT_EPOCHS = 3
QAT_LR = 5e-5
QAT_BATCH_SIZE = 8

# Name of the CPU you are benchmarking on (for honest "named hardware" latency).
TARGET_CPU_NAME = "Kaggle CPU (Intel Xeon, fbgemm)"


def out(path: str) -> str:
    """Resolve a filename inside OUTPUT_DIR (creates the directory)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, path)


# =================================================================================
# 3. ROBUST MODULE LOADER
# =================================================================================

def _load_by_file(mod_name: str, file_name: str):
    here = Path(__file__).resolve().parent
    file_path = here / file_name
    if not file_path.exists():
        return None
    spec = importlib.util.spec_from_file_location(mod_name, str(file_path))
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def load_pinpoint():
    """Return the teacher module (PinPoint) however it is available."""
    for name in ("PinPoint", "pinpoint_main", "PinPoint_main"):
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    mod = _load_by_file("PinPoint", "PinPoint-main.py")
    if mod is None:
        raise ImportError(
            "Could not import the teacher module. Provide it as `PinPoint.py` or "
            "keep `PinPoint-main.py` next to this file."
        )
    return mod


def load_distill():
    """Return the student module (Distill) however it is available."""
    for name in ("Distill", "Distill_PinPoint"):
        try:
            return importlib.import_module(name)
        except Exception:
            pass
    mod = _load_by_file("Distill", "Distill-student.py")
    if mod is None:
        raise ImportError(
            "Could not import the student module. Provide it as `Distill.py` or "
            "keep `Distill-student.py` next to this file."
        )
    return mod


def apply_data_paths(config) -> None:
    """Point a PinPoint Config instance at the LAV-DF paths configured above."""
    config.DATA_DIRECTORY = LAVDF_DATA_DIRECTORY
    config.METADATA_PATH = LAVDF_METADATA_PATH


def banner(title: str) -> None:
    print("=" * 70)
    print(title)
    print("=" * 70)
