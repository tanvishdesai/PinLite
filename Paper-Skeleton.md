# PIN-Lite: Efficient and Explainable Multimodal Deepfake Detection via Knowledge Distillation

> **Conference Paper Skeleton** — Sections, tables, figures, and content prompts

---

## Abstract (~150-200 words)
**Content:**
- **Problem**: Deepfake detection models are large and computationally expensive, limiting edge deployment.
- **Approach**: Knowledge distillation + structured pruning to compress a multimodal audio-visual transformer.
- **Key Novelty**: Explainability Preservation Score (EPS) — ensuring compressed models retain attention-based explanations.
- **Results**: 8.6× smaller, 2.2× faster, 97.5% accuracy, EPS=0.60.
- **Conclusion**: Practical deepfake detection for resource-constrained environments.

---

## 1. Introduction (~1-1.5 pages)

### 1.1 Problem Statement
- Rise of deepfakes and societal threat
- Need for real-time detection on edge devices (phones, cameras)
- Current models are too large for deployment

### 1.2 Motivation
- Existing compression methods focus only on accuracy
- Explainability is critical for trust in forensics
- Gap: No work on preserving explainability during compression

### 1.3 Contributions
> **Bulleted list of 3-4 contributions:**
1. PIN-Lite: A compressed multimodal deepfake detector via knowledge distillation
2. Explainability Preservation Score (EPS): A novel metric for XAI fidelity
3. Comprehensive evaluation on LAV-DF benchmark
4. Analysis of Pareto trade-offs between efficiency and explainability

---

## 2. Related Work (~1 page)

### 2.1 Deepfake Detection
- Face manipulation detection (FaceForensics++, Celeb-DF)
- Audio-visual approaches (LAVDF, AVoiD-DF)
- Cite: [Rossler 2019, Li 2020, Cai 2022]

### 2.2 Model Compression
- Knowledge distillation (Hinton 2015)
- Structured pruning (Li 2017)
- Quantization (Jacob 2018)
- Cite: [Hinton 2015, Howard 2017, Han 2016]

### 2.3 Explainable AI for Deepfakes
- Attention-based explanations
- Saliency maps and Integrated Gradients
- Cite: [Selvaraju 2017, Sundararajan 2017]

---

## 3. Method (~2-2.5 pages)

### 3.1 Teacher Model: PinPoint Architecture

> **Figure 1**: PinPoint Architecture Diagram
> - Video encoder (ResNet-18)
> - Audio encoder (CNN-GRU)
> - Gated Cross-Attention layers
> - Classification head

**Content:**
- Multimodal fusion via cross-attention
- Attention maps as explainability proxy

### 3.2 Knowledge Distillation for PIN-Lite

> **Figure 2**: Distillation Pipeline Diagram
> - Teacher (frozen) → Student (trainable)
> - Hard loss, Soft loss, Attention loss

**Content:**
- Student architecture (MobileNetV3 backbone, reduced dimensions)
- Distillation loss formulation:
  ```
  L_total = α·L_hard + (1-α)·L_soft + β·L_attention
  ```

> **Table 1**: Architecture Comparison (Teacher vs Student)
| Component | Teacher | Student |
|-----------|---------|---------|
| Backbone | ResNet-18 | MobileNetV3-Small |
| Embed Dim | 256 | 128 |
| Attention Heads | 8 | 4 |
| Transformer Layers | 3 | 2 |
| Parameters | 15M | 1.69M |

### 3.3 Structured Pruning
- Attention head importance ranking (L1-norm)
- Iterative prune → fine-tune cycle
- Making pruning permanent

### 3.4 Explainability Preservation Score (EPS)

> **Figure 3**: EPS Calculation Diagram
> - Teacher attention map vs Student attention map
> - Spearman correlation + IoU visualization

**Content:**
- Formal definition:
  ```
  EPS = 0.5 × Corr(A_T, A_S) + 0.5 × IoU(Top20%_T, Top20%_S)
  ```
- Interpretation: Measures how well compressed model preserves teacher's "focus"

---

## 4. Experimental Setup (~0.5-1 page)

### 4.1 Dataset
- LAV-DF (Localized Audio-Visual DeepFake)
- Train/Val/Test split statistics
- Real vs Fake distribution

### 4.2 Implementation Details
- Framework: PyTorch
- Hardware: NVIDIA GPU (specify)
- Training hyperparameters (epochs, LR, batch size)
- Distillation hyperparameters (α, β, temperature)

### 4.3 Evaluation Metrics
- Classification: Accuracy, Precision, Recall, F1, AUC
- Efficiency: Parameters, FLOPs, Latency, Model Size
- Explainability: EPS (our metric)

---

## 5. Results (~1.5-2 pages)

### 5.1 Main Results

> **Table 2**: Comprehensive Benchmark Results
| Model | Size | Params | FLOPs | Latency | Accuracy | F1 | EPS |
|-------|------|--------|-------|---------|----------|-----|-----|
| Base (Teacher) | 57.32 MB | 15M | 18.71G | 82.5 ms | 97.37% | 98.21% | 1.00 |
| Distilled | 6.62 MB | 1.69M | 0.78G | 37.6 ms | 97.53% | 98.34% | 0.60 |
| Pruned | 6.62 MB | 1.69M | 0.78G | 39.4 ms | 97.38% | 98.24% | 0.58 |

**Key Findings:**
- 8.6× model size reduction (57→6.6 MB)
- 2.2× faster inference (82→37 ms)
- Accuracy maintained (97.37%→97.53%)
- 60% explainability preservation (EPS=0.60)

### 5.2 Pareto Analysis

> **Figure 4**: Pareto Frontier — Accuracy vs Latency
> - All models plotted, showing distilled in optimal region

> **Figure 5**: Pareto Frontier — EPS vs Latency
> - Trade-off between explainability and speed

**Content:**
- Distilled model achieves best Pareto efficiency
- Acceptable EPS trade-off for significant speed gains

### 5.3 Qualitative Analysis

> **Figure 6**: Attention Map Comparison (2×3 grid)
> - Row 1: Real video (Teacher vs Student attention)
> - Row 2: Fake video (Teacher vs Student attention)

**Content:**
- Visual confirmation of attention similarity
- Student focuses on same facial/audio regions as teacher

### 5.4 Ablation Study (Optional)

> **Table 3**: Effect of Distillation Hyperparameters
| α | β | Accuracy | EPS |
|---|---|----------|-----|
| 0.3 | 0.3 | 97.1% | 0.55 |
| 0.5 | 0.3 | 97.5% | 0.60 |
| 0.7 | 0.3 | 96.8% | 0.48 |

---

## 6. Discussion (~0.5 page)

### 6.1 Quantization Failure
- Quantized model collapsed to 52% accuracy
- Hypothesis: GRU and attention layers not INT8-friendly
- Future work: Mixed-precision or attention-specific quantization

### 6.2 EPS Interpretation
- 60% preservation is acceptable but not perfect
- Trade-off: Perfect fidelity would require larger student

### 6.3 Limitations
- Single dataset (LAV-DF) — needs cross-dataset validation
- No edge hardware benchmarks (Jetson, Raspberry Pi)
- EPS metric is new — needs community adoption

---

## 7. Conclusion (~0.25 page)

**Summary:**
- Presented PIN-Lite: efficient multimodal deepfake detector
- Introduced EPS metric for explainability preservation
- Achieved 8.6× compression with 60% explainability preservation
- Practical for edge deployment without sacrificing accuracy

**Future Work:**
- Cross-dataset generalization
- Edge device deployment (ONNX, TensorRT)
- Improved quantization strategies

---

## References (~20-30 citations)
Standard conference format (IEEE, ACM, etc.)

---

## Supplementary Material (if allowed)

### A. Detailed Architecture Diagrams
### B. Additional Attention Map Visualizations
### C. Training Curves
### D. Statistical Significance Tests

---

## Checklist Before Submission

- [ ] All figures generated at 300 DPI
- [ ] Tables formatted per conference style
- [ ] References in correct format
- [ ] Page limit checked
- [ ] Anonymization (if blind review)
- [ ] Code/data availability statement
