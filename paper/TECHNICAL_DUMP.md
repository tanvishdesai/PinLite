# Technical Dump - PIN-Lite

**Project:** PIN-Lite: Lightweight Multimodal Deepfake Detection with Explainability Preservation
**Date compiled:** 2026-05-14
**Code location:** `C:\Users\DELL\Desktop\code_playground\Multi-Modal\PinLite`
**Results location:** `C:\Users\DELL\Desktop\code_playground\Multi-Modal\PinLite\pinlite results v3`
**Ground-truth result source:** `paper/pinlite_paper (4).pdf` plus result logs and benchmark CSV. The draft uses the comprehensive benchmark CSV as the main quantitative table and notes pipeline-run discrepancies separately.

---

## 1. Problem and Task Definition

**Task type:** Binary deepfake classification for audio-visual video clips.

**Input modalities:** RGB video frames plus audio MFCC features.

**Output:** Binary real/fake logit and probability; teacher/student architectures also output an auxiliary temporal-offset head and a final-layer cross-attention map.

**Evaluation setting:** Supervised in-domain evaluation on LAV-DF, efficient model comparison across compressed variants, and zero-shot cross-dataset evaluation on FakeAVCeleb as an explanation-fidelity stress test.

---

## 2. Datasets

### Dataset 1

**Name:** Localized Audio Visual DeepFake Dataset (LAV-DF)

**Citation:** Cai et al., "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization," DICTA 2022, https://arxiv.org/abs/2204.06228.

**Access:** The code uses Kaggle/Hugging Face-style preprocessed data paths. The public dataset card is at https://huggingface.co/datasets/ControlNet/LAV-DF.

**Total samples in local preprocessed experiment:** `[TO VERIFY from full metadata]`

**Train split:** 3400 entries loaded in logs.

**Validation split:** `[TO VERIFY from full metadata]`

**Test split:** 1550 entries loaded in logs.

**Class distribution:**

- Test real: 405 samples (from classification reports in attention-variant logs).
- Test fake: 1145 samples.
- Train distribution: `[TO VERIFY from full metadata]`

**Imbalance handling:** Teacher training calculates `pos_weight = num_real / num_fake` and uses it in `BCEWithLogitsLoss`.

**Preprocessing applied:**

- Video: 30 frames per sample; frames resized/padded/subsampled to 128 x 128; ImageNet normalization mean `[0.485, 0.456, 0.406]`, std `[0.229, 0.224, 0.225]`.
- Audio: MFCC features with 13 coefficients; variable sequence length padded per batch; code examples use 400 time steps for export/SHAP-style dummy inputs.
- Temporal alignment: two MFCC frames per video frame are assumed for offset/synchronization losses.

**Augmentation applied (training only):**

- JPEG degradation, random noise, Gaussian blur, downsample-resize degradation, color jitter, random erasing.
- Curriculum phase 1 masks 50-80% of video frames.
- Modality dropout probability: 0.15 during teacher training.

### Dataset 2

**Name:** FakeAVCeleb

**Citation:** Used for zero-shot cross-dataset evaluation in the rough PDF and `Cross-Dataset-Eval.py`; exact dataset citation should be verified before submission.

**Access:** Kaggle preprocessed path in code.

**Total samples / splits:** `[TO VERIFY from FakeAVCeleb metadata]`

**Use in paper:** Zero-shot cross-dataset evaluation table only.

---

## 3. Architecture

### 3.1 High-level Description

**Model name:** PIN-Lite

**One-sentence description:** PIN-Lite is a compact audio-visual cross-attention detector trained to preserve both the classification behavior and attention-map evidence of a larger multimodal teacher.

**Novel components:**

- Attention-aware distillation objective: combines hard-label BCE, teacher-logit KD, and cross-attention-map MSE.
- Explainability Preservation Score (EPS): compares teacher and student cross-attention maps using Spearman rank correlation and top-region IoU.
- Compression pipeline: distillation -> pruning -> quantization, evaluated jointly with accuracy, latency, memory, and EPS.
- Attention-variant analysis: evaluates Linear Attention, Multi-Query Attention, and Low-Rank Attention under the same EPS protocol.

### 3.2 Input Processing

**Video input:**

- Resolution: 128 x 128 RGB.
- Frames per sample: 30.
- Normalization: ImageNet mean/std.
- Special preprocessing: subsample if longer than 30 frames; zero-pad if shorter.

**Audio input:**

- Feature type: MFCC.
- Feature dimensions: 13 coefficients.
- Duration per sample: variable; padded within batch.
- Alignment assumption: two MFCC frames per video frame.

### 3.3 Teacher Feature Extractors / Backbones

**Visual backbone:**

- Architecture: ResNet-18, ImageNet-pretrained.
- Frozen layers: first six convolutional modules frozen.
- Output feature dimension: projected to 256.
- XAI compatibility: in-place ReLU and residual additions are patched.

**Audio backbone / encoder:**

- Architecture: Conv1D(13 -> 64) + Conv1D(64 -> 128) + LayerNorm + GRU.
- Output feature dimension: 256.
- Trained from scratch.

**Fusion module:**

- Three gated cross-attention blocks.
- Each block: LayerNorm, audio-to-video multi-head cross-attention, sigmoid gate, audio self-attention, feed-forward network.
- Heads: 8.
- Embedding dimension: 256.
- Dropout: 0.1.

**Prediction heads:**

- Classification head: linear projection from pooled audio features to one logit.
- Auxiliary offset head: linear projection to 11 temporal-offset classes (`MAX_OFFSET = 5`).

### 3.4 Student Feature Extractors / Backbones

**Visual backbone:**

- Architecture: MobileNetV3-Small, ImageNet-pretrained.
- Frozen layers: first three feature blocks frozen.
- Output feature dimension: MobileNet 576 channels projected to 128.

**Audio backbone:**

- Same CNN-GRU pattern as teacher, adapted to 128-dimensional output.

**Fusion module:**

- Two gated cross-attention blocks by default.
- Heads: 4.
- Embedding dimension: 128.
- Dropout: 0.15.

**Prediction heads:**

- Same binary classification and offset heads as teacher.

---

## 4. Training Procedure

### Teacher Training

**Hardware:** NVIDIA GPU acceleration used in logs; exact GPU model and VRAM `[TO VERIFY]`.

**Software:** PyTorch; exact version `[TO VERIFY]`.

**Optimizer:** AdamW.

**Learning rate:** 1e-4.

**Weight decay:** 1e-4.

**Learning rate schedule:** CosineAnnealingLR over `EPOCHS * len(train_loader)`.

**Batch size:** 4.

**Total epochs:** 15.

**Losses:**

- Classification: BCEWithLogitsLoss with optional class `pos_weight`.
- Offset: CrossEntropyLoss on real samples.
- Synchronization attention loss: direct MSE to a diagonal target, diagonal dominance penalty, and temporal smoothness penalty.

**Teacher total loss:**

```text
L_teacher = 1.0 * L_cls + 0.5 * L_offset + 3.0 * L_sync
L_sync = 1.0 * L_direct + 0.5 * L_dominance + 0.2 * L_smoothness
```

**Curriculum:**

- Epochs 1-2: video masking.
- Epochs 3-5: sync-focus phase with offset loss downweighted by 0.1.
- Epochs 6-15: full training.

**Regularization:**

- Dropout 0.1 in attention/FFN blocks.
- Modality dropout probability 0.15.
- Gradient clipping max norm 1.0.
- Mixed-precision training via GradScaler.

**Model selection:** best validation loss.

### Student Distillation

**Optimizer:** AdamW.

**Learning rate:** 2e-4.

**Weight decay:** inherited teacher config, 1e-4.

**Learning rate schedule:** CosineAnnealingLR.

**Batch size:** 8.

**Epochs:** 20.

**Distillation hyperparameters:** `alpha = 0.5`, `beta = 5.0`, `T = 2.0` in the default code path; ablations also show strong performance for `alpha = 0.3`, `beta = 3.0` in 10-epoch sweeps.

**Student total loss:**

```text
L_student = alpha * L_hard + (1 - alpha) * L_soft + beta * L_attn
```

where `L_hard` is BCE against ground-truth labels, `L_soft` is temperature-scaled KL divergence between teacher and student logits, and `L_attn` is MSE between teacher and student final cross-attention maps.

### Pruning

**Procedure:** iterative global L1 unstructured pruning over Linear and Conv2d layers.

**Iterations:** 3.

**Amount per iteration:** 20%.

**Fine-tuning after each iteration:** 3 epochs.

**Fine-tuning loss:** 0.5 hard BCE + 0.5 soft KL distillation; attention loss disabled during pruning fine-tuning because attention-map MSE conflicted with pruning hooks.

**Important implementation note:** `Combined-Pipeline.py` names this "structured pruning" in prints, but the actual implementation uses `prune.L1Unstructured` globally across prunable layers. The paper should call it "iterative global L1 unstructured pruning" unless the implementation is changed.

### Quantization

**Dynamic INT8:** PyTorch dynamic quantization of Linear and GRU layers.

**Static INT8:** attempted but failed in logs due to unsupported `empty_strided` on quantized tensors.

**ONNX quantization:** attempted but failed because the export specialized dynamic audio length to 400.

**FP16:** mixed-precision/autocast evaluation as an alternative.

---

## 5. Evaluation Metrics

**Accuracy:**

```text
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision:**

```text
Precision = TP / (TP + FP)
```

**Recall:**

```text
Recall = TP / (TP + FN)
```

**F1-score:**

```text
F1 = 2 * Precision * Recall / (Precision + Recall)
```

**AUC:** area under the ROC curve.

**Latency:** average inference time per sample in milliseconds, as reported by benchmark scripts.

**Peak VRAM:** maximum GPU memory recorded during evaluation.

**Explainability Preservation Score (EPS):**

Teacher and student final-layer cross-attention maps are flattened and compared with Spearman rank correlation and IoU of top-attended regions. The primary score reported in the comprehensive benchmark is consistent with the high-correlation-weight setting (`w = 0.7`) in `EPS-Enhanced.py`.

```text
EPS_w = w * Spearman(A_T, A_S) + (1 - w) * IoU_top20(A_T, A_S)
```

where `A_T` and `A_S` are teacher and student attention maps. `IoU_top20` binarizes each map by retaining the top 20% attended positions and computes intersection-over-union. Bootstrap confidence intervals use 1000 bootstrap iterations over 500 samples in `EPS-Enhanced.py`.

---

## 6. Results

### 6.1 Main Comprehensive Benchmark

Source: `pinlite results v3/comprehensive_benchmark_v3.csv`.

| Model | Size (MB) | Params (M) | Inference (ms) | Peak VRAM (MB) | Accuracy | Precision | Recall | F1 | AUC | EPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Base teacher | 57.32 | 15.00 | 98.62 | 886.10 | 0.9737 | 0.9843 | 0.9798 | 0.9821 | 0.9683 | 1.0000 |
| Distilled | 6.62 | 1.69 | 45.93 | 555.30 | 0.9753 | 0.9727 | 0.9944 | 0.9834 | 0.9584 | 0.6091 |
| Pruned | 6.62 | 1.69 | 44.18 | 555.30 | 0.9738 | 0.9710 | 0.9940 | 0.9824 | 0.9558 | 0.5887 |
| FP16 | 3.31 | 1.69 | 37.93 | 569.30 | 0.9752 | 0.9726 | 0.9943 | 0.9833 | 0.9582 | 0.6091 |
| Linear-Attn | 6.64 | 1.69 | 38.56 | 710.42 | 0.6082 | 0.7859 | 0.6421 | 0.7068 | 0.5780 | 0.0333 |
| MQA | 6.25 | 1.60 | 36.35 | 695.21 | 0.9800 | 0.9805 | 0.9926 | 0.9865 | 0.9689 | 0.6055 |
| LowRank | 6.26 | 1.60 | 38.28 | 703.82 | 0.9625 | 0.9714 | 0.9778 | 0.9746 | 0.9489 | 0.5719 |
| Combined | 5.29 | 1.22 | 171.29 | 0.00 | 0.9822 | 0.9837 | 0.9922 | 0.9879 | 0.9733 | 0.6093 |

**Takeaway:** The distilled and combined models preserve or improve in-domain accuracy while reducing the teacher footprint substantially; Linear Attention is the clear failure mode because its EPS collapses.

### 6.2 Compression Numbers

- Distilled parameter reduction: 15.0M -> 1.69M, an 88.7% reduction.
- Distilled size reduction: 57.32 MB -> 6.62 MB, an 8.66x reduction.
- Distilled latency speedup: 98.62 ms -> 45.93 ms, a 2.15x speedup.
- Combined size reduction: 57.32 MB -> 5.29 MB, a 10.83x reduction.
- Combined parameter reduction: 15.0M -> 1.22M, a 12.30x reduction.

### 6.3 Pipeline Run Results

Source: `pinlite results v3/combined-pipeline-results.txt`. This isolated run differs from the comprehensive benchmark and should be treated as a secondary pipeline trace.

| Stage | Accuracy | Precision | Recall | F1 | AUC | Latency (ms) | Size (MB) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Distilled input | 0.9865 | 0.9887 | 0.9930 | 0.9908 | 0.9805 | 47.62 | 6.62 |
| + Pruned | 0.9865 | 0.9938 | 0.9878 | 0.9908 | 0.9852 | 42.44 | 6.63 |
| + Quantized final | 0.9865 | 0.9938 | 0.9878 | 0.9908 | 0.9852 | 185.60 | 5.29 |

**Takeaway:** The pipeline run confirms that pruning and dynamic INT8 quantization can preserve classification metrics, but dynamic INT8 CPU inference is slower than GPU evaluation on this platform.

### 6.4 Ablation Study Results

**Loss-weight sweep:** 10-epoch student distillation runs.

| Alpha | Beta | Accuracy | F1 | AUC | Params (M) |
|---:|---:|---:|---:|---:|---:|
| 0.3 | 1.0 | 0.8981 | 0.9319 | 0.8560 | 1.69 |
| 0.3 | 3.0 | 0.9806 | 0.9870 | 0.9709 | 1.69 |
| 0.3 | 5.0 | 0.9639 | 0.9757 | 0.9468 | 1.69 |
| 0.5 | 1.0 | 0.9716 | 0.9808 | 0.9616 | 1.69 |
| 0.5 | 3.0 | 0.8948 | 0.9293 | 0.8578 | 1.69 |
| 0.5 | 5.0 | 0.9606 | 0.9736 | 0.9406 | 1.69 |

**Temperature sweep:**

| Temperature | Accuracy | F1 | AUC | Params (M) |
|---:|---:|---:|---:|---:|
| 1.0 | 0.9600 | 0.9730 | 0.9442 | 1.69 |
| 2.0 | 0.9619 | 0.9746 | 0.9391 | 1.69 |
| 4.0 | 0.8865 | 0.9237 | 0.8457 | 1.69 |
| 8.0 | 0.8871 | 0.9233 | 0.8566 | 1.69 |

**Depth sweep:**

| Layers | Accuracy | F1 | AUC | Params (M) |
|---:|---:|---:|---:|---:|
| 1 | 0.8819 | 0.9226 | 0.8180 | 1.41 |
| 2 | 0.8948 | 0.9305 | 0.8419 | 1.69 |
| 3 | 0.9742 | 0.9826 | 0.9618 | 1.98 |

**Pruning-rate sweep:**

| Pruning rate | Actual sparsity | Accuracy | F1 | AUC |
|---:|---:|---:|---:|---:|
| 0.0 | 0.0000 | 0.9865 | 0.9909 | 0.9805 |
| 0.2 | 0.0072 | 0.9826 | 0.9883 | 0.9738 |
| 0.5 | 0.0078 | 0.9852 | 0.9899 | 0.9828 |
| 0.6 | 0.0080 | 0.9794 | 0.9862 | 0.9645 |
| 0.8 | 0.0085 | 0.9845 | 0.9896 | 0.9768 |

### 6.5 EPS Analysis

Source: `pinlite results v3/enhanced-eps-results.txt`.

| Model | Spearman mean | Spearman 95% CI | IoU mean | IoU 95% CI | EPS w=0.7 | EPS 95% CI |
|---|---:|---|---:|---|---:|---|
| Distilled | 0.6559 | [0.6369, 0.6788] | 0.4947 | [0.4762, 0.5113] | 0.6076 | [0.5886, 0.6268] |
| Pruned | 0.6505 | [0.6302, 0.6698] | 0.4782 | [0.4600, 0.4946] | 0.5988 | [0.5800, 0.6172] |
| Linear-Attn | -0.0417 | [-0.0516, -0.0307] | 0.0911 | [0.0858, 0.0970] | -0.0019 | [-0.0100, 0.0071] |
| MQA | 0.6517 | [0.6326, 0.6707] | 0.4966 | [0.4802, 0.5139] | 0.6052 | [0.5875, 0.6228] |
| LowRank | 0.5749 | [0.5534, 0.5957] | 0.4739 | [0.4553, 0.4932] | 0.5446 | [0.5241, 0.5649] |
| Combined | 0.6555 | [0.6329, 0.6749] | 0.4929 | [0.4759, 0.5124] | 0.6067 | [0.5873, 0.6261] |

**Takeaway:** EPS separates explanation-preserving variants from efficient but attention-breaking variants.

### 6.6 Cross-Dataset / Generalization Results

Source: rough PDF table; raw CSV not found in repository.

| Model | FakeAVCeleb Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Base teacher | 0.507 | 0.109 | 0.502 |
| Distilled | 0.495 | 0.110 | 0.490 |
| Pruned | 0.498 | 0.113 | 0.493 |
| MQA | 0.507 | 0.098 | 0.502 |
| Linear-Attn | 0.515 | 0.295 | 0.511 |
| LowRank | 0.579 | 0.307 | 0.574 |

**Takeaway:** High-EPS students copy the teacher's out-of-domain failure pattern, which is evidence of behavioral fidelity rather than generalization success.

### 6.7 Methods Comparison Table

| Method | Year | Modality | Core technique | Compression | Interpretability | Key limitation |
|---|---:|---|---|---|---|---|
| M2TR | 2022 | RGB + frequency | Multi-scale transformer and cross-modality fusion | None | Limited | Heavy image-focused detector |
| AV-Lip-Sync+ | 2025 | Audio-video | AV-HuBERT + temporal CNN + face encoder | None | Limited | Large pretrained components |
| CAD | 2025 | Audio-video | Cross-modal alignment and distillation | Parameter-efficient modules | Limited | Not an edge compression study |
| DF-P2E | 2025 | Visual + language | Classifier, Grad-CAM, captioning, LLM explanations | 4-bit LLM quantization | Natural-language explanations | Very large explanation stack |
| Karathanasis et al. | 2025 | Visual | KD, pruning, quantization, adapters | Yes | None | Image-only and accuracy-centric |
| PIN-Lite | 2026 | Audio-video | Attention-aware KD, pruning, quantization, EPS | Yes | Cross-attention EPS | Needs broader cross-dataset validation |

---

## 7. Figures and Diagrams Needed

**Figure 1:**

- Type: architecture/workflow diagram.
- Proposed title: PIN-Lite compression and evaluation framework.
- What it shows: teacher model, attention-aware distillation into MobileNetV3 student, pruning, quantization, attention variants, and EPS measurement.
- Data source: `pinlite results v3/PIN-Lite Compression Framework Overview.png`.
- Priority: essential.

**Figure 2:**

- Type: EPS computation workflow.
- Proposed title: Explainability Preservation Score pipeline.
- What it shows: teacher/student attention extraction, flattening/ranking, top-20% binarization, Spearman, IoU, weighted EPS, bootstrap confidence interval.
- Data source: `pinlite results v3/EPS Computation Pipeline.png`.
- Priority: essential.

**Figure 3:**

- Type: component-level EPS results chart.
- Proposed title: Attention-fidelity components across model variants.
- What it shows: Spearman, IoU, cosine, and EPS differences among distilled, pruned, MQA, LowRank, Linear, and combined models.
- Data source: `pinlite results v3/EPS_Component_Breakdown_v3.png`.
- Priority: essential.

**Figure 4:**

- Type: Pareto results chart.
- Proposed title: Accuracy-latency-EPS Pareto analysis.
- What it shows: model variants plotted by latency versus accuracy and latency versus EPS; highlights MQA, FP16, and combined models.
- Data source: `pinlite results v3/Pareto_All_Models_v3.png`.
- Priority: essential.

**Figure 5:**

- Type: qualitative attention-map visualization.
- Proposed title: Teacher and student cross-attention maps on real and fake examples.
- What it shows: aligned teacher/student attention maps for representative true positive, true negative, and failure cases.
- Data source: attention visualization code in `PinPoint-main.py`; actual figure not yet found.
- Priority: recommended.

---

## 8. Limitations To Carry Into Draft

1. EPS currently compares flattened attention maps and top-k overlap, which may miss temporal structure such as diagonal continuity or localized shifts.
2. The main evaluation is LAV-DF-centered; the FakeAVCeleb cross-dataset experiment shows behavioral fidelity but not robust generalization.
3. The pruning implementation is unstructured and prunes only selected layers; it does not guarantee hardware speedups without sparse-kernel support.
4. Dynamic INT8 quantization reduced size but increased latency in the observed CPU benchmark.
5. Hardware measurements are incomplete for actual edge platforms such as Jetson, mobile NPUs, or embedded CPUs.

---

## 9. Claim Verification Checklist

- [x] Main accuracy / performance claim supported by section 6.1.
- [x] Compression claim supported by section 6.2.
- [x] Ablation claim supported by section 6.4.
- [x] EPS claim supported by section 6.5.
- [x] Attention-variant claim supported by sections 6.1 and 6.5.
- [x] Cross-dataset behavior claim supported by section 6.6, but raw CSV needs verification before submission.
- [x] Methods comparison table populated in section 6.7.
- [ ] Exact hardware/software stack needs verification before final submission.
- [ ] Dev split count and train class distribution need verification from metadata.

---

## 10. Structural Completeness Cross-reference

| Mandatory element from VENUE_NOTES.md | Data present? | Section | Action needed |
|---|---|---|---|
| Methods comparison table | Yes | 6.7 | Use in related work or experiments |
| Main results table | Yes | 6.1 | Use comprehensive benchmark as primary |
| Ablation table | Yes | 6.4 | Keep claims modest due mixed runs |
| Per-condition/model breakdown | Yes | 6.1, 6.4, 6.5 | Present by model variant |
| Custom metric with formula | Yes | 5 | Define EPS with numbered equation |
| Qualitative visualizations | Partial | 7 | Use placeholders until final attention-map examples are generated |
| Computational cost analysis | Yes | 6.1, 6.2 | Discuss dynamic INT8 latency caveat |
| Cross-dataset evaluation | Yes, secondary | 6.6 | Verify raw CSV before submission |
| Limitations | Yes | 8 | Include at least four concrete limitations |
| Declarations | Planned | Draft | Include Springer-required statements |

**Gaps identified:** hardware/software version, full metadata class distribution, raw cross-dataset CSV, and final qualitative attention examples.

