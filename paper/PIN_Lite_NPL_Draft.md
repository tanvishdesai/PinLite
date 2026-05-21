# PIN-Lite: Lightweight Multimodal Deepfake Detection with Explainability Preservation

**Palak Parmar**, **Chintan Bhatt**, **SantoshKumar Bharti**

**Target journal:** Neural Processing Letters

**Draft status:** Rough journal draft prepared from project code, result logs, and `paper/pinlite_paper (4).pdf`. This draft intentionally rewrites the paper structure and prose rather than following the rough PDF.

---

## Abstract

Multimodal deepfake detectors increasingly rely on large audio-visual transformers and foundation-model backbones, which limits their use on edge devices where fast and private media verification is needed. Existing compression studies reduce model size and latency, but they usually evaluate success through accuracy alone, leaving open whether a compact model still relies on the same forensic evidence as its teacher. We propose PIN-Lite, an explainability-preserving compression framework for multimodal deepfake detection. PIN-Lite distills a gated audio-visual cross-attention teacher into a MobileNetV3-based student using hard-label supervision, softened teacher logits, and attention-map transfer; then it evaluates pruning, INT8/FP16 quantization, and efficient attention variants under the same protocol. We also introduce the Explainability Preservation Score (EPS), which compares teacher and student cross-attention maps through Spearman rank correlation and top-attended-region overlap. On LAV-DF, the distilled student reduces parameters from 15.0M to 1.69M and model size from 57.32 MB to 6.62 MB while maintaining comparable accuracy (97.53% versus 97.37%) and retaining EPS = 0.609. The combined compressed model reaches 5.29 MB and 98.22% accuracy. Across attention variants, Multi-Query Attention achieves the best accuracy-efficiency-fidelity tradeoff (98.00% accuracy, EPS = 0.606), while Linear Attention collapses explanation fidelity (EPS = 0.033). These results show that explainability-aware evaluation is necessary when compressing forensic AI systems.

**Keywords:** deepfake detection; multimodal learning; knowledge distillation; explainability preservation; model compression; attention mechanisms

---

## 1 Introduction

Deepfake generation has shifted from a novelty to a practical risk for journalism, legal evidence, identity verification, and online moderation. Modern manipulation pipelines can alter faces, voices, and audio-visual synchronization in ways that are difficult for humans to detect reliably. This makes automatic deepfake detection necessary, but the deployment setting matters: a detector used on a mobile phone, camera, or local verification terminal must be fast, compact, and privacy-preserving enough to run near the data source.

The strongest recent detectors increasingly use multimodal features and transformer-based fusion. This direction is well motivated because audio-visual forgeries often leave inconsistencies between speech, lip motion, and facial dynamics. However, accuracy gains often arrive with large backbones and high inference cost. Audio-visual detectors based on self-supervised speech-vision encoders or large visual-language models can be effective, but their parameter counts and memory requirements make them difficult to place on edge devices. A practical forensic detector needs more than high benchmark accuracy; it must also fit the hardware envelope where verification is needed.

Model compression offers a natural response to this deployment problem. Knowledge distillation, pruning, and quantization can reduce the size and latency of deep neural networks, and recent studies show that compressed deepfake detectors can retain in-domain classification performance. Yet forensic detection has an additional requirement that ordinary compression metrics do not capture. A compressed model may produce the correct label while attending to evidence that differs from the teacher's evidence. In a forensic setting, this is not a harmless implementation detail. Analysts may need to explain why a video was flagged, and a compact model that reaches the right answer for the wrong reason is less trustworthy than its accuracy suggests.

This paper addresses the joint problem of efficiency and explanation fidelity. We propose PIN-Lite, a compression framework for a multimodal audio-visual deepfake detector based on gated cross-attention. The original teacher uses a ResNet-18 visual encoder, a CNN-GRU audio encoder, and stacked gated cross-attention layers. PIN-Lite replaces the teacher visual pathway with a MobileNetV3-Small student, reduces the embedding dimension and attention depth, and trains the student with an attention-aware distillation objective that aligns the teacher and student cross-attention maps. The compressed models are then evaluated not only by accuracy, F1-score, latency, parameter count, and memory footprint, but also by whether their cross-modal attention remains faithful to the teacher.

The central evaluation tool is the Explainability Preservation Score (EPS). EPS is an architecture-agnostic metric for attention-map fidelity. It compares teacher and student final-layer cross-attention maps using Spearman rank correlation and intersection-over-union over top-attended regions. This gives a compact scalar measure of whether the student preserves the teacher's audio-visual reasoning pattern. EPS also exposes a failure mode that accuracy-centric evaluation would miss: efficient attention replacements can be small and fast while destroying attention structure.

Our contributions are as follows:

1. We introduce PIN-Lite, an audio-visual deepfake detector compression framework that combines attention-aware knowledge distillation, iterative pruning, and post-training quantization.
2. We define EPS, a quantitative metric for preserving teacher cross-attention explanations during compression.
3. We evaluate standard and efficient attention variants under a joint accuracy-efficiency-explainability protocol, showing that Multi-Query Attention preserves the best tradeoff while Linear Attention fails to preserve explanations.
4. We provide ablation studies over distillation temperature, loss weights, student depth, and pruning rate, offering practical guidance for compact multimodal forensic models.
5. We report in-domain LAV-DF results and a zero-shot FakeAVCeleb analysis that distinguish generalization from behavioral fidelity.

---

## 2 Related Work

### 2.1 Multimodal Deepfake Detection

Early deepfake detectors focused heavily on visual artifacts, but audio-visual manipulation makes single-modality evidence incomplete. M2TR [1] combines RGB and frequency-domain streams with multi-scale transformers to capture local image forgery cues. Its success illustrates the value of multimodal feature fusion, but the method remains image/frequency oriented and does not address audio-visual synchronization or edge compression.

Audio-visual detectors extend this idea by modeling consistency between speech and face dynamics. AV-Lip-Sync+ [2] uses AV-HuBERT features and temporal modeling to exploit inconsistency between acoustic and visual streams, while CAD [3] combines cross-modal alignment and distillation to reconcile semantic synchronization cues with modality-specific artifacts. These approaches demonstrate that multimodal evidence is essential, but they rely on comparatively large backbones or do not evaluate the effect of compression on explanation fidelity.

Interpretability-oriented deepfake work has also begun to move beyond binary labels. DF-P2E [4] combines classification, saliency, captioning, and language-model-based explanation for non-expert users. This is valuable for human-facing explanation, but it pushes the system toward heavier visual-language components. PIN-Lite takes a different route: instead of generating explanations with an additional model stack, it treats the cross-attention map inside the detector as the explanation object that must survive compression.

The gap is therefore not merely "detect deepfakes better." The gap is to make multimodal detectors compact while preserving the evidence structure that makes their predictions inspectable.

### 2.2 Compression for Efficient Deepfake Detection

Knowledge distillation trains a compact student to match a larger teacher's softened predictions [5]. Pruning removes parameters or structures, and quantization reduces numerical precision for efficient inference [6]. These tools are widely used in efficient deep learning, and MobileNetV3 provides a strong lightweight visual backbone for mobile settings [7].

In deepfake detection, Karathanasis et al. [8] evaluated compression and transfer-learning techniques for image-based detectors and found that high compression can preserve same-domain performance but may degrade cross-domain behavior. This result is important because it shows both the promise and fragility of compression in forensic tasks. However, their evaluation remains mostly accuracy-centered and image-only. PIN-Lite extends this line of work to audio-visual detection and adds an explicit explanation-fidelity metric.

Compression is not automatically useful unless it translates into deployment benefits. Dynamic INT8 quantization, for example, can reduce storage but may increase latency on hardware without optimized INT8 execution. For this reason, PIN-Lite reports model size, parameter count, latency, peak memory, and EPS together instead of treating a smaller file as sufficient evidence of deployability.

### 2.3 Explanation-Preserving Distillation and Attention Efficiency

Several studies show that conventional distillation may transfer predictions without transferring the teacher's reasons. XDistillation [9] and e2KD [10] explicitly align student explanations with teacher explanations, improving the likelihood that the student is "right for the right reasons." Interpretability-aware pruning [11] similarly argues that pruning decisions should consider which model components carry relevant evidence, not only which weights are small.

PIN-Lite builds on this insight in the multimodal setting. The teacher's cross-attention map already indicates how audio frames attend to video frames, so explanation transfer can be performed directly on the attention structure. This avoids training an auxiliary explanation generator and gives a domain-relevant fidelity signal.

Efficient attention mechanisms create a second pressure point. Linear Attention [12] reduces attention complexity by replacing softmax attention with kernelized operations, while Multi-Query Attention [13] shares key and value projections to reduce memory bandwidth. These changes can improve efficiency, but they may also change the geometry of the attention map. PIN-Lite therefore evaluates attention variants with EPS, not only with accuracy and latency.

---

## 3 Proposed Method

[DIAGRAM PLACEHOLDER]
Type: architecture diagram
Title: PIN-Lite compression and evaluation framework
Shows: The figure should show the ResNet-18/CNN-GRU gated cross-attention teacher on the left, the MobileNetV3/CNN-GRU student on the right, and the training losses between them: hard-label BCE, softened-logit KL divergence, and attention-map MSE. A second row should show pruning, INT8/FP16 quantization, attention variants, and EPS evaluation. The reader should conclude that compression and explanation preservation are evaluated as a single pipeline.
Priority: essential
[END PLACEHOLDER]

### 3.1 Teacher Model

The teacher is a multimodal audio-visual detector with three components: a video encoder, an audio encoder, and gated cross-attention fusion. Given a video tensor \(x_v \in R^{T \times 3 \times 128 \times 128}\) with \(T = 30\) frames, an ImageNet-pretrained ResNet-18 extracts per-frame visual features. The early ResNet layers are frozen, and the pooled feature vector is projected to a 256-dimensional embedding:

```text
V = W_v pool(ResNet18(x_v)) + P_v.                                      (1)
```

Here, \(P_v\) is a learned temporal positional embedding. The ResNet implementation is patched to avoid in-place operations so that gradient-based explanation methods remain compatible.

The audio pathway receives 13-dimensional MFCC sequences. Two one-dimensional convolution layers map the MFCC sequence from 13 to 64 and then 128 channels. Layer normalization and a GRU then produce a 256-dimensional audio sequence:

```text
A = GRU(LN(Conv1D_2(ReLU(Conv1D_1(x_a))))) + P_a.                       (2)
```

The positional term \(P_a\) is sinusoidal. The code assumes two MFCC frames per video frame for the auxiliary offset and synchronization objectives.

Audio and video are fused through three gated cross-attention blocks. In each block, normalized audio queries attend to normalized video keys and values:

```text
C = MHA(Q = LN(A), K = LN(V), V = LN(V)).                                (3)
```

The cross-attention output is added to the audio sequence, gated with a learned sigmoid projection, passed through audio self-attention, and refined through a feed-forward network. The final audio sequence is mean-pooled and passed to a binary classification head. A separate offset head predicts temporal misalignment classes from \(-5\) to \(+5\) frames.

### 3.2 PIN-Lite Student

The student keeps the teacher's input-output interface but reduces the expensive components. The visual encoder is replaced with MobileNetV3-Small pretrained on ImageNet, with the first three feature blocks frozen. Its 576-channel pooled representation is projected to a 128-dimensional embedding. The student keeps the CNN-GRU audio encoder pattern but reduces the embedding dimension from 256 to 128, reduces attention heads from 8 to 4, and uses two gated cross-attention layers by default.

This design preserves compatibility between teacher and student attention maps. Even though the internal embedding dimension changes, the final cross-attention map remains an audio-time by video-time matrix, allowing direct fidelity measurement.

### 3.3 Attention-Aware Knowledge Distillation

The student is trained with a three-part objective:

```text
L_student = alpha L_hard + (1 - alpha) L_soft + beta L_attn.             (4)
```

The hard-label term is binary cross-entropy:

```text
L_hard = BCEWithLogits(z_s, y).                                          (5)
```

The soft distillation term is the temperature-scaled KL divergence between teacher and student logits:

```text
L_soft = T^2 KL(softmax(z_t / T) || softmax(z_s / T)).                   (6)
```

The attention-transfer term is mean squared error between the teacher and student final cross-attention maps:

```text
L_attn = || A_t - A_s ||_F^2.                                            (7)
```

The default student training configuration uses \(T = 2.0\), \(\alpha = 0.5\), and \(\beta = 5.0\), with AdamW, learning rate \(2 \times 10^{-4}\), batch size 8, cosine annealing, mixed precision, and gradient clipping with maximum norm 1.0. Ablations show that smaller \(\alpha\) and moderate attention weight can be effective in shorter training schedules.

### 3.4 Pruning and Quantization

After distillation, PIN-Lite applies iterative global L1 unstructured pruning to Linear and Conv2d layers. Each pruning cycle removes the lowest-magnitude weights, makes the pruning mask permanent, fine-tunes the student for three epochs, and reapplies the saved zero mask. Fine-tuning uses a hard-label BCE term and a softened-logit KL term; the attention loss is disabled during pruning recovery because the PyTorch pruning hooks created graph conflicts when backpropagating through returned attention maps.

Dynamic INT8 quantization is then applied to Linear and GRU layers. This reduces storage size but shifts inference to CPU kernels in the observed benchmark, so latency must be interpreted by hardware backend. FP16 mixed-precision evaluation is also included as an alternative path for GPU-equipped edge devices.

### 3.5 Explainability Preservation Score

[DIAGRAM PLACEHOLDER]
Type: workflow
Title: Explainability Preservation Score pipeline
Shows: For each sample, teacher and student final-layer cross-attention maps are extracted, flattened, ranked, and thresholded into top-20% masks. The workflow computes Spearman rank correlation, top-region IoU, and their weighted combination. A bootstrap loop estimates 95% confidence intervals. The reader should see EPS as a metric over reasoning fidelity, not classification accuracy.
Priority: essential
[END PLACEHOLDER]

EPS measures whether a compressed model preserves the teacher's cross-modal attention structure. Let \(A_t\) and \(A_s\) denote teacher and student attention maps for the same input. We define:

```text
EPS_w(A_t, A_s) = w rho_s(vec(A_t), vec(A_s)) +
                  (1 - w) IoU(top_k(A_t), top_k(A_s)).                  (8)
```

Here, \(\rho_s\) is Spearman rank correlation, \(top_k(\cdot)\) retains the top 20% attended positions, and IoU is the intersection-over-union of the resulting binary masks. The benchmark reports EPS using a correlation-heavy setting consistent with \(w = 0.7\), and the sensitivity analysis also reports \(w = 0.3\) and \(w = 0.5\). Bootstrap confidence intervals are computed over 500 samples with 1000 bootstrap iterations.

---

## 4 Experiments

### 4.1 Dataset and Setup

Experiments use LAV-DF for supervised evaluation. The preprocessed logs report 3400 training samples and 1550 test samples. The test split contains 405 real and 1145 fake samples in the available classification reports. Each video sample is represented by 30 RGB frames at 128 x 128 resolution and a padded sequence of 13-dimensional MFCC audio features.

The teacher is trained for 15 epochs using AdamW with learning rate \(1 \times 10^{-4}\), weight decay \(1 \times 10^{-4}\), batch size 4, cosine annealing, and a curriculum that begins with video masking, moves to synchronization-focused training, and finishes with full training. The teacher loss combines classification, offset prediction, and synchronization attention losses.

The student is trained for 20 epochs with AdamW, learning rate \(2 \times 10^{-4}\), batch size 8, and the distillation objective in Eq. (4). Model selection uses validation loss or validation accuracy depending on the script path. Hardware was NVIDIA GPU accelerated, but exact GPU model, CUDA version, and PyTorch version should be verified before submission.

### 4.2 Main Results on LAV-DF

Table 1 reports the comprehensive benchmark across the teacher, compressed variants, and efficient attention variants.

**Table 1. Comprehensive LAV-DF benchmark for PIN-Lite variants.**

| Model | Size (MB) | Params (M) | Latency (ms) | Peak VRAM (MB) | Accuracy | F1 | AUC | EPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher | 57.32 | 15.00 | 98.62 | 886.10 | 0.9737 | 0.9821 | 0.9683 | 1.0000 |
| Distilled | 6.62 | 1.69 | 45.93 | 555.30 | 0.9753 | 0.9834 | 0.9584 | 0.6091 |
| Pruned | 6.62 | 1.69 | 44.18 | 555.30 | 0.9738 | 0.9824 | 0.9558 | 0.5887 |
| FP16 | 3.31 | 1.69 | 37.93 | 569.30 | 0.9752 | 0.9833 | 0.9582 | 0.6091 |
| Linear-Attn | 6.64 | 1.69 | 38.56 | 710.42 | 0.6082 | 0.7068 | 0.5780 | 0.0333 |
| MQA | 6.25 | 1.60 | 36.35 | 695.21 | 0.9800 | 0.9865 | 0.9689 | 0.6055 |
| LowRank | 6.26 | 1.60 | 38.28 | 703.82 | 0.9625 | 0.9746 | 0.9489 | 0.5719 |
| Combined | 5.29 | 1.22 | 171.29 | 0.00 | 0.9822 | 0.9879 | 0.9733 | 0.6093 |

The distilled student reduces parameters by 88.7% and model size by 8.66x while slightly improving accuracy over the teacher. This indicates that the teacher has compressible redundancy and that attention-aware distillation transfers sufficient task behavior to the student.

The combined model obtains the smallest deployed footprint among the main pipeline variants, reducing the file size from 57.32 MB to 5.29 MB. Its latency is higher in the comprehensive benchmark because dynamic INT8 inference runs on CPU in the tested PyTorch path; this result should not be interpreted as an inherent failure of quantization, but as evidence that deployment claims must be hardware-specific.

### 4.3 Attention Variant Analysis

The attention-variant results show why accuracy is not enough for compact forensic models. Multi-Query Attention improves the accuracy-latency tradeoff while preserving EPS close to the distilled student. Linear Attention, by contrast, reduces the model to a compact form but collapses both accuracy and explanation fidelity in the comprehensive benchmark.

[DIAGRAM PLACEHOLDER]
Type: results chart
Title: Accuracy-latency-EPS Pareto analysis
Shows: Two panels. Panel (a) plots accuracy versus latency for teacher, distilled, pruned, FP16, combined, Linear-Attn, MQA, and LowRank. Panel (b) plots EPS versus latency for the same models. The figure should highlight that MQA lies on the desirable frontier, while Linear Attention is dominated because it loses attention fidelity.
Priority: essential
[END PLACEHOLDER]

The key finding is that efficient attention changes are not interchangeable. MQA shares keys and values while preserving per-head query diversity, so the resulting attention geometry remains close to the teacher. Linear Attention replaces softmax attention with a kernelized approximation that can flatten or distort attention structure; in this task, that distortion is visible as an EPS collapse.

### 4.4 Explainability Preservation

Table 2 decomposes EPS into its rank-correlation and top-region-overlap components.

**Table 2. Attention-map fidelity with 95% bootstrap confidence intervals.**

| Model | Spearman | Spearman 95% CI | IoU | IoU 95% CI | EPS (w=0.7) | EPS 95% CI |
|---|---:|---|---:|---|---:|---|
| Distilled | 0.6559 | [0.6369, 0.6788] | 0.4947 | [0.4762, 0.5113] | 0.6076 | [0.5886, 0.6268] |
| Pruned | 0.6505 | [0.6302, 0.6698] | 0.4782 | [0.4600, 0.4946] | 0.5988 | [0.5800, 0.6172] |
| Linear-Attn | -0.0417 | [-0.0516, -0.0307] | 0.0911 | [0.0858, 0.0970] | -0.0019 | [-0.0100, 0.0071] |
| MQA | 0.6517 | [0.6326, 0.6707] | 0.4966 | [0.4802, 0.5139] | 0.6052 | [0.5875, 0.6228] |
| LowRank | 0.5749 | [0.5534, 0.5957] | 0.4739 | [0.4553, 0.4932] | 0.5446 | [0.5241, 0.5649] |
| Combined | 0.6555 | [0.6329, 0.6749] | 0.4929 | [0.4759, 0.5124] | 0.6067 | [0.5873, 0.6261] |

The distilled, MQA, and combined variants preserve a non-trivial share of the teacher's attention structure. Pruning causes only a small additional reduction in EPS. LowRank attention remains partially aligned, but less so than MQA. Linear Attention is the only variant with near-zero or negative rank correlation, which means its attention ordering no longer resembles the teacher's.

[DIAGRAM PLACEHOLDER]
Type: comparison chart
Title: Component-level EPS analysis across model variants
Shows: Grouped bars for Spearman, top-20% IoU, cosine similarity, and EPS for each model. The chart should make clear that cosine similarity can remain high even when rank structure collapses, while Spearman and EPS separate Linear Attention from the other variants.
Priority: essential
[END PLACEHOLDER]

### 4.5 Ablation Studies

Table 3 summarizes the strongest ablation patterns. The loss-weight sweep shows that attention transfer weight matters substantially. In 10-epoch runs, \(\alpha = 0.3\), \(\beta = 3.0\) reaches 98.06% accuracy, while \(\alpha = 0.3\), \(\beta = 1.0\) reaches only 89.81%. This supports the view that cross-attention supervision is not merely decorative; it stabilizes the student training signal.

**Table 3. Distillation and architecture ablations.**

| Ablation | Setting | Accuracy | F1 | AUC | Params (M) |
|---|---|---:|---:|---:|---:|
| Loss weights | alpha=0.3, beta=1.0 | 0.8981 | 0.9319 | 0.8560 | 1.69 |
| Loss weights | alpha=0.3, beta=3.0 | 0.9806 | 0.9870 | 0.9709 | 1.69 |
| Loss weights | alpha=0.5, beta=1.0 | 0.9716 | 0.9808 | 0.9616 | 1.69 |
| Temperature | T=1.0 | 0.9600 | 0.9730 | 0.9442 | 1.69 |
| Temperature | T=2.0 | 0.9619 | 0.9746 | 0.9391 | 1.69 |
| Temperature | T=4.0 | 0.8865 | 0.9237 | 0.8457 | 1.69 |
| Depth | 1 layer | 0.8819 | 0.9226 | 0.8180 | 1.41 |
| Depth | 2 layers | 0.8948 | 0.9305 | 0.8419 | 1.69 |
| Depth | 3 layers | 0.9742 | 0.9826 | 0.9618 | 1.98 |

Low distillation temperatures are preferable in this binary classification setting. Higher temperatures \(T = 4.0\) and \(T = 8.0\) reduce accuracy sharply in the ablation logs, suggesting that excessive softening may remove useful decision-boundary information. The depth sweep shows that a three-layer student improves accuracy substantially, but the two-layer default offers a smaller footprint and was used in the main compressed student configuration.

### 4.6 Cross-Dataset Behavioral Fidelity

Table 4 reports zero-shot evaluation on FakeAVCeleb, using numbers from the rough result ledger. These results should be interpreted as behavioral-fidelity evidence rather than as a claim of strong cross-dataset generalization.

**Table 4. Zero-shot FakeAVCeleb evaluation.**

| Model | Accuracy | F1 | AUC |
|---|---:|---:|---:|
| Teacher | 0.507 | 0.109 | 0.502 |
| Distilled | 0.495 | 0.110 | 0.490 |
| Pruned | 0.498 | 0.113 | 0.493 |
| MQA | 0.507 | 0.098 | 0.502 |
| Linear-Attn | 0.515 | 0.295 | 0.511 |
| LowRank | 0.579 | 0.307 | 0.574 |

The teacher performs near chance on FakeAVCeleb, revealing substantial domain shift from LAV-DF. High-EPS variants reproduce this failure mode closely: MQA matches the teacher's accuracy and AUC, while the distilled and pruned models remain near the teacher. This is not desirable generalization, but it is evidence that high-EPS students behave like the teacher under distribution shift. In contrast, low-fidelity variants diverge unpredictably, which is exactly the kind of behavior EPS is designed to flag.

---

## 5 Discussion

### 5.1 Accuracy Alone Hides Forensic Risk

PIN-Lite shows that compression must be evaluated as a multi-objective problem. A model can be small, fast, and accurate on an in-domain test set while failing to preserve the teacher's explanation structure. In forensic AI, this matters because the explanation is part of the system's practical value. A detector that flags a video but cannot preserve stable evidence patterns is harder to trust in settings where analysts, journalists, or legal professionals need to inspect the basis of a prediction.

EPS is not proposed as a universal explanation metric. It is a targeted fidelity measure for attention-based multimodal detectors. Its usefulness comes from matching the model's own reasoning interface: the cross-attention map indicates which audio-video temporal correspondences influence the decision. If a compressed model changes that map completely, its label agreement with the teacher is less reassuring.

### 5.2 Why MQA Works Better Than Linear Attention

The attention-variant results suggest that preserving the softmax attention structure is important in this task. MQA reduces parameters and memory movement by sharing key and value projections, but it keeps query diversity across heads. This appears to retain the shape of cross-modal attention while improving efficiency.

Linear Attention makes a deeper change: it replaces softmax attention with a kernelized approximation. That change may be useful for long-sequence autoregressive modeling, but the LAV-DF attention maps are short audio-video alignment structures where the relative ordering of attended positions carries forensic meaning. In PIN-Lite, the linear approximation destroys that ordering, which EPS reveals directly.

### 5.3 Deployment Implications

The compressed models offer different deployment choices. The distilled model provides a strong general balance: 1.69M parameters, 6.62 MB, and a 2.15x latency improvement over the teacher in the comprehensive benchmark. FP16 gives the smallest GPU-friendly representation among the tested student variants. The combined INT8 model gives the smallest file size but needs hardware-aware optimization before claiming latency benefits. On devices with efficient INT8 acceleration, dynamic or static quantization may become more attractive; on GPU-equipped edge devices, FP16 may be the practical path.

### 5.4 Limitations

First, EPS flattens attention maps before computing rank correlation. This captures global ordering but does not fully model temporal structure, diagonal continuity, or localized shifts between audio and video. Future versions should evaluate structural similarity on the two-dimensional attention map and temporal alignment metrics that respect the audio-video grid.

Second, the main evidence is LAV-DF-centered. The FakeAVCeleb experiment is useful because it shows behavioral fidelity under shift, but it also shows that neither the teacher nor the students generalize well in zero-shot form. Stronger claims require training and evaluation across LAV-DF, FakeAVCeleb, IDForge, and in-the-wild manipulation sources.

Third, the pruning implementation is unstructured and does not guarantee real latency gains on ordinary hardware. Structured channel pruning or attention-head pruning could provide more predictable acceleration, especially if guided by EPS-aware importance measures.

Fourth, the quantization results are backend-dependent. Dynamic INT8 reduces storage but increases latency in the observed CPU path. A fair deployment study should evaluate optimized ONNX Runtime, TensorRT, mobile NPU, and Jetson backends.

Fifth, the current draft still needs exact hardware/software version reporting, full train/dev/test class distribution, and raw cross-dataset CSV verification before submission.

---

## 6 Conclusion

PIN-Lite addresses the practical problem of deploying multimodal deepfake detection under resource constraints without discarding the explanation signal needed for forensic trust. By combining attention-aware distillation, pruning, quantization, and EPS-based evaluation, PIN-Lite compresses a 15.0M-parameter teacher into compact student variants while preserving in-domain accuracy and a measurable portion of the teacher's cross-attention structure.

The main empirical lesson is that compactness and accuracy are not enough. MQA preserves both performance and attention fidelity, whereas Linear Attention demonstrates that an efficient architectural substitution can destroy explanation structure. This finding supports the broader recommendation that compressed forensic models should be evaluated with task metrics, efficiency metrics, and explanation-fidelity metrics together.

Future work will extend PIN-Lite to larger cross-dataset evaluations, improve EPS with temporal-structural similarity, develop structured EPS-guided pruning, and test INT8/FP16 deployment on physical edge hardware.

---

## Statements and Declarations

**Funding:** Not applicable / to be confirmed by authors.

**Competing interests:** The authors declare no competing interests / to be confirmed.

**Data availability:** LAV-DF is publicly available subject to its dataset terms. Preprocessed metadata and split details should be released or described upon publication.

**Code availability:** Code availability statement to be finalized. Recommended: release training, evaluation, EPS, and plotting scripts at publication.

**Author contributions:** To be finalized by authors.

**AI-assisted writing disclosure:** This draft was prepared with AI-assisted drafting support from Codex and requires full author review, correction, and approval before submission. Springer Nature's current guidance states that LLMs do not qualify for authorship and substantive use beyond copy editing should be documented appropriately.

---

## References

[1] Wang J, Wu Z, Ouyang W, Han X, Chen J, Lim S-N, Jiang Y-G. M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection. ICMR 2022. https://arxiv.org/abs/2104.09770

[2] Shahzad SA, Hashmi A, Peng Y-T, Tsao Y, Wang H-M. AV-Lip-Sync+: Leveraging AV-HuBERT to Exploit Multimodal Inconsistency for Video Deepfake Detection of Frontal Face Videos. IEEE Transactions on Human-Machine Systems, 2025. https://arxiv.org/abs/2311.02733

[3] Du Y, Wang Z, Luo Y, Piao C, Yan Z, Li H, Yuan L. CAD: A General Multimodal Framework for Video Deepfake Detection via Cross-Modal Alignment and Distillation. arXiv 2025. https://arxiv.org/abs/2505.15233

[4] Tariq S, Woo SS, Singh P, Irmalasari I, Gupta S, Gupta D. From Prediction to Explanation: Multimodal, Explainable, and Interactive Deepfake Detection Framework for Non-Expert Users. ACM MM 2025. https://arxiv.org/abs/2508.07596

[5] Hinton G, Vinyals O, Dean J. Distilling the Knowledge in a Neural Network. arXiv 2015. https://arxiv.org/abs/1503.02531

[6] Jacob B, Kligys S, Chen B, Zhu M, Tang M, Howard A, Adam H, Kalenichenko D. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference. CVPR 2018. https://arxiv.org/abs/1712.05877

[7] Howard A, Sandler M, Chu G, Chen L-C, Chen B, Tan M, Wang W, Zhu Y, Pang R, Vasudevan V, Le QV, Adam H. Searching for MobileNetV3. ICCV 2019. https://arxiv.org/abs/1905.02244

[8] Karathanasis A, Violos J, Kompatsiaris I, Papadopoulos S. A Brief Review for Compression and Transfer Learning Techniques in DeepFake Detection. arXiv 2025. https://arxiv.org/abs/2504.21066

[9] Alharbi R, Vu MN, Thai MT. Learning Interpretation with Explainable Knowledge Distillation. arXiv 2021. https://arxiv.org/abs/2111.06945

[10] Parchami-Araghi A, Bohle M, Rao S, Schiele B. Good Teachers Explain: Explanation-Enhanced Knowledge Distillation. ECCV 2024. https://arxiv.org/abs/2402.03119

[11] Malik N, Seth P, Singh NK, Chitroda C, Sankarapu VK. Interpretability-Aware Pruning for Efficient Medical Image Analysis. arXiv 2025. https://arxiv.org/abs/2507.08330

[12] Katharopoulos A, Vyas A, Pappas N, Fleuret F. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020. https://arxiv.org/abs/2006.16236

[13] Shazeer N. Fast Transformer Decoding: One Write-Head is All You Need. arXiv 2019. https://arxiv.org/abs/1911.02150

[14] Vaswani A, Shazeer N, Parmar N, Uszkoreit J, Jones L, Gomez AN, Kaiser L, Polosukhin I. Attention Is All You Need. NeurIPS 2017. https://arxiv.org/abs/1706.03762

[15] Cai Z, Stefanov K, Dhall A, Hayat M. Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization. DICTA 2022. https://arxiv.org/abs/2204.06228

[16] He K, Zhang X, Ren S, Sun J. Deep Residual Learning for Image Recognition. CVPR 2016. https://arxiv.org/abs/1512.03385

[17] Chakraborty S, Chatterjee K, Dey P. Detection of Image Tampering Using Deep Learning, Error Levels and Noise Residuals. Neural Processing Letters 56, 112 (2024). https://link.springer.com/article/10.1007/s11063-024-11448-9

[18] Pasen M, Boza V. Merging of Neural Networks. Neural Processing Letters 56, 8 (2024). https://link.springer.com/article/10.1007/s11063-024-11445-y

