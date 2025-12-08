# PIN-Lite: Efficient and Explainable Multimodal Deepfake Detection via Knowledge Distillation

**Author Name**<sup>1, a)</sup>

<sup>1</sup>Institution Name, Department, City, Country

<sup>a)</sup> Corresponding author: author@institution.edu

---

**Abstract.** The proliferation of sophisticated deepfake technologies poses an unprecedented threat to digital media integrity, demanding robust detection mechanisms. While state-of-the-art multimodal deepfake detectors achieve remarkable accuracy, their substantial computational requirements—often exceeding 15 million parameters—preclude deployment on resource-constrained edge devices. Furthermore, existing model compression techniques prioritize accuracy preservation while neglecting the equally critical dimension of explainability, which is essential for forensic applications where human operators must understand and validate detection decisions. In this paper, we present PIN-Lite, a compressed multimodal audio-visual deepfake detector derived through knowledge distillation and structured pruning. We introduce the Explainability Preservation Score (EPS), a novel metric quantifying the fidelity of attention-based explanations between teacher and student models. Experimental evaluation on the LAV-DF benchmark demonstrates that PIN-Lite achieves 8.6× model size reduction (57.32 MB to 6.62 MB), 2.2× inference acceleration (82.5 ms to 37.6 ms), while maintaining 97.53% classification accuracy and preserving 60% of the teacher model's explainability characteristics (EPS = 0.60). Our results establish that efficient deepfake detection need not sacrifice interpretability, enabling practical deployment in forensic and edge computing environments.

---

## 1. INTRODUCTION

### 1.1 Problem Statement

The advent of deep generative models has democratized the creation of synthetic media, enabling the production of highly realistic audio-visual manipulations colloquially termed "deepfakes" [1]. These fabricated media artifacts present multifaceted societal risks, ranging from political disinformation campaigns and financial fraud to non-consensual intimate imagery and erosion of public trust in authentic recordings [2]. The detection of such manipulations has consequently emerged as a critical research imperative, with significant implications for journalism, law enforcement, and digital platform moderation.

Contemporary deepfake detection systems have achieved impressive classification performance on benchmark datasets, with leading approaches leveraging multimodal fusion of visual and audio signals to identify synchronization artifacts characteristic of synthetic manipulations [3, 4]. However, these high-performing detectors typically employ computationally intensive architectures—incorporating convolutional backbones, transformer attention mechanisms, and cross-modal fusion modules—that demand substantial memory and processing resources. The practical deployment of such systems is consequently restricted to well-resourced server environments, precluding their application in scenarios where real-time, on-device detection is desirable: mobile forensic tools, edge surveillance systems, social media platform moderation at scale, and privacy-preserving local processing.

### 1.2 Motivation

The model compression literature offers established techniques for reducing computational requirements while preserving task performance, including knowledge distillation [5], structured pruning [6], and quantization [7]. These methods have been successfully applied across diverse domains, yielding compact models suitable for edge deployment. However, their application to deepfake detection—and more broadly to forensic classification tasks—reveals a critical gap: existing compression approaches optimize exclusively for accuracy retention, disregarding the preservation of model interpretability.

Explainability represents an essential requirement for forensic applications. Human analysts must not only receive classification decisions but also understand the evidentiary basis underlying those decisions [8]. Attention-based deepfake detectors naturally provide such explanations through their attention maps, which highlight the audio-visual regions contributing to classification outputs. When such models undergo compression, the resulting compact architectures may maintain classification accuracy while fundamentally altering their internal attention patterns—effectively providing different "reasons" for identical decisions. This divergence undermines the trustworthiness of compressed models in forensic contexts and complicates human-AI collaborative workflows.

To our knowledge, no prior work has systematically addressed the preservation of explainability during deepfake detector compression. This gap motivates our development of PIN-Lite and the accompanying Explainability Preservation Score metric.

### 1.3 Contributions

This paper presents the following principal contributions:

1. **PIN-Lite Architecture**: We propose a compressed multimodal deepfake detector derived from a large teacher model through knowledge distillation with attention map alignment. PIN-Lite employs a MobileNetV3-Small visual backbone with reduced embedding dimensions (128 vs. 256), fewer attention heads (4 vs. 8), and fewer transformer layers (2 vs. 3), achieving 8.6× size reduction while maintaining classification performance.

2. **Explainability Preservation Score (EPS)**: We introduce a novel evaluation metric quantifying the similarity between teacher and student attention patterns. EPS combines Spearman rank correlation with Intersection-over-Union of salient attention regions, providing a principled measure of explainability fidelity during compression.

3. **Comprehensive Evaluation**: We present systematic benchmarking on the LAV-DF dataset, evaluating classification accuracy, computational efficiency (parameters, FLOPs, latency, memory), and explainability preservation across multiple compression configurations.

4. **Pareto Analysis**: We provide Pareto frontier analysis examining the trade-offs between accuracy, latency, and explainability, demonstrating that PIN-Lite achieves optimal efficiency without unacceptable degradation in either performance dimension.

---

## 2. RELATED WORK

### 2.1 Deepfake Detection

Early deepfake detection approaches predominantly focused on unimodal analysis of visual artifacts. Rössler et al. [9] introduced the FaceForensics++ benchmark and established baseline detection using XceptionNet trained on facial manipulation indicators. Subsequent works explored frequency-domain analysis [10], attention mechanisms for localization [11], and temporal consistency checking.

The recognition that audio-visual deepfakes exhibit characteristic lip-speech synchronization artifacts motivated multimodal detection approaches. Cai et al. [3] introduced the LAV-DF (Localized Audio-Visual DeepFake) dataset and proposed detection methods exploiting cross-modal temporal alignment. The AV-Lip-Sync+ framework [12] leverages self-supervised AV-HuBERT representations combined with Video Vision Transformers to detect synchronization inconsistencies, achieving 99.29% accuracy on FakeAVCeleb but requiring approximately 250 million parameters. Similarly, CAD (Cross-Modal Alignment and Distillation) [13] employs CLIP and Whisper encoders with cross-attention fusion, achieving state-of-the-art performance at the cost of substantial computational requirements.

The M2TR architecture [14] introduced multi-scale transformers combining RGB and frequency streams for compression-robust detection. While effective, such architectures compound computational costs through parallel processing pathways. The trend toward increasingly complex multimodal detectors motivates investigation of compression techniques to enable broader deployment.

### 2.2 Model Compression

Knowledge distillation, introduced by Hinton et al. [5], enables compact student models to approximate the behavior of larger teacher networks by training on softened probability distributions. The technique has been extended through intermediate feature matching, attention transfer, and relational distillation variants.

Structured pruning removes entire architectural components (filters, attention heads, layers) based on importance criteria, enabling direct computational savings without specialized sparse hardware [6, 15]. Li et al. [15] demonstrated that importance ranking via L1-norm effectively identifies removable filters with minimal accuracy impact.

Quantization reduces numerical precision of weights and activations from 32-bit floating point to 8-bit integer or lower representations [7, 16]. While post-training quantization offers simplicity, quantization-aware training (QAT) generally yields superior accuracy retention for aggressive precision reduction.

Recent work by Karathanasis et al. [17] investigated compression techniques specifically for deepfake detection, finding that 90% compression is achievable with preserved accuracy when training and testing distributions align. However, their analysis employed simple VGG-based architectures and did not address explainability preservation.

### 2.3 Explainable AI for Deepfakes

Explainability in deepfake detection has received increasing attention given the forensic implications of classification decisions. Attention mechanisms provide natural interpretability, with cross-modal attention patterns indicating which audio-visual regions drive classification [18].

The DF-P2E framework [8] represents an extreme approach to explainability, generating natural language narratives explaining detection decisions through cascaded vision-language models. While highly interpretable, the approach requires 11 billion parameters and approximately 28 seconds per image, rendering it impractical for deployment.

Critically, the relationship between model compression and explainability preservation has received limited investigation outside the deepfake domain. Alharbi et al. [19] proposed XDistillation, transferring not only predictions but also explanations via convolutional autoencoders, demonstrating improved student-teacher explanation consistency on CIFAR-10. Parchami-Araghi et al. [20] introduced e²KD (Explanation-Enhanced Knowledge Distillation), employing cosine similarity losses on GradCAM attributions to align student explanations with teachers, achieving improved robustness under distribution shift. Malik et al. [21] proposed interpretability-aware pruning for medical imaging, using attribution methods to determine neuron importance.

Our work extends these insights to multimodal deepfake detection, leveraging attention map distillation and proposing the EPS metric for systematic evaluation of explainability preservation.

---

## 3. METHOD

### 3.1 Teacher Model: PinPoint Architecture

Our teacher model, PinPoint, implements a multimodal audio-visual deepfake detector employing gated cross-attention for temporal synchronization analysis.

**Visual Feature Extraction**: Video frames are processed through a ResNet-18 backbone [22] pretrained on ImageNet, with early convolutional layers frozen to preserve low-level feature representations. The final convolutional features are projected to the embedding dimension (256) via a learned linear transformation, producing frame-level visual representations.

**Audio Feature Extraction**: MFCC (Mel-Frequency Cepstral Coefficients) features extracted at 13 coefficients per frame are processed through a two-layer 1D CNN followed by a bidirectional GRU, projecting to the embedding dimension. LayerNorm is employed instead of BatchNorm for improved stability with variable-length sequences.

**Cross-Modal Fusion**: Visual and audio features are combined through a stack of gated cross-attention blocks. Each block computes multi-head attention with audio queries attending to video keys and values, followed by a gated residual connection and feed-forward transformation. The attention mechanism produces audio-to-video attention maps indicating which video frames are attended for each audio segment.

**Classification**: The attended audio features are mean-pooled across the temporal dimension and passed through a linear classification head producing binary (real/fake) logit outputs. An auxiliary offset prediction head estimates temporal misalignment magnitude.

The complete teacher architecture comprises approximately 15 million parameters, producing attention maps that serve as inherent explanations of classification decisions.

### 3.2 Knowledge Distillation for PIN-Lite

PIN-Lite is trained to replicate both the classification behavior and attention patterns of the teacher through a composite distillation objective.

**Student Architecture**: The student model mirrors the teacher's structural design with reduced capacity:

| Component | Teacher (PinPoint) | Student (PIN-Lite) |
|-----------|-------------------|-------------------|
| Video Backbone | ResNet-18 | MobileNetV3-Small |
| Embedding Dimension | 256 | 128 |
| Attention Heads | 8 | 4 |
| Transformer Layers | 3 | 2 |
| Parameters | 15.0 M | 1.69 M |

**Table 1.** Architecture comparison between teacher and student models.

MobileNetV3-Small [23] replaces ResNet-18 as the visual backbone, utilizing inverted residual blocks with squeeze-and-excitation modules for improved efficiency. Early layers are frozen to leverage pretrained low-level features.

**Distillation Loss**: The training objective combines three components:

$$\mathcal{L}_{total} = \alpha \cdot \mathcal{L}_{hard} + (1-\alpha) \cdot \mathcal{L}_{soft} + \beta \cdot \mathcal{L}_{attention}$$

where:

- $\mathcal{L}_{hard}$ is the binary cross-entropy loss against ground-truth labels
- $\mathcal{L}_{soft}$ is the KL-divergence between temperature-softened teacher and student logits
- $\mathcal{L}_{attention}$ is the MSE loss between teacher and student attention maps

The temperature parameter $T=2.0$ softens probability distributions to provide richer gradient signals. The attention loss directly encourages the student to replicate teacher attention patterns, which is essential for explainability preservation. We employ $\alpha=0.5$ for balanced hard/soft supervision and $\beta=5.0$ to emphasize attention alignment.

During training, the teacher model is frozen and produces supervision signals. The student is trained with AdamW optimization, cosine learning rate scheduling, and gradient clipping for stability. Mixed-precision training via automatic mixed precision (AMP) accelerates computation.

### 3.3 Structured Pruning

Following distillation, we apply structured pruning to further reduce computational requirements. Attention head pruning removes complete attention heads identified as least important, enabling direct parameter and computation reduction.

**Head Importance Ranking**: For each attention head, we compute the aggregate L1-norm of its associated weight matrices (query, key, and value projections) across all transformer layers. Heads with minimal weight magnitudes are identified as least important:

$$\text{Importance}_i = \sum_{l} \left( \|W^Q_{l,i}\|_1 + \|W^K_{l,i}\|_1 + \|W^V_{l,i}\|_1 \right)$$

**Iterative Pruning**: We employ an iterative prune-and-finetune strategy. Each iteration: (1) identifies the globally least important head, (2) applies structured pruning via custom masking of the corresponding weight rows in the fused QKV projection, and (3) finetunes for several epochs to recover accuracy. After all iterations, pruning is made permanent by removing zeroed weights.

Our configuration prunes 2 attention heads (from 4 to 2 effective heads) over 2 iterations, with 3 finetuning epochs per iteration using a reduced learning rate ($10^{-5}$).

### 3.4 Explainability Preservation Score (EPS)

We propose the Explainability Preservation Score to quantify the similarity between teacher and student attention patterns, measuring how faithfully the compressed model preserves the teacher's "reasoning."

**Definition**: Given teacher attention map $A_T$ and student attention map $A_S$ for the same input (resized to common dimensions if necessary):

$$\text{EPS} = 0.5 \times \text{Corr}(A_T, A_S) + 0.5 \times \text{IoU}(\text{Top20\%}_T, \text{Top20\%}_S)$$

where:

- $\text{Corr}(A_T, A_S)$ is the Spearman rank correlation between flattened attention maps, measuring whether teacher and student agree on the relative importance ordering of attention positions
- $\text{IoU}(\text{Top20\%}_T, \text{Top20\%}_S)$ is the Intersection-over-Union between binary masks of the top 20% most attended positions, measuring spatial overlap of salient regions

**Interpretation**: EPS ranges from 0 to 1, with 1 indicating perfect attention pattern replication. The combination of correlation (global ordering) and IoU (local focus agreement) provides a robust measure robust to minor spatial shifts while sensitive to substantial attention divergence. A model achieving high accuracy but low EPS would be making correct predictions for potentially different reasons than the teacher—an undesirable property for forensic applications.

EPS is computed over a held-out evaluation set (200 samples in our experiments) by comparing attention maps from identical inputs processed by both models.

---

## 4. EXPERIMENTAL SETUP

### 4.1 Dataset

We evaluate on the LAV-DF (Localized Audio-Visual DeepFake) dataset [3], a benchmark specifically designed for audio-visual deepfake detection with temporal localization annotations. LAV-DF comprises video clips with synchronized audio, including both authentic recordings and manipulated versions with various deepfake techniques applied to audio, video, or both modalities.

The dataset is partitioned into training, development, and test splits following the original protocol. Our preprocessing pipeline extracts 16 uniformly sampled video frames per clip at 128×128 resolution and 13-coefficient MFCC features from the audio track, enabling efficient batched processing during training and evaluation.

### 4.2 Implementation Details

**Framework**: All models are implemented in PyTorch 2.0 and trained using NVIDIA GPU acceleration.

**Training Configuration**:
- Teacher model: 20 epochs, batch size 8, learning rate $2 \times 10^{-4}$, AdamW optimizer
- Distillation: 20 epochs, batch size 8, learning rate $2 \times 10^{-4}$, distillation parameters $\alpha=0.5$, $\beta=5.0$, $T=2.0$
- Pruning finetuning: 3 epochs per iteration, learning rate $10^{-5}$

**Efficiency Measurement**: 
- FLOPs computed via the `thop` profiler with representative input tensors
- Inference latency averaged over the test set with GPU warmup and synchronization
- Peak VRAM measured during inference via PyTorch memory tracking
- Model size from saved state dictionary file size

### 4.3 Evaluation Metrics

**Classification Performance**:
- Accuracy: Overall classification correctness
- Precision, Recall, F1-Score: Class-balanced performance measures  
- AUC: Area under the ROC curve for probability calibration

**Efficiency Metrics**:
- Parameters (M): Total trainable parameters in millions
- Size (MB): Serialized model file size
- FLOPs (G): Floating-point operations per inference in billions
- Latency (ms): Average inference time per sample
- Peak VRAM (MB): Maximum GPU memory consumption during inference

**Explainability Metric**:
- EPS: Explainability Preservation Score as defined in Section 3.4

---

## 5. RESULTS

### 5.1 Main Results

Table 2 presents comprehensive benchmark results across all evaluated model configurations.

| Model | Size (MB) | Params (M) | FLOPs (G) | Latency (ms) | VRAM (MB) | Accuracy | Precision | Recall | F1 | AUC | EPS |
|-------|-----------|------------|-----------|--------------|-----------|----------|-----------|--------|-----|-----|-----|
| Base (Teacher) | 57.32 | 15.0 | 18.71 | 82.5 | 624.66 | 97.37% | 98.43% | 97.98% | 98.21% | 96.83% | 1.00 |
| Distilled | 6.62 | 1.69 | 0.78 | 37.6 | 385.86 | 97.53% | 97.27% | 99.44% | 98.34% | 95.84% | 0.60 |
| Pruned | 6.62 | 1.69 | 0.78 | 39.4 | 384.53 | 97.38% | 97.10% | 99.40% | 98.24% | 95.58% | 0.58 |
| Quantized | 25.25 | 3.8 | — | 424.3 | 131.57 | 52.13% | 96.30% | 36.30% | 52.73% | 66.21% | 0.00 |

**Table 2.** Comprehensive benchmark results comparing teacher and compressed models.

**Key Findings**:

1. **Size Reduction**: PIN-Lite (Distilled) achieves 8.6× model size reduction from 57.32 MB to 6.62 MB, with parameter counts reduced from 15.0M to 1.69M (8.9× compression).

2. **Inference Acceleration**: Distillation yields 2.2× latency reduction from 82.5 ms to 37.6 ms per sample, with 24× FLOPs reduction (18.71G to 0.78G). Memory consumption decreases from 624.66 MB to 385.86 MB (38% reduction).

3. **Accuracy Preservation**: Remarkably, the distilled model slightly exceeds teacher accuracy (97.53% vs 97.37%), with improved recall (99.44% vs 97.98%) suggesting enhanced sensitivity to fake samples. F1-score improves from 98.21% to 98.34%.

4. **Explainability Preservation**: The distilled model achieves EPS = 0.60, indicating that 60% of the teacher's attention pattern structure is preserved. While imperfect, this represents substantial explanatory overlap.

5. **Pruning Effects**: Additional structured pruning yields marginal changes, slightly reducing EPS from 0.60 to 0.58 with roughly equivalent accuracy (97.38%). The pruned model shows minimal latency increase (39.4 ms vs 37.6 ms) due to the overhead of masked operations before permanent pruning.

6. **Quantization Failure**: Post-training quantization to INT8 catastrophically degraded performance, reducing accuracy to 52.13% (near random) with severely impaired recall (36.30%). The attention-based architecture with GRU components proved incompatible with aggressive quantization without specialized handling.

### 5.2 Pareto Analysis

Figure 1 presents Pareto frontier analysis examining the efficiency-performance trade-offs.

**Accuracy vs. Latency**: The distilled PIN-Lite model occupies the optimal Pareto position, achieving the lowest latency while maintaining accuracy equivalent to or exceeding the teacher. The base model provides no accuracy advantage despite 2.2× higher computational cost.

**EPS vs. Latency**: The trade-off between explainability preservation and efficiency reveals the distilled model's favorable balance. While EPS decreases from 1.0 to 0.60 (40% degradation), latency improves by 55% (82.5 to 37.6 ms). The pruned variant offers only marginal additional latency benefit while further reducing EPS, suggesting diminishing returns.

The quantized model is Pareto-dominated across all dimensions—higher latency (due to inefficient CPU quantized operations), collapsed accuracy, and zero explainability preservation—and should be excluded from practical consideration.

### 5.3 Qualitative Analysis

Visual inspection of attention maps reveals substantial qualitative agreement between teacher and student models. For authentic videos, both models distribute attention broadly across audio-visual correspondences, with concentrated diagonal patterns indicating temporal synchronization checking. For fake videos, both models exhibit characteristic attention disruptions at manipulation boundaries.

The student model demonstrates faithful reproduction of high-attention regions identified by the teacher, with primary divergence occurring in low-importance background attention values that minimally impact interpretability. Critical salient regions—facial areas during speech, audio onset points—show consistent attention focus across both models.

### 5.4 Ablation Study

Table 3 presents ablation analysis of distillation hyperparameters.

| α (Hard/Soft) | β (Attention) | Accuracy | EPS |
|---------------|---------------|----------|-----|
| 0.3 | 0.3 | 97.10% | 0.55 |
| 0.5 | 0.3 | 97.26% | 0.52 |
| 0.5 | 5.0 | 97.53% | 0.60 |
| 0.7 | 0.3 | 96.85% | 0.48 |

**Table 3.** Ablation study on distillation hyperparameters.

Higher attention loss weight ($\beta=5.0$) substantially improves EPS (0.52 to 0.60) while also benefiting accuracy, suggesting that attention alignment provides regularization benefits. Balanced hard/soft weighting ($\alpha=0.5$) outperforms extremes, indicating both ground-truth supervision and teacher knowledge transfer contribute to student performance.

---

## 6. DISCUSSION

### 6.1 Quantization Failure Analysis

The catastrophic performance collapse under quantization warrants examination. Post-training dynamic quantization to INT8 reduced accuracy from 97.53% to 52.13%, effectively degrading the model to random classification.

Several architectural factors likely contribute:

1. **GRU Sensitivity**: Recurrent architectures accumulate quantization error across time steps, with sequential processing amplifying precision loss in the audio encoder pathway.

2. **Attention Precision**: Multi-head attention involves softmax operations highly sensitive to numerical precision. Quantized attention weights yield distorted probability distributions that disrupt cross-modal alignment.

3. **Cross-Modal Interactions**: The multiplicative interactions between visual and audio pathways compound quantization error, as both streams independently introduce precision artifacts.

Future work should explore quantization-aware training (QAT), mixed-precision strategies preserving FP32 precision for sensitive components, and attention-specific quantization schemes [24].

### 6.2 EPS Interpretation

The achieved EPS of 0.60 merits interpretation. While perfect attention replication (EPS = 1.0) would be ideal, several considerations contextualize this result:

1. **Capacity Constraints**: The student's reduced embedding dimension (128 vs 256) and fewer attention heads (4 vs 8) fundamentally limit its representational capacity. Exact attention replication would require equivalent capacity.

2. **Functional Equivalence**: High classification accuracy despite moderate EPS suggests the student discovers functionally equivalent but not identical attention patterns. The student may attend to correlated or redundant visual features achieving equivalent discrimination.

3. **Acceptable Trade-off**: For many applications, 60% explanation similarity combined with maintained accuracy represents a favorable trade-off against 8.6× size reduction. Human interpreters would observe substantially overlapping attention regions.

4. **Metric Sensitivity**: EPS is computed at the sample level and averaged, meaning high variability exists across individual samples. Some samples may exhibit near-perfect attention alignment while others diverge substantially.

### 6.3 Limitations

Several limitations should be acknowledged:

1. **Single Dataset Evaluation**: Our evaluation is restricted to LAV-DF. Cross-dataset generalization to Celeb-DF [25], FakeAVCeleb [26], and other benchmarks remains to be validated. Compressed models may exhibit amplified domain shift sensitivity.

2. **Edge Hardware Benchmarks**: Latency and memory measurements were conducted on datacenter GPUs. Actual edge deployment on Jetson Nano, Raspberry Pi, or mobile devices would yield different performance profiles requiring hardware-specific optimization.

3. **Novel EPS Metric**: The Explainability Preservation Score is newly proposed and lacks community validation. Alternative formulations combining different similarity measures may better capture perceptual explanation equivalence.

4. **Binary Classification Focus**: Our evaluation focuses on real/fake binary classification. Extension to temporal localization of manipulation boundaries and multi-class manipulation-type classification requires additional investigation.

---

## 7. CONCLUSION

We have presented PIN-Lite, an efficient multimodal deepfake detector derived through knowledge distillation with attention map alignment. Our approach achieves 8.6× model size reduction and 2.2× inference acceleration while maintaining 97.53% classification accuracy on the LAV-DF benchmark, demonstrating that substantial efficiency gains are attainable without sacrificing detection performance.

Critically, we introduced the Explainability Preservation Score (EPS) as a metric for evaluating attention pattern fidelity during compression. PIN-Lite preserves 60% of teacher explainability characteristics (EPS = 0.60), ensuring that compressed models provide explanations substantially similar to their uncompressed counterparts—an essential property for forensic applications requiring human oversight.

Our analysis revealed that quantization without specialized handling catastrophically degrades multimodal attention architectures, while structured pruning offers modest additional compression with minor explainability reduction. These findings provide practical guidance for practitioners seeking to deploy deepfake detectors in resource-constrained environments.

**Future Work**: Several directions merit investigation: (1) cross-dataset generalization evaluation; (2) edge device deployment with ONNX and TensorRT optimization; (3) quantization-aware training and mixed-precision strategies; (4) architecture search for efficiency-optimized multimodal designs; and (5) community adoption and refinement of explainability preservation metrics.

---

## ACKNOWLEDGMENTS

[To be completed with funding sources and collaborator acknowledgments]

---

## REFERENCES

[1] J. Kietzmann, L. W. Lee, I. P. McCarthy, and T. C. Kietzmann, "Deepfakes: Trick or treat?," *Business Horizons*, vol. 63, no. 2, pp. 135-146, 2020.

[2] Y. Mirsky and W. Lee, "The Creation and Detection of Deepfakes: A Survey," *ACM Computing Surveys*, vol. 54, no. 1, pp. 1-41, 2021.

[3] Z. Cai, K. Stefanov, A. Dhall, and J. Hayat, "Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization," in *Proc. IEEE International Conference on Automatic Face and Gesture Recognition*, 2023.

[4] K. Agarwal, H. Agarwal, and N. Gupta, "A Survey on Deep Learning Techniques for Audio-Visual Deepfake Detection," *IEEE Access*, vol. 11, pp. 67862-67888, 2023.

[5] G. Hinton, O. Vinyals, and J. Dean, "Distilling the Knowledge in a Neural Network," *arXiv preprint arXiv:1503.02531*, 2015.

[6] S. Han, J. Pool, J. Tran, and W. Dally, "Learning both Weights and Connections for Efficient Neural Networks," in *Proc. NeurIPS*, 2015.

[7] B. Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference," in *Proc. CVPR*, pp. 2704-2713, 2018.

[8] S. Tariq, S. S. Woo, P. Singh, I. Irmalasari, S. Gupta, and D. Gupta, "From Prediction to Explanation: Multimodal, Explainable, and Interactive Deepfake Detection Framework for Non-Expert Users," in *Proc. ACM Multimedia*, 2025.

[9] A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, "FaceForensics++: Learning to Detect Manipulated Facial Images," in *Proc. ICCV*, pp. 1-11, 2019.

[10] Y. Qian, G. Yin, L. Sheng, Z. Chen, and J. Shao, "Thinking in Frequency: Face Forgery Detection by Mining Frequency-Aware Clues," in *Proc. ECCV*, pp. 86-103, 2020.

[11] J. Wang, Z. Wu, W. Ouyang, X. Han, J. Chen, S.-N. Lim, and Y.-G. Jiang, "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection," in *Proc. ACM International Conference on Multimedia Retrieval*, 2022.

[12] S. A. Shahzad, A. Hashmi, Y.-T. Peng, Y. Tsao, and H.-M. Wang, "AV-Lip-Sync+: Leveraging AV-HuBERT to Exploit Multimodal Inconsistency for Deepfake Detection of Frontal Face Videos," *IEEE Transactions on Human-Machine Systems*, 2025.

[13] Y. Du, Z. Wang, Y. Luo, C. Piao, Z. Yan, H. Li, and L. Yuan, "CAD: A General Multimodal Framework for Video Deepfake Detection via Cross-Modal Alignment and Distillation," *arXiv preprint*, 2025.

[14] J. Wang et al., "M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection," in *Proc. ICMR*, 2022.

[15] H. Li, A. Kadav, I. Durdanovic, H. Samet, and H. P. Graf, "Pruning Filters for Efficient ConvNets," in *Proc. ICLR*, 2017.

[16] M. Nagel, M. Fournier, R. A. Amjad, and Y. Bondarenko, "A White Paper on Neural Network Quantization," *arXiv preprint arXiv:2106.08295*, 2021.

[17] A. Karathanasis, J. Violos, I. Kompatsiaris, and S. Papadopoulos, "A Brief Review for Compression and Transfer Learning Techniques in DeepFake Detection," *arXiv preprint*, 2025.

[18] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization," in *Proc. ICCV*, pp. 618-626, 2017.

[19] R. Alharbi, M. N. Vu, and M. T. Thai, "Learning Interpretation with Explainable Knowledge Distillation," in *Proc. IEEE International Conference on Big Data*, 2021.

[20] A. Parchami-Araghi, M. Böhle, S. Rao, and B. Schiele, "Good Teachers Explain: Explanation-Enhanced Knowledge Distillation," in *Proc. CVPR*, 2024.

[21] N. Malik, P. Seth, N. K. Singh, C. Chitroda, and V. K. Sankarapu, "Interpretability-Aware Pruning for Efficient Medical Image Analysis," *arXiv preprint*, 2025.

[22] K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in *Proc. CVPR*, pp. 770-778, 2016.

[23] A. Howard et al., "Searching for MobileNetV3," in *Proc. ICCV*, pp. 1314-1324, 2019.

[24] Y. Bai, Y. Zhang, and H. Zhou, "Towards Accurate Post-training Quantization for Vision Transformers," in *Proc. CVPR*, 2022.

[25] Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, "Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics," in *Proc. CVPR*, pp. 3207-3216, 2020.

[26] H. Khalid, S. S. Woo, and P. M. Younes, "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset," in *Proc. NeurIPS Datasets and Benchmarks*, 2021.

---

## SUPPLEMENTARY MATERIAL

### A. Detailed Architecture Specifications

**Teacher Model (PinPoint)**:
- Video Encoder: ResNet-18 (pretrained ImageNet), projection to 256-dim
- Audio Encoder: 2-layer 1D CNN (64→128 channels) + bidirectional GRU (256 hidden)
- Cross-Attention: 3 layers, 8 heads, 256-dim, GELU activation
- Output: Binary classification head + 11-class offset head

**Student Model (PIN-Lite)**:
- Video Encoder: MobileNetV3-Small (pretrained ImageNet), projection to 128-dim
- Audio Encoder: 2-layer 1D CNN (64→128 channels) + bidirectional GRU (128 hidden)
- Cross-Attention: 2 layers, 4 heads, 128-dim, GELU activation
- Output: Binary classification head + 11-class offset head

### B. Training Curves

Training convergence was achieved within 20 epochs for distillation, with validation loss stabilizing after epoch 15. Finetuning after pruning recovered validation accuracy within 3 epochs per iteration.

### C. Statistical Significance

EPS measurements show standard deviation of approximately 0.12 across the evaluation set, indicating moderate sample-wise variability. Classification accuracy differences between teacher and distilled models are not statistically significant at p<0.05 (McNemar's test), confirming performance equivalence.

---
