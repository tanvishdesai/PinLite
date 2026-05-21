# Literature List - PIN-Lite

**Project:** PIN-Lite: Lightweight Multimodal Deepfake Detection with Explainability Preservation
**Date compiled:** 2026-05-14
**Target venue:** Neural Processing Letters
**Total papers:** 15

---

## Direct Predecessors

### 1. M2TR: Multi-modal Multi-scale Transformers for Deepfake Detection
**Authors:** Junke Wang et al.
**Venue / Year:** ICMR 2022
**Link:** https://arxiv.org/abs/2104.09770
**Why it belongs:** It is a transformer-based deepfake detector that combines RGB and frequency-domain evidence, making it a close predecessor in multimodal/fusion-based forgery detection.
**Key differentiator:** PIN-Lite focuses on audio-visual detection, edge efficiency, and explanation preservation rather than image RGB/frequency fusion alone.

### 2. AV-Lip-Sync+: Leveraging AV-HuBERT to Exploit Multimodal Inconsistency for Video Deepfake Detection
**Authors:** Sahibzada Adil Shahzad et al.
**Venue / Year:** IEEE Transactions on Human-Machine Systems, 2025
**Link:** https://arxiv.org/abs/2311.02733
**Why it belongs:** It is a strong audio-visual deepfake detector using self-supervised multimodal features and a face encoder.
**Key differentiator:** PIN-Lite targets compact deployment and explanation fidelity, while AV-Lip-Sync+ uses large pretrained components.

### 3. CAD: A General Multimodal Framework for Video Deepfake Detection via Cross-Modal Alignment and Distillation
**Authors:** Yuxuan Du et al.
**Venue / Year:** arXiv 2025
**Link:** https://arxiv.org/abs/2505.15233
**Why it belongs:** It directly addresses multimodal deepfake detection with alignment and distillation.
**Key differentiator:** PIN-Lite evaluates compression stages and preserves teacher attention maps quantitatively through EPS.

### 4. From Prediction to Explanation: Multimodal, Explainable, and Interactive Deepfake Detection Framework for Non-Expert Users
**Authors:** Shahroz Tariq et al.
**Venue / Year:** ACM MM 2025
**Link:** https://arxiv.org/abs/2508.07596
**Why it belongs:** It frames deepfake detection around explanation and non-expert interpretability.
**Key differentiator:** PIN-Lite preserves internal cross-attention explanations under model compression rather than generating human-facing narrative explanations with large models.

### 5. A Brief Review for Compression and Transfer Learning Techniques in DeepFake Detection
**Authors:** Andreas Karathanasis et al.
**Venue / Year:** arXiv 2025
**Link:** https://arxiv.org/abs/2504.21066
**Why it belongs:** It is the closest work on compressing deepfake detectors for edge deployment.
**Key differentiator:** PIN-Lite extends compression to multimodal audio-visual detection and introduces explicit explanation-preservation measurement.

---

## Methodological Foundations

### 6. Do You Really Mean That? Content Driven Audio-Visual Deepfake Dataset and Multimodal Method for Temporal Forgery Localization
**Authors:** Zhixi Cai et al.
**Venue / Year:** DICTA 2022
**Link:** https://arxiv.org/abs/2204.06228
**Why it belongs:** It introduces LAV-DF, the benchmark used for PIN-Lite evaluation.

### 7. Distilling the Knowledge in a Neural Network
**Authors:** Geoffrey Hinton, Oriol Vinyals, Jeff Dean
**Venue / Year:** arXiv 2015
**Link:** https://arxiv.org/abs/1503.02531
**Why it belongs:** It defines the teacher-student knowledge distillation paradigm used in PIN-Lite.

### 8. Attention Is All You Need
**Authors:** Ashish Vaswani et al.
**Venue / Year:** NeurIPS 2017
**Link:** https://arxiv.org/abs/1706.03762
**Why it belongs:** It establishes the attention mechanism underlying the gated cross-attention blocks.

### 9. Searching for MobileNetV3
**Authors:** Andrew Howard et al.
**Venue / Year:** ICCV 2019
**Link:** https://arxiv.org/abs/1905.02244
**Why it belongs:** PIN-Lite uses MobileNetV3-Small as the lightweight visual student backbone.

### 10. Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference
**Authors:** Benoit Jacob et al.
**Venue / Year:** CVPR 2018
**Link:** https://arxiv.org/abs/1712.05877
**Why it belongs:** It motivates the INT8 quantization stage used for edge-oriented deployment.

---

## Gap-Exposing Papers

### 11. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention
**Authors:** Angelos Katharopoulos et al.
**Venue / Year:** ICML 2020
**Link:** https://arxiv.org/abs/2006.16236
**Why it belongs:** Linear attention motivates one efficient attention variant evaluated in PIN-Lite.
**Gap statement:** The efficiency gain does not guarantee preservation of forensic attention maps.

### 12. Fast Transformer Decoding: One Write-Head is All You Need
**Authors:** Noam Shazeer
**Venue / Year:** arXiv 2019
**Link:** https://arxiv.org/abs/1911.02150
**Why it belongs:** It introduces the multi-query attention idea evaluated as a PIN-Lite attention variant.
**Gap statement:** Prior MQA work targets decoder efficiency, not explanation preservation in multimodal forensic detection.

### 13. Learning Interpretation with Explainable Knowledge Distillation
**Authors:** Raed Alharbi, Minh N. Vu, My T. Thai
**Venue / Year:** arXiv 2021
**Link:** https://arxiv.org/abs/2111.06945
**Why it belongs:** It shows that conventional KD may fail to transfer explanations and proposes explanation-aware distillation.
**Gap statement:** Existing explanation-aware KD is evaluated mainly on image classification and does not measure audio-visual cross-attention fidelity.

### 14. Good Teachers Explain: Explanation-Enhanced Knowledge Distillation
**Authors:** Amin Parchami-Araghi et al.
**Venue / Year:** ECCV 2024
**Link:** https://arxiv.org/abs/2402.03119
**Why it belongs:** It strengthens the argument that students should learn the same evidence as teachers.
**Gap statement:** It does not address multimodal deepfake detection or edge compression pipelines.

### 15. Interpretability-Aware Pruning for Efficient Medical Image Analysis
**Authors:** Nikita Malik et al.
**Venue / Year:** arXiv 2025
**Link:** https://arxiv.org/abs/2507.08330
**Why it belongs:** It connects model pruning with interpretability preservation.
**Gap statement:** It uses attribution-guided pruning in medical imaging, leaving open how to measure explanation fidelity across compressed multimodal forensic models.

---

## Related Work Subsection Plan

**Subsection A: Multimodal Deepfake Detection**
Papers included: 1, 2, 3, 4, 6
Gap statement draft: Current multimodal detectors increasingly combine audio, video, and explanation components, but their compute footprint and explanation behavior under compression remain under-explored.

**Subsection B: Model Compression for Deployment**
Papers included: 5, 7, 9, 10
Gap statement draft: Compression methods reduce footprint, but accuracy retention alone is insufficient for forensic use because it does not show whether the compact model relies on the same evidence.

**Subsection C: Attention-Efficient and Explanation-Aware Learning**
Papers included: 8, 11, 12, 13, 14, 15
Gap statement draft: Efficient attention and explanation-aware learning have developed separately; PIN-Lite evaluates them jointly by measuring attention-map fidelity after compression.

**Synthesis / closing gap paragraph draft:**
Prior work shows that multimodal cues improve deepfake detection, compression can reduce model footprint, and explanation-aware learning can align student and teacher behavior. What is missing is a compact audio-visual detector whose efficiency gains are evaluated together with a quantitative measure of explanation preservation.

