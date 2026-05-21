# Phase 0 Orientation - PIN-Lite

**Project:** PIN-Lite: Lightweight Multimodal Deepfake Detection with Explainability Preservation
**Target venue:** Neural Processing Letters
**Date:** 2026-05-14

## Five Plain-English Answers

1. **What problem exists in the world that motivates this work?**
Deepfake videos are now plausible enough that people need automatic tools to check whether audio and video evidence can be trusted. The strongest detectors are usually too large for phones, cameras, and other edge devices, and many of them are hard to explain when a forensic analyst needs to justify a decision.

2. **Why do current methods fail to solve it adequately?**
Current multimodal detectors often obtain high accuracy by using large transformer or foundation-model backbones, which makes deployment expensive and slow. Compression work can make models smaller, but it usually measures success only by accuracy and ignores whether the compressed model still attends to the same evidence as the original model.

3. **What did we do differently?**
PIN-Lite compresses a multimodal audio-visual teacher into a lightweight student while explicitly preserving the teacher's cross-attention explanations. It combines attention-aware knowledge distillation, pruning, quantization, and an Explainability Preservation Score (EPS) that measures how faithfully a compressed model preserves the teacher's attention structure.

4. **What specific evidence shows it works?**
On LAV-DF, the distilled PIN-Lite student reduces parameters from 15.0M to 1.69M and model size from 57.32 MB to 6.62 MB, while achieving 97.53% accuracy compared with the teacher's 97.37%. The comprehensive benchmark reports EPS = 0.609 for the distilled model and 0.609 for the combined model, while the combined model reaches 5.29 MB and 98.22% accuracy. The attention-variant study also shows that Multi-Query Attention preserves accuracy and EPS, whereas Linear Attention collapses explanation fidelity despite remaining compact.

5. **Who benefits, and how?**
Forensic analysts, journalists, platform moderators, and edge-device developers benefit because the detector becomes easier to deploy without discarding the explanation signal needed for trust. The work also gives researchers a metric and experimental protocol for checking whether compressed forensic models remain faithful to their teachers.

## Two-Sentence Core Contribution

PIN-Lite is an explainability-preserving compression framework for multimodal deepfake detection that turns a 15.0M-parameter gated cross-attention teacher into compact student variants using attention-aware distillation, pruning, quantization, and efficient attention. Its main contribution is not only that the compressed model remains accurate, but that it quantifies and preserves the teacher's cross-modal attention behavior through EPS, exposing failures that accuracy alone hides.

