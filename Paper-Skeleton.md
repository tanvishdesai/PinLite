# PIN-Lite: Efficient and Explainable Multimodal Deepfake Detection via Knowledge Distillation

## Abstract
Briefly summarize the problem (deepfakes, computational cost of detection), the solution (PIN-Lite: Model Compression with Explainability Preservation), and the key results.
*   **Key Achievement**: Reduced model size by ~8.5x (57MB -> 6.6MB) and improved inference speed by ~2.2x (82.5ms -> 37.6ms) while *maintaining* or even slightly improving accuracy.
*   **Novelty**: Introduction of the Explainability Preservation Score (EPS) to ensure the compressed model remains trustworthy.

## 1. Introduction
*   **Context**: The proliferation of deepfakes and the need for robust, multimodal detection.
*   **Problem**: Current SOTA models (like PinPoint) are computationally expensive and hard to deploy on edge devices.
*   **Gap**: Most compression techniques focus only on accuracy/class probabilities, ignoring the "reasoning" (explainability) of the model.
*   **Proposed Solution**: PIN-Lite. A pipeline involving Knowledge Distillation (KD) and Structured Pruning.
*   **Contributions**:
    1.  Lightweight architecture (PIN-Lite) achieved through KD and Pruning.
    2.  Proposal of **EPS (Explainability Preservation Score)** to quantify feature alignment between teacher and student.
    3.  A study on the trade-offs between quantization, pruning, and distillation.

## 2. Related Work
*   Multimodal Deepfake Detection (cite PinPoint, etc.).
*   Model Compression (Distillation, Pruning, Quantization).
*   Explainable AI (XAI) in Deepfake Detection.

## 3. Methodology

### 3.1 The Teacher Model: PinPoint
*   Brief description of the original model (ResNet backbone + Transformer + Fusion).
*   Mention it serves as the ground truth for both labels and attention maps.

### 3.2 The Student Model: PIN-Lite
*   **Architecture**: Reduced embedding dimension, fewer heads, lighter backbone (if applicable).
*   **Knowledge Distillation Strategy**:
    *   Loss function: $L_{total} = \alpha L_{hard} + (1-\alpha)L_{soft} + \beta L_{attn}$
    *   **Attention Map Distillation**: Explicitly aligning the attention weights of the student to the teacher.

### 3.3 Structured Pruning
*   Method: Iterative pruning of attention heads based on L1-norm importance.
*   Goal: Remove redundant heads in the Student model to further reduce FLOPs.

### 3.4 Direct Quantization (Teacher)
*   Describe the experiment of directly quantizing the *Original Teacher* model (Post-Training Quantization / QAT).
*   Goal: To test if the heavy teacher can be compressed effectively without architectural search.

### 3.5 Explainability Preservation Score (EPS)
*   **Definition**: A composite metric to evaluate if the student "looks" at the same evidence as the teacher.
*   **Formula**: $EPS = 0.5 \times \text{SpearmanCorr}(S_T, S_S) + 0.5 \times \text{IoU}(S_T^{top20\%}, S_S^{top20\%})$
*   **Rationale**: High accuracy without correct reasoning is dangerous (Clever Hans effect).

## 4. Experimental Setup
*   **Dataset**: LAV-DF (Localized Audio-Visual DeepFake) dataset.
*   **Baselines**: Original PinPoint (Teacher).
*   **Metrics**: Accuracy, F1, AUC, EPS, Latency (ms), Model Size (MB), FLOPs (G).

## 5. Results and Analysis

### 5.1 Comprehensive Performance Comparison
This section compares the Teacher, Distilled, Pruned, and Quantized models.

**[Table 1 Needed]: Comparative Results of Model Variants**
*   *Source Data*: `comprehensive_benchmark_results_v2.csv`
*   *Columns*: Model, Acc(%), F1, AUC, Params(M), Size(MB), Latency(ms), EPS.
*   *Key Finding*: The **Distilled** model achieves the best trade-off. The **Quantized Teacher** suffers significant accuracy drop (52%) and high latency (likely due to CPU-bound INT8 execution or lack of optimization), proving that simple quantization of the complex Teacher is insufficient.

| Model | Accuracy | F1-Score | AUC | Latency (ms) | Size (MB) | EPS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Teacher (Base)** | 97.37% | 0.9821 | 0.9683 | 82.50 | 57.32 | 1.0 (Ref) |
| **Distilled** | **97.53%** | **0.9834** | 0.9584 | **37.60** | **6.62** | 0.60 |
| **Pruned** | 97.38% | 0.9824 | 0.9558 | 39.41 | 6.62 | 0.58 |
| **Quantized (Teacher)** | 52.13% | 0.5273 | 0.6621 | 424.34 | 25.25 | N/A |

### 5.2 Efficiency vs. Accuracy (Pareto Analysis)
**[Figure 1 Needed]: Pareto Frontier (Accuracy vs. Latency)**
*   *Description*: A scatter plot showing Accuracy on Y-axis and Latency on X-axis.
*   *Visual*: The Distilled/Pruned models should be in the top-left corner (High Acc, Low Latency).
*   *File to use*: `Pareto_Accuracy_vs_Latency_v2.png`

### 5.3 Explainability Preservation
**[Figure 2 Needed]: Pareto Frontier (EPS vs. Latency)**
*   *Description*: Plotting EPS against Latency.
*   *Analysis*: While Distilled/Pruned models lose some explainability fidelity (EPS ~0.6) compared to the teacher (EPS=1.0), they maintain respectable alignment while being 2x faster.
*   *File to use*: `Pareto_EPS_vs_Latency_v2.png`

## 6. Discussion
*   **Effectiveness of Distillation**: Reducing the model size by ~8x while increasing accuracy (+0.16%) suggests the Teacher was over-parameterized.
*   **Failure of Direct Quantization**: The Quantized Teacher model failed to maintain performance. This justifies the need for the Distillation approach (PIN-Lite) rather than just quantizing the legacy model.
*   **The Value of EPS**: It provides a secondary check. Even though Pruned and Distilled have similar accuracy, Distilled has slightly higher EPS (0.60 vs 0.58), suggesting it preserves the teacher's reasoning better.

## 7. Conclusion
*   Summary of PIN-Lite's success.
*   Final recommendation for deployment: The **Distilled** model is the best candidate for edge deployment.
