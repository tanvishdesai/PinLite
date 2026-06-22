# Project 2 — PIN-Lite Elevation Plan
### Objective 2: Lightweight / Efficient Multimodal Detection

> **Scope.** Grounded assessment from `pinlite results v3/comprehensive_benchmark_v3.csv`, the paper draft `paper/PIN_Lite_NPL_Draft.md`, `paper/STRUCTURAL_AUDIT.md`, and the `research/` GAQ workspace. Then a **re-centering** of the contribution away from EPS-as-headline toward a real technical contribution: **correct quantization of gated cross-attention (GAQ)**. All numbers below are read from the CSV/draft, not invented.

---

## 1. Current situation (grounded)

**What PIN-Lite is.** A compression pipeline for the PinPoint teacher: attention-aware knowledge distillation (`L = α·L_hard + (1−α)·L_soft + β·L_attn`) into a MobileNetV3-Small student → ℓ1 iterative pruning → INT8/FP16 post-training quantization, plus an efficient-attention study (MQA / low-rank / linear) and a custom metric, **EPS = w1·Spearman(A_T,A_S) + w2·TopK-IoU(A_T,A_S)**. Target venue in draft: **Neural Processing Letters**.

**The actual benchmark numbers (`comprehensive_benchmark_v3.csv`):**

| Model | Size MB | Params M | Latency ms | Acc | AUC | EPS |
|---|---|---|---|---|---|---|
| Base (teacher) | 57.32 | 15.0 | 98.62 | 0.9737 | 0.9683 | 1.000 |
| Distilled | 6.62 | 1.69 | 45.93 | 0.9753 | 0.9584 | 0.6091 |
| **Pruned** | **6.62** | **1.69** | 44.18 | 0.9738 | 0.9558 | **0.5887** |
| FP16 | 3.31 | 1.69 | 37.93 | 0.9752 | 0.9582 | 0.6091 |
| MQA | 6.25 | 1.60 | 36.35 | 0.9800 | 0.9689 | 0.6055 |
| LowRank | 6.26 | 1.60 | 38.28 | 0.9625 | 0.9489 | 0.5719 |
| Linear-Attn | 6.64 | 1.69 | 38.56 | **0.6082** | 0.578 | **0.0333** |
| Combined (PIN-Lite) | 5.29 | 1.22 | 171.29† | 0.9822 | 0.9733 | 0.6093 |

†INT8 on CPU.

### Honest critique — the three real problems (you already sensed these)

1. **Pruning did literally nothing.** `Pruned` has the **exact same size (6.62 MB) and params (1.69 M) as `Distilled`**, and its EPS actually *dropped* (0.6091 → 0.5887). The ℓ1 pruning either was not applied structurally (so no params were physically removed) or was fully recovered/reverted. As a "contribution," pruning is currently a liability — a reviewer who reads the table will see it does nothing and lose trust in the rest. **Pruning must be either fixed (made structural and real) or removed.**
2. **The EPS headline story does not hold up in the final numbers.** The intended hook was "Linear Attention keeps high accuracy but loses explanation fidelity (accuracy ≠ reasoning)." But in this CSV, Linear-Attn collapses on **accuracy too** (0.6082) — so it is not a clean "good accuracy, bad EPS" case; it just fails. Meanwhile all the *sensible* variants cluster in a narrow EPS band (0.57–0.61). So EPS has **poor dynamic range among reasonable models** and its one dramatic data point is confounded by an accuracy collapse. EPS as currently defined and evidenced is not a strong standalone contribution. (Your instinct is right.)
3. **"Combined" is slower than the teacher** (171 ms vs 98.6 ms) because INT8 dynamic quant has no optimized kernel on the test CPU. The "speedup" claim only survives in the FP16/GPU column (37.9 ms). The efficiency story is currently inconsistent.

Additional gaps from `STRUCTURAL_AUDIT.md`: hardware/software versions missing, FakeAVCeleb cross-dataset CSV missing (table relies on a "rough PDF ledger"), no real edge-hardware deployment, figures not final.

### Verdict on the current contribution

EPS + KD + pruning + PTQ is an *engineering bundle*, not a sharp technical contribution — and one component (pruning) is broken while another (EPS) is under-powered. NPL is a modest venue and even there the pruning result is a problem. **The project needs a real technical nucleus.** You already pointed at it.

---

## 2. The re-centering — GAQ as the technical contribution

You said: *"there is a problem with quantization where it does not quantize the attention mechanism well — maybe that's a genuine contribution."* **Yes. This is exactly right, and you have already scaffolded it** in `research/` (the `gaq_*` files, `quant_gated_attention_main.md`). Make this the spine of PIN-Lite v2.

### Why attention quantization is a *real, unsolved-enough* problem
- PyTorch's `nn.MultiheadAttention` **cannot be quantized directly** — that is why your `gaq_attention_refactor.py` decomposes it into explicit Q/K/V/out linear ops. This decomposition is a prerequisite no compression paper in the deepfake space has done for *gated cross-modal* attention.
- The hard parts are well-documented in the LLM-quantization literature but **unstudied for multimodal forensic cross-attention**: (i) **softmax** is numerically fragile under INT8 (large dynamic range, outliers); (ii) **LayerNorm** accumulates error; (iii) the **sigmoid gate** in PinPoint multiplies attention output — quantization error in the gate compounds; (iv) cross-attention has *asymmetric* token roles (audio query vs video key/value) so per-tensor scales are mismatched.
- So the contribution is concrete: **a hybrid-precision quantization scheme for gated cross-attention** (INT8 for Q/K/V/out-proj/gate/FFN linear ops, FP32 for softmax + LayerNorm + the gate's sigmoid), with **per-channel scales** and **QAT** to recover accuracy, that — uniquely — preserves *both* accuracy and the explanation (attention map). This is a defensible, generalizable, signal-processing-flavored contribution.

### EPS gets demoted to its correct role
EPS stops being "the contribution" and becomes **the measurement instrument that proves GAQ preserves reasoning.** Reframed honestly: *"naive attention quantization corrupts the attention map (we show this with EPS); GAQ's hybrid scheme keeps EPS high."* That is a much more credible use of EPS — as a diagnostic that motivates a method — than as a standalone metric. Also tighten EPS:
- Add a **behavioral** fidelity check alongside the map-similarity (deletion/insertion agreement between teacher and quantized student), so EPS is grounded in causal faithfulness, not just map correlation. This directly answers "EPS is just comparing attention maps."

### The GAQ phase plan (already defined in your workspace)
- **P0** — FP32 teacher reproducibility + EPS baseline on CPU.
- **P1** — explicit Q/K/V refactor parity test (the refactor must be numerically identical to `nn.MultiheadAttention` before quantizing). *(code exists: `gaq_attention_refactor.py`)*
- **P2** — first hybrid PTQ (INT8 Q/K/V/out/gate/FFN, FP32 softmax+LN), calibrated on ~200 LAV-DF clips.
- **P3** — per-channel scales + outlier handling for softmax inputs.
- **P4** — QAT to recover the accuracy/EPS lost by PTQ.
- **P5** — INT4 / mixed-precision frontier + ablation of which sub-modules tolerate low precision.

**Locked success bar (from your `quant_gated_attention_main.md`):** accuracy drop ≤ 1.0 pt, EPS drop ≤ 0.05, CPU speedup ≥ 1.3× vs FP32 teacher. Keep this bar; it is sensible and falsifiable.

---

## 3. What to do — concrete

### 3.1 Fix or remove the broken pruning
- **Decision:** drop unstructured ℓ1 pruning from the headline (it produced no size change). If you want a pruning result, do **structured channel pruning of the MobileNetV3 backbone** (which holds ~55% of params and is currently untouched) and report *real* param/size reduction. Only keep it if it produces a genuine, measured reduction; otherwise cut it cleanly and say the compression comes from KD + GAQ.

### 3.2 Make the efficiency story consistent
- Report latency on a **named target** (e.g., ONNX Runtime on a fixed CPU, and/or a Jetson if accessible). Use the `ONNX-Export.py` you already have. Show GAQ-INT8 on a backend that *has* INT8 kernels (fbgemm/qnnpack/ONNX-RT) so the speedup is real, and stop reporting the 171 ms dynamic-INT8 number as if it were the deployment latency.
- Report the FP16-GPU and INT8-CPU(optimized) paths separately and honestly, as two deployment targets.

### 3.3 Strengthen evaluation
- Add **AUC/EER/AP** to every row (you have logits). Note the teacher AUC 0.9683 here differs from PinPoint's run — reconcile which teacher checkpoint is canonical and state it.
- **Cross-dataset row** (train LAV-DF → test FakeAVCeleb) for the *quantized* model — does quantization hurt generalization more than in-domain accuracy? This is a real open question (Karathanasis et al. found compression degrades cross-domain) and a cheap, novel result.
- **EPS + behavioral faithfulness** for: FP32 teacher, naive-INT8 attention (the strawman), GAQ-INT8, GAQ-QAT. The expected money-table: naive-INT8 tanks EPS; GAQ keeps it.

### 3.4 What to run on Kaggle
1. `gaq P0` — teacher FP32 + EPS/faithfulness baseline. *(CPU)*
2. `gaq P1` — Q/K/V parity (must pass before anything else). *(CPU, `gaq_attention_refactor.py`)*
3. `gaq P2/P3` — hybrid PTQ + per-channel scales, calibrate on LAV-DF-200. *(CPU/T4)*
4. `gaq P4` — QAT recovery, few epochs. *(P100)*
5. `naive_int8_strawman.py` — quantize attention naively (incl. softmax/LN in INT8) to produce the failure baseline that motivates GAQ. *(CPU)*
6. `onnx_latency.py` — ONNX-RT INT8 latency on a fixed CPU + (optional) Jetson. *(CPU/edge)*
7. `crossdataset_quantized.py` — LAV-DF→FAVC AUC for FP32 vs GAQ. *(T4)*

### 3.5 Target result table (the paper's Table 1)

| Model | Size | Params | Latency (named HW) | Acc | AUC | EPS | Faithfulness (del/ins) |
|---|---|---|---|---|---|---|---|
| Teacher FP32 | — | — | — | — | — | 1.00 | ref |
| Distilled student | — | — | — | — | — | — | — |
| **Naive INT8 attention** (strawman) | — | — | — | — | — | **low** | **low** |
| **GAQ-INT8 (ours)** | — | — | — | — | — | **high** | **high** |
| **GAQ-QAT (ours)** | — | — | — | — | — | **highest** | **highest** |

The story the table tells: *quantizing gated cross-attention naively destroys the explanation; our hybrid GAQ scheme is the first to compress a multimodal forensic detector while keeping accuracy AND faithful reasoning, verified on real edge hardware.*

---

## 4. Narrative integration

- PIN-Lite is **Pillar 2**: it takes Project 1's *faithful* detector and makes it deployable **without losing the faithfulness** — which is only a meaningful claim because Project 1 established the attention is faithful. GAQ is the mechanism; EPS+faithfulness is the proof.
- The GAQ scheme is reusable: the certified detector in Project 3 and the adaptive detector in Project 4 are also attention-based, so "how to quantize gated cross-attention" is a thesis-wide tool, not a one-off.
- Keep the thesis line consistent: *the synchronization attention is the object we detect with (P1), compress (P2), certify (P3), and adapt (P4).*

---

## 5. Venue

- **Current draft target:** Neural Processing Letters (Q2/Q3) — too modest for a real GAQ contribution.
- **Recommended:** **ICASSP 2027** (the quantization-of-attention + audio-visual angle is a perfect signal-processing fit and ICASSP rewards a sharp technical method) as the conference; or **IEEE TMM / Pattern Recognition** (Q1) for a journal version that includes the full EPS+faithfulness+edge-deployment study.
- **Bar:** a working GAQ scheme meeting the locked success bar (≤1 pt acc drop, ≤0.05 EPS drop, ≥1.3× real speedup), the naive-INT8 strawman, and real-hardware latency. That is a clean ICASSP paper.

---

## 6. Risk assessment

| Risk | Likelihood | Mitigation |
|---|---|---|
| GAQ PTQ misses the ≤1 pt / ≤0.05 EPS bar | medium | That is what P4 (QAT) is for; QAT reliably recovers PTQ loss. The success bar already anticipates this. |
| INT8 has no fast kernel on available hardware | medium | Use ONNX-RT / qnnpack / fbgemm which *do*; report on a named backend. Don't repeat the 171 ms dynamic-INT8 mistake. |
| EPS still looks under-powered | low (now demoted) | EPS is now a diagnostic supporting GAQ + paired with behavioral faithfulness, not the headline. |
| Pruning keeps producing nothing | n/a | Drop it or make it structural; no longer load-bearing. |
| Reviewer: "attention quantization is solved in LLMs" | medium | It is **not** solved for *gated cross-modal* attention with a sigmoid gate in a forensic setting, and not jointly with explanation preservation. State the distinction explicitly and cite LLM-quant work as the closest prior. |

**Most likely failure mode of the *old* plan:** the broken pruning + weak EPS get the paper desk-rejected or stuck at a low venue. **The fix:** lead with GAQ, demote EPS to a diagnostic, drop/repair pruning.
