# PIN-Lite v2 — GAQ Results (master reference)

**Single source of truth for the GAQ (Gated-Attention Quantization) elevation.** All
numbers here are read from the run logs in this folder; the figure is `gaq_frontier.png`.
Use this file when writing the draft.

- **In-domain dataset:** LAV-DF (3400 train / 1550 test).
- **Teacher:** PinPoint (ResNet-18 + CNN-GRU + 3× gated cross-attention), 15.0 M params.
- **Student:** PinpointTransformerLite (MobileNetV3-Small + CNN-GRU + 2× gated cross-attention), 1.69 M params.
- **Hardware (latency):** Kaggle CPU (Intel Xeon, fbgemm backend).
- **The contribution:** GAQ keeps softmax / LayerNorm / the sigmoid gate in FP32 and
  quantizes only the Q/K/V/out/gate/FFN linears (per-channel, percentile-calibrated).
  The naive strawman additionally quantizes softmax/LN/sigmoid.

---

## Headline (one paragraph)

At INT8 the naive and GAQ schemes are indistinguishable (both ≈0.57 EPS, ≈0.986 acc) —
INT8 is too gentle to separate them. **The contribution is visible only on the precision
frontier:** as bit-width drops to INT4 and INT3, the naive scheme's attention-map fidelity
(EPS) collapses (0.563 → 0.158, a 72 % loss) *and* its accuracy falls (0.986 → 0.955),
while **GAQ holds both** — EPS 0.575 → 0.516 and accuracy ≈0.985 down to 3 bits. GAQ thus
enables aggressive low-bit compression of gated cross-attention while preserving both the
prediction and the forensic explanation; naive quantization silently destroys the
explanation. The CPU latency benefit of INT8 is ~1.0× (no speedup) — **the deployment win
is model size / compression depth, not CPU latency.**

---

## Table 1 — LAV-DF main benchmark (INT8)

Source: `gaq_results.csv`. Speedup is vs the FP32 **teacher** (latency 822.6 ms).
Faith-agree = Spearman agreement of per-sample deletion scores vs the teacher.

| Model | Size MB | Params M | Latency ms | Acc | AUC | EPS | Spearman | IoU | Del-AUC | Ins-AUC | Faith-agree | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher FP32 | 57.32 | 14.999 | 822.6 | 0.9806 | 0.9970 | 1.0000 | 1.0000 | 1.0000 | 0.7008 | 0.7007 | 1.0000 | 1.00 |
| Distilled student (FP32) | 6.62 | 1.695 | 314.6 | 0.9865 | 0.9975 | 0.5719 | 0.6454 | 0.4983 | 0.6940 | 0.6988 | 0.7407 | 2.61 |
| Naive INT8 attention (strawman) | 4.60 | 1.695 | 257.4 | 0.9865 | 0.9978 | 0.5634 | 0.6410 | 0.4858 | 0.7002 | 0.7016 | 0.6914 | 3.20 |
| GAQ-INT8 (PTQ, ours) | 4.60 | 1.695 | 260.0 | 0.9871 | 0.9981 | 0.5748 | 0.6551 | 0.4944 | 0.6939 | 0.6987 | 0.7337 | 3.16 |
| GAQ-QAT (ours) | 4.60 | 1.695 | 361.4 | 0.9858 | 0.9978 | 0.5896 | 0.6722 | 0.5070 | 0.6982 | 0.6950 | 0.7171 | 2.28 |

**Read-out:** At INT8 every quantized variant matches the FP32 student on accuracy and EPS;
the naive↔GAQ EPS gap is only +0.011. This table alone does **not** motivate GAQ — it
shows INT8 quantization of this block is benign. The motivation is Table 2.

> Note on the "Speedup" column: 3.2× is dominated by the student being smaller than the
> teacher, **not** by quantization. Quantization-attributable CPU speedup is ~1.0× (Table 3).
> Do not present 3.2× as a quantization result.

---

## Table 2 — Precision frontier (THE headline result)

Source: `gaq_frontier.csv`. Figure: `gaq_frontier.png`. EPS p10 = 10th-percentile
(worst-case) per-sample EPS — the tail where corruption shows first.

| Bits | Scheme | Acc | AUC | EPS | Spearman | IoU | EPS p10 | EPS p50 | Del-AUC | Faith-agree |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | Naive | 0.9865 | 0.9978 | 0.5634 | 0.6410 | 0.4858 | 0.1842 | 0.6670 | 0.7002 | 0.6914 |
| 8 | **GAQ** | 0.9871 | 0.9981 | 0.5748 | 0.6552 | 0.4944 | 0.2198 | 0.6705 | 0.6939 | 0.7338 |
| 6 | Naive | 0.9845 | 0.9958 | 0.5277 | 0.6016 | 0.4538 | 0.0993 | 0.6399 | 0.7212 | 0.7357 |
| 6 | **GAQ** | 0.9871 | 0.9981 | 0.5737 | 0.6534 | 0.4941 | 0.2184 | 0.6684 | 0.6938 | 0.7341 |
| 4 | Naive | 0.9735 | 0.9968 | 0.3024 | 0.2970 | 0.3078 | 0.0798 | 0.3519 | 0.7817 | 0.7483 |
| 4 | **GAQ** | 0.9871 | 0.9980 | 0.5599 | 0.6385 | 0.4814 | 0.1996 | 0.6575 | 0.6947 | 0.7297 |
| 3 | Naive | 0.9548 | 0.9943 | 0.1582 | 0.1958 | 0.1207 | 0.0125 | 0.1804 | 0.7919 | 0.7438 |
| 3 | **GAQ** | 0.9852 | 0.9987 | 0.5160 | 0.5750 | 0.4570 | 0.1307 | 0.6207 | 0.6969 | 0.7476 |

**EPS gap (GAQ − naive) — the curves fan apart as precision drops:**

| Bits | GAQ EPS | Naive EPS | Gap |
|---:|---:|---:|---:|
| INT8 | 0.5748 | 0.5634 | **+0.0115** |
| INT6 | 0.5737 | 0.5277 | **+0.0460** |
| INT4 | 0.5599 | 0.3024 | **+0.2576** |
| INT3 | 0.5160 | 0.1582 | **+0.3578** |

**Read-out:**
- GAQ is nearly flat from 8→3 bits (EPS −0.059, acc −0.002). You can quantize gated
  cross-attention to **3 bits** and keep both the answer and the explanation.
- Naive collapses: EPS −0.405 (−72 %), accuracy −0.032. At INT3 the worst-decile EPS is
  0.0125 — the explanation is destroyed for the hardest samples while top-line accuracy
  (0.955) still looks "acceptable." This is the silent-failure argument.
- AUC stays high for naive even when EPS collapses (e.g. INT4 naive AUC 0.997 with EPS
  0.302) — concrete proof that **accuracy/AUC cannot detect explanation corruption**.

---

## Table 3 — CPU latency (named hardware, honest)

Source: `gaq_latency.csv`. Single-sample forward, Kaggle CPU (Intel Xeon, fbgemm).

| Backend | Latency ms | Speedup vs FP32 student |
|---|---:|---:|
| PyTorch FP32 student | 311.1 | 1.00× |
| PyTorch dynamic-INT8 GAQ | 308.9 | **1.01×** |
| ONNX Runtime FP32 | — | export failed (`onnxscript` missing) |
| ONNX Runtime INT8 (GAQ) | — | export failed (`onnxscript` missing) |

**Read-out:** Dynamic INT8 gives **no CPU speedup** here — PyTorch dynamic quant only
shrinks weights, and the attention matmuls are too small for the INT8 kernels to win on
this CPU. The deployment claim must be **size** (FP32 student 6.62 MB → INT8 4.60 MB ≈
1.4×; INT4/INT3 would be smaller still), not latency. State this as a limitation; do not
report 3.2× as a quantization speedup. ONNX path was not completed (missing dependency);
not pursued because the contribution does not rest on latency.

---

## Table 4 — Cross-dataset (LAV-DF → FakeAVCeleb, zero-shot)

Source: `gaq_crossdataset.csv`. FakeAVCeleb test set (2120 samples).

| Model | Acc | F1 | AUC |
|---|---:|---:|---:|
| FP32 student | 0.4953 | 0.1098 | 0.6140 |
| GAQ-INT8 (ours) | 0.4792 | 0.0754 | 0.5679 |

**Read-out:** Both are near chance — this is a **base-model generalization failure** (the
student does not transfer LAV-DF → FAVC), not a GAQ failure. GAQ adds a small extra
degradation (ΔAUC = −0.046). Frame honestly: quantization slightly compounds an existing
domain-shift weakness; do not over-claim. Consider this a limitation row, not a result.

---

## What to take into the draft

1. **Lead with Table 2 + `gaq_frontier.png`.** That is the paper. The story: *naive low-bit
   quantization of gated cross-attention silently destroys the forensic explanation while
   accuracy looks fine; GAQ preserves explanation and accuracy down to 3 bits.*
2. **EPS is now a diagnostic, not the headline** — it is the instrument that detects the
   silent failure the frontier exposes. Pair it with deletion/insertion faithfulness.
3. **Efficiency axis = compression depth (bits / size), not CPU latency.** Be explicit that
   CPU INT8 latency is ~1.0× and that the win is model size and the ability to push to INT4/3.
4. **Demote QAT** (Table 1, last row): slower than PTQ, slightly lower accuracy — it does not
   pay for itself here. Keep as "recovers residual PTQ loss," not a headline.
5. **AUC-vs-EPS divergence** (Table 2, INT4 naive) is a strong standalone sentence: a metric
   reviewers trust (AUC 0.997) is blind to a failure EPS catches.

---

## Provenance / caveats

- Tables 1, 4 and the teacher latency (822.6 ms): from the first elevation session
  (`gaq_results.csv`, `gaq_crossdataset.csv`, `gaq_teacher_latency.txt`), unchanged.
- Table 2 (`gaq_frontier.csv`): reconstructed from the `gaq_p5_frontier.py` console log of
  the 2026-06-23 session. `insertion_auc` and `spearman_p10` columns were not echoed to the
  console and are omitted here; the full CSV with those columns is regenerable on Kaggle.
- Table 3 (`gaq_latency.csv`): from the 2026-06-23 `gaq_onnx_latency.py` run (FP32 311.1 ms,
  INT8 308.9 ms). ONNX rows blank — export aborted on missing `onnxscript`. CPU latency
  varies run-to-run (a prior session logged ~257 ms); the ~1.0× INT8/FP32 ratio is stable.
