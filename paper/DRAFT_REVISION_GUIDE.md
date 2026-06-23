# PIN-Lite — Draft Revision Guide

Checklist for revising `paper/PIN_Lite_NPL_Draft.md` (and the Springer `.tex`) given the
**GAQ re-centering** and the new results in `elevation-results/RESULTS.md`. Read this
alongside that results file — every number cited below is sourced there.

> **The one-sentence repositioning:** the paper is no longer "we compressed a detector and
> invented EPS." It is **"naive low-bit quantization of gated cross-attention silently
> destroys the forensic explanation while accuracy looks fine; GAQ is the first scheme to
> preserve both, down to 3 bits."** EPS becomes the *instrument* that proves this, not the
> contribution.

---

## 0. Before you touch prose: the decision you must make first

**Venue.** The current draft targets Neural Processing Letters (NPL). The GAQ contribution
is a sharper, signal-processing-flavored result that fits **ICASSP 2027** (or a Q1 journal
like IEEE TMM / Pattern Recognition). Decide now — it changes length, format, and framing:
- **ICASSP:** 4 pages, lead hard with the frontier figure + the method, cut the breadth.
- **NPL (keep):** keep the broader compression study but make GAQ the spine and the frontier
  the centerpiece table/figure.

Everything below assumes **GAQ is the spine regardless of venue.**

---

## 1. Reconcile the two result generations (do this before writing any table)

The current draft's tables come from the **old** `comprehensive_benchmark_v3` run; the new
results come from the GAQ sessions. **They disagree and cannot both appear.** Examples:

| Quantity | Old draft | New GAQ results |
|---|---|---|
| Teacher accuracy | 0.9737 | 0.9806 |
| Teacher AUC | 0.9683 | 0.9970 |
| Student accuracy | 0.9753 | 0.9865 |
| Student EPS | 0.609 | 0.5719 |

These differ because the runs used different teacher checkpoints / eval protocols / EPS
sample counts. **Pick the GAQ generation as canonical** (it is the one with the frontier,
faithfulness, and a named teacher checkpoint) and regenerate/replace every table from
`elevation-results/`. State the teacher checkpoint and eval protocol explicitly. Do **not**
mix old and new numbers in the same paper.

---

## 2. Section-by-section edit list (current draft → target)

### Title & Abstract (lines 1–15)
- Retitle around GAQ, e.g. *"Quantizing Gated Cross-Attention without Losing the
  Explanation: Low-Bit Multimodal Deepfake Detection."*
- Rewrite the abstract: drop the MQA/Linear-Attn headline and the "combined 5.29 MB" hook.
  New arc: teacher→student KD (context) → **GAQ hybrid-precision quantization** (method) →
  **frontier result** (naive collapses at INT4/3, GAQ holds) → EPS+faithfulness as proof.
- Replace the closing sentence ("explainability-aware evaluation is necessary…") with the
  GAQ claim.

### §1 Introduction (lines 19–38)
- Keep the edge-deployment motivation (good).
- Rewrite the contributions list (lines 31–38). New contributions:
  1. **GAQ**, a hybrid-precision quantization scheme for *gated cross-modal* attention
     (INT-Q/K/V/out/gate/FFN, FP32 softmax/LN/sigmoid, per-channel, calibrated, + QAT) —
     enabled by an explicit Q/K/V decomposition of `nn.MultiheadAttention`.
  2. The **precision-frontier** demonstration that naive quantization destroys the
     attention-map explanation at low bit-width while GAQ preserves it to 3 bits.
  3. EPS **+ behavioral faithfulness** (deletion/insertion) as a paired diagnostic that
     detects this silent failure — and the finding that **AUC is blind to it**.
  4. (Optional/secondary) the KD compression pipeline and attention-variant study.
- **Remove** "iterative pruning" from the contributions — it produced no size change.

### §2 Related Work (lines 41–67)
- **Add a new subsection: quantization of attention / LLM quantization.** This is the
  closest prior and a likely reviewer objection ("attention quantization is solved in
  LLMs"). Cite LLM.int8(), SmoothQuant, GPTQ, AWQ (add to refs) and state the distinction:
  none address a **gated cross-modal** attention with a sigmoid gate in a **forensic**
  setting, and none evaluate **explanation preservation**. This is essential.
- Keep §2.3 (explanation-preserving distillation) — it now supports the EPS-as-diagnostic
  framing.
- Trim the M2TR/AV-Lip-Sync breadth if going to ICASSP.

### §3 Method (lines 71–162)
- §3.1–3.3 (teacher, student, KD): keep, condense. This is now *context*, not contribution.
- **§3.4 must be rewritten as the GAQ method**, not "pruning and quantization." Content to
  add (it is in `gaq_core.py`, currently absent from the draft):
  - explicit Q/K/V/out decomposition of `nn.MultiheadAttention` (prerequisite for quantizing);
  - the hybrid policy (which ops INT, which stay FP32) and **why** (softmax dynamic range,
    LN error accumulation, the gate's sigmoid compounding error);
  - per-channel weight scales + percentile activation calibration;
  - QAT with straight-through estimator + the KD objective.
- **Delete the unstructured-pruning paragraph** (lines 141–142) or move it to a one-line
  limitation. It is broken (Distilled and Pruned are identical at 6.62 MB / 1.69 M).
- §3.5 EPS: keep, but add the **deletion/insertion behavioral faithfulness** definition
  (it is implemented and reported now) and reframe EPS as a diagnostic.

### §4 Experiments (lines 166–269)
- **§4.2 Table 1:** replace with the new INT8 table from `RESULTS.md` Table 1 (Teacher /
  Distilled / Naive-INT8 / GAQ-INT8-PTQ / GAQ-QAT). **Fix the "Combined 171 ms" claim** — do
  not report it as latency. Add a sentence that INT8 CPU speedup is ~1.0× and the win is size.
- **NEW §4.3 Precision frontier (the centerpiece).** Insert `RESULTS.md` Table 2 + the
  EPS-gap table + `gaq_frontier.png` as **Figure 1**. This is the paper's money result.
  Make the AUC-blindness point explicitly: at INT4 naive AUC is 0.997 while EPS is 0.302.
- **§4.3 Attention variants (old):** demote or cut. The Linear-Attn "EPS collapse" is
  **confounded** — in the v3 CSV Linear-Attn also collapses on *accuracy* (0.61), so it is
  not a clean "good accuracy, bad EPS" case. If kept, present honestly as a secondary study
  and stop calling it the headline. MQA-as-best-tradeoff is orthogonal to GAQ.
- **§4.4 EPS table:** keep as supporting evidence; align numbers to the GAQ generation.
- **§4.6 Cross-dataset:** replace with `RESULTS.md` Table 4 (FP32 student 0.614 AUC vs
  GAQ-INT8 0.568). Frame as a **limitation**: both near chance → base-model domain-shift
  failure, GAQ adds a small −0.046 AUC. Do not over-claim "behavioral fidelity."

### §5 Discussion (lines 273–301)
- §5.1 (accuracy hides forensic risk): keep — the frontier now *proves* it with dynamic range.
- §5.2 (MQA vs Linear): demote/cut with the variant study.
- §5.3 Deployment: rewrite honestly — **size/compression-depth story**, not latency. INT8 =
  1.0× CPU; the lever is pushing to INT4/3 (smaller weights) which only GAQ survives.
- §5.4 Limitations: keep and expand (see §4 checklist below). The pruning, cross-dataset,
  and latency honesty items all live here now.

### §6 Conclusion (lines 305–311)
- Re-anchor on GAQ + the frontier. Drop "MQA preserves both" as the main lesson.

---

## 3. Tables & figures — concrete swap list

| In draft | Action |
|---|---|
| Table 1 (comprehensive v3) | **Replace** with RESULTS.md Table 1 (GAQ generation). |
| — | **Add Table 2 = precision frontier** (RESULTS.md Table 2) — new centerpiece. |
| — | **Add Figure 1 = `gaq_frontier.png`** — the two fan-out panels. |
| Table 2 (EPS bootstrap CI) | Keep; regenerate CIs on the GAQ generation if possible. |
| Table 3 (ablations) | Keep — still valid KD context. |
| Table 4 (FAVC, "rough ledger") | **Replace** with measured RESULTS.md Table 4. |
| Pareto fig, EPS-component fig | Demote/cut with the attention-variant study. |

---

## 4. Reviewer-proofing checklist (from STRUCTURAL_AUDIT + the new gaps)

- [ ] **Report exact hardware/software**: CPU (Intel Xeon, fbgemm), GPU model, CUDA, PyTorch
      version. The draft flags these as missing (§4.1, line 174; §5.4, line 301).
- [ ] **Latency honesty**: state INT8 CPU speedup = ~1.0×; lead efficiency with **size**.
      Never present the 3.2× (student-vs-teacher) as a quantization speedup.
- [ ] **Pruning**: removed from headline; one-line limitation only.
- [ ] **Cross-dataset**: presented as limitation (near-chance), not generalization win.
- [ ] **EPS definition consistency**: the draft uses w=0.7 in places and the code uses
      w1=0.5 — state the exact weighting used for the reported numbers and be consistent.
- [ ] **Name the teacher checkpoint** and reconcile why teacher AUC differs from prior runs
      (draft already flags this, §3.5 region / audit).
- [ ] **Anticipate "LLM quant solved this"**: the new related-work subsection must explicitly
      distinguish gated cross-modal + sigmoid gate + explanation preservation + forensic.
- [ ] **Honest negative**: the INT8 result is benign (no contrast). Say so, then show the
      frontier is where GAQ matters — this *strengthens* credibility, don't hide it.

---

## 5. References to add

- LLM.int8() (Dettmers et al., 2022)
- SmoothQuant (Xiao et al., 2023)
- GPTQ (Frantar et al., 2023)
- AWQ (Lin et al., 2024)
- (optional) a QAT reference beyond Jacob et al. [6] if QAT is kept in the method.

These anchor the "closest prior work" and let you draw the gated-cross-modal /
explanation-preservation distinction that is the paper's defensibility.

---

## 6. Suggested target outline (ICASSP-style; compress for NPL as needed)

1. Intro — edge forensic detection; accuracy ≠ reasoning; GAQ contribution.
2. Related work — multimodal deepfake; compression; **attention quantization** (new).
3. Method — teacher/student/KD (brief) → **GAQ** (the meat) → EPS + faithfulness (diagnostic).
4. Experiments — setup → INT8 table → **frontier (Fig 1 + Table 2)** → faithfulness/AUC-blindness → cross-dataset limitation.
5. Discussion / limitations — size-not-latency; INT8 benign, low-bit is the regime; honesty.
6. Conclusion.

**Centerpiece to write first:** the frontier paragraph + Figure 1 caption. Everything else
supports it.
