# Quantized Gated-Attention Main

## Main Hypothesis
Full PinPoint can be quantized end-to-end, including gated attention, while preserving task performance and explainability under the locked success bar:
- Accuracy drop <= 1.0 point versus FP32 teacher baseline.
- EPS drop <= 0.05 versus FP32 teacher baseline.
- CPU latency speedup >= 1.3x versus FP32 teacher baseline.

## Current Best Candidate
<!-- AUTO_CURRENT_BEST_START -->
No completed candidate is logged yet.
<!-- AUTO_CURRENT_BEST_END -->

## Validated/Rejected Hypotheses
Use this section to keep short statements and evidence links.

| Hypothesis ID | Status | Evidence |
| --- | --- | --- |
| H-INIT | In progress | Waiting for Cycle 1 runs |

## Failure Analysis
<!-- AUTO_FAILURE_START -->
No failures logged yet.
<!-- AUTO_FAILURE_END -->

## Next 3 Runs
<!-- AUTO_NEXT3_START -->
1. `GAQ-P0-001` - Teacher FP32 reproducibility on CPU and EPS baseline extraction.
2. `GAQ-P1-001` - Explicit Q/K/V parity test against original `nn.MultiheadAttention`.
3. `GAQ-P2-001` - First hybrid PTQ attempt with softmax/LN in higher precision.
<!-- AUTO_NEXT3_END -->

## What to run in Kaggle now
<!-- AUTO_KAGGLE_START -->
1. `python research/gaq_experiment.py --exp_id GAQ-P0-001 --phase P0 --seed 11 --backend fp32_cpu --attention_quant_policy fp32_reference --qat_epochs 0 --calibration_profile none --eval_profile cpu_edge_v1 --status planned`
2. `python research/gaq_experiment.py --exp_id GAQ-P1-001 --phase P1 --seed 11 --backend fp32_cpu --attention_quant_policy explicit_qkv_refactor --qat_epochs 0 --calibration_profile none --eval_profile parity_v1 --run_parity --status completed`
3. `python research/gaq_experiment.py --exp_id GAQ-P2-001 --phase P2 --seed 11 --backend fbgemm --attention_quant_policy hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32 --qat_epochs 0 --calibration_profile lavdf_200 --eval_profile cpu_edge_v1 --status planned`
<!-- AUTO_KAGGLE_END -->
