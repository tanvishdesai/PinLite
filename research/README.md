# GAQ Research Workspace

This folder is the autonomous control plane for the Quantized Gated-Attention main track.

## Files
- `quant_gated_attention_main.md`: human-readable progress and next-run guidance.
- `experiment_registry.csv`: machine-readable registry (single source of run history).
- `agent_state.json`: agent memory, phase gates, best candidate, and 3-command queue.
- `gaq_experiment.py`: standardized experiment logger/orchestrator CLI.
- `gaq_attention_refactor.py`: explicit Q/K/V parity utility for P1.
- `snapshot_pinpoint_gated_attention.py`: frozen copy of original full-model gated-attention path.
- `snapshot_pinlite_student.py`: frozen copy of PinLite student architecture using gated attention.

## Standard CLI (required interface)
Use this interface for all future run logging:

```bash
python research/gaq_experiment.py \
  --exp_id <ID> \
  --phase <P0|P1|P2|P3|P4|P5> \
  --seed <int> \
  --backend <name> \
  --attention_quant_policy <policy> \
  --qat_epochs <int> \
  --calibration_profile <profile> \
  --eval_profile <profile> \
  [--acc <float> --f1 <float> --auc <float> --eps <float> --spearman <float> --iou <float> --latency_cpu_ms <float> --size_mb <float> --speedup_vs_teacher <float>] \
  [--run_parity --parity_trials <int>] \
  [--status planned|completed|failed]
```

At the end of every call, the script prints one parseable summary line:
- `GAQ_SUMMARY {...}`

It also prints exactly 3 ranked Kaggle commands for the next cycle.

## Typical Kaggle workflow
1. Run your actual training/eval script in Kaggle (for example P0 baseline or P2 PTQ run).
2. Call `gaq_experiment.py` with measured metrics to log the run and update phase/state.
3. Execute the top command from the emitted 3-command queue.

## Every Time You Open Your Coding Agent
1. Open these three files first:
2. `research/quant_gated_attention_main.md`
3. `research/agent_state.json`
4. `research/experiment_registry.csv`
5. Ask the agent to parse the latest queue and give you the exact next Kaggle run command.
6. Run that command in Kaggle.
7. Paste Kaggle outputs (metrics + latency + size + EPS) back to the agent.
8. Ask the agent to log completion via `gaq_experiment.py` and regenerate the next 3 ranked commands.
9. Repeat until the current phase gate passes, then continue to the next phase.

## First 3 Commands To Run (Cycle 1)
1. `python research/gaq_experiment.py --exp_id GAQ-P0-001 --phase P0 --seed 11 --backend fp32_cpu --attention_quant_policy fp32_reference --qat_epochs 0 --calibration_profile none --eval_profile cpu_edge_v1 --status planned`
2. `python research/gaq_experiment.py --exp_id GAQ-P1-001 --phase P1 --seed 11 --backend fp32_cpu --attention_quant_policy explicit_qkv_refactor --qat_epochs 0 --calibration_profile none --eval_profile parity_v1 --run_parity --status completed`
3. `python research/gaq_experiment.py --exp_id GAQ-P2-001 --phase P2 --seed 11 --backend fbgemm --attention_quant_policy hybrid_int8_qkv_out_gate_ffn_softmaxln_fp32 --qat_epochs 0 --calibration_profile lavdf_200 --eval_profile cpu_edge_v1 --status planned`
