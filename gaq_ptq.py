"""
====================================================================================
GAQ P2/P3 — Hybrid post-training quantization + the naive INT8 strawman
====================================================================================
This is the script that produces the paper's money table contrast:

  * NAIVE INT8 attention (strawman): quantizes the *whole* attention path including
    softmax / LayerNorm / the sigmoid gate to INT8. We expect the attention map to
    be corrupted -> EPS collapses, faithfulness drops.

  * GAQ-INT8 (ours): INT8 on Q/K/V/out-proj + gate-linear + FFN linears with
    per-output-channel weight scales, calibrated on LAV-DF-200; softmax + LayerNorm
    + sigmoid kept in FP32. We expect accuracy AND the attention map preserved.

Accuracy / EPS / faithfulness are measured with calibrated fake-quant (faithful INT8
numerics with full control). Deployment latency + size come from the REAL dynamic
INT8 backend (fbgemm/qnnpack), which — because attention is now explicit Linears —
actually reaches the attention projections that nn.MultiheadAttention blocked.

Run (after gaq_p0_baseline.py so the teacher-latency reference exists):
    python gaq_ptq.py
Outputs: rows appended to gaq_results.csv; calibrated GAQ model saved for QAT.
====================================================================================
"""

import os

import torch

import gaq_config as C
import gaq_core as G


def _read_teacher_latency(default=None):
    p = C.out("gaq_teacher_latency.txt")
    if os.path.exists(p):
        try:
            return float(open(p).read().strip())
        except Exception:
            return default
    return default


def main():
    C.banner("GAQ P2/P3 — HYBRID PTQ vs NAIVE INT8 STRAWMAN")
    G.set_seed(C.SEED)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    results_csv = C.out("gaq_results.csv")

    print("\n[1/6] Loading teacher + student ...")
    teacher, t_cfg = G.load_teacher(device=gpu)
    student, s_cfg = G.load_student(device=gpu)

    print("\n[2/6] Dataloaders (test for eval, train for calibration) ...")
    loaders = G.make_loaders(s_cfg, splits=("train", "test"),
                             batch_size=C.CALIBRATION_BATCH_SIZE)
    test_loader = loaders["test"]
    calib_loader = loaders["train"]
    calib_batches = max(1, C.CALIBRATION_SAMPLES // C.CALIBRATION_BATCH_SIZE)

    # Teacher faithfulness reference (computed once, reused for agreement).
    print("\n[3/6] Teacher faithfulness reference ...")
    faith_teacher = G.deletion_insertion_scores(
        teacher, test_loader, gpu, C.FAITHFULNESS_SAMPLES)
    teacher_del = faith_teacher.get("per_sample_deletion", [])
    teacher_lat = _read_teacher_latency()

    def eval_quant(model, name, source_note):
        acc = G.evaluate_accuracy(model, test_loader, gpu)
        eps = G.compute_eps(teacher, model, test_loader, gpu, C.EPS_SAMPLES)
        faith = G.deletion_insertion_scores(model, test_loader, gpu, C.FAITHFULNESS_SAMPLES)
        agree = G.faithfulness_agreement(teacher_del, faith.get("per_sample_deletion", []))
        # Real-backend deployment latency + size.
        real = G.to_dynamic_int8(model)
        batch = G.get_one_batch(test_loader)
        lat = G.measure_cpu_latency(real, batch["video"], batch["audio"],
                                    C.LATENCY_REPEATS, C.LATENCY_WARMUP)
        size = G.actual_state_dict_size_mb(real)
        params = G.count_parameters(model) / 1e6
        speed = round(teacher_lat / max(lat, 1e-6), 2) if teacher_lat else ""
        print(f"\n  >>> {name}")
        print(f"      acc={acc.get('acc')}, auc={acc.get('auc')}, eps={eps.get('eps')} "
              f"(sp={eps.get('spearman')}, iou={eps.get('iou')})")
        print(f"      faith del={faith.get('deletion_auc')}, ins={faith.get('insertion_auc')}, "
              f"agree={agree:.4f}")
        print(f"      latency={lat:.1f}ms, size={size:.2f}MB, speedup={speed}")
        G.append_result(results_csv, {
            "model": name,
            "acc": acc.get("acc"), "f1": acc.get("f1"), "auc": acc.get("auc"),
            "eps": eps.get("eps"), "spearman": eps.get("spearman"), "iou": eps.get("iou"),
            "deletion_auc": faith.get("deletion_auc"),
            "insertion_auc": faith.get("insertion_auc"),
            "faith_agreement": round(agree, 4),
            "latency_cpu_ms": round(lat, 2), "size_mb": round(size, 2),
            "params_m": round(params, 3), "speedup_vs_teacher": speed,
            "source_note": source_note,
        })

    # ---- Naive INT8 strawman -------------------------------------------------
    print("\n[4/6] NAIVE INT8 strawman (quantizes softmax/LN/sigmoid too) ...")
    naive = G.build_quantizable_student(student, num_bits=8, per_channel=False,
                                        observer="minmax")
    G.calibrate(naive, calib_loader, gpu, calib_batches, policy="naive")
    eval_quant(naive, "Naive INT8 attention (strawman)",
               "Naive PTQ: softmax/LN/sigmoid quantized, per-tensor")

    # ---- GAQ-INT8 (ours) -----------------------------------------------------
    print("\n[5/6] GAQ-INT8 PTQ (per-channel; softmax/LN/sigmoid FP32) ...")
    gaq = G.build_quantizable_student(student, num_bits=8, per_channel=True,
                                      observer="percentile")
    G.calibrate(gaq, calib_loader, gpu, calib_batches, policy="gaq")
    eval_quant(gaq, "GAQ-INT8 (PTQ, ours)",
               "Hybrid PTQ: INT8 linears (per-channel, percentile calib), softmax/LN/sigmoid FP32")

    # Save the calibrated GAQ model so QAT can resume from it.
    print("\n[6/6] Saving calibrated GAQ model for QAT ...")
    torch.save(gaq.state_dict(), C.out("gaq_int8_ptq.pth"))
    print("Saved -> gaq_int8_ptq.pth")
    print("\nPTQ phase complete.")


if __name__ == "__main__":
    main()
