"""
====================================================================================
GAQ P4 — Quantization-aware training (recovers the accuracy/EPS lost by PTQ)
====================================================================================
Starts from the calibrated GAQ-INT8 model, keeps the fake-quant nodes active
(straight-through estimator), and fine-tunes a few epochs against the frozen FP32
teacher with the distillation objective (hard BCE + soft KL + attention MSE). This
is the phase that reliably pulls accuracy/EPS back inside the locked success bar.

Run (after gaq_ptq.py):
    python gaq_qat.py
Outputs: row "GAQ-QAT (ours)" appended to gaq_results.csv; gaq_int8_qat.pth saved.
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
    C.banner("GAQ P4 — QUANTIZATION-AWARE TRAINING")
    G.set_seed(C.SEED)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    results_csv = C.out("gaq_results.csv")

    print("\n[1/5] Loading teacher + student ...")
    teacher, t_cfg = G.load_teacher(device=gpu)
    student, s_cfg = G.load_student(device=gpu)

    print("\n[2/5] Building GAQ quantizable student + calibration ...")
    gaq = G.build_quantizable_student(student, num_bits=8, per_channel=True,
                                      observer="percentile")
    loaders = G.make_loaders(s_cfg, splits=("train", "test"), batch_size=C.QAT_BATCH_SIZE,
                             shuffle=False)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Resume calibrated state if PTQ already produced it; else calibrate fresh.
    ptq_ckpt = C.out("gaq_int8_ptq.pth")
    if os.path.exists(ptq_ckpt):
        print(f"  Resuming calibrated observers from {ptq_ckpt}")
        gaq.load_state_dict(torch.load(ptq_ckpt, map_location="cpu"), strict=False)
        G.set_model_policy(gaq, "gaq")
        for fq in G._all_act_fq(gaq):
            if torch.isfinite(fq.scale).all():
                fq.enabled = True
                fq.calibrating = False
        G.set_model_policy(gaq, "gaq")
    else:
        calib_batches = max(1, C.CALIBRATION_SAMPLES // C.QAT_BATCH_SIZE)
        G.calibrate(gaq, train_loader, gpu, calib_batches, policy="gaq")
    gaq.to(gpu)

    print(f"\n[3/5] QAT fine-tune ({C.QAT_EPOCHS} epochs, lr={C.QAT_LR}) ...")
    G.qat_finetune(gaq, teacher, train_loader, gpu, epochs=C.QAT_EPOCHS, lr=C.QAT_LR,
                   kd_alpha=getattr(s_cfg, "KD_ALPHA", 0.5),
                   kd_beta=getattr(s_cfg, "KD_BETA", 5.0),
                   kd_temperature=getattr(s_cfg, "KD_TEMPERATURE", 2.0),
                   policy="gaq")

    print("\n[4/5] Evaluating GAQ-QAT ...")
    acc = G.evaluate_accuracy(gaq, test_loader, gpu)
    eps = G.compute_eps(teacher, gaq, test_loader, gpu, C.EPS_SAMPLES)
    faith_teacher = G.deletion_insertion_scores(teacher, test_loader, gpu, C.FAITHFULNESS_SAMPLES)
    faith = G.deletion_insertion_scores(gaq, test_loader, gpu, C.FAITHFULNESS_SAMPLES)
    agree = G.faithfulness_agreement(
        faith_teacher.get("per_sample_deletion", []),
        faith.get("per_sample_deletion", []))

    real = G.to_dynamic_int8(gaq)
    batch = G.get_one_batch(test_loader)
    lat = G.measure_cpu_latency(real, batch["video"], batch["audio"],
                                C.LATENCY_REPEATS, C.LATENCY_WARMUP)
    size = G.actual_state_dict_size_mb(real)
    params = G.count_parameters(gaq) / 1e6
    teacher_lat = _read_teacher_latency()
    speed = round(teacher_lat / max(lat, 1e-6), 2) if teacher_lat else ""

    print(f"  acc={acc.get('acc')}, auc={acc.get('auc')}, eps={eps.get('eps')}, "
          f"agree={agree:.4f}, latency={lat:.1f}ms, size={size:.2f}MB")

    G.append_result(results_csv, {
        "model": "GAQ-QAT (ours)",
        "acc": acc.get("acc"), "f1": acc.get("f1"), "auc": acc.get("auc"),
        "eps": eps.get("eps"), "spearman": eps.get("spearman"), "iou": eps.get("iou"),
        "deletion_auc": faith.get("deletion_auc"),
        "insertion_auc": faith.get("insertion_auc"),
        "faith_agreement": round(agree, 4),
        "latency_cpu_ms": round(lat, 2), "size_mb": round(size, 2),
        "params_m": round(params, 3), "speedup_vs_teacher": speed,
        "source_note": f"QAT {C.QAT_EPOCHS}ep over calibrated GAQ-INT8",
    })

    print("\n[5/5] Saving QAT model ...")
    torch.save(gaq.state_dict(), C.out("gaq_int8_qat.pth"))
    print("Saved -> gaq_int8_qat.pth\nQAT phase complete.")


if __name__ == "__main__":
    main()
