"""
====================================================================================
GAQ P0 — FP32 baselines + EPS/faithfulness reference
====================================================================================
Establishes the reference row of Table 1 on a *named* device:
  - Teacher (FP32): EPS = 1.00 by definition; deletion/insertion = reference.
  - Distilled student (FP32): accuracy/AUC, EPS vs teacher, faithfulness, CPU
    latency, size, params.

These numbers anchor the locked success bar (acc drop <= 1pt, EPS drop <= 0.05,
speedup >= 1.3x) used by every later GAQ phase.

Run (after writing PinPoint.py / Distill.py / gaq_config.py / gaq_core.py):
    python gaq_p0_baseline.py
Outputs: gaq_results.csv (rows: "Teacher FP32", "Distilled student (FP32)")
====================================================================================
"""

import torch

import gaq_config as C
import gaq_core as G


def main():
    C.banner("GAQ P0 — FP32 BASELINE + EPS/FAITHFULNESS REFERENCE")
    G.set_seed(C.SEED)
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    results_csv = C.out("gaq_results.csv")

    print("\n[1/4] Loading teacher and student ...")
    teacher, t_cfg = G.load_teacher(device=gpu)
    student, s_cfg = G.load_student(device=gpu)

    print("\n[2/4] Building dataloaders (LAV-DF test) ...")
    loaders = G.make_loaders(s_cfg, splits=("test",), batch_size=C.EVAL_BATCH_SIZE)
    test_loader = loaders["test"]

    print("\n[3/4] Accuracy / AUC ...")
    teacher_acc = G.evaluate_accuracy(teacher, test_loader, gpu)
    student_acc = G.evaluate_accuracy(student, test_loader, gpu)
    print(f"  Teacher : {teacher_acc}")
    print(f"  Student : {student_acc}")

    print("\n[4/4] EPS + behavioral faithfulness ...")
    # Student EPS vs teacher
    eps_student = G.compute_eps(teacher, student, test_loader, gpu, C.EPS_SAMPLES)
    print(f"  Student EPS: {eps_student}")

    faith_teacher = G.deletion_insertion_scores(
        teacher, test_loader, gpu, C.FAITHFULNESS_SAMPLES)
    faith_student = G.deletion_insertion_scores(
        student, test_loader, gpu, C.FAITHFULNESS_SAMPLES)
    agree = G.faithfulness_agreement(
        faith_teacher.get("per_sample_deletion", []),
        faith_student.get("per_sample_deletion", []))
    print(f"  Teacher faith: del={faith_teacher.get('deletion_auc')}, "
          f"ins={faith_teacher.get('insertion_auc')}")
    print(f"  Student faith: del={faith_student.get('deletion_auc')}, "
          f"ins={faith_student.get('insertion_auc')}, agree={agree:.4f}")

    # Latency + size on CPU (named hardware) — single-sample
    batch = G.get_one_batch(test_loader)
    teacher_lat = G.measure_cpu_latency(
        teacher, batch["video"], batch["audio"], C.LATENCY_REPEATS, C.LATENCY_WARMUP)
    student_lat = G.measure_cpu_latency(
        student, batch["video"], batch["audio"], C.LATENCY_REPEATS, C.LATENCY_WARMUP)
    print(f"\n  CPU latency [{C.TARGET_CPU_NAME}]: teacher={teacher_lat:.1f}ms, "
          f"student={student_lat:.1f}ms")

    teacher_size = G.actual_state_dict_size_mb(teacher)
    student_size = G.actual_state_dict_size_mb(student)
    teacher_params = G.count_parameters(teacher) / 1e6
    student_params = G.count_parameters(student) / 1e6

    # Persist the teacher latency so later phases can compute speedup honestly.
    with open(C.out("gaq_teacher_latency.txt"), "w") as f:
        f.write(str(teacher_lat))

    G.append_result(results_csv, {
        "model": "Teacher FP32",
        "acc": teacher_acc.get("acc"), "f1": teacher_acc.get("f1"),
        "auc": teacher_acc.get("auc"),
        "eps": 1.0, "spearman": 1.0, "iou": 1.0,
        "deletion_auc": faith_teacher.get("deletion_auc"),
        "insertion_auc": faith_teacher.get("insertion_auc"),
        "faith_agreement": 1.0,
        "latency_cpu_ms": round(teacher_lat, 2), "size_mb": round(teacher_size, 2),
        "params_m": round(teacher_params, 3), "speedup_vs_teacher": 1.0,
        "source_note": f"FP32 reference @ {C.TARGET_CPU_NAME}",
    })
    G.append_result(results_csv, {
        "model": "Distilled student (FP32)",
        "acc": student_acc.get("acc"), "f1": student_acc.get("f1"),
        "auc": student_acc.get("auc"),
        "eps": eps_student.get("eps"), "spearman": eps_student.get("spearman"),
        "iou": eps_student.get("iou"),
        "deletion_auc": faith_student.get("deletion_auc"),
        "insertion_auc": faith_student.get("insertion_auc"),
        "faith_agreement": round(agree, 4),
        "latency_cpu_ms": round(student_lat, 2), "size_mb": round(student_size, 2),
        "params_m": round(student_params, 3),
        "speedup_vs_teacher": round(teacher_lat / max(student_lat, 1e-6), 2),
        "source_note": f"FP32 distilled student @ {C.TARGET_CPU_NAME}",
    })
    print("\nP0 baseline complete. Reference row written.")


if __name__ == "__main__":
    main()
