"""
====================================================================================
GAQ P5 — PRECISION FRONTIER (the experiment that produces the naive-vs-GAQ contrast)
====================================================================================
WHY THIS EXISTS
At INT8 the naive strawman barely degrades (EPS 0.563 vs GAQ 0.575) because 8 bits
is too gentle: the task is saturated and the attention-map corruption washes out by
the time it propagates through the gate and the classifier head. That leaves the
paper without its money shot.

This script sweeps the weight/activation bit-width DOWN (8 -> 6 -> 4 -> 3) for BOTH
schemes and measures acc / AUC / EPS / faithfulness at each step. The hypothesis the
whole contribution rests on:

    As precision drops, the NAIVE scheme (softmax/LN/sigmoid in low-bit) collapses
    on EPS + faithfulness FIRST, while GAQ (those ops kept FP32) holds the
    attention map far longer.

If true, this is the frontier figure: two EPS-vs-bits curves that fan apart. That is
the ICASSP-grade result. If the curves DON'T fan apart even at 3 bits, that is also a
real (negative) finding — it means INT8 quantization of this block is simply benign,
and the honest paper claim shrinks to "1.4x smaller, iso-accuracy, iso-reasoning."

Run (after gaq_p0_baseline.py, so the teacher reference + latency exist):
    python gaq_p5_frontier.py
Outputs:
    gaq_frontier.csv         — one row per (bits, policy) with acc/eps/faith + EPS tail
    gaq_frontier_eps.csv     — per-sample EPS arrays (for distribution/tail plots)
====================================================================================
"""

import csv

import numpy as np
import torch

import gaq_config as C
import gaq_core as G

# Bit-widths to sweep. 8 reproduces the existing rows; 6/4/3 are the frontier.
FRONTIER_BITS = [8, 6, 4, 3]


def main():
    C.banner("GAQ P5 — PRECISION FRONTIER (naive vs GAQ as bits drop)")
    G.set_seed(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n[1/4] Loading teacher + student ...")
    teacher, _ = G.load_teacher(device=dev)
    student, s_cfg = G.load_student(device=dev)

    print("\n[2/4] Dataloaders (test for eval, train for calibration) ...")
    loaders = G.make_loaders(s_cfg, splits=("train", "test"),
                             batch_size=C.CALIBRATION_BATCH_SIZE)
    test_loader = loaders["test"]
    calib_loader = loaders["train"]
    calib_batches = max(1, C.CALIBRATION_SAMPLES // C.CALIBRATION_BATCH_SIZE)

    print("\n[3/4] Teacher faithfulness reference ...")
    faith_teacher = G.deletion_insertion_scores(
        teacher, test_loader, dev, C.FAITHFULNESS_SAMPLES)
    teacher_del = faith_teacher.get("per_sample_deletion", [])

    rows = []
    persample_rows = []

    def run_one(bits, policy):
        # GAQ keeps softmax/LN/sigmoid FP32 (policy="gaq") with per-channel weights +
        # percentile calibration; the naive strawman quantizes everything per-tensor.
        if policy == "gaq":
            model = G.build_quantizable_student(
                student, num_bits=bits, per_channel=True, observer="percentile")
            calib_policy = "gaq"
        else:
            model = G.build_quantizable_student(
                student, num_bits=bits, per_channel=False, observer="minmax")
            calib_policy = "naive"
        G.calibrate(model, calib_loader, dev, calib_batches, policy=calib_policy)

        acc = G.evaluate_accuracy(model, test_loader, dev)
        eps = G.compute_eps(teacher, model, test_loader, dev, C.EPS_SAMPLES)
        faith = G.deletion_insertion_scores(model, test_loader, dev, C.FAITHFULNESS_SAMPLES)
        agree = G.faithfulness_agreement(teacher_del, faith.get("per_sample_deletion", []))

        tag = f"{policy.upper()}-INT{bits}"
        print(f"\n  >>> {tag}")
        print(f"      acc={acc.get('acc'):.4f} auc={acc.get('auc'):.4f} "
              f"eps={eps.get('eps'):.4f} (sp={eps.get('spearman'):.4f}, "
              f"iou={eps.get('iou'):.4f})")
        print(f"      EPS tail: p10={eps.get('eps_p10'):.4f} p50={eps.get('eps_p50'):.4f} "
              f"| faith del={faith.get('deletion_auc'):.4f} agree={agree:.4f}")

        rows.append({
            "bits": bits, "policy": policy,
            "acc": acc.get("acc"), "auc": acc.get("auc"),
            "eps": eps.get("eps"), "spearman": eps.get("spearman"),
            "iou": eps.get("iou"),
            "eps_p10": eps.get("eps_p10"), "eps_p50": eps.get("eps_p50"),
            "spearman_p10": eps.get("spearman_p10"),
            "deletion_auc": faith.get("deletion_auc"),
            "insertion_auc": faith.get("insertion_auc"),
            "faith_agreement": round(agree, 4),
        })
        for v in eps.get("per_sample_eps", []):
            persample_rows.append({"bits": bits, "policy": policy, "eps": v})

    print("\n[4/4] Sweeping bit-widths ...")
    for bits in FRONTIER_BITS:
        print(f"\n----- {bits}-bit -----")
        run_one(bits, "naive")
        run_one(bits, "gaq")

    # ---- write frontier table ------------------------------------------------
    fcsv = C.out("gaq_frontier.csv")
    fields = ["bits", "policy", "acc", "auc", "eps", "spearman", "iou",
              "eps_p10", "eps_p50", "spearman_p10",
              "deletion_auc", "insertion_auc", "faith_agreement"]
    with open(fcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved -> {fcsv}")

    pcsv = C.out("gaq_frontier_eps.csv")
    with open(pcsv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["bits", "policy", "eps"])
        w.writeheader()
        w.writerows(persample_rows)
    print(f"Saved -> {pcsv}")

    # ---- print the contrast so you can read the verdict immediately ----------
    print("\n" + "=" * 70)
    print("FRONTIER VERDICT (EPS gap = GAQ - naive at each bit-width)")
    print("=" * 70)
    by = {(r["bits"], r["policy"]): r for r in rows}
    for bits in FRONTIER_BITS:
        g = by.get((bits, "gaq"))
        n = by.get((bits, "naive"))
        if g and n and g["eps"] is not None and n["eps"] is not None:
            gap = g["eps"] - n["eps"]
            flag = "  <-- GAQ pulls ahead" if gap > 0.03 else ""
            print(f"  INT{bits}:  GAQ eps={g['eps']:.4f}  naive eps={n['eps']:.4f}  "
                  f"gap={gap:+.4f}{flag}")
    print("\nIf the gap widens as bits drop, the GAQ contribution is demonstrated.")


if __name__ == "__main__":
    main()
