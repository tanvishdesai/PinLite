"""
====================================================================================
GAQ — Assemble Table 1 + check the locked success bar
====================================================================================
Reads gaq_results.csv (P0/PTQ/QAT rows) plus the optional latency / cross-dataset
CSVs and prints the paper's money table as Markdown, then evaluates the locked
success bar. Accuracy/EPS drops are measured against the FP32 *student* (the model
we quantize); speedup is measured against the FP32 teacher:

    accuracy drop <= 1.0 pt    EPS drop <= 0.05    CPU speedup >= 1.3x

Run (after the phase scripts):
    python gaq_build_table.py
Outputs: gaq_table1.md
====================================================================================
"""

import csv
import os

import gaq_config as C


def _read_rows(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _fmt(x, nd=4):
    v = _f(x)
    return "—" if v is None else f"{v:.{nd}f}"


def main():
    C.banner("GAQ — TABLE 1 + SUCCESS BAR")
    rows = _read_rows(C.out("gaq_results.csv"))
    if not rows:
        raise SystemExit("gaq_results.csv not found — run the phase scripts first.")

    # Order rows the way the paper tells the story.
    order = ["Teacher FP32", "Distilled student (FP32)",
             "Naive INT8 attention (strawman)", "GAQ-INT8 (PTQ, ours)", "GAQ-QAT (ours)"]
    by_name = {r["model"]: r for r in rows}
    ordered = [by_name[n] for n in order if n in by_name]
    ordered += [r for r in rows if r["model"] not in order]

    header = ("| Model | Size MB | Params M | Latency ms | Acc | AUC | EPS | "
              "Spearman | IoU | Del-AUC | Ins-AUC | Faith-agree | Speedup |")
    sep = "|" + "---|" * 13
    lines = [header, sep]
    for r in ordered:
        lines.append("| {m} | {sz} | {pa} | {lt} | {ac} | {au} | {ep} | {sp} | {iou} | "
                     "{da} | {ia} | {fa} | {spd} |".format(
                         m=r["model"], sz=_fmt(r.get("size_mb"), 2),
                         pa=_fmt(r.get("params_m"), 3), lt=_fmt(r.get("latency_cpu_ms"), 1),
                         ac=_fmt(r.get("acc")), au=_fmt(r.get("auc")), ep=_fmt(r.get("eps")),
                         sp=_fmt(r.get("spearman")), iou=_fmt(r.get("iou")),
                         da=_fmt(r.get("deletion_auc")), ia=_fmt(r.get("insertion_auc")),
                         fa=_fmt(r.get("faith_agreement")), spd=_fmt(r.get("speedup_vs_teacher"), 2)))
    table_md = "\n".join(lines)
    print("\n" + table_md + "\n")

    # ---- Success-bar check (GAQ rows vs the FP32 student) --------------------
    # The model we quantize is the distilled FP32 student, so accuracy/EPS drops
    # are measured against IT, not the teacher. (EPS is the student's attention
    # agreement with the teacher; the teacher's own EPS is trivially 1.0, so it
    # is not a meaningful reference.) Speedup remains vs the FP32 teacher — that
    # is exactly what the speedup_vs_teacher column already holds.
    ref = by_name.get("Distilled student (FP32)")
    print("Locked success bar (vs FP32 student; speedup vs FP32 teacher): "
          f"acc drop <= {C.ACC_DROP_LIMIT}, EPS drop <= {C.EPS_DROP_LIMIT}, "
          f"speedup >= {C.SPEEDUP_MIN}")
    bar_md = ["", "## Success-bar check", ""]
    if ref:
        r_acc, r_eps = _f(ref.get("acc")), _f(ref.get("eps"))
        for name in ("GAQ-INT8 (PTQ, ours)", "GAQ-QAT (ours)"):
            r = by_name.get(name)
            if not r:
                continue
            acc, eps, spd = _f(r.get("acc")), _f(r.get("eps")), _f(r.get("speedup_vs_teacher"))
            pa = (acc is not None and r_acc is not None and acc >= r_acc - C.ACC_DROP_LIMIT)
            pe = (eps is not None and r_eps is not None and eps >= r_eps - C.EPS_DROP_LIMIT)
            ps = (spd is not None and spd >= C.SPEEDUP_MIN)
            verdict = "PASS ✅" if (pa and pe and ps) else "needs work ⚠️"
            line = (f"- **{name}**: acc {'✓' if pa else '✗'}  "
                    f"eps {'✓' if pe else '✗'}  speed {'✓' if ps else '✗'}  -> {verdict}")
            print(line)
            bar_md.append(line)

    # ---- Append auxiliary tables ---------------------------------------------
    extra_md = []
    lat = _read_rows(C.out("gaq_latency.csv"))
    if lat:
        extra_md += ["", "## Named-hardware latency", "",
                     "| Backend | Latency ms | Hardware |", "|---|---|---|"]
        for r in lat:
            extra_md.append(f"| {r['backend']} | {r['latency_ms']} | {r['hardware']} |")
    cross = _read_rows(C.out("gaq_crossdataset.csv"))
    if cross:
        extra_md += ["", "## Cross-dataset (LAV-DF -> FakeAVCeleb)", "",
                     "| Model | Acc | F1 | AUC |", "|---|---|---|---|"]
        for r in cross:
            extra_md.append(f"| {r['model']} | {r['favc_acc']} | {r['favc_f1']} | {r['favc_auc']} |")

    # ---- Precision frontier (the headline contrast) --------------------------
    front = _read_rows(C.out("gaq_frontier.csv"))
    if front:
        extra_md += ["", "## Precision frontier — naive vs GAQ as bits drop", "",
                     "| Bits | Policy | Acc | AUC | EPS | EPS p10 | Faith-agree |",
                     "|---|---|---|---|---|---|---|"]
        # sort by bits desc, naive before gaq for readability
        front_sorted = sorted(front, key=lambda r: (-int(r["bits"]), r["policy"]))
        for r in front_sorted:
            extra_md.append(
                f"| INT{r['bits']} | {r['policy']} | {_fmt(r.get('acc'))} | "
                f"{_fmt(r.get('auc'))} | {_fmt(r.get('eps'))} | "
                f"{_fmt(r.get('eps_p10'))} | {_fmt(r.get('faith_agreement'))} |")
        # EPS gap verdict (GAQ - naive) per bit-width
        by_bp = {(r["bits"], r["policy"]): r for r in front}
        extra_md += ["", "**EPS gap (GAQ − naive) — widening gap = GAQ contribution:**", ""]
        for bits in sorted({r["bits"] for r in front}, key=lambda b: -int(b)):
            g, n = by_bp.get((bits, "gaq")), by_bp.get((bits, "naive"))
            ge, ne = _f(g.get("eps")) if g else None, _f(n.get("eps")) if n else None
            if ge is not None and ne is not None:
                extra_md.append(f"- INT{bits}: GAQ {ge:.4f} − naive {ne:.4f} = **{ge - ne:+.4f}**")

    out_md = "# PIN-Lite v2 — GAQ Table 1\n\n" + table_md + "\n" + \
             "\n".join(bar_md) + "\n" + "\n".join(extra_md) + "\n"
    with open(C.out("gaq_table1.md"), "w", encoding="utf-8") as f:
        f.write(out_md)
    print(f"\nSaved -> {C.out('gaq_table1.md')}")


if __name__ == "__main__":
    main()
