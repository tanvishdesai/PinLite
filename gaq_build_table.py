"""
====================================================================================
GAQ — Assemble Table 1 + check the locked success bar
====================================================================================
Reads gaq_results.csv (P0/PTQ/QAT rows) plus the optional latency / cross-dataset
CSVs and prints the paper's money table as Markdown, then evaluates the locked
success bar against the FP32 teacher reference:

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

    # ---- Success-bar check (GAQ rows vs teacher) -----------------------------
    teacher = by_name.get("Teacher FP32")
    print("Locked success bar (vs FP32 teacher): "
          f"acc drop <= {C.ACC_DROP_LIMIT}, EPS drop <= {C.EPS_DROP_LIMIT}, "
          f"speedup >= {C.SPEEDUP_MIN}")
    bar_md = ["", "## Success-bar check", ""]
    if teacher:
        t_acc, t_eps = _f(teacher.get("acc")), _f(teacher.get("eps"))
        for name in ("GAQ-INT8 (PTQ, ours)", "GAQ-QAT (ours)"):
            r = by_name.get(name)
            if not r:
                continue
            acc, eps, spd = _f(r.get("acc")), _f(r.get("eps")), _f(r.get("speedup_vs_teacher"))
            pa = (acc is not None and t_acc is not None and acc >= t_acc - C.ACC_DROP_LIMIT)
            pe = (eps is not None and t_eps is not None and eps >= t_eps - C.EPS_DROP_LIMIT)
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

    out_md = "# PIN-Lite v2 — GAQ Table 1\n\n" + table_md + "\n" + \
             "\n".join(bar_md) + "\n" + "\n".join(extra_md) + "\n"
    with open(C.out("gaq_table1.md"), "w", encoding="utf-8") as f:
        f.write(out_md)
    print(f"\nSaved -> {C.out('gaq_table1.md')}")


if __name__ == "__main__":
    main()
