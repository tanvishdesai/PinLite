"""
====================================================================================
GAQ — Plot the precision frontier (paper figure)
====================================================================================
Reads gaq_frontier.csv and draws EPS-vs-bits and accuracy-vs-bits for naive vs GAQ.
The money figure: the two EPS curves should fan apart as bits drop.

Run (after gaq_p5_frontier.py):
    python gaq_plot_frontier.py
Outputs: gaq_frontier.png
====================================================================================
"""

import csv

import gaq_config as C


def _read(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("install matplotlib: pip install matplotlib")

    rows = _read(C.out("gaq_frontier.csv"))
    if not rows:
        raise SystemExit("gaq_frontier.csv not found — run gaq_p5_frontier.py first.")

    def series(policy, col):
        pts = [(int(r["bits"]), float(r[col])) for r in rows
               if r["policy"] == policy and r[col] not in ("", None)]
        pts.sort(key=lambda x: x[0])
        return [p[0] for p in pts], [p[1] for p in pts]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, title in ((axes[0], "eps", "EPS (attention-map fidelity)"),
                           (axes[1], "acc", "Accuracy")):
        xn, yn = series("naive", col)
        xg, yg = series("gaq", col)
        ax.plot(xn, yn, "o--", color="#c0392b", label="Naive (softmax/LN/sigmoid quantized)")
        ax.plot(xg, yg, "s-", color="#2471a3", label="GAQ (ours, those ops FP32)")
        ax.set_xlabel("weight/activation bits")
        ax.set_title(title)
        ax.invert_xaxis()  # precision drops left-to-right
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("GAQ precision frontier — lower bits expose the naive scheme")
    fig.tight_layout()
    out = C.out("gaq_frontier.png")
    fig.savefig(out, dpi=150)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
