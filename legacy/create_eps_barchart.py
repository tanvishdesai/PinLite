"""
Figure 4: EPS Component Breakdown — Grouped bar chart
Spearman ρ_s, IoU_20, and Cosine Similarity for each model variant.
Error bars show 95 % bootstrap confidence intervals.
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "text.usetex": False,
})

# === DATA (from enhanced-eps-results.txt, 500 samples, bootstrap 95% CIs) ===
models = ["Teacher", "Distilled", "Pruned", "MQA", "LowRank", "Linear\nAttn", "Combined"]

# Spearman ρ_s
spearman_mean = np.array([1.0000, 0.6559, 0.6505, 0.6517, 0.5749, -0.0417, 0.6555])
spearman_ci_lo = np.array([1.0000, 0.6369, 0.6302, 0.6326, 0.5534, -0.0516, 0.6329])
spearman_ci_hi = np.array([1.0000, 0.6788, 0.6698, 0.6707, 0.5957, -0.0307, 0.6749])

# IoU_20
iou_mean = np.array([1.0000, 0.4947, 0.4782, 0.4966, 0.4739, 0.0911, 0.4929])
iou_ci_lo = np.array([1.0000, 0.4762, 0.4600, 0.4802, 0.4553, 0.0858, 0.4759])
iou_ci_hi = np.array([1.0000, 0.5113, 0.4946, 0.5139, 0.4932, 0.0970, 0.5124])

# Cosine similarity
cos_mean = np.array([1.0000, 0.9189, 0.9008, 0.9124, 0.8910, 0.6643, 0.9175])
cos_ci_lo = np.array([1.0000, 0.9162, 0.8987, 0.9097, 0.8876, 0.6575, 0.9150])
cos_ci_hi = np.array([1.0000, 0.9215, 0.9030, 0.9150, 0.8943, 0.6710, 0.9199])

# Compute symmetric error-bar half-widths
sp_err = np.array([spearman_mean - spearman_ci_lo, spearman_ci_hi - spearman_mean])
io_err = np.array([iou_mean - iou_ci_lo, iou_ci_hi - iou_mean])
co_err = np.array([cos_mean - cos_ci_lo, cos_ci_hi - cos_mean])

# Teacher has 0 error
sp_err[:, 0] = 0
io_err[:, 0] = 0
co_err[:, 0] = 0

n = len(models)
x = np.arange(n)
width = 0.24

# Color palette — IEEE-friendly, colorblind-safe
c_spearman = "#2D6A9F"   # slate blue
c_iou      = "#E07B3C"   # warm orange
c_cosine   = "#4E9A5D"   # forest green

fig, ax = plt.subplots(figsize=(7.16, 3.4))  # IEEE single-column width ≈ 3.5 in, double ≈ 7.16 in

bars1 = ax.bar(x - width, spearman_mean, width, yerr=sp_err,
               label=r"Spearman $\rho_s$", color=c_spearman, edgecolor="white",
               linewidth=0.4, capsize=2.5, error_kw={"linewidth": 0.8, "capthick": 0.8})

bars2 = ax.bar(x, iou_mean, width, yerr=io_err,
               label=r"IoU$_{20}$", color=c_iou, edgecolor="white",
               linewidth=0.4, capsize=2.5, error_kw={"linewidth": 0.8, "capthick": 0.8})

bars3 = ax.bar(x + width, cos_mean, width, yerr=co_err,
               label="Cosine", color=c_cosine, edgecolor="white",
               linewidth=0.4, capsize=2.5, error_kw={"linewidth": 0.8, "capthick": 0.8})

# Formatting
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Score")
ax.set_ylim(-0.12, 1.12)
ax.axhline(y=0, color="grey", linewidth=0.5, linestyle="-")

# Add a subtle reference line at 1.0 for the teacher baseline
ax.axhline(y=1.0, color="#BBBBBB", linewidth=0.5, linestyle="--", zorder=0)

# Legend
ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="#CCCCCC")

# Light grid on y-axis
ax.yaxis.grid(True, linestyle=":", linewidth=0.4, alpha=0.6)
ax.set_axisbelow(True)

# Spine cleanup
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)

plt.tight_layout()

out_path = r"c:\Users\DELL\Desktop\code_playground\Multi-Modal\PinLite\pinlite results v3\EPS_Component_Breakdown_v3.png"
fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
print(f"✅ Saved → {out_path}")

# Also save as PDF for LaTeX inclusion
pdf_path = out_path.replace(".png", ".pdf")
fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
print(f"✅ Saved → {pdf_path}")

plt.close()
