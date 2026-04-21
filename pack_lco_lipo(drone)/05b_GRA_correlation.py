"""
05b_gray_relational_analysis.py
================================
Computes Gray Relational Grade (GRG) between each manual feature
and SOH, following Fan et al. (2025).

Reads  : drone_manual_feats.npy
         drone_index.csv
Writes : drone_gra_results.png

Run from pack2/ directory:
    python 05b_gray_relational_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ── Load data ──────────────────────────────────────────────────────────────
feats = np.load("drone_manual_feats.npy")
index = pd.read_csv("drone_index.csv")
soh   = index["soh"].values

FEAT_NAMES = [
    "Vd_std", "Vd_diff", "Vmax_std", "Vmax_mean", "Vmin_diff",
    "Td_max", "Td_diff", "Tmax_min", "Tmin_std", "Tmin_mean",
    "Tmin_min", "Tmin_diff", "ir_mean", "T_mean"
]

# ── Gray Relational Analysis ───────────────────────────────────────────────
def gray_relational_grade(x, y, rho=0.5):
    """
    Compute the Gray Relational Grade between sequence x and reference y.
    
    Steps:
    1. Normalise both sequences to [0,1] (initial value processing)
    2. Compute absolute difference at each point
    3. Compute gray relational coefficient using distinguishing coefficient rho
    4. Average coefficients to get the grade
    
    rho = 0.5 is the standard value used in most GRA applications.
    Higher rho = more weight to global differences.
    Lower rho = more weight to local minimum differences.
    """
    # Normalise to [0,1]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-10)
    y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)

    # Absolute difference sequence
    delta = np.abs(x_norm - y_norm)

    delta_min = delta.min()
    delta_max = delta.max()

    # Gray relational coefficient at each point
    gamma = (delta_min + rho * delta_max) / (delta + rho * delta_max + 1e-10)

    # Gray Relational Grade = mean of coefficients
    return float(gamma.mean())

# Sort by cycle for correct temporal ordering
sort_idx = np.argsort(index["cycle"].values)
soh_sorted  = soh[sort_idx]
feats_sorted = feats[sort_idx]

# Compute GRG for each feature
grg_values  = []
pearson_r   = []
for i in range(feats.shape[1]):
    col   = feats_sorted[:, i]
    valid = ~np.isnan(col)
    grg   = gray_relational_grade(col[valid], soh_sorted[valid])
    pr, _ = pearsonr(col[valid], soh_sorted[valid])
    grg_values.append(grg)
    pearson_r.append(pr)

grg_values = np.array(grg_values)
pearson_r  = np.array(pearson_r)

# Sort by GRG descending
sort_grg = np.argsort(grg_values)[::-1]

print("Gray Relational Grade (GRG) vs Pearson r — ranked by GRG:")
print(f"{'Rank':>4}  {'Feature':12s}  {'GRG':>8}  {'|Pearson r|':>11}  {'Agreement':>10}")
print("-" * 55)
for rank, i in enumerate(sort_grg):
    agree = "YES" if (grg_values[i] > 0.6) == (abs(pearson_r[i]) > 0.3) else "DIFF"
    print(f"  {rank+1:2d}  {FEAT_NAMES[i]:12s}  {grg_values[i]:.4f}  "
          f"{abs(pearson_r[i]):11.4f}  {agree:>10}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

x      = np.arange(len(FEAT_NAMES))
colors = ["steelblue" if g >= 0.6 else "darkorange" for g in grg_values]

# GRG bar chart
bars = axes[0].bar(x, grg_values, color=colors, alpha=0.8,
                   edgecolor="black", lw=0.5)
axes[0].axhline(0.6, color="red", lw=1.5, linestyle="--",
                label="GRG = 0.6 (strong threshold)")
axes[0].axhline(0.5, color="orange", lw=1, linestyle=":",
                label="GRG = 0.5 (moderate threshold)")
axes[0].set_xticks(x)
axes[0].set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Gray Relational Grade (GRG)")
axes[0].set_title("Gray Relational Analysis\n"
                  "(blue = strong GRG ≥ 0.6, orange = weaker)")
axes[0].set_ylim(0, 1)
for bar, val in zip(bars, grg_values):
    axes[0].text(bar.get_x() + bar.get_width()/2, val + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=7)
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.4)

# GRG vs |Pearson r| scatter
sc = axes[1].scatter(np.abs(pearson_r), grg_values, s=60,
                     color="steelblue", alpha=0.8, edgecolors="black", lw=0.5)
for i, name in enumerate(FEAT_NAMES):
    axes[1].annotate(name, (abs(pearson_r[i]), grg_values[i]),
                     textcoords="offset points", xytext=(5, 3), fontsize=7)
axes[1].axhline(0.6, color="red",    lw=1, linestyle="--", alpha=0.6,
                label="GRG = 0.6")
axes[1].axvline(0.3, color="green",  lw=1, linestyle="--", alpha=0.6,
                label="|r| = 0.3")
axes[1].set_xlabel("|Pearson r| with SOH")
axes[1].set_ylabel("Gray Relational Grade (GRG)")
axes[1].set_title("GRG vs |Pearson r|\n"
                  "(top-right = strong by both methods)")
axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
axes[1].legend(fontsize=8); axes[1].grid(alpha=0.4)

plt.suptitle("Feature Informativeness: Gray Relational Analysis vs Pearson Correlation",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_gra_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → drone_gra_results.png")
print("\nDone.")