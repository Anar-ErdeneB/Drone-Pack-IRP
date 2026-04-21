"""
05_feature_correlation.py
=========================
Computes and plots the correlation between manual features and SOH,
replicating the analysis approach of Fan et al. (2025).

Reads  : drone_manual_feats.npy
         drone_index.csv
Writes : drone_feature_correlation.png

Run from pack2/ directory:
    python 05_feature_correlation.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr

# ── Load data ──────────────────────────────────────────────────────────────
feats  = np.load("drone_manual_feats.npy")       # (352, 14)
index  = pd.read_csv("drone_index.csv")
soh    = index["soh"].values                      # smoothed SOH

FEAT_NAMES = [
    "Vd_std",    "Vd_diff",   "Vmax_std",  "Vmax_mean",
    "Vmin_diff", "Td_max",    "Td_diff",   "Tmax_min",
    "Tmin_std",  "Tmin_mean", "Tmin_min",  "Tmin_diff",
    "ir_mean",   "T_mean"
]

FEAT_LABELS = [
    "Vd_std\n(pack V spread std)",
    "Vd_diff\n(pack V spread range)",
    "Vmax_std\n(std of max pack V)",
    "Vmax_mean\n(mean of max pack V)",
    "Vmin_diff\n(range of min pack V)",
    "Td_max\n(max temp spread)",
    "Td_diff\n(range of temp spread)",
    "Tmax_min\n(min of max pack T)",
    "Tmin_std\n(std of min pack T)",
    "Tmin_mean\n(mean of min pack T)",
    "Tmin_min\n(min of min pack T)",
    "Tmin_diff\n(range of min pack T)",
    "ir_mean\n(mean cell IR [mΩ])",
    "T_mean\n(ambient temp [°C])"
]

n_feats = feats.shape[1]

# ── Compute correlations ───────────────────────────────────────────────────
pearson_r  = []
pearson_p  = []
spearman_r = []
spearman_p = []

for i in range(n_feats):
    col = feats[:, i]
    valid = ~np.isnan(col)
    pr, pp = pearsonr(col[valid],  soh[valid])
    sr, sp = spearmanr(col[valid], soh[valid])
    pearson_r.append(pr);   pearson_p.append(pp)
    spearman_r.append(sr);  spearman_p.append(sp)

pearson_r  = np.array(pearson_r)
spearman_r = np.array(spearman_r)

print("Feature correlation with SOH:")
print(f"{'Feature':12s}  {'Pearson r':>10}  {'p-value':>10}  "
      f"{'Spearman r':>10}  {'p-value':>10}  {'|r| > 0.3':>10}")
print("-" * 70)
for i, name in enumerate(FEAT_NAMES):
    sig = "***" if pearson_p[i] < 0.001 else ("**" if pearson_p[i] < 0.01
          else ("*" if pearson_p[i] < 0.05 else ""))
    strong = "YES" if abs(pearson_r[i]) > 0.3 else ""
    print(f"  {name:12s}  {pearson_r[i]:>10.4f}  {pearson_p[i]:>10.2e}  "
          f"{spearman_r[i]:>10.4f}  {spearman_p[i]:>10.2e}  "
          f"{strong:>6} {sig}")

# ── Sort by absolute Pearson correlation ──────────────────────────────────
sort_idx = np.argsort(np.abs(pearson_r))[::-1]
print(f"\nFeatures ranked by |Pearson r| with SOH:")
for rank, i in enumerate(sort_idx):
    print(f"  {rank+1:2d}. {FEAT_NAMES[i]:12s}: r = {pearson_r[i]:+.4f}")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 1: Correlation bar chart (Fan et al. style)
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

x     = np.arange(n_feats)
colors_p = ["steelblue" if r >= 0 else "darkorange" for r in pearson_r]
colors_s = ["steelblue" if r >= 0 else "darkorange" for r in spearman_r]

# Pearson
bars = axes[0].bar(x, pearson_r, color=colors_p, alpha=0.8, edgecolor="black", lw=0.5)
axes[0].axhline(0,    color="black", lw=0.8)
axes[0].axhline( 0.3, color="green", lw=1.5, linestyle="--", alpha=0.7,
                 label="|r| = 0.3 threshold")
axes[0].axhline(-0.3, color="green", lw=1.5, linestyle="--", alpha=0.7)
axes[0].set_xticks(x)
axes[0].set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=8)
axes[0].set_ylabel("Pearson Correlation with SOH")
axes[0].set_title("Pearson Correlation: Manual Features vs SOH\n"
                  "(blue = positive, orange = negative)")
axes[0].set_ylim(-1, 1)
axes[0].legend(fontsize=8); axes[0].grid(axis="y", alpha=0.4)

# Add value labels on bars
for bar, val in zip(bars, pearson_r):
    ypos = val + 0.02 if val >= 0 else val - 0.05
    axes[0].text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

# Spearman
bars2 = axes[1].bar(x, spearman_r, color=colors_s, alpha=0.8,
                    edgecolor="black", lw=0.5)
axes[1].axhline(0,    color="black", lw=0.8)
axes[1].axhline( 0.3, color="green", lw=1.5, linestyle="--", alpha=0.7,
                 label="|r| = 0.3 threshold")
axes[1].axhline(-0.3, color="green", lw=1.5, linestyle="--", alpha=0.7)
axes[1].set_xticks(x)
axes[1].set_xticklabels(FEAT_NAMES, rotation=45, ha="right", fontsize=8)
axes[1].set_ylabel("Spearman Correlation with SOH")
axes[1].set_title("Spearman Correlation: Manual Features vs SOH\n"
                  "(blue = positive, orange = negative)")
axes[1].set_ylim(-1, 1)
axes[1].legend(fontsize=8); axes[1].grid(axis="y", alpha=0.4)

for bar, val in zip(bars2, spearman_r):
    ypos = val + 0.02 if val >= 0 else val - 0.05
    axes[1].text(bar.get_x() + bar.get_width()/2, ypos,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=7)

plt.suptitle("Feature Correlation Analysis — Drone Battery Pack",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_feature_correlation_bars.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → drone_feature_correlation_bars.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 2: Scatter plots — each feature vs SOH (top 8 by |r|)
# ══════════════════════════════════════════════════════════════════════════
top8 = sort_idx[:8]
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

cmap   = plt.cm.RdYlGn_r
cycles = index["cycle"].values
norm_c = plt.Normalize(cycles.min(), cycles.max())

for plot_idx, feat_idx in enumerate(top8):
    ax  = axes[plot_idx]
    col = feats[:, feat_idx]
    sc  = ax.scatter(col, soh, c=cycles, cmap=cmap, s=15, alpha=0.7)
    # Trend line
    valid = ~np.isnan(col)
    z     = np.polyfit(col[valid], soh[valid], 1)
    xr    = np.linspace(col[valid].min(), col[valid].max(), 100)
    ax.plot(xr, np.polyval(z, xr), "k--", lw=1.5, alpha=0.7)

    ax.set_xlabel(FEAT_LABELS[feat_idx], fontsize=8)
    ax.set_ylabel("SOH", fontsize=8)
    ax.set_title(f"r = {pearson_r[feat_idx]:+.3f}  "
                 f"ρ = {spearman_r[feat_idx]:+.3f}",
                 fontsize=9)
    ax.grid(True, alpha=0.4)
    plt.colorbar(sc, ax=ax, label="Cycle", shrink=0.8)

plt.suptitle("Top 8 Features by |Pearson r| — Scatter vs SOH\n"
             "(colored by cycle number: red=early, green=late)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_feature_correlation_scatter.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → drone_feature_correlation_scatter.png")

# ══════════════════════════════════════════════════════════════════════════
# FIGURE 3: Correlation heatmap between all features + SOH
# ══════════════════════════════════════════════════════════════════════════
data_matrix = np.column_stack([feats, soh])
col_names   = FEAT_NAMES + ["SOH"]
corr_matrix = np.zeros((len(col_names), len(col_names)))

for i in range(len(col_names)):
    for j in range(len(col_names)):
        valid = ~(np.isnan(data_matrix[:, i]) | np.isnan(data_matrix[:, j]))
        r, _  = pearsonr(data_matrix[valid, i], data_matrix[valid, j])
        corr_matrix[i, j] = r

fig, ax = plt.subplots(figsize=(14, 12))
im = ax.imshow(corr_matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.8)

ax.set_xticks(range(len(col_names)))
ax.set_yticks(range(len(col_names)))
ax.set_xticklabels(col_names, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(col_names, fontsize=8)

# Annotate cells
for i in range(len(col_names)):
    for j in range(len(col_names)):
        val = corr_matrix[i, j]
        color = "white" if abs(val) > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=6, color=color)

# Highlight SOH row/column
ax.axhline(len(col_names)-1.5, color="gold", lw=2)
ax.axvline(len(col_names)-1.5, color="gold", lw=2)

plt.title("Pearson Correlation Heatmap — Manual Features + SOH\n"
          "(gold lines highlight SOH row/column)",
          fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_feature_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → drone_feature_correlation_heatmap.png")
print("\nDone.")