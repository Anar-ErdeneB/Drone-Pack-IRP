"""
05_drone_soh_smoothing.py
Compare Butterworth filter vs Moving Average for SOH smoothing.
Run from pack2/ directory.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import isotonic_regression

df = pd.read_csv("soh_labels_drone.csv").sort_values("cycle").reset_index(drop=True)
cycles  = df["cycle"].values
soh_raw = df["soh_raw"].values

print(f"Cycles: {len(cycles)}")
print(f"SOH range: {soh_raw.min():.4f} – {soh_raw.max():.4f}")
print(f"SOH std  : {soh_raw.std():.4f}")

# ── Butterworth filter (various cutoff frequencies) ────────────────────────
def butterworth(data, cutoff, order=3):
    """Zero-phase low-pass Butterworth filter."""
    nyq = 0.5
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

butter_options = {
    "Butterworth (cutoff=0.05)": butterworth(soh_raw, 0.05),
    "Butterworth (cutoff=0.10)": butterworth(soh_raw, 0.10),
    "Butterworth (cutoff=0.20)": butterworth(soh_raw, 0.20),
}

# ── Moving average (various window sizes) ─────────────────────────────────
def moving_avg(data, window):
    """Centered moving average."""
    return pd.Series(data).rolling(
        window, center=True, min_periods=1).mean().values

ma_options = {
    "Moving Avg (w=5)" : moving_avg(soh_raw, 5),
    "Moving Avg (w=10)": moving_avg(soh_raw, 10),
    "Moving Avg (w=20)": moving_avg(soh_raw, 20),
}

# ── Isotonic regression (monotone baseline) ────────────────────────────────
iso = isotonic_regression(soh_raw, increasing=False).x

# ── Print stats for each method ────────────────────────────────────────────
print("\nSmoothing method comparison:")
print(f"{'Method':35s}  {'Min':>6}  {'Max':>6}  {'Std':>6}  "
      f"{'Monotone':>8}  {'RMSE vs raw':>11}")

all_methods = {**butter_options, **ma_options, "Isotonic": iso}

for name, vals in all_methods.items():
    is_mono  = all(vals[i] >= vals[i+1] or
                   abs(vals[i]-vals[i+1]) < 1e-6
                   for i in range(len(vals)-1))
    rmse     = np.sqrt(np.mean((vals - soh_raw)**2))
    print(f"  {name:35s}  {vals.min():.4f}  {vals.max():.4f}  "
          f"{vals.std():.4f}  {'Yes' if is_mono else 'No':>8}  {rmse:.6f}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# ── Top left: Butterworth comparison ──────────────────────────────────────
axes[0,0].scatter(cycles, soh_raw, s=8, color="gray",
                  alpha=0.4, label="Raw SOH", zorder=1)
colors = ["steelblue", "darkorange", "green"]
for (name, vals), col in zip(butter_options.items(), colors):
    axes[0,0].plot(cycles, vals, lw=2, color=col, label=name)
axes[0,0].axhline(0.8, color="red", lw=1.5, linestyle="--", label="EOL 80%")
axes[0,0].set_title("Butterworth Filter — different cutoff frequencies")
axes[0,0].set_xlabel("Cycle"); axes[0,0].set_ylabel("SOH")
axes[0,0].legend(fontsize=8); axes[0,0].grid(True)
axes[0,0].set_ylim(0.74, 1.06)

# ── Top right: Moving average comparison ──────────────────────────────────
axes[0,1].scatter(cycles, soh_raw, s=8, color="gray",
                  alpha=0.4, label="Raw SOH", zorder=1)
for (name, vals), col in zip(ma_options.items(), colors):
    axes[0,1].plot(cycles, vals, lw=2, color=col, label=name)
axes[0,1].axhline(0.8, color="red", lw=1.5, linestyle="--", label="EOL 80%")
axes[0,1].set_title("Moving Average — different window sizes")
axes[0,1].set_xlabel("Cycle"); axes[0,1].set_ylabel("SOH")
axes[0,1].legend(fontsize=8); axes[0,1].grid(True)
axes[0,1].set_ylim(0.74, 1.06)

# ── Bottom left: Best of each vs isotonic ─────────────────────────────────
axes[1,0].scatter(cycles, soh_raw, s=8, color="gray",
                  alpha=0.4, label="Raw SOH", zorder=1)
axes[1,0].plot(cycles, butter_options["Butterworth (cutoff=0.10)"],
               lw=2, color="steelblue", label="Butterworth (cutoff=0.10)")
axes[1,0].plot(cycles, ma_options["Moving Avg (w=10)"],
               lw=2, color="darkorange", label="Moving Avg (w=10)")
axes[1,0].plot(cycles, iso, lw=2, color="purple",
               linestyle="--", label="Isotonic regression")
axes[1,0].axhline(0.8, color="red", lw=1.5, linestyle="--", label="EOL 80%")
axes[1,0].set_title("Best of each method — head-to-head")
axes[1,0].set_xlabel("Cycle"); axes[1,0].set_ylabel("SOH")
axes[1,0].legend(fontsize=8); axes[1,0].grid(True)
axes[1,0].set_ylim(0.74, 1.06)

# ── Bottom right: Residuals (raw - smoothed) ──────────────────────────────
for (name, vals, col) in [
    ("Butterworth (0.10)", butter_options["Butterworth (cutoff=0.10)"], "steelblue"),
    ("Moving Avg (w=10)",  ma_options["Moving Avg (w=10)"],             "darkorange"),
    ("Isotonic",           iso,                                          "purple"),
]:
    residuals = soh_raw - vals
    axes[1,1].plot(cycles, residuals, lw=0.8, alpha=0.8,
                   label=f"{name} (std={residuals.std():.4f})", color=col)
axes[1,1].axhline(0, color="black", lw=1)
axes[1,1].set_title("Residuals (raw − smoothed)\nSmaller = more smoothing")
axes[1,1].set_xlabel("Cycle"); axes[1,1].set_ylabel("Residual")
axes[1,1].legend(fontsize=8); axes[1,1].grid(True)

plt.suptitle("SOH Smoothing Comparison — Drone Battery Dataset",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("soh_smoothing_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved → soh_smoothing_comparison.png")