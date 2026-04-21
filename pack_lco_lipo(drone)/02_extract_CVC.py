"""
02_extract_charge_curves.py
===========================
Extracts end-aligned per-cell charge voltage curves from the 4 pack
charge Excel files.

Reads  : data/UUT2_BatteryPackP{1,2,3,4}_Charge.xlsx
Writes : drone_charge_curves.npy   (N, 101) float32
         drone_curve_index.csv     (N, 3)   cycle, total_mah, n_packs

Curve construction:
  - Average Pack Volts [V] / 6 across all available packs (>= MIN_PACKS)
  - End-align on Capacity Added [mAh] axis (index 0 = end of charge)
  - Resample to N_POINTS using linear interpolation
  - Apply Gaussian smoothing (sigma=2)
  - Valid range check: cell voltage must be 2.5-4.5 V
  - Outlier removal: RMSE vs median curve > mean + 3*std

Run from pack2/ directory:
    python 02_extract_charge_curves.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage     import gaussian_filter1d

# Configuration
DATA_DIR       = "data"
N_POINTS       = 101
N_CELLS        = 6
SIGMA          = 2
MIN_PACKS      = 2
OUTLIER_SIGMA  = 3.0
OUT_NPY        = "drone_charge_curves.npy"
OUT_IDX        = "drone_curve_index.csv"

CHG_FILES = {
    "P1": os.path.join(DATA_DIR, "UUT2_BatteryPackP1_Charge.xlsx"),
    "P2": os.path.join(DATA_DIR, "UUT2_BatteryPackP2_Charge.xlsx"),
    "P3": os.path.join(DATA_DIR, "UUT2_BatteryPackP3_Charge.xlsx"),
    "P4": os.path.join(DATA_DIR, "UUT2_BatteryPackP4_Charge.xlsx"),
}

# Load all charge sheets
print("Reading charge files ...")
chg_data = {p: {} for p in ["P1", "P2", "P3", "P4"]}

for pack, fpath in CHG_FILES.items():
    xl     = pd.ExcelFile(fpath)
    sheets = [s for s in xl.sheet_names
              if s not in ("Readme", "Missing Data")]
    for s in sheets:
        try:
            df = pd.read_excel(fpath, sheet_name=s)
            df.columns = [c.replace("\n", " ").strip() for c in df.columns]
            cycle = int(s)
            if len(df) > 3:
                chg_data[pack][cycle] = df
        except Exception:
            pass
    print(f"  {pack}: {len(chg_data[pack])} sheets  "
          f"({min(chg_data[pack].keys())}-{max(chg_data[pack].keys())})")


def extract_curve(cycle_num):
    pack_curves = []
    total_mahs  = []

    for pack in ["P1", "P2", "P3", "P4"]:
        if cycle_num not in chg_data[pack]:
            continue
        df = chg_data[pack][cycle_num]

        v_cols   = [c for c in df.columns if "Pack Volts"     in c]
        cap_cols = [c for c in df.columns if "Capacity Added" in c]
        if not v_cols or not cap_cols:
            continue

        v_pack  = df[v_cols[0]].values.astype(float) / N_CELLS
        cap_mah = df[cap_cols[0]].values.astype(float)
        tot_mah = float(cap_mah[-1])

        if tot_mah < 100 or len(v_pack) < 5:
            continue

        cap_rem = tot_mah - cap_mah
        x = cap_rem[::-1].copy()
        y = v_pack[::-1].copy()
        _, uid = np.unique(x, return_index=True)
        x, y   = x[uid], y[uid]
        if len(x) < 4:
            continue

        f    = interp1d(x, y, kind="linear", bounds_error=False,
                        fill_value=(y[0], y[-1]))
        y_rs = f(np.linspace(0, x.max(), N_POINTS))
        pack_curves.append(y_rs)
        total_mahs.append(tot_mah)

    if len(pack_curves) < MIN_PACKS:
        return None, None, 0

    curve = gaussian_filter1d(np.mean(pack_curves, axis=0), sigma=SIGMA)

    if curve.min() < 2.5 or curve.max() > 4.5:
        return None, None, 0

    return curve.astype(np.float32), float(np.mean(total_mahs)), len(pack_curves)


# Determine candidate cycles
all_chg_cycles = set()
for p in ["P1", "P2", "P3", "P4"]:
    all_chg_cycles |= set(chg_data[p].keys())

candidate_cycles = sorted(
    c for c in all_chg_cycles
    if sum(c in chg_data[p] for p in ["P1","P2","P3","P4"]) >= MIN_PACKS
)
print(f"\nCandidate cycles (>={MIN_PACKS} packs): {len(candidate_cycles)}")

# Extract all curves
print("Extracting curves ...")
curves_list = []
index_rows  = []
skipped     = []

for cycle in candidate_cycles:
    curve, tot_mah, n_packs = extract_curve(cycle)
    if curve is None:
        skipped.append(cycle)
        continue
    curves_list.append(curve)
    index_rows.append({"cycle": cycle, "total_mah": tot_mah, "n_packs": n_packs})

curves_raw = np.array(curves_list, dtype=np.float32)
print(f"  Extracted before outlier removal : {len(curves_raw)}")
print(f"  Skipped (insufficient data)      : {len(skipped)}")

# Outlier removal: RMSE vs median curve
print(f"\nRemoving outlier curves (threshold = mean + {OUTLIER_SIGMA} * std) ...")

median_curve   = np.median(curves_raw, axis=0)
rmse           = np.sqrt(((curves_raw - median_curve)**2).mean(axis=1))
rmse_mean      = rmse.mean()
rmse_std       = rmse.std()
threshold      = rmse_mean + OUTLIER_SIGMA * rmse_std
outlier_mask   = rmse > threshold
outlier_cycles = [index_rows[i]["cycle"] for i in range(len(index_rows))
                  if outlier_mask[i]]

print(f"  RMSE mean={rmse_mean:.5f}  std={rmse_std:.5f}  threshold={threshold:.5f}")
print(f"  Outlier cycles removed : {outlier_cycles}")

keep       = ~outlier_mask
charge_arr = curves_raw[keep]
index_df   = pd.DataFrame([r for r, k in zip(index_rows, keep) if k])
print(f"  Retained after removal : {len(charge_arr)}")

# Save
np.save(OUT_NPY, charge_arr)
index_df.to_csv(OUT_IDX, index=False)
print(f"\nSaved -> {OUT_NPY}  {charge_arr.shape}")
print(f"Saved -> {OUT_IDX}  {index_df.shape}")

# Plot
x_axis     = np.linspace(0, 1, N_POINTS)
cycles_arr = index_df["cycle"].values
cmap       = plt.cm.RdYlGn_r
norm_c     = plt.Normalize(cycles_arr.min(), cycles_arr.max())

fig, axes = plt.subplots(1, 3, figsize=(20, 5))

for i in range(len(charge_arr)):
    axes[0].plot(x_axis, charge_arr[i],
                 color=cmap(norm_c(cycles_arr[i])), alpha=0.3, lw=0.6)
n = len(charge_arr)
axes[0].plot(x_axis, charge_arr[:n//4].mean(0),  "g-", lw=2.5, label="Early mean")
axes[0].plot(x_axis, charge_arr[-n//4:].mean(0), "r-", lw=2.5, label="Late mean")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_c), ax=axes[0], label="Cycle")
axes[0].set_xlabel("<- End of charge    Start of charge ->")
axes[0].set_ylabel("Cell voltage (V)")
axes[0].set_title(f"Charge curves after outlier removal  (n={len(index_df)})")
axes[0].legend(fontsize=9); axes[0].grid(True)

axes[1].scatter(range(len(rmse)), rmse, s=10, alpha=0.6,
                c=["red" if o else "steelblue" for o in outlier_mask])
axes[1].axhline(threshold, color="red",  lw=1.5, linestyle="--",
                label=f"Threshold={threshold:.5f}")
axes[1].axhline(rmse_mean, color="gray", lw=1,   linestyle=":",
                label=f"Mean={rmse_mean:.5f}")
axes[1].set_xlabel("Curve index"); axes[1].set_ylabel("RMSE vs median curve")
axes[1].set_title(f"Outlier detection\nRemoved: {outlier_mask.sum()} "
                  f"cycle(s) {outlier_cycles}")
axes[1].legend(fontsize=8); axes[1].grid(True)

axes[2].plot(index_df["cycle"], index_df["total_mah"],
             "o-", ms=3, lw=1, color="steelblue")
axes[2].set_xlabel("Cycle"); axes[2].set_ylabel("Capacity Added (mAh)")
axes[2].set_title("Total mAh charged per cycle")
axes[2].grid(True)

plt.suptitle("Drone Battery - Charge Curve Extraction", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_charge_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> drone_charge_curves.png")
print("\nDone.")