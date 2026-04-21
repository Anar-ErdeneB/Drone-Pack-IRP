"""
02_extract_calce.py
===================
Complete extraction pipeline for CALCE CS2 battery dataset.

Dataset:
    4 LCO pouch cells (CS2_35, CS2_36, CS2_37, CS2_38)
    CC-CV charge at 0.5C to 4.2V, discharge at 1C to 2.7V
    Room temperature (~25 degrees C), no temperature sensor
    ~1000 cycles per cell

Reads  : data/CS2_35/*.xlsx  (sheet: Channel_1-008)
         data/CS2_36/*.xlsx  (sheet: Channel_1-009)
         data/CS2_37/*.xlsx  (sheet: Channel_1-010)
         data/CS2_38/*.xlsx  (sheet: Channel_1-011)

Outputs (saved in processed/):
    calce_curves.npy         (N, 101) float32  -- charge voltage curves
    calce_soh.npy            (N,)     float32  -- SOH labels 0-1
    calce_temperature.npy    (N,)     float32  -- mean charge temperature
    calce_cell_ids.npy       (N,)     str      -- cell identifier per sample
    calce_index.csv          (N, 9)            -- human-readable reference

Every row i across all 4 arrays corresponds to the same cycle.

Run from source domain directory:
    python 02_extract_calce.py
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage    import gaussian_filter1d
from scipy.signal     import butter, filtfilt

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR   = "data"
OUT_DIR    = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS   = 101      # autoencoder input length
SIGMA      = 2        # Gaussian smoothing sigma for charge curves
T_DEFAULT  = 25.0     # fixed temperature (no sensor in CS2 cells)
SOH_MIN    = 0.75     # filter threshold -- exclude post-EOL cycles
BW_CUTOFF  = 0.10     # Butterworth SOH filter cutoff frequency
BW_ORDER   = 3        # Butterworth filter order

# Each CS2 cell uses a different channel sheet name
CS2_CELLS = {
    "CS2_33": "Channel_1-006",
    "CS2_34": "Channel_1-007",
    "CS2_35": "Channel_1-008",
    "CS2_36": "Channel_1-009",
    "CS2_37": "Channel_1-010",
    "CS2_38": "Channel_1-011",
}

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def butterworth_smooth(data, cutoff=BW_CUTOFF, order=BW_ORDER):
    """Zero-phase Butterworth low-pass filter, output capped at [0, 1]."""
    b, a = butter(order, cutoff / 0.5, btype="low", analog=False)
    return np.clip(filtfilt(b, a, data), 0.0, 1.0)


def load_cell(cell_id, sheet_name):
    """
    Load all xlsx files for one cell, sort chronologically by Date_Time,
    reindex Cycle_Index to be globally unique, return one DataFrame.
    """
    cell_dir = os.path.join(DATA_DIR, cell_id)
    files    = sorted(glob.glob(os.path.join(cell_dir, "*.xlsx")))
    if not files:
        print(f"  WARNING: no files found in {cell_dir}")
        return None

    dfs = []
    for fpath in files:
        try:
            df = pd.read_excel(fpath, sheet_name=sheet_name)
            dfs.append(df)
        except Exception as e:
            print(f"  WARNING: {os.path.basename(fpath)}: {e}")

    if not dfs:
        return None

    # Sort by first timestamp for chronological order
    dfs.sort(key=lambda d: d["Date_Time"].iloc[0])

    # Reindex Cycle_Index to be globally unique across all files
    combined     = []
    cycle_offset = 0
    for df in dfs:
        df = df.copy()
        df["Cycle_Index"] = df["Cycle_Index"] + cycle_offset
        cycle_offset      = df["Cycle_Index"].max()
        combined.append(df)

    full_df = pd.concat(combined, ignore_index=True)
    full_df = full_df.sort_values("Date_Time").reset_index(drop=True)
    return full_df


def extract_capacity_per_cycle(full_df):
    """
    Discharge_Capacity(Ah) is cumulative and never resets to zero.
    Capacity per cycle = last value - first value during discharge phase.
    Returns dict: {cycle_idx: capacity_ah}
    """
    cap_per_cycle = {}
    for cyc_idx, cyc_df in full_df.groupby("Cycle_Index"):
        dis_mask = cyc_df["Current(A)"] < -0.05
        if dis_mask.sum() < 5:
            continue
        dis_df  = cyc_df[dis_mask]
        q_start = float(dis_df["Discharge_Capacity(Ah)"].iloc[0])
        q_end   = float(dis_df["Discharge_Capacity(Ah)"].iloc[-1])
        q_diff  = q_end - q_start
        if q_diff > 0.1:
            cap_per_cycle[cyc_idx] = q_diff
    return cap_per_cycle


def extract_charge_curve(cyc_df):
    """
    Extract end-aligned charge voltage curve from one cycle.

    Steps:
      1. Filter to CC phase: Current(A) > 0.05A
      2. Reverse time axis so index 0 = end of charge (fully charged)
      3. Resample to N_POINTS using linear interpolation
      4. Apply Gaussian smoothing (sigma=2) to remove artifacts

    Returns (curve float32, t_mean float) or (None, None).
    """
    cc_mask = cyc_df["Current(A)"] > 0.05
    if cc_mask.sum() < 10:
        return None, None

    cc_df = cyc_df[cc_mask].copy()
    v     = cc_df["Voltage(V)"].values.astype(float)
    time  = cc_df["Test_Time(s)"].values.astype(float)

    if v.min() < 2.5 or v.max() > 4.5:
        return None, None

    # End-align: reverse so position 0 = end of charge
    time_rev = (time[-1] - time)[::-1]
    v_rev    = v[::-1]

    _, uid          = np.unique(time_rev, return_index=True)
    time_rev, v_rev = time_rev[uid], v_rev[uid]

    if len(time_rev) < 4:
        return None, None

    f    = interp1d(time_rev, v_rev, kind="linear",
                    bounds_error=False,
                    fill_value=(v_rev[0], v_rev[-1]))
    y_rs = f(np.linspace(0, time_rev.max(), N_POINTS))
    y_rs = gaussian_filter1d(y_rs, sigma=SIGMA)

    return y_rs.astype(np.float32), T_DEFAULT


# ══════════════════════════════════════════════════════════════════════════
# STEP 1-4: EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("CALCE CS2 -- Battery Charge Curve Extraction")
print("=" * 60)

all_curves  = []
all_records = []

for cell_id, sheet_name in CS2_CELLS.items():
    print(f"\n{cell_id}  (sheet: {sheet_name})")

    full_df = load_cell(cell_id, sheet_name)
    if full_df is None:
        continue

    print(f"  Total rows     : {len(full_df)}")
    print(f"  Cycle range    : {full_df['Cycle_Index'].min()} - "
          f"{full_df['Cycle_Index'].max()}")
    print(f"  Date range     : {full_df['Date_Time'].min()} -> "
          f"{full_df['Date_Time'].max()}")

    cap_per_cycle = extract_capacity_per_cycle(full_df)

    if not cap_per_cycle:
        print(f"  No valid discharge capacities found -- skipping")
        continue

    q_max = max(cap_per_cycle.values())
    print(f"  Discharge cycles with capacity : {len(cap_per_cycle)}")
    print(f"  Q_max = {q_max:.4f} Ah")
    print(f"  SOH range (raw): "
          f"{min(cap_per_cycle.values())/q_max:.4f} - "
          f"{max(cap_per_cycle.values())/q_max:.4f}")

    cell_curves, cell_records = [], []
    charge_num = 0

    for cyc_idx in sorted(full_df["Cycle_Index"].unique()):
        cyc_df = full_df[full_df["Cycle_Index"] == cyc_idx]

        if (cyc_df["Current(A)"] > 0.05).sum() < 10:
            continue

        soh, q_ah = None, None
        for look in [cyc_idx, cyc_idx + 1, cyc_idx - 1]:
            if look in cap_per_cycle:
                q_ah = cap_per_cycle[look]
                soh  = q_ah / q_max
                break
        if soh is None:
            continue

        curve, t_mean = extract_charge_curve(cyc_df)
        if curve is None:
            continue

        charge_num += 1

        cell_curves.append(curve)
        cell_records.append({
            "cell_id"   : cell_id,
            "group"     : "CALCE",
            "charge_num": charge_num,
            "cyc_idx"   : cyc_idx,
            "soh"       : soh,
            "q_ah"      : q_ah,
            "q_max"     : q_max,
            "t_mean"    : t_mean,
        })

    print(f"  Extracted charge curves : {len(cell_curves)}")
    if cell_curves:
        soh_v = [r["soh"] for r in cell_records]
        print(f"  SOH range (extracted)   : "
              f"{min(soh_v):.4f} - {max(soh_v):.4f}")

    all_curves.extend(cell_curves)
    all_records.extend(cell_records)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════
curves_arr = np.array(all_curves, dtype=np.float32)
index_df   = pd.DataFrame(all_records)

print(f"\n{'='*60}")
print(f"BEFORE SOH FILTERING")
print(f"{'='*60}")
print(f"  Total samples : {len(index_df)}")
print(f"  SOH range     : {index_df['soh'].min():.4f} - "
      f"{index_df['soh'].max():.4f}")
for cell_id, grp in index_df.groupby("cell_id"):
    print(f"    {cell_id}: {len(grp):4d} cycles  "
          f"SOH {grp['soh'].min():.4f}-{grp['soh'].max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: FILTER post-EOL cycles below SOH_MIN
# ══════════════════════════════════════════════════════════════════════════
print(f"\nFiltering to SOH >= {SOH_MIN} ...")
keep_mask  = index_df["soh"] >= SOH_MIN
n_removed  = (~keep_mask).sum()
index_df   = index_df[keep_mask].reset_index(drop=True)
curves_arr = curves_arr[keep_mask.values]

print(f"  Removed  : {n_removed} post-EOL cycles")
print(f"  Retained : {len(index_df)} cycles")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6b: OUTLIER CURVE REMOVAL
# ══════════════════════════════════════════════════════════════════════════
print(f"\nRemoving outlier curves "
      f"(threshold = mean + 3.0*std) ...")

OUTLIER_SIGMA = 3.0

median_curve = np.median(curves_arr, axis=0)
rmse         = np.sqrt(((curves_arr - median_curve)**2).mean(axis=1))
threshold    = rmse.mean() + OUTLIER_SIGMA * rmse.std()
outlier_mask = rmse > threshold

outlier_info = list(zip(
    index_df["cell_id"][outlier_mask].tolist(),
    index_df["charge_num"][outlier_mask].tolist()
))
print(f"  RMSE mean={rmse.mean():.5f}  std={rmse.std():.5f}  "
      f"threshold={threshold:.5f}")
print(f"  Outliers removed : {outlier_mask.sum()}  "
      f"(first 10: {outlier_info[:10]})")

keep_mask2 = ~outlier_mask
curves_arr = curves_arr[keep_mask2]
index_df   = index_df[keep_mask2].reset_index(drop=True)
print(f"  Retained         : {len(index_df)} cycles")
# ══════════════════════════════════════════════════════════════════════════
# STEP 7: SOH SMOOTHING per cell
# ══════════════════════════════════════════════════════════════════════════
print(f"\nApplying Butterworth smoothing "
      f"(order={BW_ORDER}, cutoff={BW_CUTOFF}) per cell ...")

index_df["soh_raw"]    = index_df["soh"].copy()
index_df["soh_smooth"] = index_df["soh"].copy()

for cell_id, grp in index_df.groupby("cell_id"):
    grp_sorted = grp.sort_values("charge_num")
    smooth     = butterworth_smooth(grp_sorted["soh"].values)
    index_df.loc[grp_sorted.index, "soh_smooth"] = smooth

index_df["soh"] = index_df["soh_smooth"]

print(f"  Per-cell SOH after smoothing:")
for cell_id, grp in index_df.groupby("cell_id"):
    print(f"    {cell_id}: {len(grp):4d} cycles  "
          f"raw {grp['soh_raw'].min():.4f}-{grp['soh_raw'].max():.4f}  "
          f"smooth {grp['soh_smooth'].min():.4f}-"
          f"{grp['soh_smooth'].max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: BUILD FINAL 4 ARRAYS
# ══════════════════════════════════════════════════════════════════════════
soh_arr   = index_df["soh"].values.astype(np.float32)
temp_arr  = index_df["t_mean"].values.astype(np.float32)
ids_arr   = index_df["cell_id"].values          # str array

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print(f"\nValidation:")
assert len(curves_arr) == len(soh_arr) == len(temp_arr) == len(ids_arr), \
    "Array length mismatch!"
print(f"  All arrays length   : {len(soh_arr)}  -- aligned OK")
print(f"  curves_arr shape    : {curves_arr.shape}")
print(f"  soh_arr shape       : {soh_arr.shape}")
print(f"  temp_arr shape      : {temp_arr.shape}")
print(f"  ids_arr shape       : {ids_arr.shape}")
print(f"  V range             : {curves_arr.min():.4f} - "
      f"{curves_arr.max():.4f} V")
print(f"  SOH range           : {soh_arr.min():.4f} - {soh_arr.max():.4f}")
print(f"  Temp range          : {temp_arr.min():.1f} - "
      f"{temp_arr.max():.1f} C")
print(f"  Unique cell IDs     : {np.unique(ids_arr).tolist()}")
print(f"  NaNs in curves      : {np.isnan(curves_arr).sum()}")
print(f"  NaNs in soh         : {np.isnan(soh_arr).sum()}")
print(f"  NaNs in temp        : {np.isnan(temp_arr).sum()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: SAVE
# ══════════════════════════════════════════════════════════════════════════
np.save(os.path.join(OUT_DIR, "calce_curves.npy"),      curves_arr)
np.save(os.path.join(OUT_DIR, "calce_soh.npy"),         soh_arr)
np.save(os.path.join(OUT_DIR, "calce_temperature.npy"), temp_arr)
np.save(os.path.join(OUT_DIR, "calce_cell_ids.npy"),    ids_arr)
index_df.to_csv(os.path.join(OUT_DIR, "calce_index.csv"), index=False)

print(f"\nSaved -> calce_curves.npy       {curves_arr.shape}")
print(f"Saved -> calce_soh.npy          {soh_arr.shape}")
print(f"Saved -> calce_temperature.npy  {temp_arr.shape}")
print(f"Saved -> calce_cell_ids.npy     {ids_arr.shape}")
print(f"Saved -> calce_index.csv        {index_df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: PLOTS
# ══════════════════════════════════════════════════════════════════════════
cell_colors = {
    "CS2_33": "purple",
    "CS2_34": "brown",
    "CS2_35": "steelblue",
    "CS2_36": "darkorange",
    "CS2_37": "green",
    "CS2_38": "crimson",
}

# ── Plot A: SOH degradation trajectories ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for cell_id, grp in index_df.groupby("cell_id"):
    grp   = grp.sort_values("charge_num")
    color = cell_colors[cell_id]
    axes[0].plot(grp["charge_num"], grp["soh_raw"],
                 lw=0.5, color=color, alpha=0.35)
    axes[0].plot(grp["charge_num"], grp["soh_smooth"],
                 lw=2.0, color=color,
                 label=f"{cell_id} (n={len(grp)}, "
                       f"min={grp['soh_smooth'].min():.3f})")

axes[0].axhline(0.80, color="red",   lw=1.5, linestyle="--", label="EOL 80%")
axes[0].axhline(0.75, color="black", lw=0.8, linestyle=":",  label="Filter 75%")
axes[0].axhline(1.00, color="black", lw=0.8, linestyle="--")
axes[0].set_xlabel("Cycle number")
axes[0].set_ylabel("SOH  (Q / Q_max,  normalised 0-1)")
axes[0].set_title("CALCE CS2 -- SOH Degradation\n"
                  "(faint=raw, solid=Butterworth smoothed)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)
axes[0].set_ylim(0.70, 1.05)

# Normalised lifecycle
common_x   = np.linspace(0, 1, 300)
norm_trajs = []
for cell_id, grp in index_df.groupby("cell_id"):
    grp    = grp.sort_values("charge_num")
    x_norm = (grp["charge_num"].values - 1) / (grp["charge_num"].max() - 1)
    f      = np.interp(common_x, x_norm, grp["soh_smooth"].values)
    norm_trajs.append(f)
    axes[1].plot(common_x, f, lw=1.5, alpha=0.7,
                 color=cell_colors[cell_id], label=cell_id)

mean_traj = np.mean(norm_trajs, axis=0)
std_traj  = np.std(norm_trajs,  axis=0)
axes[1].plot(common_x, mean_traj, "k-", lw=2.5, label="Mean", zorder=5)
axes[1].fill_between(common_x,
                     mean_traj - std_traj,
                     mean_traj + std_traj,
                     alpha=0.15, color="black", label="+/-1 std")
axes[1].axhline(0.80, color="red",   lw=1.5, linestyle="--", label="EOL 80%")
axes[1].axhline(0.75, color="black", lw=0.8, linestyle=":")
axes[1].axhline(1.00, color="black", lw=0.8, linestyle="--")
axes[1].set_xlabel("Normalised lifecycle  (0=start, 1=end of test)")
axes[1].set_ylabel("SOH  (normalised 0-1)")
axes[1].set_title("CALCE CS2 -- Normalised Lifecycle\n"
                  "(cells aligned to same 0-1 axis)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)
axes[1].set_ylim(0.70, 1.05)

plt.suptitle(f"CALCE CS2 -- Final SOH Labels\n"
             f"(Butterworth order={BW_ORDER}, cutoff={BW_CUTOFF}, "
             f"SOH >= {SOH_MIN})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calce_soh_final.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> calce_soh_final.png")

# ── Plot B: charge curves overview ────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
axes = axes.flatten()
x_axis = np.linspace(0, 1, N_POINTS)
cmap   = plt.cm.RdYlGn

for plot_idx, (cell_id, grp) in enumerate(index_df.groupby("cell_id")):
    ax       = axes[plot_idx]
    soh_vals = grp["soh"].values
    norm_c   = plt.Normalize(soh_vals.min(), soh_vals.max())
    c_idx    = index_df[index_df["cell_id"] == cell_id].index.tolist()

    for i, idx in enumerate(c_idx):
        ax.plot(x_axis, curves_arr[idx],
                color=cmap(norm_c(soh_vals[i])), alpha=0.3, lw=0.5)

    n     = len(c_idx)
    early = curves_arr[c_idx[:n//4]].mean(0)
    late  = curves_arr[c_idx[-n//4:]].mean(0)
    ax.plot(x_axis, early, "g-", lw=2.5,
            label=f"Early (SOH~{soh_vals[:n//4].mean():.3f})")
    ax.plot(x_axis, late,  "r-", lw=2.5,
            label=f"Late  (SOH~{soh_vals[-n//4:].mean():.3f})")

    ax.set_title(f"{cell_id}\nn={n}  "
                 f"SOH: {soh_vals.min():.3f}-{soh_vals.max():.3f}",
                 fontsize=9)
    ax.set_xlabel("<- End of charge    Start ->", fontsize=8)
    ax.set_ylabel("Voltage (V)", fontsize=8)
    ax.legend(fontsize=7); ax.grid(True)

plt.suptitle(f"CALCE CS2 -- Charge Curves per Cell  (SOH >= {SOH_MIN})\n"
             "(green=early healthy, red=late degraded, colored by SOH)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "calce_soh_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> calce_soh_overview.png")

print("\nDone.")