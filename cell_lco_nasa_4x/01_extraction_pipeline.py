"""
01_extract_nasa.py
==================
Complete extraction pipeline for NASA battery dataset (Group A only).

Dataset:
    4 LCO cylindrical cells (B0005, B0006, B0007, B0018)
    CC-CV charge at 1.5A to 4.2V, discharge at 2A CC
    Room temperature (~24 degrees C)
    ~170 cycles per cell
    Group B (B0029-B0032) excluded: only 40 cycles, <13% SOH degradation

Reads  : data/B0005.mat
         data/B0006.mat
         data/B0007.mat
         data/B0018.mat

Outputs (saved in processed/):
    nasa_curves.npy          (N, 101) float32  -- charge voltage curves
    nasa_soh.npy             (N,)     float32  -- SOH labels 0-1
    nasa_temperature.npy     (N,)     float32  -- mean charge temperature
    nasa_cell_ids.npy        (N,)     str      -- cell identifier per sample
    nasa_index.csv           (N, 10)           -- human-readable reference

Every row i across all 4 arrays corresponds to the same cycle.

Notes:
    - SOH = Q_cycle / Q_max per cell, normalised to 0-1
    - Butterworth smoothing (order=3, cutoff=0.10) applied to SOH labels
    - Outlier curves removed: RMSE vs median > mean + 3*std
    - Charge curves end-aligned on time axis (CC phase only)
    - Resampled to 101 points with Gaussian smoothing sigma=2
    - Temperature measured directly from Temperature_measured field

Run from source domain directory:
    python 01_extract_nasa.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage    import gaussian_filter1d
from scipy.signal     import butter, filtfilt

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR  = "data"
OUT_DIR   = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS      = 101      # autoencoder input length
SIGMA         = 2        # Gaussian smoothing sigma for charge curves
BW_CUTOFF     = 0.03     # Butterworth SOH filter cutoff frequency
BW_ORDER      = 3        # Butterworth filter order
OUTLIER_SIGMA = 3.0      # outlier threshold: mean + N*std of RMSE

# Group A only
CELLS = {
    "B0005": {"group": "NASA_A", "temp": 24, "cutoff_v": 2.7, "dis_i": 2.0},
    "B0006": {"group": "NASA_A", "temp": 24, "cutoff_v": 2.5, "dis_i": 2.0},
    "B0007": {"group": "NASA_A", "temp": 24, "cutoff_v": 2.2, "dis_i": 2.0},
    "B0018": {"group": "NASA_A", "temp": 24, "cutoff_v": 2.5, "dis_i": 2.0},
}

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def butterworth_smooth(data, cutoff=BW_CUTOFF, order=BW_ORDER):
    """Zero-phase Butterworth low-pass filter, output capped at [0, 1]."""
    b, a = butter(order, cutoff / 0.5, btype="low", analog=False)
    return np.clip(filtfilt(b, a, data), 0.0, 1.0)


def load_mat(cell_id):
    """Load NASA .mat file and return cycle array."""
    fpath = os.path.join(DATA_DIR, f"{cell_id}.mat")
    mat   = sio.loadmat(fpath, simplify_cells=True)
    return mat[cell_id]["cycle"]


def extract_capacity(data):
    """Extract scalar discharge capacity (Ah) from discharge cycle data."""
    cap = np.array(data["Capacity"])
    return float(cap) if cap.ndim == 0 else float(cap.max())


def extract_charge_curve(data):
    """
    Extract end-aligned charge voltage curve from a charge cycle data dict.

    Steps:
      1. Filter to CC phase: Current_measured > 0.05A
      2. Reverse time axis so index 0 = end of charge (fully charged)
      3. Resample to N_POINTS using linear interpolation
      4. Apply Gaussian smoothing (sigma=2) to remove artifacts

    Returns (curve float32, t_mean float) or (None, None).
    """
    v    = np.array(data["Voltage_measured"],     dtype=float)
    i    = np.array(data["Current_measured"],     dtype=float)
    t    = np.array(data["Temperature_measured"], dtype=float)
    time = np.array(data["Time"],                 dtype=float)

    # CC phase only
    cc_mask = i > 0.05
    if cc_mask.sum() < 10:
        return None, None

    v_cc    = v[cc_mask]
    t_cc    = t[cc_mask]
    time_cc = time[cc_mask]

    if v_cc.min() < 2.5 or v_cc.max() > 4.5:
        return None, None

    # End-align: reverse so index 0 = end of charge
    time_rev = (time_cc[-1] - time_cc)[::-1]
    v_rev    = v_cc[::-1]

    _, uid          = np.unique(time_rev, return_index=True)
    time_rev, v_rev = time_rev[uid], v_rev[uid]

    if len(time_rev) < 4:
        return None, None

    f    = interp1d(time_rev, v_rev, kind="linear",
                    bounds_error=False,
                    fill_value=(v_rev[0], v_rev[-1]))
    y_rs = f(np.linspace(0, time_rev.max(), N_POINTS))
    y_rs = gaussian_filter1d(y_rs, sigma=SIGMA)

    return y_rs.astype(np.float32), float(t_cc.mean())


# ══════════════════════════════════════════════════════════════════════════
# STEP 1-4: EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("NASA Battery Dataset -- Group A Extraction")
print("=" * 60)

all_curves  = []
all_records = []

for cell_id, meta in CELLS.items():
    fpath = os.path.join(DATA_DIR, f"{cell_id}.mat")
    if not os.path.exists(fpath):
        print(f"\n{cell_id} -- FILE NOT FOUND: {fpath}")
        continue

    print(f"\n{cell_id}  (Group {meta['group']}, "
          f"T={meta['temp']}C, "
          f"dis={meta['dis_i']}A, "
          f"cutoff={meta['cutoff_v']}V)")

    cycles = load_mat(cell_id)

    # Build discharge capacity map
    cap_map = {}
    for idx, cyc in enumerate(cycles):
        if isinstance(cyc, dict) and cyc.get("type") == "discharge":
            data = cyc["data"]
            if "Capacity" in data:
                cap_map[idx] = extract_capacity(data)

    if not cap_map:
        print(f"  No discharge capacity found -- skipping")
        continue

    q_max = max(cap_map.values())
    print(f"  Discharge cycles with capacity : {len(cap_map)}")
    print(f"  Q_max = {q_max:.4f} Ah")
    print(f"  SOH range (raw): "
          f"{min(cap_map.values())/q_max:.4f} - "
          f"{max(cap_map.values())/q_max:.4f}")

    discharge_indices = [i for i, c in enumerate(cycles)
                         if isinstance(c, dict)
                         and c.get("type") == "discharge"
                         and i in cap_map]

    charge_num = 0

    for idx, cyc in enumerate(cycles):
        if not isinstance(cyc, dict) or cyc.get("type") != "charge":
            continue

        next_dis = [d for d in discharge_indices if d > idx]
        if not next_dis:
            continue
        dis_idx = next_dis[0]
        soh     = cap_map[dis_idx] / q_max

        curve, t_mean = extract_charge_curve(cyc["data"])
        if curve is None:
            continue

        charge_num += 1
        all_curves.append(curve)
        all_records.append({
            "cell_id"   : cell_id,
            "group"     : meta["group"],
            "charge_num": charge_num,
            "chg_idx"   : idx,
            "dis_idx"   : dis_idx,
            "soh"       : soh,
            "q_ah"      : cap_map[dis_idx],
            "q_max"     : q_max,
            "t_mean"    : t_mean,
        })

    print(f"  Extracted charge curves : {charge_num}")
    if charge_num > 0:
        cell_soh = [r["soh"] for r in all_records
                    if r["cell_id"] == cell_id]
        print(f"  SOH range (extracted)   : "
              f"{min(cell_soh):.4f} - {max(cell_soh):.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: ASSEMBLE
# ══════════════════════════════════════════════════════════════════════════
curves_arr = np.array(all_curves, dtype=np.float32)
index_df   = pd.DataFrame(all_records)

print(f"\n{'='*60}")
print(f"BEFORE FILTERING/SMOOTHING")
print(f"{'='*60}")
print(f"  Total samples : {len(index_df)}")
print(f"  SOH range     : {index_df['soh'].min():.4f} - "
      f"{index_df['soh'].max():.4f}")
for cell_id, grp in index_df.groupby("cell_id"):
    print(f"    {cell_id}: {len(grp):3d} cycles  "
          f"SOH {grp['soh'].min():.4f}-{grp['soh'].max():.4f}  "
          f"T_mean {grp['t_mean'].mean():.1f}C")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5b: FILTER post-EOL cycles below SOH_MIN
# ══════════════════════════════════════════════════════════════════════════
SOH_MIN = 0.75
print(f"\nFiltering to SOH >= {SOH_MIN} ...")
keep_soh   = index_df["soh"] >= SOH_MIN
n_removed  = (~keep_soh).sum()
index_df   = index_df[keep_soh].reset_index(drop=True)
curves_arr = curves_arr[keep_soh.values]
print(f"  Removed  : {n_removed} post-EOL cycles")
print(f"  Retained : {len(index_df)} cycles")
for cell_id, grp in index_df.groupby("cell_id"):
    print(f"    {cell_id}: {len(grp):3d} cycles  "
          f"SOH {grp['soh'].min():.4f}-{grp['soh'].max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: OUTLIER CURVE REMOVAL
# ══════════════════════════════════════════════════════════════════════════
print(f"\nRemoving outlier curves "
      f"(threshold = mean + {OUTLIER_SIGMA}*std) ...")

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
print(f"  Outliers removed : {outlier_mask.sum()}  {outlier_info}")

keep_mask  = ~outlier_mask
curves_arr = curves_arr[keep_mask]
index_df   = index_df[keep_mask].reset_index(drop=True)
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
    print(f"    {cell_id}: {len(grp):3d} cycles  "
          f"raw {grp['soh_raw'].min():.4f}-{grp['soh_raw'].max():.4f}  "
          f"smooth {grp['soh_smooth'].min():.4f}-"
          f"{grp['soh_smooth'].max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: BUILD FINAL 4 ARRAYS
# ══════════════════════════════════════════════════════════════════════════
soh_arr  = index_df["soh"].values.astype(np.float32)
temp_arr = index_df["t_mean"].values.astype(np.float32)
ids_arr  = index_df["cell_id"].values

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
np.save(os.path.join(OUT_DIR, "nasa_curves.npy"),      curves_arr)
np.save(os.path.join(OUT_DIR, "nasa_soh.npy"),         soh_arr)
np.save(os.path.join(OUT_DIR, "nasa_temperature.npy"), temp_arr)
np.save(os.path.join(OUT_DIR, "nasa_cell_ids.npy"),    ids_arr)
index_df.to_csv(os.path.join(OUT_DIR, "nasa_index.csv"), index=False)

print(f"\nSaved -> nasa_curves.npy        {curves_arr.shape}")
print(f"Saved -> nasa_soh.npy           {soh_arr.shape}")
print(f"Saved -> nasa_temperature.npy   {temp_arr.shape}")
print(f"Saved -> nasa_cell_ids.npy      {ids_arr.shape}")
print(f"Saved -> nasa_index.csv         {index_df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: PLOTS
# ══════════════════════════════════════════════════════════════════════════
cell_colors = {
    "B0005": "steelblue",
    "B0006": "darkorange",
    "B0007": "green",
    "B0018": "crimson",
}

# ── Plot A: SOH degradation ────────────────────────────────────────────────
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
axes[0].axhline(1.00, color="black", lw=0.8, linestyle="--")
axes[0].set_xlabel("Cycle number")
axes[0].set_ylabel("SOH  (Q / Q_max,  normalised 0-1)")
axes[0].set_title("NASA Group A -- SOH Degradation\n"
                  "(faint=raw, solid=Butterworth smoothed)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)
axes[0].set_ylim(0.50, 1.05)

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
axes[1].axhline(1.00, color="black", lw=0.8, linestyle="--")
axes[1].set_xlabel("Normalised lifecycle  (0=start, 1=end of test)")
axes[1].set_ylabel("SOH  (normalised 0-1)")
axes[1].set_title("NASA Group A -- Normalised Lifecycle\n"
                  "(cells aligned to same 0-1 axis)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)
axes[1].set_ylim(0.50, 1.05)

plt.suptitle(f"NASA Battery Dataset Group A -- Final SOH Labels\n"
             f"(Butterworth order={BW_ORDER}, cutoff={BW_CUTOFF})",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nasa_soh_final.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> nasa_soh_final.png")

# ── Plot B: charge curves per cell ────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
x_axis = np.linspace(0, 1, N_POINTS)
cmap   = plt.cm.RdYlGn

for plot_idx, (cell_id, grp) in enumerate(index_df.groupby("cell_id")):
    ax       = axes[plot_idx]
    soh_vals = grp["soh"].values
    norm_c   = plt.Normalize(soh_vals.min(), soh_vals.max())
    c_idx    = index_df[index_df["cell_id"] == cell_id].index.tolist()

    for i, idx in enumerate(c_idx):
        ax.plot(x_axis, curves_arr[idx],
                color=cmap(norm_c(soh_vals[i])), alpha=0.3, lw=0.6)

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

plt.suptitle("NASA Group A -- Charge Curves per Cell\n"
             "(green=early healthy, red=late degraded, colored by SOH)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "nasa_soh_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> nasa_soh_overview.png")

print("\nDone.")