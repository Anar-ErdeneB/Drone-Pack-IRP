"""
03_extract_lco.py
=================
Complete extraction pipeline for LCO LiPo large-format cell dataset.

Dataset:
    1 LCO LiPo large-format cell (22 Ah nominal)
    Same chemistry as drone target domain (LCO LiPo)
    RPT-based cycling -- only 46 measurement points
    Temperature ~20-21 degrees C

Reads  : data/Aging_Dataset_Cycling.mat

Outputs (saved in processed/):
    lco_curves.npy          (N, 101) float32  -- charge voltage curves
    lco_soh.npy             (N,)     float32  -- SOH labels 0-1
    lco_temperature.npy     (N,)     float32  -- mean charge temperature
    lco_cell_ids.npy        (N,)     str      -- all "lco_cell_0"
    lco_index.csv           (N, 6)            -- human-readable reference

Every row i across all 4 arrays corresponds to the same cycle.

Notes:
    - SOH = discharge capacity / nominal capacity (22 Ah), normalised 0-1
    - Cycle 750 dip fixed via linear interpolation (measurement artifact)
    - Charge curves end-aligned on capacity axis (index 0 = end of charge)
    - Resampled to 101 points -- consistent with NASA and CALCE pipelines
    - Only 46 RPT cycles (not every cycle) -- sparse but same chemistry as drone
    - SOH range: 0.824 - 0.993 (pre-EOL only, no filter needed)

Run from LCO cell directory:
    python 03_extract_lco.py
"""

import os
import numpy as np
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DATA_PATH   = os.path.join("data", "Aging_Dataset_Cycling.mat")
OUT_DIR     = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

N_POINTS     = 101       # autoencoder input length -- must match other datasets
NOMINAL_CAP  = 22.0      # Ah -- rated capacity of this cell
CELL_ID_STR  = "lco_cell_0"
MAX_CYCLE    = 1125      # exclude cycles beyond this

# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
def is_valid(struct):
    """Check if a matlab struct field contains real data."""
    if struct is None:
        return False
    if isinstance(struct, (int, float, np.integer, np.floating)):
        return False
    if isinstance(struct, np.ndarray) and struct.size == 0:
        return False
    return True


def get_field(struct, field):
    """Extract a field from a matlab struct as a flat numpy array."""
    if isinstance(struct, dict):
        return np.array(struct[field]).flatten()
    else:
        return np.array(getattr(struct, field)).flatten()


def compute_capacity_ah(struct, current_field):
    """Compute total capacity (Ah) by integrating current over time."""
    t  = get_field(struct, "Time")
    i  = np.abs(get_field(struct, current_field))
    dt = np.diff(t, prepend=t[0])
    return np.sum(i * dt) / 3600.0


def resample_end_aligned(v, q, n_points=101):
    """
    Resample voltage curve to n_points aligned at end of charge.
    Following Fan et al. Fig 6b -- reverse capacity axis so all
    curves share the same endpoint (end of charge = index 0).

    This is consistent with NASA and CALCE pipelines which also
    use end-alignment (index 0 = end of charge).
    """
    q_from_end = q[-1] - q
    q_max      = q_from_end[0]
    if q_max <= 0:
        return None
    q_norm_rev = q_from_end / q_max
    q_flipped  = q_norm_rev[::-1]
    v_flipped  = v[::-1]
    q_grid     = np.linspace(0, 1, n_points)
    return np.interp(q_grid, q_flipped, v_flipped)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LCO LiPo Cell -- Extraction Pipeline")
print("=" * 60)
print(f"\nLoading {DATA_PATH} ...")

cycling = scipy.io.loadmat(DATA_PATH, simplify_cells=True)
D = cycling["D"]

rpt_a_cycles   = D[0, 5]
discharge_rpta = D[0, 0]
charge_rpta    = D[0, 1]

print(f"  RPT cycle indices available : {len(rpt_a_cycles)}")
print(f"  Discharge repeats shape     : {discharge_rpta.shape}")
print(f"  Charge repeats shape        : {charge_rpta.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: EXTRACT DISCHARGE CAPACITIES -> SOH LABELS
# ══════════════════════════════════════════════════════════════════════════
print(f"\nExtracting discharge capacities ...")

cyc_all = []
cap_dis = []

for idx in range(len(rpt_a_cycles)):
    cyc = rpt_a_cycles[idx]
    if cyc >= MAX_CYCLE:
        continue
    caps = []
    for j in range(discharge_rpta.shape[1]):
        s = discharge_rpta[idx, j]
        if is_valid(s):
            try:
                cap = compute_capacity_ah(s, "I_EL")
                if cap > 10.0:
                    caps.append(cap)
            except Exception:
                pass
    if caps:
        cyc_all.append(cyc)
        cap_dis.append(np.median(caps))

cyc_all = np.array(cyc_all)
cap_dis = np.array(cap_dis)

soh_raw = cap_dis / NOMINAL_CAP
print(f"  Valid RPT cycles  : {len(cyc_all)}")
print(f"  Nominal capacity  : {NOMINAL_CAP} Ah")
print(f"  SOH range (raw)   : {soh_raw.min():.4f} - {soh_raw.max():.4f}")
print(f"  Values above 1.0  : {(soh_raw > 1.0).sum()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: FIX CYCLE 750 MEASUREMENT ARTIFACT
# ══════════════════════════════════════════════════════════════════════════
print(f"\nFixing cycle 750 measurement artifact ...")

if 750 in cyc_all:
    idx_750 = np.where(cyc_all == 750)[0][0]
    idx_725 = np.where(cyc_all == 725)[0][0]
    idx_775 = np.where(cyc_all == 775)[0][0]
    soh_before = soh_raw[idx_750]
    soh_raw[idx_750] = np.interp(
        750,
        [cyc_all[idx_725], cyc_all[idx_775]],
        [soh_raw[idx_725], soh_raw[idx_775]]
    )
    print(f"  Cycle 750: {soh_before:.4f} -> {soh_raw[idx_750]:.4f} (interpolated)")
else:
    print(f"  Cycle 750 not found -- skipping fix")

# Clip to [0, 1]
soh_arr = np.clip(soh_raw, 0.0, 1.0).astype(np.float32)
print(f"  SOH range (final) : {soh_arr.min():.4f} - {soh_arr.max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: EXTRACT CHARGE CURVES + TEMPERATURE
# ══════════════════════════════════════════════════════════════════════════
print(f"\nExtracting charge curves ...")

curves_list  = []
temp_list    = []
valid_cycles = []

for idx in range(len(rpt_a_cycles)):
    cyc = rpt_a_cycles[idx]
    if cyc >= MAX_CYCLE:
        continue

    # Collect valid charge repeats
    candidates = []
    for j in range(charge_rpta.shape[1]):
        s = charge_rpta[idx, j]
        if is_valid(s):
            try:
                cap = compute_capacity_ah(s, "I_PS")
                if cap > 10.0:
                    candidates.append((cap, s))
            except Exception:
                pass

    if not candidates:
        continue

    # Pick repeat closest to median capacity
    caps_list = [c[0] for c in candidates]
    med_cap   = np.median(caps_list)
    best_idx  = np.argmin(np.abs(np.array(caps_list) - med_cap))
    _, best   = candidates[best_idx]

    # Extract signals
    t  = get_field(best, "Time")
    v  = get_field(best, "V_batt")
    i  = get_field(best, "I_PS")
    T  = get_field(best, "T_batt")
    dt = np.diff(t, prepend=t[0])
    q  = np.cumsum(i * dt) / 3600.0

    # End-aligned resample to N_POINTS
    curve = resample_end_aligned(v, q, N_POINTS)
    if curve is None:
        continue

    curves_list.append(curve)
    temp_list.append(float(np.mean(T)))
    valid_cycles.append(cyc)

curves_arr = np.array(curves_list, dtype=np.float32)
temp_arr   = np.array(temp_list,   dtype=np.float32)
valid_cycles = np.array(valid_cycles)

print(f"  Extracted curves  : {len(curves_arr)}")
print(f"  V range           : {curves_arr.min():.4f} - {curves_arr.max():.4f} V")
print(f"  Temp range        : {temp_arr.min():.2f} - {temp_arr.max():.2f} C")
print(f"  End of charge V   : {curves_arr[:, 0].mean():.4f} V (mean)")
print(f"  Start of charge V : {curves_arr[:, -1].mean():.4f} V (mean)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: VERIFY ALIGNMENT
# ══════════════════════════════════════════════════════════════════════════
print(f"\nVerifying alignment ...")
assert np.all(valid_cycles == cyc_all), \
    "Cycle mismatch between SOH labels and charge curves!"
print(f"  SOH cycles    : {len(cyc_all)}")
print(f"  Curve cycles  : {len(valid_cycles)}")
print(f"  Aligned       : OK")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: BUILD FINAL 4 ARRAYS
# ══════════════════════════════════════════════════════════════════════════
ids_arr = np.array([CELL_ID_STR] * len(curves_arr), dtype=object)

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: VALIDATION
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
# STEP 8: SAVE
# ══════════════════════════════════════════════════════════════════════════
index_df = pd.DataFrame({
    "cell_id"   : ids_arr,
    "group"     : "LCO_LiPo",
    "charge_num": range(1, len(valid_cycles) + 1),
    "cycle"     : valid_cycles,
    "soh"       : soh_arr,
    "t_mean"    : temp_arr,
})

np.save(os.path.join(OUT_DIR, "lco_curves.npy"),      curves_arr)
np.save(os.path.join(OUT_DIR, "lco_soh.npy"),         soh_arr)
np.save(os.path.join(OUT_DIR, "lco_temperature.npy"), temp_arr)
np.save(os.path.join(OUT_DIR, "lco_cell_ids.npy"),    ids_arr)
index_df.to_csv(os.path.join(OUT_DIR, "lco_index.csv"), index=False)

print(f"\nSaved -> lco_curves.npy        {curves_arr.shape}")
print(f"Saved -> lco_soh.npy           {soh_arr.shape}")
print(f"Saved -> lco_temperature.npy   {temp_arr.shape}")
print(f"Saved -> lco_cell_ids.npy      {ids_arr.shape}")
print(f"Saved -> lco_index.csv         {index_df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: PLOTS
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
x_axis = np.linspace(0, 1, N_POINTS)
cmap   = plt.cm.RdYlGn
norm_c = plt.Normalize(soh_arr.min(), soh_arr.max())

# SOH trajectory
axes[0].plot(valid_cycles, soh_arr, "o-", color="steelblue",
             markersize=4, lw=1.5)
axes[0].axhline(0.80, color="red",  lw=1.5, linestyle="--", label="EOL 80%")
axes[0].axhline(0.75, color="gray", lw=1.0, linestyle=":",  label="0.75")
axes[0].set_xlabel("Cycle number")
axes[0].set_ylabel("SOH (0-1)")
axes[0].set_title(f"SOH trajectory  ({len(valid_cycles)} RPT cycles)")
axes[0].legend(fontsize=8); axes[0].grid(True)
axes[0].set_ylim(0.75, 1.05)

# Charge curves colored by SOH
for i in range(len(curves_arr)):
    axes[1].plot(x_axis, curves_arr[i],
                 color=cmap(norm_c(soh_arr[i])), alpha=0.5, lw=1.0)
n = len(curves_arr)
axes[1].plot(x_axis, curves_arr[:n//4].mean(0),  "g-", lw=2.5,
             label=f"Early (SOH~{soh_arr[:n//4].mean():.3f})")
axes[1].plot(x_axis, curves_arr[-n//4:].mean(0), "r-", lw=2.5,
             label=f"Late  (SOH~{soh_arr[-n//4:].mean():.3f})")
axes[1].set_xlabel("<- End of charge    Start ->")
axes[1].set_ylabel("Voltage (V)")
axes[1].set_title("Charge curves colored by SOH\n"
                  "(green=early healthy, red=late degraded)")
axes[1].legend(fontsize=8); axes[1].grid(True)

# Temperature
axes[2].plot(valid_cycles, temp_arr, "o-", color="tomato",
             markersize=4, lw=1.5)
axes[2].set_xlabel("Cycle number")
axes[2].set_ylabel("Mean temperature (C)")
axes[2].set_title("Mean battery temperature per RPT cycle")
axes[2].grid(True)

plt.suptitle(f"LCO LiPo Cell -- Final Dataset  "
             f"(SOH {soh_arr.min():.3f}-{soh_arr.max():.3f})\n"
             f"22 Ah nominal, same chemistry as drone target domain",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lco_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> lco_overview.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("FINAL LCO DATASET SUMMARY")
print("=" * 55)
print(f"  Samples (N)         : {len(index_df)}")
print(f"  Cycle range         : {valid_cycles.min()} - {valid_cycles.max()}")
print(f"  SOH range           : {soh_arr.min():.4f} - {soh_arr.max():.4f}")
print(f"  Degradation         : "
      f"{(soh_arr.max() - soh_arr.min())*100:.1f}%")
print(f"  Temperature range   : {temp_arr.min():.1f} - "
      f"{temp_arr.max():.1f} C")
print(f"  Chemistry           : LCO LiPo (same as drone target domain)")
print(f"  Nominal capacity    : {NOMINAL_CAP} Ah")
print(f"\n  Output files:")
print(f"    lco_curves.npy        {curves_arr.shape}  -- autoencoder input")
print(f"    lco_soh.npy           {soh_arr.shape}   -- LSTM label")
print(f"    lco_temperature.npy   {temp_arr.shape}   -- T_mean feature")
print(f"    lco_cell_ids.npy      {ids_arr.shape}   -- cell identifier")
print(f"    lco_index.csv         {index_df.shape}")
print("\nDone.")