"""
04_combine_source.py
====================
Combines all source domain cell datasets into one unified dataset
for autoencoder and LSTM pretraining.

Source datasets:
    NASA Group A  : 4 cells  (B0005, B0006, B0007, B0018)
    CALCE CS2     : 6 cells  (CS2_33, CS2_34, CS2_35, CS2_36, CS2_37, CS2_38)
    LCO LiPo      : 1 cell   (lco_cell_0)
    Total         : 11 cells

Directory structure (run from COMBINED/ directory):
    ../cell_lco_nasa_4x/processed/    -- NASA input files
    ../cell_lco_calce_6x/processed/   -- CALCE input files
    ../cell_lco_lipo_1x/processed/    -- LCO input files
    processed/                        -- combined output files (created here)

Writes (to COMBINED/processed/):
    source_curves.npy        (N_total, 101) float32
    source_soh.npy           (N_total,)     float32
    source_temperature.npy   (N_total,)     float32
    source_cell_ids.npy      (N_total,)     str
    source_index.csv         (N_total, *)
    source_overview.png

Every row i across all 4 arrays corresponds to the same cycle.

Framework notes (Fan et al. 2025):
    Stage 1 -- Autoencoder pretraining:
        Input  : source_curves.npy  (all cycles, order irrelevant)
        Target : reconstruct input (unsupervised)
        Output : 24-dim latent encoding per cycle

    Stage 2 -- LSTM pretraining:
        Input  : [latent(24) | T_mean(1) | zeros(13)] = 38-dim per cycle
        Cycles grouped as per-cell temporal sequences using cell_ids
        Target : SOH per cycle

Run from COMBINED/ directory:
    python 04_combine_source.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════

# Input directories -- relative to COMBINED/ directory
NASA_DIR  = "../cell_lco_nasa_4x/processed"
CALCE_DIR = "../cell_lco_calce_6x/processed"
LCO_DIR   = "../cell_lco_lipo_1x/processed"

# Output directory -- inside COMBINED/
OUT_DIR   = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Maps dataset name -> (input directory, file prefix)
DATASETS = {
    "NASA" : (NASA_DIR,  "nasa"),
    "CALCE": (CALCE_DIR, "calce"),
    "LCO"  : (LCO_DIR,   "lco")
}

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Source Domain -- Combining All Datasets")
print("=" * 60)

loaded = {}

for name, (data_dir, prefix) in DATASETS.items():
    print(f"\n  Loading {name} from {data_dir} ...")

    curves = np.load(os.path.join(data_dir, f"{prefix}_curves.npy"))
    soh    = np.load(os.path.join(data_dir, f"{prefix}_soh.npy"))
    temp   = np.load(os.path.join(data_dir, f"{prefix}_temperature.npy"))
    ids    = np.load(os.path.join(data_dir, f"{prefix}_cell_ids.npy"),
                     allow_pickle=True)
    index  = pd.read_csv(os.path.join(data_dir, f"{prefix}_index.csv"))

    loaded[name] = {
        "curves": curves,
        "soh"   : soh,
        "temp"  : temp,
        "ids"   : ids,
        "index" : index,
    }

    n_cells = len(np.unique(ids))
    print(f"    curves shape : {curves.shape}")
    print(f"    SOH range    : {soh.min():.4f} - {soh.max():.4f}")
    print(f"    Temp range   : {temp.min():.1f} - {temp.max():.1f} C")
    print(f"    Cells ({n_cells}): {np.unique(ids).tolist()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: VALIDATE CONSISTENCY ACROSS DATASETS
# ══════════════════════════════════════════════════════════════════════════
print(f"\nValidating consistency ...")

N_POINTS = 101
for name, d in loaded.items():
    assert d["curves"].shape[1] == N_POINTS, \
        f"{name}: curve length {d['curves'].shape[1]} != {N_POINTS}"
    assert len(d["curves"]) == len(d["soh"]) == \
           len(d["temp"])   == len(d["ids"]), \
        f"{name}: array length mismatch"
    assert d["curves"].dtype == np.float32, \
        f"{name}: curves dtype {d['curves'].dtype} != float32"
    assert d["soh"].dtype == np.float32, \
        f"{name}: soh dtype {d['soh'].dtype} != float32"
    assert d["temp"].dtype == np.float32, \
        f"{name}: temp dtype {d['temp'].dtype} != float32"
    print(f"  {name}: OK")

# Check no cell_id overlap across datasets
all_ids_flat = []
for name, d in loaded.items():
    all_ids_flat.extend(np.unique(d["ids"]).tolist())
assert len(all_ids_flat) == len(set(all_ids_flat)), \
    "Duplicate cell IDs across datasets!"
print(f"  No duplicate cell IDs: OK")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: CONCATENATE
# ══════════════════════════════════════════════════════════════════════════
print(f"\nConcatenating ...")

source_curves = np.concatenate(
    [d["curves"] for d in loaded.values()], axis=0)
source_soh    = np.concatenate(
    [d["soh"]    for d in loaded.values()], axis=0)
source_temp   = np.concatenate(
    [d["temp"]   for d in loaded.values()], axis=0)
source_ids    = np.concatenate(
    [d["ids"]    for d in loaded.values()], axis=0)

# Build combined index with dataset label
index_parts = []
for name, d in loaded.items():
    idx = d["index"].copy()
    idx["dataset"] = name
    index_parts.append(idx)
source_index = pd.concat(index_parts, ignore_index=True)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print(f"\nValidation:")
assert len(source_curves) == len(source_soh) == \
       len(source_temp)   == len(source_ids), \
    "Combined array length mismatch!"
print(f"  All arrays length   : {len(source_soh)}  -- aligned OK")
print(f"  source_curves shape : {source_curves.shape}")
print(f"  source_soh shape    : {source_soh.shape}")
print(f"  source_temp shape   : {source_temp.shape}")
print(f"  source_ids shape    : {source_ids.shape}")
print(f"  V range             : {source_curves.min():.4f} - "
      f"{source_curves.max():.4f} V")
print(f"  SOH range           : {source_soh.min():.4f} - "
      f"{source_soh.max():.4f}")
print(f"  Temp range          : {source_temp.min():.1f} - "
      f"{source_temp.max():.1f} C")
print(f"  Total unique cells  : {len(np.unique(source_ids))}")
print(f"  NaNs in curves      : {np.isnan(source_curves).sum()}")
print(f"  NaNs in soh         : {np.isnan(source_soh).sum()}")
print(f"  NaNs in temp        : {np.isnan(source_temp).sum()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: PER-CELL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print(f"\nPer-cell summary:")
print(f"  {'Cell':12s}  {'Dataset':6s}  {'N':>5}  "
      f"{'SOH min':>8}  {'SOH max':>8}  {'T_mean':>8}")
print(f"  {'-'*62}")

for dataset_name, d in loaded.items():
    for cell_id in np.unique(d["ids"]):
        mask = source_ids == cell_id
        n    = mask.sum()
        print(f"  {cell_id:12s}  {dataset_name:6s}  {n:>5}  "
              f"{source_soh[mask].min():>8.4f}  "
              f"{source_soh[mask].max():>8.4f}  "
              f"{source_temp[mask].mean():>7.1f}C")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE
# ══════════════════════════════════════════════════════════════════════════
np.save(os.path.join(OUT_DIR, "source_curves.npy"),      source_curves)
np.save(os.path.join(OUT_DIR, "source_soh.npy"),         source_soh)
np.save(os.path.join(OUT_DIR, "source_temperature.npy"), source_temp)
np.save(os.path.join(OUT_DIR, "source_cell_ids.npy"),    source_ids)
source_index.to_csv(os.path.join(OUT_DIR, "source_index.csv"), index=False)

print(f"\nSaved -> source_curves.npy        {source_curves.shape}")
print(f"Saved -> source_soh.npy           {source_soh.shape}")
print(f"Saved -> source_temperature.npy   {source_temp.shape}")
print(f"Saved -> source_cell_ids.npy      {source_ids.shape}")
print(f"Saved -> source_index.csv         {source_index.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: PLOTS
# ══════════════════════════════════════════════════════════════════════════
dataset_colors = {
    "NASA" : "steelblue",
    "CALCE": "darkorange",
    "LCO"  : "green",
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
x_axis = np.linspace(0, 1, N_POINTS)
cmap   = plt.cm.RdYlGn

# ── Plot 1: SOH distribution per dataset ──────────────────────────────────
for name, d in loaded.items():
    axes[0].hist(d["soh"], bins=40, alpha=0.6,
                 color=dataset_colors[name],
                 label=f"{name} (n={len(d['soh'])})")
axes[0].axvline(0.75, color="red",  lw=1.5, linestyle="--",
                label="SOH = 0.75")
axes[0].axvline(0.80, color="gray", lw=1.0, linestyle=":",
                label="SOH = 0.80")
axes[0].set_xlabel("SOH")
axes[0].set_ylabel("Count")
axes[0].set_title("SOH Distribution per Dataset")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)

# ── Plot 2: All charge curves colored by SOH ──────────────────────────────
norm_c = plt.Normalize(source_soh.min(), source_soh.max())
for i in range(0, len(source_curves), 5):  # every 5th for speed
    axes[1].plot(x_axis, source_curves[i],
                 color=cmap(norm_c(source_soh[i])), alpha=0.15, lw=0.5)
q4_idx = np.where(source_soh >= np.percentile(source_soh, 75))[0]
q1_idx = np.where(source_soh <= np.percentile(source_soh, 25))[0]
axes[1].plot(x_axis, source_curves[q4_idx].mean(0), "g-", lw=2.5,
             label=f"High SOH Q4 (mean={source_soh[q4_idx].mean():.3f})")
axes[1].plot(x_axis, source_curves[q1_idx].mean(0), "r-", lw=2.5,
             label=f"Low SOH  Q1 (mean={source_soh[q1_idx].mean():.3f})")
plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm_c),
             ax=axes[1], label="SOH")
axes[1].set_xlabel("<- End of charge    Start ->")
axes[1].set_ylabel("Voltage (V)")
axes[1].set_title(f"All Source Curves (n={len(source_curves)})\n"
                  f"colored by SOH")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

# ── Plot 3: Temperature distribution ──────────────────────────────────────
for name, d in loaded.items():
    axes[2].hist(d["temp"], bins=30, alpha=0.6,
                 color=dataset_colors[name],
                 label=f"{name} mean={d['temp'].mean():.1f}C")
axes[2].set_xlabel("T_mean (C)")
axes[2].set_ylabel("Count")
axes[2].set_title("Temperature Feature Distribution\n"
                  "(manual feature [0] during pretraining)")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.4)

plt.suptitle(f"Source Domain -- Combined Dataset\n"
             f"NASA (4 cells) + CALCE (6 cells) + LCO (1 cell) = "
             f"11 cells, {len(source_curves)} cycles",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "source_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> source_overview.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SOURCE DOMAIN SUMMARY")
print("=" * 60)
print(f"  Total cycles        : {len(source_soh)}")
print(f"  Total cells         : {len(np.unique(source_ids))}")
print(f"  SOH range           : {source_soh.min():.4f} - "
      f"{source_soh.max():.4f}")
print(f"  V range             : {source_curves.min():.4f} - "
      f"{source_curves.max():.4f} V")
print(f"\n  Per dataset:")
for name, d in loaded.items():
    n_cells = len(np.unique(d["ids"]))
    print(f"    {name:6s}: {len(d['soh']):5d} cycles  "
          f"{n_cells} cell(s)  "
          f"SOH {d['soh'].min():.3f}-{d['soh'].max():.3f}  "
          f"T {d['temp'].min():.1f}-{d['temp'].max():.1f}C")

print(f"\n  Framework usage (Fan et al. 2025):")
print(f"    Stage 1 -- Autoencoder:")
print(f"      Input  : source_curves.npy  {source_curves.shape}")
print(f"      Output : 24-dim latent per cycle")
print(f"    Stage 2 -- LSTM pretraining:")
print(f"      Input  : [latent(24) | T_mean(1) | zeros(13)] = 38-dim")
print(f"      Cycles grouped per cell in temporal order via cell_ids")
print(f"      Target : source_soh.npy  {source_soh.shape}")
print(f"\nDone.")