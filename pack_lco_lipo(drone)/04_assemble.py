"""
04_assemble_dataset.py
======================
Assembles the final aligned dataset for the drone battery pack
(target domain) and saves in the consistent 4-file structure.

Reads  : soh_labels_drone.csv      (from 01_extract_soh.py)
         drone_charge_curves.npy   (from 02_extract_charge_curves.py)
         drone_curve_index.csv     (from 02_extract_charge_curves.py)
         drone_manual_feats.npy    (from 03_extract_manual_feats.py)
         drone_feat_index.csv      (from 03_extract_manual_feats.py)

Writes : drone_curves.npy          (N, 101) float32  -- charge voltage curves
         drone_soh.npy             (N,)     float32  -- SOH labels 0-1
         drone_temperature.npy     (N,)     float32  -- T_mean per cycle
         drone_cell_ids.npy        (N,)     str      -- all "drone_pack"
         drone_manual_feats.npy    (N, 14)  float32  -- 14-dim pack features
         drone_index.csv           (N, *)            -- human-readable reference

Every row i across all arrays corresponds to the same cycle.

Feature vector layout (14-dim):
    [0]  Vd_std      [1]  Vd_diff     [2]  Vmax_std    [3]  Vmax_mean
    [4]  Vmin_diff   [5]  Td_max      [6]  Td_diff     [7]  Tmax_min
    [8]  Tmin_std    [9]  Tmin_mean   [10] Tmin_min    [11] Tmin_diff
    [12] ir_mean     [13] T_mean

Note:
    During fine-tuning the full 14-dim manual feature vector is used.
    During source pretraining only index [13] (T_mean) is used;
    indices [0-12] are zero-masked (no pack data for source cells).

Run from pack2/ directory:
    python 04_assemble_dataset.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FEAT_NAMES = ["Vd_std","Vd_diff","Vmax_std","Vmax_mean","Vmin_diff",
              "Td_max","Td_diff","Tmax_min","Tmin_std","Tmin_mean",
              "Tmin_min","Tmin_diff","ir_mean","T_mean"]

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD INTERMEDIATE FILES
# ══════════════════════════════════════════════════════════════════════════
print("Loading intermediate files ...")

df_soh   = pd.read_csv("soh_labels_drone.csv")
df_curve = pd.read_csv("drone_curve_index.csv")
df_feat  = pd.read_csv("drone_feat_index.csv")

curves_raw = np.load("drone_charge_curves.npy")   # (N_curve, 101)
feats_raw  = np.load("drone_manual_feats.npy")     # (N_feat,  14)

print(f"  SOH labels     : {len(df_soh)} cycles  "
      f"({df_soh['cycle'].min()}-{df_soh['cycle'].max()})")
print(f"  Charge curves  : {len(df_curve)} cycles  "
      f"({df_curve['cycle'].min()}-{df_curve['cycle'].max()})")
print(f"  Manual feats   : {len(df_feat)} cycles  "
      f"({df_feat['cycle'].min()}-{df_feat['cycle'].max()})")
print(f"  curves_raw shape : {curves_raw.shape}")
print(f"  feats_raw  shape : {feats_raw.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: FIND VALID CYCLES (INTERSECTION)
# ══════════════════════════════════════════════════════════════════════════
df_curve["_row"] = range(len(df_curve))
df_feat["_row"]  = range(len(df_feat))

soh_cycles   = set(df_soh["cycle"].tolist())
curve_cycles = set(df_curve["cycle"].tolist())
feat_cycles  = set(df_feat["cycle"].tolist())

valid_cycles = sorted(soh_cycles & curve_cycles & feat_cycles)

print(f"\n  SOH cycles        : {len(soh_cycles)}")
print(f"  Curve cycles      : {len(curve_cycles)}")
print(f"  Feat cycles       : {len(feat_cycles)}")
print(f"  Valid (intersect) : {len(valid_cycles)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: ASSEMBLE ALIGNED ARRAYS
# ══════════════════════════════════════════════════════════════════════════
print("\nAligning arrays ...")

soh_lookup   = df_soh.set_index("cycle")
curve_lookup = df_curve.set_index("cycle")
feat_lookup  = df_feat.set_index("cycle")

curves_aligned = []
feats_aligned  = []
index_rows     = []

for cycle in valid_cycles:
    soh_row = soh_lookup.loc[cycle]
    soh     = float(soh_row["soh_smooth"])
    soh_raw = float(soh_row["soh_raw"])
    q_ah    = float(soh_row["q_ah"])

    c_row   = curve_lookup.loc[cycle]
    c_idx   = int(c_row["_row"])
    curve   = curves_raw[c_idx]
    tot_mah = float(c_row["total_mah"])
    n_packs = int(c_row["n_packs"])

    f_row   = feat_lookup.loc[cycle]
    f_idx   = int(f_row["_row"])
    feat    = feats_raw[f_idx]

    curves_aligned.append(curve)
    feats_aligned.append(feat)
    index_rows.append({
        "cell_id"  : "drone_pack",
        "group"    : "drone",
        "cycle"    : cycle,
        "soh"      : soh,
        "soh_raw"  : soh_raw,
        "q_ah"     : q_ah,
        "total_mah": tot_mah,
        "n_packs"  : n_packs,
        "t_mean"   : float(feat[13]),
    })

curves_arr = np.array(curves_aligned, dtype=np.float32)
feats_arr  = np.array(feats_aligned,  dtype=np.float32)
index_df   = pd.DataFrame(index_rows)

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD CONSISTENT 4 ARRAYS + MANUAL FEATS
# ══════════════════════════════════════════════════════════════════════════
soh_arr  = index_df["soh"].values.astype(np.float32)
temp_arr = feats_arr[:, 13].astype(np.float32)
ids_arr  = np.array(["drone_pack"] * len(curves_arr), dtype=object)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print("\nValidation:")
assert len(curves_arr) == len(soh_arr) == len(temp_arr) == \
       len(ids_arr) == len(feats_arr), "Array length mismatch!"
print(f"  All arrays length   : {len(soh_arr)}  -- aligned OK")
print(f"  curves_arr shape    : {curves_arr.shape}")
print(f"  soh_arr shape       : {soh_arr.shape}")
print(f"  temp_arr shape      : {temp_arr.shape}")
print(f"  ids_arr shape       : {ids_arr.shape}")
print(f"  feats_arr shape     : {feats_arr.shape}")
print(f"  V range             : {curves_arr.min():.4f} - "
      f"{curves_arr.max():.4f} V")
print(f"  SOH range           : {soh_arr.min():.4f} - {soh_arr.max():.4f}")
print(f"  Temp range          : {temp_arr.min():.1f} - "
      f"{temp_arr.max():.1f} C")
print(f"  Unique cell IDs     : {np.unique(ids_arr).tolist()}")
print(f"  NaNs in curves      : {np.isnan(curves_arr).sum()}")
print(f"  NaNs in soh         : {np.isnan(soh_arr).sum()}")
print(f"  NaNs in temp        : {np.isnan(temp_arr).sum()}")
feat_nans = np.isnan(feats_arr).sum(axis=0)
print(f"  NaNs in feats       : {feat_nans.sum()}")
if feat_nans.sum() > 0:
    for name, cnt in zip(FEAT_NAMES, feat_nans):
        if cnt > 0:
            print(f"    {name}: {cnt} NaNs")
assert list(index_df["cycle"]) == sorted(index_df["cycle"].tolist()), \
    "Cycles not sorted!"
print(f"  Cycles sorted       : OK")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE
# ══════════════════════════════════════════════════════════════════════════
np.save("drone_curves.npy",       curves_arr)
np.save("drone_soh.npy",          soh_arr)
np.save("drone_temperature.npy",  temp_arr)
np.save("drone_cell_ids.npy",     ids_arr)
np.save("drone_manual_feats.npy", feats_arr)
index_df.to_csv("drone_index.csv", index=False)

print(f"\nSaved -> drone_curves.npy        {curves_arr.shape}")
print(f"Saved -> drone_soh.npy           {soh_arr.shape}")
print(f"Saved -> drone_temperature.npy   {temp_arr.shape}")
print(f"Saved -> drone_cell_ids.npy      {ids_arr.shape}")
print(f"Saved -> drone_manual_feats.npy  {feats_arr.shape}")
print(f"Saved -> drone_index.csv         {index_df.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: PLOTS
# ══════════════════════════════════════════════════════════════════════════
cmap   = plt.cm.RdYlGn
norm_c = plt.Normalize(soh_arr.min(), soh_arr.max())
x_axis = np.linspace(0, 1, curves_arr.shape[1])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# SOH trajectory
axes[0,0].plot(index_df["cycle"], soh_arr, "o-", ms=3, lw=1,
               color="steelblue")
axes[0,0].axhline(0.8, color="red", lw=1.5, linestyle="--", label="EOL 80%")
axes[0,0].set_xlabel("Cycle"); axes[0,0].set_ylabel("SOH")
axes[0,0].set_title(f"SOH trajectory  (n={len(index_df)} cycles)")
axes[0,0].legend(); axes[0,0].grid(True)

# Charge curves colored by SOH
for i in range(len(curves_arr)):
    axes[0,1].plot(x_axis, curves_arr[i],
                   color=cmap(norm_c(soh_arr[i])), alpha=0.3, lw=0.6)
q1 = np.where(soh_arr <= np.percentile(soh_arr, 25))[0]
q4 = np.where(soh_arr >= np.percentile(soh_arr, 75))[0]
axes[0,1].plot(x_axis, curves_arr[q1].mean(0), "r-", lw=2.5,
               label=f"Low SOH  (mean={soh_arr[q1].mean():.3f})")
axes[0,1].plot(x_axis, curves_arr[q4].mean(0), "g-", lw=2.5,
               label=f"High SOH (mean={soh_arr[q4].mean():.3f})")
axes[0,1].set_xlabel("<- End of charge    Start ->")
axes[0,1].set_ylabel("Cell voltage (V)")
axes[0,1].set_title("Charge curves colored by SOH")
axes[0,1].legend(fontsize=8); axes[0,1].grid(True)

# ir_mean over lifecycle
axes[1,0].scatter(index_df["cycle"], feats_arr[:, 12],
                  c=soh_arr, cmap=cmap, s=20, alpha=0.7)
axes[1,0].set_xlabel("Cycle")
axes[1,0].set_ylabel("ir_mean")
axes[1,0].set_title("Feature [12] ir_mean over lifecycle")
axes[1,0].grid(True)

# Vd_std over lifecycle
axes[1,1].scatter(index_df["cycle"], feats_arr[:, 0],
                  c=soh_arr, cmap=cmap, s=20, alpha=0.7)
axes[1,1].set_xlabel("Cycle")
axes[1,1].set_ylabel("Vd_std")
axes[1,1].set_title("Feature [0] Vd_std over lifecycle")
axes[1,1].grid(True)

plt.suptitle("Drone Battery Pack -- Final Aligned Dataset",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_dataset_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> drone_dataset_overview.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("FINAL DRONE DATASET SUMMARY")
print("=" * 55)
print(f"  Samples (N)         : {len(index_df)}")
print(f"  Cycle range         : {index_df['cycle'].min()} - "
      f"{index_df['cycle'].max()}")
print(f"  SOH range           : {soh_arr.min():.4f} - {soh_arr.max():.4f}")
print(f"  Degradation         : "
      f"{(soh_arr.max()-soh_arr.min())*100:.1f}%")
print(f"\n  Output files:")
print(f"    drone_curves.npy        {curves_arr.shape}   -- autoencoder input")
print(f"    drone_soh.npy           {soh_arr.shape}     -- LSTM label")
print(f"    drone_temperature.npy   {temp_arr.shape}     -- manual feat [13]")
print(f"    drone_cell_ids.npy      {ids_arr.shape}     -- sequence grouping")
print(f"    drone_manual_feats.npy  {feats_arr.shape}  -- full 14-dim vector")
print(f"\n  Feature ranges:")
for i, name in enumerate(FEAT_NAMES):
    col   = feats_arr[:, i]
    valid = col[~np.isnan(col)]    # add this line
    if len(valid) > 0:
        print(f"    [{i:2d}] {name:12s}: "
              f"{valid.min():.4f} - {valid.max():.4f}  "
              f"(mean={valid.mean():.4f})")
    else:
        print(f"    [{i:2d}] {name:12s}: all NaN")
print("\nDone.")