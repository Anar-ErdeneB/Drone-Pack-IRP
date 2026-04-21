"""
migrate_drone_outputs.py
=========================
Migrates old drone pipeline outputs to the consistent 4-file structure.

Old files (kept untouched):
    drone_charge_curves.npy
    drone_manual_feats.npy
    drone_index.csv
    soh_labels_drone.csv

New files (created):
    drone_curves.npy         (N, 101) float32
    drone_soh.npy            (N,)     float32
    drone_temperature.npy    (N,)     float32
    drone_cell_ids.npy       (N,)     str
    drone_index.csv          (N, *)   replaces old one

Run from pack2/ directory:
    python migrate_drone_outputs.py
"""

import numpy as np
import pandas as pd
import os

# ── Load old files ─────────────────────────────────────────────────────────
curves_old = np.load("drone_charge_curves.npy")
feats_old  = np.load("drone_manual_feats.npy")
index_old  = pd.read_csv("drone_index.csv")

print(f"Loaded:")
print(f"  drone_charge_curves.npy : {curves_old.shape}")
print(f"  drone_manual_feats.npy  : {feats_old.shape}")
print(f"  drone_index.csv         : {index_old.shape}")
print(f"  drone_index columns     : {list(index_old.columns)}")

# ── Build new arrays ───────────────────────────────────────────────────────
# curves: direct rename
curves_arr = curves_old.astype(np.float32)

# soh: from index (already smoothed and normalised 0-1)
soh_arr    = index_old["soh"].values.astype(np.float32)

# temperature: T_mean is feature index [13] in manual feats
# (index 13 = T_mean based on our feature definition)
temp_arr   = feats_old[:, 13].astype(np.float32)

# cell_ids: all same pack — use "drone_pack" for all
ids_arr    = np.array(["drone_pack"] * len(curves_arr), dtype=object)

# ── Validation ─────────────────────────────────────────────────────────────
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

# ── Build clean index ──────────────────────────────────────────────────────
index_new = pd.DataFrame({
    "cell_id" : ids_arr,
    "group"   : "drone",
    "cycle"   : index_old["cycle"].values,
    "soh"     : soh_arr,
    "soh_raw" : index_old["soh_raw"].values if "soh_raw" in index_old.columns
                else soh_arr,
    "q_ah"    : index_old["q_ah"].values,
    "t_mean"  : temp_arr,
})

# ── Save new files ─────────────────────────────────────────────────────────
np.save("drone_curves.npy",      curves_arr)
np.save("drone_soh.npy",         soh_arr)
np.save("drone_temperature.npy", temp_arr)
np.save("drone_cell_ids.npy",    ids_arr)
index_new.to_csv("drone_index.csv", index=False)

print(f"\nSaved -> drone_curves.npy        {curves_arr.shape}")
print(f"Saved -> drone_soh.npy           {soh_arr.shape}")
print(f"Saved -> drone_temperature.npy   {temp_arr.shape}")
print(f"Saved -> drone_cell_ids.npy      {ids_arr.shape}")
print(f"Saved -> drone_index.csv         {index_new.shape}")
print(f"\nOld files kept untouched:")
print(f"  drone_charge_curves.npy")
print(f"  drone_manual_feats.npy")
print(f"  soh_labels_drone.csv")
print(f"  drone_curve_index.csv")
print(f"  drone_feat_index.csv")
print(f"\nDone.")