"""
03_extract_manual_feats.py
==========================
Extracts the 14-dimensional manual feature vector per cycle from
the discharge file (voltage/temperature spread) and charge files (cell IR).

Reads  : data/UUT2_BatteryModule_Discharge.xlsx
         data/UUT2_BatteryPackP{1,2,3,4}_Charge.xlsx
Writes : drone_manual_feats.npy    (N, 14) float32
         drone_feat_index.csv      (N, 2)  cycle, ir_mean

Feature definitions (matching Fan et al. 2025, Table 5):
  [0]  Vd_std    — std of pack voltage spread during discharge
  [1]  Vd_diff   — range of pack voltage spread
  [2]  Vmax_std  — std of max pack voltage per timestep
  [3]  Vmax_mean — mean of max pack voltage per timestep
  [4]  Vmin_diff — range of min pack voltage per timestep
  [5]  Td_max    — max temperature spread across packs
  [6]  Td_diff   — range of temperature spread over time
  [7]  Tmax_min  — minimum of max pack temperature
  [8]  Tmin_std  — std of min pack temperature
  [9]  Tmin_mean — mean of min pack temperature
  [10] Tmin_min  — minimum of min pack temperature
  [11] Tmin_diff — range of min pack temperature
  [12] ir_mean   — mean cell IR at end of charge [mΩ] (all 4 packs × 6 cells)
  [13] T_mean    — mean ambient temperature [°C]

All pack voltages divided by N_CELLS=6 to give per-cell equivalent voltage.

Run from pack2/ directory:
    python 03_extract_manual_feats.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR   = "data"
DIS_FILE   = os.path.join(DATA_DIR, "UUT2_BatteryModule_Discharge.xlsx")
CHG_FILES  = {
    "P1": os.path.join(DATA_DIR, "UUT2_BatteryPackP1_Charge.xlsx"),
    "P2": os.path.join(DATA_DIR, "UUT2_BatteryPackP2_Charge.xlsx"),
    "P3": os.path.join(DATA_DIR, "UUT2_BatteryPackP3_Charge.xlsx"),
    "P4": os.path.join(DATA_DIR, "UUT2_BatteryPackP4_Charge.xlsx"),
}
N_CELLS    = 6
N_FEATS    = 14
OUT_NPY    = "drone_manual_feats.npy"
OUT_IDX    = "drone_feat_index.csv"

FEAT_NAMES = ["Vd_std","Vd_diff","Vmax_std","Vmax_mean","Vmin_diff",
              "Td_max","Td_diff","Tmax_min","Tmin_std","Tmin_mean",
              "Tmin_min","Tmin_diff","ir_mean","T_mean"]

# ── PART A: Discharge features (voltage + temperature spread) ──────────────
print("Reading discharge file for V/T features …")
xl         = pd.ExcelFile(DIS_FILE)
dis_sheets = [s for s in xl.sheet_names if s != "Readme"]

dis_feats = {}   # cycle → dict of features

for sheet in dis_sheets:
    cycle = int(sheet)
    df    = pd.read_excel(DIS_FILE, sheet_name=sheet)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    p_v_cols = [c for c in df.columns if "Voltage" in c and "UUT" in c]
    p_t_cols = [c for c in df.columns if "Temp"    in c and "UUT" in c]
    amb_col  = next(c for c in df.columns if "Ambient" in c)

    if len(p_v_cols) == 4:
        pv         = df[p_v_cols].values / N_CELLS
        vd         = pv.max(1) - pv.min(1)
        Vd_std     = float(vd.std())
        Vd_diff    = float(vd.max()   - vd.min())
        Vmax_std   = float(pv.max(1).std())
        Vmax_mean  = float(pv.max(1).mean())
        Vmin_diff  = float(pv.min(1).max() - pv.min(1).min())
    else:
        Vd_std=Vd_diff=Vmax_std=Vmax_mean=Vmin_diff = np.nan

    if len(p_t_cols) == 4:
        pt         = df[p_t_cols].values
        td         = pt.max(1) - pt.min(1)
        Td_max     = float(td.max())
        Td_diff    = float(td.max()   - td.min())
        Tmax_min   = float(pt.max(1).min())
        Tmin_std   = float(pt.min(1).std())
        Tmin_mean  = float(pt.min(1).mean())
        Tmin_min   = float(pt.min(1).min())
        Tmin_diff  = float(pt.min(1).max() - pt.min(1).min())
    else:
        Td_max=Td_diff=Tmax_min=Tmin_std=Tmin_mean=Tmin_min=Tmin_diff=np.nan

    dis_feats[cycle] = {
        "Vd_std": Vd_std, "Vd_diff": Vd_diff,
        "Vmax_std": Vmax_std, "Vmax_mean": Vmax_mean,
        "Vmin_diff": Vmin_diff,
        "Td_max": Td_max, "Td_diff": Td_diff,
        "Tmax_min": Tmax_min, "Tmin_std": Tmin_std,
        "Tmin_mean": Tmin_mean, "Tmin_min": Tmin_min,
        "Tmin_diff": Tmin_diff,
        "T_mean": float(df[amb_col].mean()),
    }

print(f"  Discharge features extracted for {len(dis_feats)} cycles")

# ── PART B: Cell IR from charge files ─────────────────────────────────────
print("Reading charge files for IR features …")
chg_ir = {}   # cycle → mean IR (mΩ)

for pack, fpath in CHG_FILES.items():
    xl     = pd.ExcelFile(fpath)
    sheets = [s for s in xl.sheet_names
              if s not in ("Readme", "Missing Data")]
    for s in sheets:
        try:
            df = pd.read_excel(fpath, sheet_name=s)
            df.columns = [c.replace("\n", " ").strip() for c in df.columns]
            cycle   = int(s)
            ir_cols = [c for c in df.columns if "Int Res" in c]
            if len(ir_cols) == N_CELLS:
                ir_vals = df[ir_cols].iloc[-1].values.astype(float)
                ir_vals = ir_vals[ir_vals > 0]
                if len(ir_vals) > 0:
                    chg_ir.setdefault(cycle, []).extend(ir_vals.tolist())
        except Exception:
            pass
    print(f"  {pack}: loaded IR for {sum(1 for p in chg_ir)} cycles so far")

# Average IR across all packs for each cycle
ir_mean_per_cycle = {
    cycle: float(np.mean(vals))
    for cycle, vals in chg_ir.items()
}
print(f"  IR available for {len(ir_mean_per_cycle)} cycles")

# ── PART C: Determine valid cycles ────────────────────────────────────────
# Use cycles that have BOTH discharge features AND charge IR
valid_cycles = sorted(
    set(dis_feats.keys()) & set(ir_mean_per_cycle.keys())
)
print(f"\n  Cycles with discharge features : {len(dis_feats)}")
print(f"  Cycles with IR data            : {len(ir_mean_per_cycle)}")
print(f"  Valid (intersection)           : {len(valid_cycles)}")

# ── PART D: Assemble feature matrix ───────────────────────────────────────
feats_list = []
index_rows = []

for cycle in valid_cycles:
    d   = dis_feats[cycle]
    ir  = ir_mean_per_cycle[cycle]

    feat = np.array([
        d["Vd_std"], d["Vd_diff"], d["Vmax_std"], d["Vmax_mean"],
        d["Vmin_diff"], d["Td_max"], d["Td_diff"], d["Tmax_min"],
        d["Tmin_std"], d["Tmin_mean"], d["Tmin_min"], d["Tmin_diff"],
        ir, d["T_mean"],
    ], dtype=np.float32)

    feats_list.append(feat)
    index_rows.append({"cycle": cycle, "ir_mean": ir})

feat_arr = np.array(feats_list, dtype=np.float32)
index_df = pd.DataFrame(index_rows)

print(f"\n  Feature matrix shape : {feat_arr.shape}")

# ── NaN check ──────────────────────────────────────────────────────────────
nan_counts = np.isnan(feat_arr).sum(axis=0)
print("\n  NaN counts per feature:")
for name, cnt in zip(FEAT_NAMES, nan_counts):
    flag = " ← CHECK" if cnt > 0 else ""
    print(f"    {name:12s}: {cnt}{flag}")

# ── Feature ranges ─────────────────────────────────────────────────────────
print("\n  Feature ranges (min – max, mean):")
for i, name in enumerate(FEAT_NAMES):
    col   = feat_arr[:, i]
    valid = col[~np.isnan(col)]
    print(f"    [{i:2d}] {name:12s}: {valid.min():.4f} – "
          f"{valid.max():.4f}  (mean={valid.mean():.4f})")

# ── Save ───────────────────────────────────────────────────────────────────
np.save(OUT_NPY, feat_arr)
index_df.to_csv(OUT_IDX, index=False)
print(f"\nSaved → {OUT_NPY}")
print(f"Saved → {OUT_IDX}")

# ── Plot: each feature over lifecycle ─────────────────────────────────────
fig, axes = plt.subplots(4, 4, figsize=(18, 14))
axes = axes.flatten()

for i, name in enumerate(FEAT_NAMES):
    axes[i].scatter(valid_cycles, feat_arr[:, i],
                    s=10, alpha=0.6, color="steelblue")
    axes[i].set_title(name, fontsize=9)
    axes[i].set_xlabel("Cycle", fontsize=7)
    axes[i].grid(True)
    axes[i].tick_params(labelsize=7)

# Hide unused subplots
for j in range(len(FEAT_NAMES), len(axes)):
    axes[j].set_visible(False)

plt.suptitle("Drone Battery — Manual Features over Lifecycle",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_manual_feats.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → drone_manual_feats.png")
print("\nDone.")