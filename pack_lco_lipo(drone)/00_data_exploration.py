"""
drone_degradation_overview.py
==============================
Reproduces the drone battery degradation overview plot.
Reads only: data/UUT2_BatteryModule_Discharge.xlsx

Run from pack2/ directory:
    python drone_degradation_overview.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data"
DIS_FILE = os.path.join(DATA_DIR, "UUT2_BatteryModule_Discharge.xlsx")

# ── Load all discharge sheets ──────────────────────────────────────────────
print("Reading discharge file ...")
xl         = pd.ExcelFile(DIS_FILE)
dis_sheets = [s for s in xl.sheet_names if s != "Readme"]
print(f"  Found {len(dis_sheets)} cycle sheets")

records    = []
dis_curves = {}   # cycle → dict of time series

for sheet in dis_sheets:
    cycle = int(sheet)
    df    = pd.read_excel(DIS_FILE, sheet_name=sheet)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    # Identify columns
    cap_col  = next(c for c in df.columns if "capacity"   in c.lower())
    v_col    = next(c for c in df.columns if c.startswith("Voltage"))
    i_col    = next(c for c in df.columns if "current"    in c.lower())
    res_col  = next(c for c in df.columns if "resistance" in c.lower())
    time_col = next(c for c in df.columns if "Time"       in c)
    amb_col  = next(c for c in df.columns if "Ambient"    in c)
    p_v_cols = [c for c in df.columns if "Voltage" in c and "UUT" in c]
    p_t_cols = [c for c in df.columns if "Temp"    in c and "UUT" in c]

    q_ah     = float(df[cap_col].iloc[-1])
    duration = float(df[time_col].iloc[-1])

    # Pack temperature spread
    if len(p_t_cols) == 4:
        pt = df[p_t_cols].values
        t_spread_mean = (pt.max(1) - pt.min(1)).mean()
    else:
        t_spread_mean = np.nan

    records.append({
        "cycle"        : cycle,
        "q_ah"         : q_ah,
        "duration_s"   : duration,
        "res_mean"     : df[res_col].mean(),
        "t_spread_mean": t_spread_mean,
    })

    # Store curves for voltage vs Ah plot
    dis_curves[cycle] = {
        "v_pack"  : df[v_col].values,
        "cum_ah"  : df[cap_col].values,
    }

df_r = pd.DataFrame(records).sort_values("cycle").reset_index(drop=True)

# ── SOH ────────────────────────────────────────────────────────────────────
# Remove outlier (cycle 349 — anomalously low capacity)
q_tmp        = df_r["q_ah"].iloc[:5].mean()
outlier_mask = df_r["q_ah"] / q_tmp < 0.3
df_r         = df_r[~outlier_mask].reset_index(drop=True)

# q_max        = df_r["q_ah"].iloc[:5].mean()   # match original plot
q_max        = df_r["q_ah"].max()
df_r["soh"]  = df_r["q_ah"] / q_max

print(f"  Valid cycles : {len(df_r)}")
print(f"  Q_max        : {q_max:.3f} Ah")
print(f"  SOH range    : {df_r['soh'].min():.4f} – {df_r['soh'].max():.4f}")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

cmap   = plt.cm.RdYlGn
cycles = df_r["cycle"].values
norm_c = plt.Normalize(cycles.min(), cycles.max())

# 1. SOH trajectory
axes[0,0].plot(df_r["cycle"], df_r["soh"], "o-", ms=2, lw=1, color="steelblue")
axes[0,0].axhline(0.8, color="red",   lw=1.5, linestyle="--", label="EOL 80%")
axes[0,0].axhline(1.0, color="black", lw=1,   linestyle="--", label="100%")
axes[0,0].set_xlabel("Cycle"); axes[0,0].set_ylabel("SOH")
axes[0,0].set_title(f"SOH trajectory (Q_max={q_max:.3f} Ah)")
axes[0,0].legend(fontsize=8); axes[0,0].grid(True)

# 2. Discharge capacity
axes[0,1].plot(df_r["cycle"], df_r["q_ah"], "o-", ms=2, lw=1, color="steelblue")
axes[0,1].axhline(q_max,       color="black", lw=1,   linestyle="--",
                  label=f"Q_max={q_max:.2f}")
axes[0,1].axhline(q_max * 0.8, color="red",   lw=1.5, linestyle="--",
                  label="EOL 80%")
axes[0,1].set_xlabel("Cycle"); axes[0,1].set_ylabel("Capacity (Ah)")
axes[0,1].set_title("Discharge capacity")
axes[0,1].legend(fontsize=8); axes[0,1].grid(True)

# 3. Mean resistance
axes[0,2].plot(df_r["cycle"], df_r["res_mean"], "o-", ms=2, lw=1,
               color="darkorange")
axes[0,2].set_xlabel("Cycle"); axes[0,2].set_ylabel("Resistance (\u03a9)")
axes[0,2].set_title("Mean resistance over lifecycle")
axes[0,2].grid(True)

# 4. Discharge V vs Ah — sample every ~12 cycles for clarity
sample_cycles = sorted(dis_curves.keys())[::12]
for c in sample_cycles:
    if c not in dis_curves:
        continue
    color = cmap(norm_c(c))
    d     = dis_curves[c]
    axes[1,0].plot(d["cum_ah"], d["v_pack"],
                   color=color, alpha=0.5, lw=0.7)
axes[1,0].set_xlabel("Cumulative Ah discharged")
axes[1,0].set_ylabel("Module voltage (V)")
axes[1,0].set_title("Discharge V vs Ah (green=early, red=late)")
axes[1,0].grid(True)

# 5. Discharge duration
axes[1,1].plot(df_r["cycle"], df_r["duration_s"], "o-", ms=2, lw=1,
               color="green")
axes[1,1].set_xlabel("Cycle"); axes[1,1].set_ylabel("Duration (s)")
axes[1,1].set_title("Discharge duration per cycle")
axes[1,1].grid(True)

# 6. Temperature spread
axes[1,2].plot(df_r["cycle"], df_r["t_spread_mean"], "o-", ms=2, lw=1,
               color="purple")
axes[1,2].set_xlabel("Cycle"); axes[1,2].set_ylabel("Temp spread (\u00b0C)")
axes[1,2].set_title("Mean pack temperature spread (inconsistency)")
axes[1,2].grid(True)

plt.suptitle("Drone Battery Pack \u2014 Degradation Overview (378 cycles)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_degradation_overview.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved \u2192 drone_degradation_overview.png")
print("Done.")