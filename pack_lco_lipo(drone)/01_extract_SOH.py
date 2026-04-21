"""
01_extract_soh.py
=================
Extracts SOH labels from the discharge Excel file.

Reads  : data/UUT2_BatteryModule_Discharge.xlsx
Writes : soh_labels_drone.csv

SOH definition (Paw & Ang 2023):
    SOH = Q_i / Q_max
    where Q_i   = total Ah discharged in cycle i (last value of capacity col)
          Q_max = maximum observed capacity across all valid cycles

Smoothing: zero-phase Butterworth low-pass filter
    order=3, cutoff=0.10, capped at 1.0

Run from pack2/ directory:
    python 01_extract_soh.py
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# ── Configuration ──────────────────────────────────────────────────────────
DATA_DIR   = "data"
DIS_FILE   = os.path.join(DATA_DIR, "UUT2_BatteryModule_Discharge.xlsx")
OUT_CSV    = "soh_labels_drone.csv"
BW_CUTOFF  = 0.10
BW_ORDER   = 3

# ── Load discharge file ────────────────────────────────────────────────────
print("Reading discharge file …")
xl         = pd.ExcelFile(DIS_FILE)
dis_sheets = [s for s in xl.sheet_names if s != "Readme"]
print(f"  Found {len(dis_sheets)} cycle sheets")

records = []
for sheet in dis_sheets:
    cycle = int(sheet)
    df    = pd.read_excel(DIS_FILE, sheet_name=sheet)
    df.columns = [c.replace("\n", " ").strip() for c in df.columns]

    cap_col = next(c for c in df.columns if "capacity" in c.lower())
    q_ah    = float(df[cap_col].iloc[-1])

    records.append({"cycle": cycle, "q_ah": q_ah})

df = pd.DataFrame(records).sort_values("cycle").reset_index(drop=True)

# ── Remove outlier cycles (anomalously low capacity, SOH < 0.3) ───────────
q_tmp        = df["q_ah"].iloc[:5].mean()
outlier_mask = df["q_ah"] / q_tmp < 0.3
removed      = df[outlier_mask]["cycle"].tolist()
df           = df[~outlier_mask].reset_index(drop=True)
print(f"  Outlier cycles removed : {removed}")

# ── SOH calculation ────────────────────────────────────────────────────────
q_max        = df["q_ah"].max()
df["soh_raw"]= df["q_ah"] / q_max
print(f"  Valid cycles : {len(df)}")
print(f"  Q_max        : {q_max:.4f} Ah  (cycle {df.loc[df['q_ah'].idxmax(), 'cycle']})")
print(f"  SOH raw      : {df['soh_raw'].min():.4f} – {df['soh_raw'].max():.4f}")

# ── Butterworth smoothing ──────────────────────────────────────────────────
def butterworth(data, cutoff=BW_CUTOFF, order=BW_ORDER):
    b, a = butter(order, cutoff / 0.5, btype="low", analog=False)
    return np.clip(filtfilt(b, a, data), 0.0, 1.0)

df["soh_smooth"] = butterworth(df["soh_raw"].values)
print(f"  SOH smoothed : {df['soh_smooth'].min():.4f} – "
      f"{df['soh_smooth'].max():.4f}")
print(f"  Degradation  : "
      f"{(df['soh_smooth'].max() - df['soh_smooth'].min())*100:.1f}%")

# ── Save ───────────────────────────────────────────────────────────────────
df.to_csv(OUT_CSV, index=False)
print(f"\nSaved → {OUT_CSV}  ({len(df)} rows)")

# ── Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

axes[0].scatter(df["cycle"], df["soh_raw"], s=8, color="gray",
                alpha=0.4, label="Raw SOH", zorder=1)
axes[0].plot(df["cycle"], df["soh_smooth"], lw=2, color="steelblue",
             label="Butterworth smoothed", zorder=3)
axes[0].axhline(0.8, color="red",   lw=1.5, linestyle="--", label="EOL 80%")
axes[0].axhline(1.0, color="black", lw=1,   linestyle="--", label="100%")
axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("SOH")
axes[0].set_title(f"SOH — Raw vs Smoothed  (Q_max={q_max:.4f} Ah)")
axes[0].legend(fontsize=9); axes[0].grid(True)
axes[0].set_ylim(0.74, 1.06)

axes[1].fill_between(df["cycle"], df["soh_smooth"], 0.8,
                     where=df["soh_smooth"] >= 0.8,
                     alpha=0.15, color="steelblue")
axes[1].fill_between(df["cycle"], df["soh_smooth"], 0.8,
                     where=df["soh_smooth"] < 0.8,
                     alpha=0.15, color="red")
axes[1].plot(df["cycle"], df["soh_smooth"], lw=2.5, color="steelblue")
axes[1].axhline(0.8, color="red",   lw=1.5, linestyle="--", label="EOL 80%")
axes[1].axhline(1.0, color="black", lw=1,   linestyle="--", label="100%")
eol = df[df["soh_smooth"] <= 0.8]["cycle"].min()
if pd.notna(eol):
    axes[1].axvline(eol, color="red", lw=1.5, linestyle=":", alpha=0.7)
    axes[1].annotate(f"EOL at cycle {int(eol)}",
                     xy=(eol, 0.80), xytext=(eol - 70, 0.82),
                     fontsize=9, color="red",
                     arrowprops=dict(arrowstyle="->", color="red"))
axes[1].set_xlabel("Cycle"); axes[1].set_ylabel("SOH")
axes[1].set_title(f"Smoothed SOH  |  "
                  f"{df['soh_smooth'].max():.3f} → {df['soh_smooth'].min():.3f}  |  "
                  f"Degradation {(df['soh_smooth'].max()-df['soh_smooth'].min())*100:.1f}%")
axes[1].legend(fontsize=9); axes[1].grid(True)
axes[1].set_ylim(0.74, 1.06)

plt.suptitle(f"Drone Battery — SOH Labels  "
             f"(Butterworth order={BW_ORDER}, cutoff={BW_CUTOFF}, capped at 1.0)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig("drone_soh_final.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved → drone_soh_final.png")
print("\nDone.")