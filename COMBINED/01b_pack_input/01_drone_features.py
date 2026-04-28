"""
build_drone_features.py
=======================
Encodes drone pack charge curves using the pretrained autoencoder
and builds the 38-dim fused feature vectors for fine-tuning.

This is the TARGET DOMAIN preparation step.

Pipeline:
    1. Load drone curves and manual features
    2. Normalise curves using saved voltage_scaler.pkl (from autoencoder)
    3. Encode normalised curves -> 24-dim latent features
    4. Normalise latent features using saved latent_scaler.pkl
    5. Normalise temperature using saved temp_scaler.pkl
    6. Normalise 13 pack features using MinMaxScaler (fitted on drone)
    7. Build 38-dim fused vector:
          [latent_scaled(24) | T_mean_scaled(1) | pack_feats_scaled(13)]
    8. Split into train (first 10%) and test (remaining 90%)
    9. Save all outputs

Key difference from source domain:
    Source: [latent(24) | T_mean(1) | zeros(13)]  -- no pack data
    Drone:  [latent(24) | T_mean(1) | pack_feats(13)]  -- full features

Reads:
    ../../pack_lco_lipo_drone/drone_curves.npy        (350, 101)
    ../../pack_lco_lipo_drone/drone_manual_feats.npy  (350, 14)
    ../../pack_lco_lipo_drone/drone_soh.npy           (350,)
    ../../pack_lco_lipo_drone/drone_index.csv
    ../00_autoencoder/processed/encoder.pt
    ../00_autoencoder/processed/voltage_scaler.pkl
    ../01_fusevector/processed/latent_scaler.pkl
    ../01_fusevector/processed/temp_scaler.pkl

Writes (to processed/):
    drone_latent.npy           (350, 24)  encoded drone curves
    drone_fused.npy            (350, 38)  fused feature vectors
    drone_soh.npy              (350,)     SOH labels
    drone_index.csv            (350, *)   metadata
    pack_scaler.pkl            MinMaxScaler for 13 pack features
    drone_overview.png         visualisation

    Fine-tuning split (10/90):
    finetune_fused.npy         (35, 38)   first 10% for fine-tuning
    finetune_soh.npy           (35,)
    test_fused.npy             (315, 38)  remaining 90% for evaluation
    test_soh.npy               (315,)

Run from COMBINED/ directory:
    python build_drone_features.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DRONE_DIR   = "../../pack_lco_lipo(drone)"
AE_DIR      = "../00_autoencoder/processed"
FEAT_DIR    = "../01a_cell_input/processed"
OUT_DIR     = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

FINETUNE_FRAC = 0.1   # first 10% of drone cycles for fine-tuning
LATENT_DIM    = 24
FUSED_DIM     = 38
INPUT_DIM     = 101
HIDDEN1       = 496
HIDDEN2       = 72

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# ENCODER DEFINITION (must match train_autoencoder.py exactly)
# ══════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1,   HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2,   LATENT_DIM), nn.ReLU(),
        )
    def forward(self, x):
        return self.net(x)

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DRONE DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Building Drone Fused Features (Target Domain)")
print("=" * 60)

curves       = np.load(os.path.join(DRONE_DIR, "drone_curves.npy"))
manual_feats = np.load(os.path.join(DRONE_DIR, "drone_manual_feats.npy"))
soh          = np.load(os.path.join(DRONE_DIR, "drone_soh.npy"))
index        = pd.read_csv(os.path.join(DRONE_DIR, "drone_index.csv"))

print(f"\nLoaded drone data:")
print(f"  drone_curves.npy        : {curves.shape}")
print(f"  drone_manual_feats.npy  : {manual_feats.shape}")
print(f"  drone_soh.npy           : {soh.shape}")
print(f"  SOH range               : {soh.min():.4f} - {soh.max():.4f}")
print(f"  Voltage range           : {curves.min():.4f} - {curves.max():.4f} V")

# Validate manual feats layout
# [0-12] = pack inconsistency features, [13] = T_mean
print(f"\nManual features layout:")
feat_names = ["Vd_std","Vd_diff","Vmax_std","Vmax_mean","Vmin_diff",
              "Td_max","Td_diff","Tmax_min","Tmin_std","Tmin_mean",
              "Tmin_min","Tmin_diff","ir_mean","T_mean"]
for i, name in enumerate(feat_names):
    col = manual_feats[:, i]
    print(f"  [{i:2d}] {name:12s}: {col.min():.4f} - {col.max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: NORMALISE DRONE CURVES USING SAVED VOLTAGE SCALER
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising drone curves using saved voltage scaler ...")

voltage_scaler = joblib.load(os.path.join(AE_DIR, "voltage_scaler.pkl"))
curves_scaled  = voltage_scaler.transform(
    curves.reshape(-1, 1)).reshape(curves.shape).astype(np.float32)

print(f"  Raw range    : {curves.min():.4f} - {curves.max():.4f} V")
print(f"  Scaled range : {curves_scaled.min():.4f} - "
      f"{curves_scaled.max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: ENCODE DRONE CURVES USING PRETRAINED ENCODER
# ══════════════════════════════════════════════════════════════════════════
print(f"\nEncoding drone curves using pretrained encoder ...")

encoder = Encoder().to(DEVICE)
encoder.load_state_dict(
    torch.load(os.path.join(AE_DIR, "encoder.pt"),
               map_location=DEVICE))
encoder.eval()

curves_t    = torch.tensor(curves_scaled, dtype=torch.float32)
latent_list = []
with torch.no_grad():
    for i in range(0, len(curves_t), 256):
        batch  = curves_t[i:i+256].to(DEVICE)
        latent = encoder(batch).cpu().numpy()
        latent_list.append(latent)

drone_latent = np.concatenate(latent_list, axis=0).astype(np.float32)
np.save(os.path.join(OUT_DIR, "drone_latent.npy"), drone_latent)

print(f"  drone_latent.npy : {drone_latent.shape}")
print(f"  Latent range     : {drone_latent.min():.4f} - "
      f"{drone_latent.max():.4f}")
print(f"Saved -> drone_latent.npy")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: NORMALISE LATENT USING SAVED LATENT SCALER
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising latent features using saved latent scaler ...")

latent_scaler  = joblib.load(os.path.join(FEAT_DIR, "latent_scaler.pkl"))
latent_scaled  = latent_scaler.transform(
    drone_latent).astype(np.float32)

print(f"  Raw range    : {drone_latent.min():.4f} - "
      f"{drone_latent.max():.4f}")
print(f"  Scaled range : {latent_scaled.min():.4f} - "
      f"{latent_scaled.max():.4f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: NORMALISE TEMPERATURE USING SAVED TEMP SCALER
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising temperature using saved temp scaler ...")

temp_raw    = manual_feats[:, 13]   # T_mean is index 13
temp_scaler = joblib.load(os.path.join(FEAT_DIR, "temp_scaler.pkl"))
temp_scaled = temp_scaler.transform(
    temp_raw.reshape(-1, 1)).flatten().astype(np.float32)

print(f"  Raw range    : {temp_raw.min():.2f} - {temp_raw.max():.2f} C")
print(f"  Scaled range : {temp_scaled.min():.4f} - "
      f"{temp_scaled.max():.4f}")

# Note: drone temp range (28.8-35.3 C) is outside source temp range
# (20.6-27.8 C) so some values will be > 1.0 after scaling
# This is expected and acceptable -- scaler clips to visible range
if temp_scaled.max() > 1.0 or temp_scaled.min() < 0.0:
    print(f"  NOTE: Drone temp outside source training range "
          f"({temp_raw.min():.1f}-{temp_raw.max():.1f} C vs "
          f"20.6-27.8 C source range)")
    print(f"  Values will be outside [0,1] -- this is expected")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: NORMALISE 13 PACK INCONSISTENCY FEATURES
# Fit scaler on all drone pack data
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising 13 pack inconsistency features ...")

pack_feats_raw = manual_feats[:, :13]   # indices 0-12
pack_scaler    = MinMaxScaler()
pack_feats_scaled = pack_scaler.fit_transform(
    pack_feats_raw).astype(np.float32)

joblib.dump(pack_scaler, os.path.join(OUT_DIR, "pack_scaler.pkl"))

print(f"  Pack feats shape : {pack_feats_raw.shape}")
print(f"  Raw range        : {pack_feats_raw.min():.4f} - "
      f"{pack_feats_raw.max():.4f}")
print(f"  Scaled range     : {pack_feats_scaled.min():.4f} - "
      f"{pack_feats_scaled.max():.4f}")
print(f"Saved -> pack_scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: BUILD 38-DIM FUSED VECTORS
# [latent_scaled(24) | T_mean_scaled(1) | pack_feats_scaled(13)]
# ══════════════════════════════════════════════════════════════════════════
print(f"\nBuilding {FUSED_DIM}-dim fused feature vectors ...")

N           = len(latent_scaled)
drone_fused = np.zeros((N, FUSED_DIM), dtype=np.float32)

drone_fused[:, :24]  = latent_scaled       # latent features
drone_fused[:, 24]   = temp_scaled         # temperature
drone_fused[:, 25:]  = pack_feats_scaled   # pack inconsistency (13 dims)

print(f"  Fused shape        : {drone_fused.shape}")
print(f"  Latent  cols 0-23  : {drone_fused[:,:24].min():.4f} - "
      f"{drone_fused[:,:24].max():.4f}")
print(f"  Temp    col  24    : {drone_fused[:,24].min():.4f} - "
      f"{drone_fused[:,24].max():.4f}")
print(f"  Pack    cols 25-37 : {drone_fused[:,25:].min():.4f} - "
      f"{drone_fused[:,25:].max():.4f}")

# Validate no NaNs
assert np.isnan(drone_fused).sum() == 0, "NaNs in drone fused features"
print(f"  NaN check          : OK")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: SORT BY CYCLE AND SPLIT 10% FINETUNE / 90% TEST
# ══════════════════════════════════════════════════════════════════════════
print(f"\nSplitting into fine-tune (10%) and test (90%) ...")

# Sort by cycle number to ensure temporal order
sort_order   = index["cycle"].argsort().values
drone_fused  = drone_fused[sort_order]
soh_sorted   = soh[sort_order]
index_sorted = index.iloc[sort_order].reset_index(drop=True)

n_total    = len(soh_sorted)
n_finetune = max(1, int(n_total * FINETUNE_FRAC))
n_test     = n_total - n_finetune

finetune_fused = drone_fused[:n_finetune]
finetune_soh   = soh_sorted[:n_finetune]
test_fused     = drone_fused[n_finetune:]
test_soh       = soh_sorted[n_finetune:]

print(f"  Total cycles    : {n_total}")
print(f"  Fine-tune (10%) : {n_finetune} cycles  "
      f"SOH {finetune_soh.min():.3f}-{finetune_soh.max():.3f}")
print(f"  Test (90%)      : {n_test} cycles  "
      f"SOH {test_soh.min():.3f}-{test_soh.max():.3f}")
print(f"  Cycle range ft  : "
      f"{index_sorted['cycle'].iloc[:n_finetune].min()} - "
      f"{index_sorted['cycle'].iloc[:n_finetune].max()}")
print(f"  Cycle range test: "
      f"{index_sorted['cycle'].iloc[n_finetune:].min()} - "
      f"{index_sorted['cycle'].iloc[n_finetune:].max()}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: SAVE ALL OUTPUTS
# ══════════════════════════════════════════════════════════════════════════
np.save(os.path.join(OUT_DIR, "drone_fused.npy"),    drone_fused)
np.save(os.path.join(OUT_DIR, "drone_soh.npy"),      soh_sorted.astype(np.float32))
np.save(os.path.join(OUT_DIR, "drone_latent.npy"),   drone_latent[sort_order])
np.save(os.path.join(OUT_DIR, "finetune_fused.npy"), finetune_fused)
np.save(os.path.join(OUT_DIR, "finetune_soh.npy"),   finetune_soh.astype(np.float32))
np.save(os.path.join(OUT_DIR, "test_fused.npy"),     test_fused)
np.save(os.path.join(OUT_DIR, "test_soh.npy"),       test_soh.astype(np.float32))
index_sorted.to_csv(os.path.join(OUT_DIR, "drone_index.csv"), index=False)

print(f"\nSaved -> drone_fused.npy         {drone_fused.shape}")
print(f"Saved -> drone_soh.npy           {soh_sorted.shape}")
print(f"Saved -> drone_latent.npy        {drone_latent.shape}")
print(f"Saved -> finetune_fused.npy      {finetune_fused.shape}")
print(f"Saved -> finetune_soh.npy        {finetune_soh.shape}")
print(f"Saved -> test_fused.npy          {test_fused.shape}")
print(f"Saved -> test_soh.npy            {test_soh.shape}")
print(f"Saved -> drone_index.csv         {index_sorted.shape}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: PLOTS
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
x_axis    = np.linspace(0, 1, FUSED_DIM)
cycles    = index_sorted["cycle"].values

# ── Plot 1: SOH with fine-tune / test split marked ────────────────────────
axes[0].plot(cycles, soh_sorted, lw=1.5, color="steelblue",
             label="SOH trajectory")
axes[0].axvspan(cycles[0], cycles[n_finetune-1],
                alpha=0.2, color="green",
                label=f"Fine-tune ({n_finetune} cycles, 10%)")
axes[0].axvspan(cycles[n_finetune], cycles[-1],
                alpha=0.1, color="red",
                label=f"Test ({n_test} cycles, 90%)")
axes[0].axvline(cycles[n_finetune], color="red", lw=2,
                linestyle="--", label="Split point")
axes[0].axhline(0.80, color="red", lw=1, linestyle=":",
                label="EOL 80%")
axes[0].set_xlabel("Cycle"); axes[0].set_ylabel("SOH")
axes[0].set_title(f"Drone SOH -- Fine-tune/Test Split\n"
                  f"(10% fine-tune = cycles "
                  f"{cycles[0]}-{cycles[n_finetune-1]})")
axes[0].legend(fontsize=7); axes[0].grid(True, alpha=0.4)

# ── Plot 2: Fused feature heatmap (sample of cycles) ─────────────────────
sample_idx = np.linspace(0, n_total-1, min(50, n_total), dtype=int)
im = axes[1].imshow(drone_fused[sample_idx].T,
                     aspect="auto", cmap="viridis")
plt.colorbar(im, ax=axes[1])
axes[1].axhline(23.5, color="red",   lw=1, linestyle="--",
                label="latent|temp boundary")
axes[1].axhline(24.5, color="orange", lw=1, linestyle="--",
                label="temp|pack boundary")
axes[1].set_xlabel("Cycle (sampled)")
axes[1].set_ylabel("Feature dimension")
axes[1].set_title("Fused Feature Heatmap (38-dim)\n"
                  "[0-23: latent | 24: temp | 25-37: pack]")
axes[1].legend(fontsize=7)

# ── Plot 3: Pack feature distributions ───────────────────────────────────
for i in range(13):
    axes[2].plot(cycles, pack_feats_scaled[:, i],
                 lw=0.5, alpha=0.6,
                 label=feat_names[i] if i < 5 else "")
axes[2].axvline(cycles[n_finetune], color="red", lw=2,
                linestyle="--", label="Split point")
axes[2].set_xlabel("Cycle")
axes[2].set_ylabel("Normalised feature value")
axes[2].set_title("Pack Inconsistency Features (normalised)\n"
                  "dims 25-37 of fused vector")
axes[2].legend(fontsize=6); axes[2].grid(True, alpha=0.3)

plt.suptitle(f"Drone Target Domain -- Fused Features ({FUSED_DIM}-dim)\n"
             f"[latent(24) | T_mean(1) | pack_feats(13)]  --  "
             f"{n_total} cycles  |  "
             f"Fine-tune: {n_finetune}  Test: {n_test}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "drone_overview.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> drone_overview.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("DRONE FEATURE BUILDING SUMMARY")
print("=" * 60)
print(f"  Total cycles      : {n_total}")
print(f"  Fine-tune (10%)   : {n_finetune} cycles")
print(f"  Test (90%)        : {n_test} cycles")
print(f"\n  Fused vector layout (38-dim):")
print(f"    dims  0-23 : latent (encoded from autoencoder)")
print(f"    dim   24   : T_mean (normalised with source temp scaler)")
print(f"    dims 25-37 : pack inconsistency features (normalised)")
print(f"\n  Scalers used:")
print(f"    voltage_scaler.pkl  -- from autoencoder training")
print(f"    latent_scaler.pkl   -- from source feature building")
print(f"    temp_scaler.pkl     -- from source feature building")
print(f"    pack_scaler.pkl     -- fitted on drone pack data (new)")
print(f"\n  Output files:")
print(f"    drone_fused.npy          {drone_fused.shape}")
print(f"    drone_soh.npy            {soh_sorted.shape}")
print(f"    finetune_fused.npy       {finetune_fused.shape}")
print(f"    finetune_soh.npy         {finetune_soh.shape}")
print(f"    test_fused.npy           {test_fused.shape}")
print(f"    test_soh.npy             {test_soh.shape}")
print(f"\n  Next step:")
print(f"    Fine-tune pretrained LSTM on finetune_fused.npy")
print(f"    Evaluate on test_fused.npy")
print("\nDone.")