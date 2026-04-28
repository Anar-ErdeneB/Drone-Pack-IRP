"""
build_features.py
=================
Builds the 38-dimensional fused feature vectors for source domain
pretraining, following Fan et al. (2025) exactly.

Pipeline (Fan et al. 2025):
    1. Load source latent features (24-dim, from autoencoder)
    2. Normalise latent features to [0,1]  <- Fan et al. explicit step
    3. Normalise temperature to [0,1]
    4. Build fused vector:
          [latent_scaled(24) | temp_scaled(1) | zeros(13)] = 38-dim
       -- zeros(13) because pack inconsistency features do not
          exist for source domain single cells
    5. Organise into per-cell temporal sequences for LSTM training
       -- LSTM must see cycles in order: cycle 1 -> 2 -> ... -> N
          per cell, not shuffled

Reads (from ../autoencoder/processed/):
    source_latent.npy       (N, 24)  raw latent features

Reads (from ../processed/):
    source_temperature.npy  (N,)     mean charge temperature
    source_soh.npy          (N,)     SOH labels
    source_cell_ids.npy     (N,)     cell identifiers
    source_index.csv        (N, *)   cycle metadata

Writes (to processed/):
    source_fused.npy         (N, 38)  fused feature vectors
    source_soh.npy           (N,)     SOH labels (copy, aligned)
    source_cell_ids.npy      (N,)     cell IDs (copy, aligned)
    latent_scaler.pkl        MinMaxScaler fitted on latent features
    temp_scaler.pkl          MinMaxScaler fitted on temperature
    cell_sequences.pkl       list of per-cell dicts for LSTM training
    feature_summary.png      visualisation of fused feature space
    feature_summary.csv      per-cell statistics

Fused vector layout (38-dim):
    dims  0-23 : normalised latent features (from autoencoder)
    dim   24   : normalised temperature
    dims 25-37 : zeros (pack inconsistency -- not available for cells)

Run from COMBINED/ directory:
    python build_features.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import joblib

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
LATENT_DIR  = "../00_autoencoder/processed"   # where source_latent.npy lives
SOURCE_DIR  = "../processed"               # where source domain files live
OUT_DIR     = "processed"               # output directory (same)
os.makedirs(OUT_DIR, exist_ok=True)

LATENT_DIM  = 24
TEMP_DIM    = 1
ZERO_DIM    = 13
FUSED_DIM   = LATENT_DIM + TEMP_DIM + ZERO_DIM   # = 38

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD ALL SOURCE DOMAIN DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Building Source Domain Fused Features")
print("Fan et al. (2025) -- 38-dim fused vector")
print("=" * 60)

latent  = np.load(os.path.join(LATENT_DIR,  "source_latent.npy"))
temp    = np.load(os.path.join(SOURCE_DIR,  "source_temperature.npy"))
soh     = np.load(os.path.join(SOURCE_DIR,  "source_soh.npy"))
ids     = np.load(os.path.join(SOURCE_DIR,  "source_cell_ids.npy"),
                  allow_pickle=True)
index   = pd.read_csv(os.path.join(SOURCE_DIR, "source_index.csv"))

print(f"\nLoaded:")
print(f"  source_latent.npy      : {latent.shape}")
print(f"  source_temperature.npy : {temp.shape}")
print(f"  source_soh.npy         : {soh.shape}")
print(f"  source_cell_ids.npy    : {ids.shape}")
print(f"  Latent range           : {latent.min():.4f} - {latent.max():.4f}")
print(f"  Temp range             : {temp.min():.1f} - {temp.max():.1f} C")
print(f"  SOH range              : {soh.min():.4f} - {soh.max():.4f}")
print(f"  Unique cells           : {len(np.unique(ids))}")

# Validate alignment
assert len(latent) == len(temp) == len(soh) == len(ids), \
    "Array length mismatch!"
print(f"  Alignment check        : OK ({len(soh)} samples)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: NORMALISE LATENT FEATURES TO [0, 1]
# Fan et al. explicitly normalise latent features before concatenation
# Scaler fitted on ALL source domain latent features
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising latent features to [0, 1] ...")

latent_scaler  = MinMaxScaler()
latent_scaled  = latent_scaler.fit_transform(latent).astype(np.float32)

print(f"  Raw range    : {latent.min():.4f} - {latent.max():.4f}")
print(f"  Scaled range : {latent_scaled.min():.4f} - "
      f"{latent_scaled.max():.4f}")

joblib.dump(latent_scaler, os.path.join(OUT_DIR, "latent_scaler.pkl"))
print(f"  Saved -> latent_scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: NORMALISE TEMPERATURE TO [0, 1]
# Scaler fitted on ALL source domain temperatures
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising temperature to [0, 1] ...")

temp_scaler  = MinMaxScaler()
temp_scaled  = temp_scaler.fit_transform(
    temp.reshape(-1, 1)).flatten().astype(np.float32)

print(f"  Raw range    : {temp.min():.2f} - {temp.max():.2f} C")
print(f"  Scaled range : {temp_scaled.min():.4f} - "
      f"{temp_scaled.max():.4f}")

joblib.dump(temp_scaler, os.path.join(OUT_DIR, "temp_scaler.pkl"))
print(f"  Saved -> temp_scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD 38-DIM FUSED FEATURE VECTORS
# [latent_scaled(24) | temp_scaled(1) | zeros(13)]
# ══════════════════════════════════════════════════════════════════════════
print(f"\nBuilding {FUSED_DIM}-dim fused feature vectors ...")

N = len(latent_scaled)
fused = np.zeros((N, FUSED_DIM), dtype=np.float32)

# dims 0-23: normalised latent
fused[:, :LATENT_DIM]    = latent_scaled

# dim 24: normalised temperature
fused[:, LATENT_DIM]     = temp_scaled

# dims 25-37: zeros (pack features not available for source cells)
# already zero from np.zeros initialisation

print(f"  Fused shape      : {fused.shape}")
print(f"  Latent cols 0-23 : min={fused[:,:24].min():.4f}  "
      f"max={fused[:,:24].max():.4f}")
print(f"  Temp col 24      : min={fused[:,24].min():.4f}  "
      f"max={fused[:,24].max():.4f}")
print(f"  Zero cols 25-37  : sum={fused[:,25:].sum():.0f} "
      f"(should be 0)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: ORGANISE INTO PER-CELL TEMPORAL SEQUENCES
# Critical for LSTM -- must see aging progression per cell in order
# cell_sequences is a list of dicts, one per cell
# ══════════════════════════════════════════════════════════════════════════
print(f"\nOrganising into per-cell temporal sequences ...")

cell_sequences = []

for cell_id in np.unique(ids):
    mask      = ids == cell_id
    cell_fuse = fused[mask]
    cell_soh  = soh[mask]
    cell_idx  = index[mask].copy()

    # Sort by charge_num to ensure temporal order
    if "charge_num" in cell_idx.columns:
        sort_order = cell_idx["charge_num"].argsort().values
    else:
        sort_order = np.arange(len(cell_fuse))

    cell_fuse = cell_fuse[sort_order]
    cell_soh  = cell_soh[sort_order]

    # Get dataset name from index
    dataset = cell_idx["dataset"].iloc[0] \
              if "dataset" in cell_idx.columns else "unknown"

    cell_sequences.append({
        "cell_id"  : cell_id,
        "dataset"  : dataset,
        "features" : cell_fuse,    # (n_cycles, 38)
        "soh"      : cell_soh,     # (n_cycles,)
        "n_cycles" : len(cell_fuse),
    })

    print(f"  {cell_id:12s} ({dataset:6s}): "
          f"{len(cell_fuse):4d} cycles  "
          f"SOH {cell_soh.min():.3f}-{cell_soh.max():.3f}")

print(f"\n  Total cells    : {len(cell_sequences)}")
print(f"  Total cycles   : {sum(s['n_cycles'] for s in cell_sequences)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: VALIDATION
# ══════════════════════════════════════════════════════════════════════════
print(f"\nValidation:")
assert fused.shape == (N, FUSED_DIM), "Fused shape mismatch"
assert np.isnan(fused).sum() == 0, "NaNs in fused features"
assert fused[:, 25:].sum() == 0, "Non-zero values in zero-padded dims"
assert fused[:, :24].min() >= 0.0, "Latent below 0"
assert fused[:, :24].max() <= 1.0, "Latent above 1"
assert fused[:, 24].min() >= 0.0, "Temp below 0"
assert fused[:, 24].max() <= 1.0, "Temp above 1"
print(f"  Shape    : {fused.shape}  OK")
print(f"  No NaNs  : OK")
print(f"  Zeros    : dims 25-37 all zero  OK")
print(f"  Range    : all values in [0,1] for dims 0-24  OK")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE
# ══════════════════════════════════════════════════════════════════════════
np.save(os.path.join(OUT_DIR, "source_fused.npy"),    fused)
np.save(os.path.join(OUT_DIR, "source_soh.npy"),      soh)
np.save(os.path.join(OUT_DIR, "source_cell_ids.npy"), ids)

with open(os.path.join(OUT_DIR, "cell_sequences.pkl"), "wb") as f:
    pickle.dump(cell_sequences, f)

# Summary CSV
summary_rows = []
for s in cell_sequences:
    summary_rows.append({
        "cell_id"  : s["cell_id"],
        "dataset"  : s["dataset"],
        "n_cycles" : s["n_cycles"],
        "soh_min"  : round(float(s["soh"].min()), 4),
        "soh_max"  : round(float(s["soh"].max()), 4),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(OUT_DIR, "feature_summary.csv"),
                  index=False)

print(f"\nSaved -> source_fused.npy       {fused.shape}")
print(f"Saved -> source_soh.npy         {soh.shape}")
print(f"Saved -> source_cell_ids.npy    {ids.shape}")
print(f"Saved -> cell_sequences.pkl     {len(cell_sequences)} sequences")
print(f"Saved -> latent_scaler.pkl")
print(f"Saved -> temp_scaler.pkl")
print(f"Saved -> feature_summary.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: PLOTS
# ══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ── Plot 1: PCA of fused features coloured by SOH ─────────────────────────
pca    = PCA(n_components=2)
fused_2d = pca.fit_transform(fused[:, :24])  # PCA on latent dims only

sc = axes[0].scatter(fused_2d[:, 0], fused_2d[:, 1],
                     c=soh, cmap="RdYlGn", alpha=0.4, s=8)
plt.colorbar(sc, ax=axes[0], label="SOH")
axes[0].set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[0].set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[0].set_title("Latent Space PCA -- coloured by SOH\n"
                  "(good = smooth SOH gradient along arc)")
axes[0].grid(True, alpha=0.3)

# ── Plot 2: PCA coloured by dataset ───────────────────────────────────────
dataset_colors = {"NASA": "steelblue", "CALCE": "darkorange",
                  "LCO": "green"}
for cell_id in np.unique(ids):
    mask    = ids == cell_id
    dataset = index[mask]["dataset"].iloc[0] \
              if "dataset" in index.columns else "unknown"
    color   = dataset_colors.get(dataset, "gray")
    axes[1].scatter(fused_2d[mask, 0], fused_2d[mask, 1],
                    color=color, alpha=0.4, s=8,
                    label=dataset if dataset not in
                    [t.get_label() for t in axes[1].collections] else "")

# Deduplicate legend
handles, labels = axes[1].get_legend_handles_labels()
seen = set()
unique = [(h, l) for h, l in zip(handles, labels)
          if not (l in seen or seen.add(l))]
axes[1].legend(*zip(*unique), fontsize=8)
axes[1].set_xlabel(
    f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
axes[1].set_ylabel(
    f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
axes[1].set_title("Latent Space PCA -- coloured by dataset")
axes[1].grid(True, alpha=0.3)

# ── Plot 3: SOH distribution per dataset ──────────────────────────────────
for dataset_name, color in dataset_colors.items():
    if "dataset" in index.columns:
        mask = index["dataset"] == dataset_name
        if mask.sum() > 0:
            axes[2].hist(soh[mask.values], bins=30, alpha=0.6,
                         color=color,
                         label=f"{dataset_name} (n={mask.sum()})")

axes[2].axvline(0.75, color="red",  lw=1.5, linestyle="--",
                label="SOH=0.75")
axes[2].axvline(1.00, color="black", lw=0.8, linestyle="--")
axes[2].set_xlabel("SOH")
axes[2].set_ylabel("Count")
axes[2].set_title("SOH Distribution per Dataset")
axes[2].legend(fontsize=8); axes[2].grid(True, alpha=0.4)

plt.suptitle(f"Source Domain Fused Features ({FUSED_DIM}-dim)\n"
             f"[latent(24) | temp(1) | zeros(13)]  --  "
             f"{len(cell_sequences)} cells, {N} cycles",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "feature_summary.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> feature_summary.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FUSED FEATURE SUMMARY")
print("=" * 60)
print(f"  Total samples       : {N}")
print(f"  Total cells         : {len(cell_sequences)}")
print(f"  Fused vector dim    : {FUSED_DIM}")
print(f"    dims  0-23 : latent (normalised to [0,1])")
print(f"    dim   24   : temperature (normalised to [0,1])")
print(f"    dims 25-37 : zeros (pack features -- source domain)")
print(f"\n  Scalers saved:")
print(f"    latent_scaler.pkl  -- apply to drone latent features")
print(f"    temp_scaler.pkl    -- apply to drone temperature")
print(f"\n  Per-cell sequences:")
print(f"  {'Cell':12s}  {'Dataset':6s}  {'N':>5}  "
      f"{'SOH min':>8}  {'SOH max':>8}")
print(f"  {'-'*50}")
for s in cell_sequences:
    print(f"  {s['cell_id']:12s}  {s['dataset']:6s}  "
          f"{s['n_cycles']:>5}  "
          f"{s['soh'].min():>8.4f}  "
          f"{s['soh'].max():>8.4f}")

print(f"\n  Ready for LSTM pretraining:")
print(f"    Input  : cell_sequences.pkl")
print(f"             each cell = sequence of 38-dim vectors in cycle order")
print(f"    Target : SOH per cycle")
print(f"\nDone.")