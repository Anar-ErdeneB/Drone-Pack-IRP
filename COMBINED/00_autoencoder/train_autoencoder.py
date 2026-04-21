"""
train_autoencoder.py
====================
Trains the autoencoder on source domain charge voltage curves.
Architecture follows Fan et al. (2025) exactly.

Architecture:
    Encoder: fc(101) -> fc(496) -> fc(72) -> fc(24)[ReLU]
    Decoder: fc(24)  -> fc(72)  -> fc(496) -> fc(101)[Sigmoid]
    Activation: ReLU on all hidden layers
    Latent layer: ReLU (constrains latent space to non-negative)
    Output layer: Sigmoid (bounds output to [0,1] -- consistent
                  with normalised input)
    Loss: MSE between normalised input and reconstructed output

Input:
    ../processed/source_curves.npy    (N, 101) float32

Outputs (saved in processed/):
    autoencoder.pt          -- full autoencoder weights
    encoder.pt              -- encoder only (used downstream)
    voltage_scaler.pkl      -- MinMaxScaler fitted on train set
    source_latent.npy       -- (N, 24) encoded source curves
    training_log.csv        -- loss per epoch
    training_curves.png     -- train/val loss plot
    ae_reconstruction.png   -- sample reconstruction quality

Normalisation:
    Voltage curves normalised to [0,1] before training.
    Scaler fitted on TRAINING data only.
    Same scaler applied to val curves and later to drone curves.
    Scaler saved as voltage_scaler.pkl for downstream use.

Training:
    - 80/20 train/val split (random, per cell)
    - Adam optimizer, lr=1e-3
    - ReduceLROnPlateau (patience=50, factor=0.5, min_lr=1e-6)
    - Early stopping (patience=100)
    - Batch size=32
    - Max epochs=1500

Run from COMBINED/autoencoder/ directory:
    python train_autoencoder.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import joblib

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
DATA_DIR   = "../processed"
OUT_DIR    = "processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Architecture (Fan et al. 2025, Table 3)
INPUT_DIM  = 101
HIDDEN1    = 496
HIDDEN2    = 72
LATENT_DIM = 24

# Training
EPOCHS      = 1500
BATCH_SIZE  = 32
LR          = 1e-3
VAL_FRAC    = 0.20
SEED        = 42

# Scheduler
LR_PATIENCE = 50
LR_FACTOR   = 0.5
LR_MIN      = 1e-6

# Early stopping
ES_PATIENCE = 100

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# MODEL DEFINITION
# ══════════════════════════════════════════════════════════════════════════
class Encoder(nn.Module):
    """
    Encoder: fc(101) -> fc(496) -> fc(72) -> fc(24)[ReLU]
    ReLU on latent layer constrains latent space to non-negative values,
    making the decoder's job easier and the latent space more interpretable.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(INPUT_DIM, HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1,   HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2,   LATENT_DIM), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """
    Decoder: fc(24) -> fc(72) -> fc(496) -> fc(101)[Sigmoid]
    Sigmoid on output bounds reconstruction to [0,1],
    consistent with normalised input curves.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(LATENT_DIM, HIDDEN2), nn.ReLU(),
            nn.Linear(HIDDEN2,    HIDDEN1), nn.ReLU(),
            nn.Linear(HIDDEN1,    INPUT_DIM), nn.Sigmoid(),
        )

    def forward(self, z):
        return self.net(z)


class Autoencoder(nn.Module):
    """Full autoencoder: Encoder + Decoder."""
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))

    def encode(self, x):
        return self.encoder(x)


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Autoencoder Training -- Fan et al. (2025)")
print("=" * 60)

curves = np.load(os.path.join(DATA_DIR, "source_curves.npy"))
ids    = np.load(os.path.join(DATA_DIR, "source_cell_ids.npy"),
                 allow_pickle=True)

print(f"\nLoaded:")
print(f"  source_curves.npy : {curves.shape}")
print(f"  Voltage range     : {curves.min():.4f} - {curves.max():.4f} V")
print(f"  Unique cells      : {len(np.unique(ids))}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: TRAIN/VAL SPLIT (per cell, 80/20)
# ══════════════════════════════════════════════════════════════════════════
print(f"\nSplitting train/val ({int((1-VAL_FRAC)*100)}/{int(VAL_FRAC*100)}) ...")

train_idx = []
val_idx   = []

np.random.seed(SEED)
for cell_id in np.unique(ids):
    cell_mask    = np.where(ids == cell_id)[0]
    n_val        = max(1, int(len(cell_mask) * VAL_FRAC))
    val_choose   = np.random.choice(cell_mask, n_val, replace=False)
    train_choose = np.setdiff1d(cell_mask, val_choose)
    train_idx.extend(train_choose.tolist())
    val_idx.extend(val_choose.tolist())

train_idx = np.array(train_idx)
val_idx   = np.array(val_idx)

X_train_raw = curves[train_idx]
X_val_raw   = curves[val_idx]

print(f"  Train samples : {len(X_train_raw)}")
print(f"  Val samples   : {len(X_val_raw)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: NORMALISE TO [0, 1]
# Scaler fitted on TRAINING data only -- applied to val and drone later
# ══════════════════════════════════════════════════════════════════════════
print(f"\nNormalising voltage curves to [0, 1] ...")

scaler = MinMaxScaler()

# Fit on train only -- reshape to (N*101, 1) for sklearn
X_train_scaled = scaler.fit_transform(
    X_train_raw.reshape(-1, 1)).reshape(X_train_raw.shape).astype(np.float32)

# Transform val using same scaler
X_val_scaled = scaler.transform(
    X_val_raw.reshape(-1, 1)).reshape(X_val_raw.shape).astype(np.float32)

# Transform all curves for final encoding step
X_all_scaled = scaler.transform(
    curves.reshape(-1, 1)).reshape(curves.shape).astype(np.float32)

print(f"  Raw range    : {curves.min():.4f} - {curves.max():.4f} V")
print(f"  Train scaled : {X_train_scaled.min():.4f} - "
      f"{X_train_scaled.max():.4f}")
print(f"  Val scaled   : {X_val_scaled.min():.4f} - "
      f"{X_val_scaled.max():.4f}")

# Save scaler -- needed for drone curves normalisation
joblib.dump(scaler, os.path.join(OUT_DIR, "voltage_scaler.pkl"))
print(f"  Saved -> voltage_scaler.pkl")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: BUILD DATALOADERS
# ══════════════════════════════════════════════════════════════════════════
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled,   dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train_t, X_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, X_val_t),
    batch_size=BATCH_SIZE, shuffle=False
)

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: INITIALISE MODEL
# ══════════════════════════════════════════════════════════════════════════
model     = Autoencoder().to(DEVICE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min",
                               patience=LR_PATIENCE,
                               factor=LR_FACTOR,
                               min_lr=LR_MIN)

n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"\nModel parameters : {n_params:,}")
print(f"Architecture     : fc({INPUT_DIM})->fc({HIDDEN1})->fc({HIDDEN2})"
      f"->fc({LATENT_DIM})[ReLU]->fc({HIDDEN2})->fc({HIDDEN1})"
      f"->fc({INPUT_DIM})[Sigmoid]")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════
print(f"\nTraining (max {EPOCHS} epochs, "
      f"early stopping patience={ES_PATIENCE}) ...")
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  "
      f"{'LR':>10}")
print("-" * 48)

train_losses = []
val_losses   = []
best_val     = float("inf")
best_epoch   = 0
es_counter   = 0
best_weights = None

for epoch in range(1, EPOCHS + 1):

    # ── Train ──────────────────────────────────────────────────────────
    model.train()
    train_loss = 0.0
    for x_batch, _ in train_loader:
        x_batch = x_batch.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), x_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(x_batch)
    train_loss /= len(X_train_t)

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, _ in val_loader:
            x_batch = x_batch.to(DEVICE)
            val_loss += criterion(model(x_batch),
                                  x_batch).item() * len(x_batch)
    val_loss /= len(X_val_t)

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # ── Early stopping ─────────────────────────────────────────────────
    if val_loss < best_val:
        best_val     = val_loss
        best_epoch   = epoch
        es_counter   = 0
        best_weights = {k: v.cpu().clone()
                        for k, v in model.state_dict().items()}
        marker = " <-- best"
    else:
        es_counter += 1
        marker = ""

    if epoch % 50 == 0 or epoch == 1 or marker:
        print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
              f"{current_lr:>10.2e}{marker}")

    if es_counter >= ES_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(best at epoch {best_epoch})")
        break

model.load_state_dict(best_weights)
print(f"\nRestored best weights from epoch {best_epoch}")
print(f"Best val MSE : {best_val:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE MODEL
# ══════════════════════════════════════════════════════════════════════════
torch.save(model.state_dict(),
           os.path.join(OUT_DIR, "autoencoder.pt"))
torch.save(model.encoder.state_dict(),
           os.path.join(OUT_DIR, "encoder.pt"))
print(f"\nSaved -> autoencoder.pt")
print(f"Saved -> encoder.pt")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: ENCODE ALL SOURCE CURVES
# ══════════════════════════════════════════════════════════════════════════
print(f"\nEncoding all source curves ...")

all_t       = torch.tensor(X_all_scaled, dtype=torch.float32)
latent_list = []
model.eval()
with torch.no_grad():
    for i in range(0, len(all_t), 256):
        batch  = all_t[i:i+256].to(DEVICE)
        latent = model.encode(batch).cpu().numpy()
        latent_list.append(latent)

source_latent = np.concatenate(latent_list, axis=0).astype(np.float32)
np.save(os.path.join(OUT_DIR, "source_latent.npy"), source_latent)

print(f"  source_latent.npy : {source_latent.shape}")
print(f"  Latent range      : {source_latent.min():.4f} - "
      f"{source_latent.max():.4f}")
print(f"Saved -> source_latent.npy")

# ══════════════════════════════════════════════════════════════════════════
# STEP 9: SAVE TRAINING LOG
# ══════════════════════════════════════════════════════════════════════════
log_df = pd.DataFrame({
    "epoch"     : range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss"  : val_losses,
})
log_df.to_csv(os.path.join(OUT_DIR, "training_log.csv"), index=False)
print(f"Saved -> training_log.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 10: PLOTS
# ══════════════════════════════════════════════════════════════════════════

# ── Plot A: Training curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_r = range(1, len(train_losses) + 1)

axes[0].plot(epochs_r, train_losses, lw=1.5,
             color="steelblue", label="Train loss")
axes[0].plot(epochs_r, val_losses, lw=1.5,
             color="darkorange", label="Val loss")
axes[0].axvline(best_epoch, color="red", lw=1,
                linestyle="--", label=f"Best epoch {best_epoch}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
axes[0].set_title("Training Loss (log scale)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)
axes[0].set_yscale("log")

# Zoomed last 20% of training
start_zoom = max(0, len(train_losses) - len(train_losses)//5)
axes[1].plot(list(epochs_r)[start_zoom:],
             train_losses[start_zoom:], lw=1.5,
             color="steelblue", label="Train loss")
axes[1].plot(list(epochs_r)[start_zoom:],
             val_losses[start_zoom:], lw=1.5,
             color="darkorange", label="Val loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MSE Loss")
axes[1].set_title(f"Last {len(train_losses)//5} epochs (zoomed)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

plt.suptitle(f"Autoencoder Training  |  "
             f"Best val MSE={best_val:.6f} at epoch {best_epoch}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> training_curves.png")

# ── Plot B: Reconstruction quality ────────────────────────────────────────
soh = np.load(os.path.join(DATA_DIR, "source_soh.npy"))

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes      = axes.flatten()
x_axis    = np.linspace(0, 1, INPUT_DIM)

soh_sorted_idx = np.argsort(soh)
sample_indices = soh_sorted_idx[
    np.linspace(0, len(soh)-1, 8, dtype=int)
]

model.eval()
with torch.no_grad():
    for plot_idx, sample_idx in enumerate(sample_indices):
        orig   = X_all_scaled[sample_idx]
        orig_t = torch.tensor(orig,
                               dtype=torch.float32).unsqueeze(0).to(DEVICE)
        recon  = model(orig_t).cpu().numpy().squeeze()
        mse    = float(np.mean((orig - recon)**2))

        axes[plot_idx].plot(x_axis, orig,  "b-",
                            lw=1.5, label="Original")
        axes[plot_idx].plot(x_axis, recon, "r--",
                            lw=1.5, label="Reconstructed")

        # Also show residual
        residual = orig - recon
        ax2 = axes[plot_idx].twinx()
        ax2.plot(x_axis, residual, "g-", lw=0.8, alpha=0.5)
        ax2.axhline(0, color="gray", lw=0.5, linestyle="--")
        ax2.set_ylabel("Residual", fontsize=6, color="green")
        ax2.tick_params(axis="y", labelsize=6, colors="green")

        axes[plot_idx].set_title(
            f"Sample {sample_idx}  SOH={soh[sample_idx]:.3f}\n"
            f"MSE={mse:.5f}",
            fontsize=8)
        axes[plot_idx].set_xlabel("<- End    Start ->", fontsize=7)
        axes[plot_idx].set_ylabel("Voltage (normalised)", fontsize=7)
        axes[plot_idx].legend(fontsize=7)
        axes[plot_idx].grid(True, alpha=0.3)

plt.suptitle("Autoencoder Reconstruction Quality\n"
             "(8 samples spread across SOH range, "
             "curves normalised to [0,1])",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "ae_reconstruction.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> ae_reconstruction.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 11: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("AUTOENCODER TRAINING SUMMARY")
print("=" * 60)
print(f"  Architecture     : fc({INPUT_DIM})->fc({HIDDEN1})->fc({HIDDEN2})"
      f"->fc({LATENT_DIM})[ReLU]->fc({HIDDEN2})->"
      f"fc({HIDDEN1})->fc({INPUT_DIM})[Sigmoid]")
print(f"  Parameters       : {n_params:,}")
print(f"  Train samples    : {len(X_train_t)}")
print(f"  Val samples      : {len(X_val_t)}")
print(f"  Epochs trained   : {len(train_losses)}")
print(f"  Best epoch       : {best_epoch}")
print(f"  Best val MSE     : {best_val:.6f}")
print(f"  Final train MSE  : {train_losses[-1]:.6f}")
print(f"\n  Output files:")
print(f"    autoencoder.pt       -- full model weights")
print(f"    encoder.pt           -- encoder only (use downstream)")
print(f"    voltage_scaler.pkl   -- scaler for drone curves")
print(f"    source_latent.npy    {source_latent.shape}")
print(f"    training_log.csv")
print(f"    training_curves.png")
print(f"    ae_reconstruction.png")
print(f"\n  Next step:")
print(f"    Encode drone curves using encoder.pt + voltage_scaler.pkl")
print(f"    Then build 38-dim fused vectors for LSTM/Transformer/GRU")
print("\nDone.")