"""
pretrain_lstm.py
================
Pretrains the LSTM model on source domain cell sequences.
Architecture and loss function follow Fan et al. (2025) exactly.

Architecture (Fan et al. 2025, Table 3):
    fc(38) -> lstm(128) -> fc(64) -> fc(8) -> fc(1)

Loss function (Fan et al. 2025, Eq. 8):
    L = L_mse + lambda1 * L_emse

    L_mse  = mean squared error over all predictions
    L_emse = mean squared error computed ONLY over predictions
             where the individual squared error exceeds L_mse.
             This penalises large errors more heavily without
             the instability of exponential weighting.
    lambda1 = 1.0 (from paper)

Training:
    - Each cell is one sequence fed in temporal order (cycle 1->N)
    - All train cells processed per epoch, cell order shuffled
    - Val split: one cell per dataset (NASA, CALCE, LCO) for diversity
    - Adam optimizer, lr=1e-3, max 1500 epochs
    - ReduceLROnPlateau + early stopping + gradient clipping

Input (from ../01_fusevector/processed/):
    cell_sequences.pkl      list of 11 per-cell dicts

Outputs (saved in processed/):
    lstm_pretrained.pt      pretrained weights
    lstm_training_log.csv   loss per epoch
    lstm_cell_results.csv   RMSE per cell
    lstm_training_curves.png
    lstm_predictions.png

Run from COMBINED/02_pretrain_lstm/ directory:
    python pretrain_lstm.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
SEQ_DIR  = "../01a_cell_input/processed"
OUT_DIR  = "00_processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Architecture (Fan et al. 2025, Table 3)
FUSED_DIM  = 38
HIDDEN_DIM = 128

# Loss (Fan et al. 2025, Eq. 8)
LAMBDA1 = 1.0

# Training
EPOCHS      = 1500
LR          = 1e-3
SEED        = 42
LR_PATIENCE = 50
LR_FACTOR   = 0.5
LR_MIN      = 1e-6
ES_PATIENCE = 150

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ══════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════
class LSTMModel(nn.Module):
    """
    LSTM SOH estimator -- Fan et al. (2025) Table 3.
    fc(38) -> lstm(128) -> fc(64) -> fc(8) -> fc(1)
    No sigmoid on output -- loss function handles scale.
    """
    def __init__(self, input_dim=38, hidden_dim=128):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, input_dim)
        self.lstm     = nn.LSTM(input_size=input_dim,
                                hidden_size=hidden_dim,
                                num_layers=1,
                                batch_first=True)
        self.head     = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(),
            nn.Linear(64,          8), nn.ReLU(),
            nn.Linear(8,           1),
        )

    def forward(self, x):
        """
        x      : (seq_len, 38)
        returns: (seq_len, 1)
        """
        x           = torch.relu(self.input_fc(x))
        out, _      = self.lstm(x.unsqueeze(0))
        return self.head(out.squeeze(0))


# ══════════════════════════════════════════════════════════════════════════
# LOSS FUNCTION (Fan et al. 2025, Eq. 8)
# ══════════════════════════════════════════════════════════════════════════
def composite_loss(pred, target, lambda1=LAMBDA1):
    """
    Fan et al. (2025) Eq. 8:
        L = L_mse + lambda1 * L_emse

    L_mse  = standard MSE over all predictions
    L_emse = MSE computed only over predictions where the
             individual squared error exceeds L_mse.
             Penalises large errors without exponential instability.
    """
    errors_sq = (pred - target) ** 2
    l_mse     = errors_sq.mean()

    large_mask = errors_sq > l_mse.detach()
    if large_mask.sum() > 0:
        l_emse = errors_sq[large_mask].mean()
    else:
        l_emse = torch.tensor(0.0, device=pred.device)

    return l_mse + lambda1 * l_emse, l_mse, l_emse


# ══════════════════════════════════════════════════════════════════════════
# STEP 1: LOAD
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("LSTM Pretraining -- Fan et al. (2025)")
print("=" * 60)

with open(os.path.join(SEQ_DIR, "cell_sequences.pkl"), "rb") as f:
    cell_sequences = pickle.load(f)

print(f"\nLoaded {len(cell_sequences)} cell sequences:")
for s in cell_sequences:
    print(f"  {s['cell_id']:12s} ({s['dataset']:6s}): "
          f"{s['n_cycles']:4d} cycles  "
          f"SOH {s['soh'].min():.3f}-{s['soh'].max():.3f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: DATASET-DIVERSE TRAIN/VAL SPLIT
# Take the last cell from each dataset as validation
# This ensures val covers all 3 chemistries/datasets
# ══════════════════════════════════════════════════════════════════════════
datasets      = ["NASA", "CALCE"]   # LCO excluded from val -- only 1 cell, use for training
val_indices   = []
train_indices = []

for ds in datasets:
    ds_cells = [i for i, s in enumerate(cell_sequences)
                if s["dataset"] == ds]
    if ds_cells:
        val_indices.append(ds_cells[-1])  # last cell per dataset

train_indices = [i for i in range(len(cell_sequences))
                 if i not in val_indices]

train_seqs = [cell_sequences[i] for i in train_indices]
val_seqs   = [cell_sequences[i] for i in val_indices]

print(f"\nDataset-diverse train/val split:")
print(f"  Train ({len(train_seqs)} cells): "
      f"{[s['cell_id'] for s in train_seqs]}")
print(f"  Val   ({len(val_seqs)} cells): "
      f"{[s['cell_id'] for s in val_seqs]}")
print(f"  Train cycles : {sum(s['n_cycles'] for s in train_seqs)}")
print(f"  Val cycles   : {sum(s['n_cycles'] for s in val_seqs)}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: INITIALISE MODEL
# ══════════════════════════════════════════════════════════════════════════
model     = LSTMModel(FUSED_DIM, HIDDEN_DIM).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min",
                               patience=LR_PATIENCE,
                               factor=LR_FACTOR,
                               min_lr=LR_MIN)

n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"\nModel parameters : {n_params:,}")
print(f"Architecture     : fc({FUSED_DIM}) -> lstm({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")
print(f"Loss             : L_mse + {LAMBDA1} * L_emse  (Fan et al. Eq.8)")

# ══════════════════════════════════════════════════════════════════════════
# STEP 4: TRAINING LOOP
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
    epoch_loss = 0.0
    total_pts  = 0

    perm = np.random.permutation(len(train_seqs))
    for idx in perm:
        seq    = train_seqs[idx]
        X      = torch.tensor(seq["features"],
                               dtype=torch.float32).to(DEVICE)
        y_true = torch.tensor(seq["soh"],
                               dtype=torch.float32).unsqueeze(1).to(DEVICE)

        optimizer.zero_grad()
        y_pred        = model(X)
        loss, _, _    = composite_loss(y_pred, y_true)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item() * len(X)
        total_pts  += len(X)

    train_loss = epoch_loss / total_pts

    # ── Validate ───────────────────────────────────────────────────────
    model.eval()
    val_loss_sum = 0.0
    val_pts      = 0

    with torch.no_grad():
        for seq in val_seqs:
            X      = torch.tensor(seq["features"],
                                   dtype=torch.float32).to(DEVICE)
            y_true = torch.tensor(seq["soh"],
                                   dtype=torch.float32).unsqueeze(1).to(DEVICE)
            y_pred = model(X)
            loss, _, _ = composite_loss(y_pred, y_true)
            val_loss_sum += loss.item() * len(X)
            val_pts      += len(X)

    val_loss = val_loss_sum / val_pts

    scheduler.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    current_lr = optimizer.param_groups[0]["lr"]

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
print(f"Best val loss : {best_val:.6f}")

# ══════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE
# ══════════════════════════════════════════════════════════════════════════
torch.save(model.state_dict(),
           os.path.join(OUT_DIR, "lstm_pretrained.pt"))
print(f"\nSaved -> lstm_pretrained.pt")

log_df = pd.DataFrame({
    "epoch"     : range(1, len(train_losses) + 1),
    "train_loss": train_losses,
    "val_loss"  : val_losses,
})
log_df.to_csv(os.path.join(OUT_DIR, "lstm_training_log.csv"), index=False)
print(f"Saved -> lstm_training_log.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 6: EVALUATE ON ALL CELLS
# ══════════════════════════════════════════════════════════════════════════
print(f"\nEvaluating on all cells ...")
model.eval()
all_results = []

with torch.no_grad():
    for seq in cell_sequences:
        X      = torch.tensor(seq["features"],
                               dtype=torch.float32).to(DEVICE)
        y_true = seq["soh"]
        y_pred = model(X).cpu().numpy().squeeze()

        rmse = float(np.sqrt(np.mean((y_pred - y_true)**2)))
        mae  = float(np.mean(np.abs(y_pred - y_true)))
        split = "val" if seq["cell_id"] in \
                [s["cell_id"] for s in val_seqs] else "train"

        all_results.append({
            "cell_id" : seq["cell_id"],
            "dataset" : seq["dataset"],
            "split"   : split,
            "n_cycles": seq["n_cycles"],
            "rmse"    : rmse,
            "mae"     : mae,
        })
        print(f"  {seq['cell_id']:12s} [{split:5s}]: "
              f"RMSE={rmse:.4f}  MAE={mae:.4f}")

results_df = pd.DataFrame(all_results)
results_df.to_csv(os.path.join(OUT_DIR, "lstm_cell_results.csv"),
                  index=False)

train_rmse = results_df[results_df["split"]=="train"]["rmse"].mean()
val_rmse   = results_df[results_df["split"]=="val"]["rmse"].mean()
print(f"\n  Mean train RMSE : {train_rmse:.4f}")
print(f"  Mean val RMSE   : {val_rmse:.4f}")
print(f"Saved -> lstm_cell_results.csv")

# ══════════════════════════════════════════════════════════════════════════
# STEP 7: PLOTS
# ══════════════════════════════════════════════════════════════════════════

# ── Plot A: Training curves ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
epochs_r  = range(1, len(train_losses) + 1)

axes[0].plot(epochs_r, train_losses, lw=1.5,
             color="steelblue", label="Train loss")
axes[0].plot(epochs_r, val_losses,   lw=1.5,
             color="darkorange", label="Val loss")
axes[0].axvline(best_epoch, color="red", lw=1, linestyle="--",
                label=f"Best epoch {best_epoch}")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("LSTM Training Loss (log scale)")
axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)
axes[0].set_yscale("log")

start_zoom = max(0, len(train_losses) - len(train_losses)//5)
axes[1].plot(list(epochs_r)[start_zoom:],
             train_losses[start_zoom:], lw=1.5,
             color="steelblue", label="Train loss")
axes[1].plot(list(epochs_r)[start_zoom:],
             val_losses[start_zoom:], lw=1.5,
             color="darkorange", label="Val loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title(f"Last {len(train_losses)//5} epochs (zoomed)")
axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)

plt.suptitle(f"LSTM Pretraining  |  "
             f"Best val loss={best_val:.6f} at epoch {best_epoch}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lstm_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> lstm_training_curves.png")

# ── Plot B: SOH predictions per cell ──────────────────────────────────────
n_cells_plot = len(cell_sequences)
ncols        = 4
nrows        = int(np.ceil(n_cells_plot / ncols))
fig, axes    = plt.subplots(nrows, ncols,
                             figsize=(ncols*5, nrows*4))
axes         = axes.flatten()

model.eval()
with torch.no_grad():
    for plot_idx, seq in enumerate(cell_sequences):
        ax     = axes[plot_idx]
        X      = torch.tensor(seq["features"],
                               dtype=torch.float32).to(DEVICE)
        y_true = seq["soh"]
        y_pred = model(X).cpu().numpy().squeeze()
        cycles = np.arange(1, len(y_true) + 1)
        rmse   = float(np.sqrt(np.mean((y_pred - y_true)**2)))

        split = "val" if seq["cell_id"] in \
                [s["cell_id"] for s in val_seqs] else "train"
        color = "darkorange" if split == "val" else "steelblue"

        ax.plot(cycles, y_true, "k-",  lw=1.5, label="True SOH")
        ax.plot(cycles, y_pred, "--",  lw=1.5,
                color=color, label=f"Predicted [{split}]")
        ax.fill_between(cycles, y_true, y_pred,
                         alpha=0.15, color=color)
        ax.set_title(f"{seq['cell_id']} ({seq['dataset']})\n"
                     f"RMSE={rmse:.4f}  n={seq['n_cycles']}",
                     fontsize=8)
        ax.set_xlabel("Cycle", fontsize=7)
        ax.set_ylabel("SOH",   fontsize=7)
        ax.legend(fontsize=6); ax.grid(True, alpha=0.3)
        ax.set_ylim(0.70, 1.05)

for j in range(n_cells_plot, len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f"LSTM Pretraining -- SOH Predictions\n"
             f"(blue=train, orange=val)  "
             f"Mean train RMSE={train_rmse:.4f}  "
             f"Mean val RMSE={val_rmse:.4f}",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lstm_predictions.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> lstm_predictions.png")

# ── Plot C: Predicted vs True scatter ─────────────────────────────────────
all_true_vals, all_pred_vals = [], []
model.eval()
with torch.no_grad():
    for seq in cell_sequences:
        X = torch.tensor(seq["features"],
                          dtype=torch.float32).to(DEVICE)
        y_pred = model(X).cpu().numpy().squeeze()
        all_true_vals.extend(seq["soh"].tolist())
        all_pred_vals.extend(y_pred.tolist())

all_true_vals = np.array(all_true_vals)
all_pred_vals = np.array(all_pred_vals)
overall_rmse  = float(np.sqrt(np.mean((all_pred_vals - all_true_vals)**2)))
overall_r2    = float(1 - np.sum((all_pred_vals - all_true_vals)**2) /
                      np.sum((all_true_vals - all_true_vals.mean())**2))

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(all_true_vals, all_pred_vals, s=2, alpha=0.3,
           color="steelblue")
lims = [all_true_vals.min(), all_true_vals.max()]
ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
ax.set_xlabel("True SOH"); ax.set_ylabel("Predicted SOH")
ax.set_title(f"Predicted vs True SOH\n"
             f"RMSE={overall_rmse:.4f}  R²={overall_r2:.4f}")
ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lstm_scatter.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> lstm_scatter.png")

# ══════════════════════════════════════════════════════════════════════════
# STEP 8: FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("LSTM PRETRAINING SUMMARY")
print("=" * 60)
print(f"  Architecture   : fc({FUSED_DIM}) -> lstm({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")
print(f"  Parameters     : {n_params:,}")
print(f"  Loss           : L_mse + {LAMBDA1} * L_emse (Fan et al. Eq.8)")
print(f"  Epochs trained : {len(train_losses)}")
print(f"  Best epoch     : {best_epoch}")
print(f"  Best val loss  : {best_val:.6f}")
print(f"  Mean train RMSE: {train_rmse:.4f}")
print(f"  Mean val RMSE  : {val_rmse:.4f}")
print(f"  Overall RMSE   : {overall_rmse:.4f}")
print(f"  Overall R²     : {overall_r2:.4f}")
print(f"\n  Val cells (one per dataset):")
for s in val_seqs:
    print(f"    {s['cell_id']:12s} ({s['dataset']})")
print(f"\n  Output files:")
print(f"    lstm_pretrained.pt")
print(f"    lstm_training_log.csv")
print(f"    lstm_cell_results.csv")
print(f"    lstm_training_curves.png")
print(f"    lstm_predictions.png")
print(f"    lstm_scatter.png")
print(f"\n  Next step:")
print(f"    Fine-tune lstm_pretrained.pt on drone pack data")
print("\nDone.")