"""
pretrain_gru.py
===============
Pretrains the GRU model on source domain cell sequences.
GRU replaces LSTM from Fan et al. (2025) -- thesis contribution.

Architecture:
    fc(38) -> gru(128) -> fc(64) -> fc(8) -> fc(1)

Loss function (Fan et al. 2025, Eq. 8):
    L = L_mse   (pure MSE; composite loss used only at fine-tune stage)

Training:
    - Each cell is one sequence fed in temporal order (cycle 1->N)
    - All train cells processed per epoch, cell order shuffled
    - Val split: last cell from NASA and CALCE for diversity
    - AdamW optimizer, lr=2e-4, max 3000 epochs
    - ReduceLROnPlateau + early stopping + gradient clipping
    - NOISE_STD=0.003 per-feature noise for regularisation

Reads (from ../01a_cell_input/processed/):
    cell_sequences.pkl

Writes (to 00_processed/):
    gru_pretrained.pt
    gru_training_curves.png
    gru_predictions.png
    gru_scatter.png
"""

import os
import math
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
SEQ_DIR  = "../01a_cell_input/processed"
OUT_DIR  = "00_processed"
os.makedirs(OUT_DIR, exist_ok=True)

FUSED_DIM  = 38
HIDDEN_DIM = 128

LAMBDA1      = 0.0        # pure MSE for pretraining
EPOCHS       = 3000
LR           = 2e-4
WEIGHT_DECAY = 2e-4
NOISE_STD    = 0.003

SEED         = 42
LR_PATIENCE  = 50
LR_FACTOR    = 0.5
LR_MIN       = 1e-6
ES_PATIENCE  = 300

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════
class GRUModel(nn.Module):
    """
    GRU SOH estimator.
    Architecture mirrors Fan et al. (2025) LSTM (Table 3),
    with nn.LSTM replaced by nn.GRU.
        fc(38) -> gru(128) -> fc(64) -> fc(8) -> fc(1)
    """
    def __init__(self, input_dim=FUSED_DIM, hidden_dim=HIDDEN_DIM):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, input_dim)
        self.gru      = nn.GRU(input_size=input_dim,
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
        x      : (seq_len, FUSED_DIM)  — one cell, T cycles
        returns: (seq_len, 1)
        """
        x      = torch.relu(self.input_fc(x))
        out, _ = self.gru(x.unsqueeze(0))          # (1, T, hidden)
        return self.head(out.squeeze(0))            # (T, 1)


# ══════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════
def composite_loss(pred, target):
    errors_sq = (pred - target) ** 2
    l_mse     = errors_sq.mean()
    if LAMBDA1 > 0:
        mask   = errors_sq > l_mse.detach()
        l_emse = errors_sq[mask].mean() if mask.sum() > 0 \
                 else torch.tensor(0.0, device=pred.device)
        return l_mse + LAMBDA1 * l_emse
    return l_mse


# ══════════════════════════════════════════════════════════════
# DATA
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("GRU Pretraining")
print("=" * 60)

with open(os.path.join(SEQ_DIR, "cell_sequences.pkl"), "rb") as f:
    cell_sequences = pickle.load(f)

print(f"\nLoaded {len(cell_sequences)} cell sequences:")
for s in cell_sequences:
    print(f"  {s['cell_id']:12s} ({s['dataset']:6s}): "
          f"{s['n_cycles']:4d} cycles  "
          f"SOH {min(s['soh']):.3f}-{max(s['soh']):.3f}")

# Dataset-diverse val split (last cell from NASA and CALCE)
val_idx = []
for ds in ["NASA", "CALCE"]:
    idx = [i for i, s in enumerate(cell_sequences) if s["dataset"] == ds]
    if idx:
        val_idx.append(idx[-1])

train_seqs = [s for i, s in enumerate(cell_sequences) if i not in val_idx]
val_seqs   = [s for i, s in enumerate(cell_sequences) if i in val_idx]

print(f"\nTrain ({len(train_seqs)} cells): "
      f"{[s['cell_id'] for s in train_seqs]}")
print(f"Val   ({len(val_seqs)}  cells): "
      f"{[s['cell_id'] for s in val_seqs]}")


# ══════════════════════════════════════════════════════════════
# MODEL + OPTIMISER
# ══════════════════════════════════════════════════════════════
model = GRUModel().to(DEVICE)
opt   = torch.optim.AdamW(model.parameters(),
                           lr=LR, weight_decay=WEIGHT_DECAY)
sched = ReduceLROnPlateau(opt, patience=LR_PATIENCE,
                           factor=LR_FACTOR, min_lr=LR_MIN)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel parameters : {n_params:,}")
print(f"Architecture     : fc({FUSED_DIM}) -> gru({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")


# ══════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════
train_losses, val_losses = [], []
best_val = 1e9
best_w   = None
es       = 0

for epoch in range(EPOCHS):

    model.train()
    np.random.shuffle(train_seqs)

    epoch_loss = 0.0
    for s in train_seqs:
        X = torch.tensor(s["features"], dtype=torch.float32).to(DEVICE)
        Y = torch.tensor(s["soh"],      dtype=torch.float32).to(DEVICE)

        X = X + NOISE_STD * torch.randn_like(X)

        pred = model(X).squeeze(-1)          # (T,)
        loss = composite_loss(pred, Y)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        epoch_loss += loss.item()

    train_loss = epoch_loss / len(train_seqs)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for s in val_seqs:
            X = torch.tensor(s["features"], dtype=torch.float32).to(DEVICE)
            Y = torch.tensor(s["soh"],      dtype=torch.float32).to(DEVICE)
            pred = model(X).squeeze(-1)
            val_loss += composite_loss(pred, Y).item()
    val_loss /= max(1, len(val_seqs))

    sched.step(val_loss)
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        best_w   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        es       = 0
    else:
        es += 1

    if epoch % 50 == 0:
        print(epoch, train_loss, val_loss)

    if es > ES_PATIENCE:
        break

model.load_state_dict(best_w)
best_epoch = int(np.argmin(val_losses)) + 1
print(f"\nBest val={best_val:.6f} at epoch {best_epoch}")


# ══════════════════════════════════════════════════════════════
# SAVE CHECKPOINT
# ══════════════════════════════════════════════════════════════
torch.save(model.state_dict(),
           os.path.join(OUT_DIR, "gru_pretrained.pt"))


# ══════════════════════════════════════════════════════════════
# METRICS HELPER
# ══════════════════════════════════════════════════════════════
def calc_metrics(true, pred):
    true, pred = np.array(true), np.array(pred)
    rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
    mae  = float(np.mean(np.abs(pred - true)))
    r2   = float(1 - np.sum((pred - true) ** 2) /
                     np.sum((true - true.mean()) ** 2))
    return rmse, mae, r2

# Build prediction cache
model.eval()
pred_cache = {}
with torch.no_grad():
    for s in cell_sequences:
        X = torch.tensor(s["features"], dtype=torch.float32).to(DEVICE)
        pred_cache[s["cell_id"]] = model(X).cpu().numpy().squeeze()

train_ids = {s["cell_id"] for i, s in enumerate(cell_sequences)
             if i not in val_idx}
val_ids   = {s["cell_id"] for i, s in enumerate(cell_sequences)
             if i in val_idx}

true_tr, pred_tr, true_va, pred_va = [], [], [], []
for s in cell_sequences:
    y = np.array(s["soh"])
    p = pred_cache[s["cell_id"]]
    if s["cell_id"] in train_ids:
        true_tr.extend(y); pred_tr.extend(p)
    else:
        true_va.extend(y); pred_va.extend(p)

rmse_tr, mae_tr, r2_tr = calc_metrics(true_tr, pred_tr)
rmse_va, mae_va, r2_va = calc_metrics(true_va, pred_va)
true_all = np.concatenate([true_tr, true_va])
pred_all = np.concatenate([pred_tr, pred_va])
rmse_all, mae_all, r2_all = calc_metrics(true_all, pred_all)

print(f"\nPretraining metrics:")
print(f"  Train : RMSE={rmse_tr:.4f}  MAE={mae_tr:.4f}  R²={r2_tr:.4f}")
print(f"  Val   : RMSE={rmse_va:.4f}  MAE={mae_va:.4f}  R²={r2_va:.4f}")
print(f"  All   : RMSE={rmse_all:.4f}  MAE={mae_all:.4f}  R²={r2_all:.4f}")


# ══════════════════════════════════════════════════════════════
# PLOT 1: TRAINING CURVES
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].plot(train_losses, label="Train loss")
axes[0].plot(val_losses,   label="Val loss")
axes[0].axvline(best_epoch - 1, color="red", linestyle="--",
                label=f"Best epoch {best_epoch}")
axes[0].set_yscale("log")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("Training Loss (log scale)")
axes[0].legend(); axes[0].grid()

zoom_start = max(0, len(train_losses) - 33)
axes[1].plot(range(zoom_start + 1, len(train_losses) + 1),
             train_losses[zoom_start:], label="Train loss")
axes[1].plot(range(zoom_start + 1, len(val_losses) + 1),
             val_losses[zoom_start:],   label="Val loss")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
axes[1].set_title(f"Last {len(train_losses) - zoom_start} epochs (zoomed)")
axes[1].legend(); axes[1].grid()

fig.suptitle(
    f"GRU Pretraining  |  Best val={best_val:.6f} at epoch {best_epoch}\n"
    f"Train RMSE={rmse_tr:.4f}  R²={r2_tr:.4f}    "
    f"Val RMSE={rmse_va:.4f}  R²={r2_va:.4f}",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gru_training_curves.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════
# PLOT 2: PER-CELL PREDICTIONS
# ══════════════════════════════════════════════════════════════
n    = len(cell_sequences)
cols = 4
rows = math.ceil(n / cols)

fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 3))
axes = axes.flatten()

for i, s in enumerate(cell_sequences):
    y = np.array(s["soh"])
    p = pred_cache[s["cell_id"]]
    rmse_c, _, r2_c = calc_metrics(y, p)
    split = "Val" if s["cell_id"] in val_ids else "Train"
    color = "orange" if split == "Val" else None

    axes[i].plot(np.arange(len(y)), y, "k-",  lw=1.2, label="True")
    axes[i].plot(np.arange(len(p)), p, "--", lw=1.2,
                 label="Pred", color=color)
    axes[i].set_title(
        f"{s['cell_id']} ({split})\nRMSE={rmse_c:.4f}  R²={r2_c:.4f}",
        fontsize=8)
    axes[i].legend(fontsize=7); axes[i].grid()

for j in range(i + 1, len(axes)):
    axes[j].axis("off")

fig.suptitle(
    f"GRU Pretraining Predictions  —  "
    f"train RMSE={rmse_tr:.4f}  val RMSE={rmse_va:.4f}",
    fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gru_predictions.png"),
            dpi=150, bbox_inches="tight")
plt.close()


# ══════════════════════════════════════════════════════════════
# PLOT 3: SCATTER (TRAIN vs VAL)
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(true_tr, pred_tr, s=4,  alpha=0.5,
           label=f"Train  R²={r2_tr:.4f}")
ax.scatter(true_va, pred_va, s=8,  alpha=0.8,
           color="orange", label=f"Val  R²={r2_va:.4f}")
lims = [true_all.min() - 0.01, true_all.max() + 0.01]
ax.plot(lims, lims, "r--", lw=1.2, label="Perfect")
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True SOH"); ax.set_ylabel("Predicted SOH")
ax.set_title(
    f"Predicted vs True SOH\n"
    f"Overall  RMSE={rmse_all:.4f}  MAE={mae_all:.4f}  R²={r2_all:.4f}",
    fontsize=10)
ax.legend(); ax.grid()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gru_scatter.png"),
            dpi=150, bbox_inches="tight")
plt.close()


print("\nSaved all outputs:")
print(f" - gru_training_curves.png")
print(f" - gru_predictions.png")
print(f" - gru_scatter.png")
print(f" - gru_pretrained.pt")
print(f"\nDone.")
