# """
# pretrain_transformer.py
# =======================
# Pretrains the Transformer model on source domain cell sequences.
# Replaces the LSTM from Fan et al. (2025) -- this is the thesis contribution.

# Version history:
#     v2: d_model=64, dropout=0.10, wd=1e-4, lr=1e-3  -> best epoch=18  (overfit)
#     v3: d_model=48, dropout=0.25, wd=1e-3, lr=5e-4  -> best epoch=24  (underfit, R²=0.49)
#     v4: d_model=48, dropout=0.10, wd=2e-4, lr=2e-4  -> balanced, R²=0.8345
#     v5: d_model=64, dropout=0.10, wd=2e-4, lr=2e-4  -> restore v2 capacity, R²=0.8682
#     v6: + causal mask + noise_std=0.003             -> prevent seq memorisation

# Key insight: v5 still overfit at epoch 41 — the bidirectional transformer was
# learning co-occurrence shortcuts across all cycle positions, not temporal
# dynamics. Two fixes: (1) causal mask so each cycle only attends to past cycles,
# forcing genuine trend learning; (2) feature noise std=0.003 so the model cannot
# memorise exact feature vectors of the 9 training sequences.

# Architecture:
#     Linear projection (38 -> 64)
#     Positional encoding (sinusoidal)
#     2 x TransformerEncoderLayer (nhead=4, dim_feedforward=128, dropout=0.10)
#       + causal self-attention mask
#     fc(64->48) -> Dropout(0.10) -> fc(32) -> ReLU -> fc(1)

# Run from COMBINED/04_pretrain_transformer/ directory:
#     python pretrain_transformer.py
# """

# import os
# import math
# import pickle
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # ══════════════════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════
# SEQ_DIR  = "../01a_cell_input/processed"
# OUT_DIR  = "processed"
# os.makedirs(OUT_DIR, exist_ok=True)

# FUSED_DIM       = 38
# D_MODEL         = 64     # restored from v2; v2 overfit because LR=1e-3, not
# NHEAD           = 4      # because 64 was too large. With LR=2e-4 it should
# NUM_LAYERS      = 2      # generalise well and give better transfer features.
# DIM_FEEDFORWARD = 128    # 2x D_MODEL
# DROPOUT         = 0.10   # same as v2 -- v3 proved 0.25 is too aggressive

# LAMBDA1      = 1.0
# EPOCHS       = 3000
# LR           = 2e-4
# WEIGHT_DECAY = 2e-4
# NOISE_STD    = 0.003     # feature noise per epoch; prevents memorising 9 train seqs
# SEED         = 42
# LR_PATIENCE  = 50
# LR_FACTOR    = 0.5
# LR_MIN       = 1e-6
# ES_PATIENCE  = 300

# torch.manual_seed(SEED)
# np.random.seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {DEVICE}")

# # ══════════════════════════════════════════════════════════════════════════
# # POSITIONAL ENCODING
# # ══════════════════════════════════════════════════════════════════════════
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)

#         pe = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len).unsqueeze(1).float()

#         div = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-math.log(10000.0) / d_model)
#         )

#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)

#         self.register_buffer("pe", pe.unsqueeze(0))  # (1, T, D)

#     def forward(self, x):
#         # x: (B, T, D)
#         T = x.size(1)
#         x = x + self.pe[:, :T, :]
#         return self.dropout(x)


# # ══════════════════════════════════════════════════════════════════════════
# # MODEL
# # ══════════════════════════════════════════════════════════════════════════
# class TransformerModel(nn.Module):
#     def __init__(self, input_dim=38, d_model=64, nhead=4,
#                  num_layers=2, dim_feedforward=128, dropout=0.10):
#         super().__init__()

#         self.input_proj = nn.Linear(input_dim, d_model)
#         self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=True,
#         )

#         self.transformer = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )

#         self.head = nn.Sequential(
#             nn.Linear(d_model, 48),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(48, 32),
#             nn.ReLU(),
#             nn.Linear(32, 1),
#         )

#     def forward(self, x):
#         # x: (B, T, F)
#         if x.dim() == 2:
#             x = x.unsqueeze(0)

#         x = self.input_proj(x)
#         x = self.pos_enc(x)

#         B, T, _ = x.shape

#         # causal mask (correct shape)
#         causal_mask = torch.triu(
#             torch.ones(T, T, device=x.device), diagonal=1
#         ).bool()

#         x = self.transformer(
#             x,
#             mask=causal_mask
#         )

#         return self.head(x).squeeze(-1)  # (B, T)


# # ══════════════════════════════════════════════════════════════════════════
# # LOSS
# # ══════════════════════════════════════════════════════════════════════════
# def composite_loss(pred, target, lambda1=LAMBDA1):
#     errors_sq  = (pred - target) ** 2
#     l_mse      = errors_sq.mean()
#     large_mask = errors_sq > l_mse.detach()
#     l_emse     = errors_sq[large_mask].mean() \
#                  if large_mask.sum() > 0 \
#                  else torch.tensor(0.0, device=pred.device)
#     return l_mse + lambda1 * l_emse, l_mse, l_emse


# # ══════════════════════════════════════════════════════════════════════════
# # STEP 1: LOAD
# # ══════════════════════════════════════════════════════════════════════════
# print("=" * 60)
# print("Transformer Pretraining v6 -- Causal Mask + Noise Augmentation")
# print("=" * 60)

# with open(os.path.join(SEQ_DIR, "cell_sequences.pkl"), "rb") as f:
#     cell_sequences = pickle.load(f)

# print(f"\nLoaded {len(cell_sequences)} cell sequences:")
# for s in cell_sequences:
#     print(f"  {s['cell_id']:12s} ({s['dataset']:6s}): "
#           f"{s['n_cycles']:4d} cycles  "
#           f"SOH {s['soh'].min():.3f}-{s['soh'].max():.3f}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 2: VAL SPLIT (B0018 + CS2_38)
# # ══════════════════════════════════════════════════════════════════════════
# val_indices = []
# for ds in ["NASA", "CALCE"]:
#     ds_cells = [i for i, s in enumerate(cell_sequences)
#                 if s["dataset"] == ds]
#     if ds_cells:
#         np.random.seed(SEED)
#         val_indices.append(np.random.choice(ds_cells))

# train_indices = [i for i in range(len(cell_sequences))
#                  if i not in val_indices]
# train_seqs    = [cell_sequences[i] for i in train_indices]
# val_seqs      = [cell_sequences[i] for i in val_indices]

# print(f"\nTrain/Val split:")
# print(f"  Train ({len(train_seqs)}): "
#       f"{[s['cell_id'] for s in train_seqs]}")
# print(f"  Val   ({len(val_seqs)}): "
#       f"{[s['cell_id'] for s in val_seqs]}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 3: INITIALISE
# # ══════════════════════════════════════════════════════════════════════════
# model = TransformerModel(
#     input_dim=FUSED_DIM, d_model=D_MODEL, nhead=NHEAD,
#     num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
#     dropout=DROPOUT).to(DEVICE)

# optimizer = torch.optim.AdamW(model.parameters(),
#                                lr=LR, weight_decay=WEIGHT_DECAY)
# scheduler = ReduceLROnPlateau(optimizer, mode="min",
#                                patience=LR_PATIENCE,
#                                factor=LR_FACTOR, min_lr=LR_MIN)

# n_params = sum(p.numel() for p in model.parameters()
#                if p.requires_grad)
# print(f"\nModel parameters : {n_params:,}")
# print(f"Architecture     : Linear({FUSED_DIM}->{D_MODEL}) -> "
#       f"Transformer(x{NUM_LAYERS}, ff={DIM_FEEDFORWARD}) -> "
#       f"fc(48)->fc(32)->fc(1)")
# print(f"Dropout          : {DROPOUT}  (light -- v3 proved 0.25 too aggressive)")
# print(f"Weight decay     : {WEIGHT_DECAY}  (slightly above v2)")
# print(f"LR               : {LR}  (key fix -- 5x lower than v2)")
# print(f"ES patience      : {ES_PATIENCE}")
# print(f"Max epochs       : {EPOCHS}")
# print(f"Loss             : L_mse + {LAMBDA1} * L_emse")

# def collate_batch(seqs):
#     lengths = [len(s["features"]) for s in seqs]
#     max_len = max(lengths)

#     X_batch = []
#     y_batch = []

#     for s in seqs:
#         X = torch.tensor(s["features"], dtype=torch.float32)
#         y = torch.tensor(s["soh"], dtype=torch.float32).unsqueeze(1)

#         pad_len = max_len - len(X)

#         if pad_len > 0:
#             X = torch.cat([X, torch.zeros(pad_len, X.shape[1])], dim=0)
#             y = torch.cat([y, torch.zeros(pad_len, 1)], dim=0)

#         X_batch.append(X)
#         y_batch.append(y)

#     return torch.stack(X_batch), torch.stack(y_batch), lengths

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 4: TRAINING LOOP
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nTraining (max {EPOCHS} epochs, ES patience={ES_PATIENCE}) ...")
# print(f"{'Epoch':>6}  {'Train Loss':>12}  {'Val Loss':>12}  "
#       f"{'LR':>10}  {'ES counter':>10}")
# print("-" * 58)

# train_losses, val_losses = [], []
# best_val     = float("inf")
# best_epoch   = 0
# es_counter   = 0
# best_weights = None

# BATCH_SIZE = 4  # small since dataset is tiny

# for epoch in range(1, EPOCHS + 1):

#     # ── Train ──────────────────────────────────────────
#     model.train()
#     epoch_loss, total_pts = 0.0, 0

#     indices = np.random.permutation(len(train_seqs))

#     for i in range(0, len(indices), BATCH_SIZE):
#         batch_indices = indices[i:i+BATCH_SIZE]
#         batch_seqs = [train_seqs[j] for j in batch_indices]

#         X, y_true, lengths = collate_batch(batch_seqs)

#         X = X.to(DEVICE)
#         y_true = y_true.to(DEVICE)

#         optimizer.zero_grad()

#         # noise
#         noise = NOISE_STD * torch.randn_like(X)
#         X_noisy = X + noise

#         y_pred = model(X_noisy)


#         y_pred = y_pred[mask]
#         y_true = y_true[mask]

#         loss, _, _ = composite_loss(y_pred, y_true)

#         loss.backward()
#         nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         optimizer.step()

#         epoch_loss += loss.item() * mask.sum().item()
#         total_pts  += mask.sum().item()

#     train_loss = epoch_loss / total_pts

#     # ── Validate ───────────────────────────────────────────────────────
#     model.eval()
#     val_loss_sum, val_pts = 0.0, 0

#     with torch.no_grad():
#         for i in range(0, len(val_seqs), BATCH_SIZE):
#             batch_seqs = val_seqs[i:i+BATCH_SIZE]

#             X, y_true, lengths = collate_batch(batch_seqs)

#             X = X.to(DEVICE)
#             y_true = y_true.to(DEVICE)

#             y_pred = model(X)

#             mask = torch.zeros_like(y_true, dtype=torch.bool)
#             for b, l in enumerate(lengths):
#                 mask[b, :l] = True

#             y_pred = y_pred.reshape(-1)[mask.reshape(-1)]
#             y_true = y_true.reshape(-1)[mask.reshape(-1)]

#             loss, _, _ = composite_loss(y_pred, y_true)

#             val_loss_sum += loss.item() * mask.sum().item()
#             val_pts      += mask.sum().item()

#     val_loss = val_loss_sum / val_pts

#     scheduler.step(val_loss)
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)
#     current_lr = optimizer.param_groups[0]["lr"]

#     if val_loss < best_val:
#         best_val     = val_loss
#         best_epoch   = epoch
#         es_counter   = 0
#         best_weights = {k: v.cpu().clone()
#                         for k, v in model.state_dict().items()}
#         marker = " <-- best"
#     else:
#         es_counter += 1
#         marker = ""

#     if epoch % 100 == 0 or epoch == 1 or marker:
#         print(f"{epoch:>6}  {train_loss:>12.6f}  {val_loss:>12.6f}  "
#               f"{current_lr:>10.2e}  {es_counter:>10}{marker}")

#     if es_counter >= ES_PATIENCE:
#         print(f"\nEarly stopping at epoch {epoch} "
#               f"(best at epoch {best_epoch})")
#         break

# model.load_state_dict(best_weights)
# print(f"\nRestored best weights from epoch {best_epoch}")
# print(f"Best val loss : {best_val:.6f}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 5: SAVE MODEL + LOG
# # ══════════════════════════════════════════════════════════════════════════
# torch.save(model.state_dict(),
#            os.path.join(OUT_DIR, "transformer_pretrained.pt"))
# pd.DataFrame({
#     "epoch"     : range(1, len(train_losses) + 1),
#     "train_loss": train_losses,
#     "val_loss"  : val_losses,
# }).to_csv(os.path.join(OUT_DIR, "transformer_training_log.csv"),
#           index=False)
# print(f"\nSaved -> transformer_pretrained.pt")
# print(f"Saved -> transformer_training_log.csv")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 6: EVALUATE -- single inference pass, single R² calculation
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nEvaluating on all cells ...")
# model.eval()
# all_results   = []
# all_true_vals = []
# all_pred_vals = []

# with torch.no_grad():
#     for seq in cell_sequences:
#         X      = torch.tensor(seq["features"],
#                                dtype=torch.float32).to(DEVICE)
#         y_true = seq["soh"]
#         y_pred = model(X).cpu().numpy().squeeze()
#         rmse   = float(np.sqrt(np.mean((y_pred - y_true)**2)))
#         mae    = float(np.mean(np.abs(y_pred - y_true)))
#         split  = "val" if seq["cell_id"] in \
#                  [s["cell_id"] for s in val_seqs] else "train"
#         all_results.append({
#             "cell_id" : seq["cell_id"],
#             "dataset" : seq["dataset"],
#             "split"   : split,
#             "n_cycles": seq["n_cycles"],
#             "rmse"    : rmse,
#             "mae"     : mae,
#         })
#         all_true_vals.extend(y_true.tolist())
#         all_pred_vals.extend(y_pred.tolist())
#         print(f"  {seq['cell_id']:12s} [{split:5s}]: "
#               f"RMSE={rmse:.4f}  MAE={mae:.4f}")

# results_df = pd.DataFrame(all_results)
# results_df.to_csv(
#     os.path.join(OUT_DIR, "transformer_cell_results.csv"), index=False)

# train_rmse   = results_df[results_df["split"] == "train"]["rmse"].mean()
# val_rmse     = results_df[results_df["split"] == "val"]["rmse"].mean()
# all_true_arr = np.array(all_true_vals)
# all_pred_arr = np.array(all_pred_vals)
# overall_rmse = float(np.sqrt(np.mean((all_pred_arr - all_true_arr)**2)))
# overall_r2   = float(
#     1 - np.sum((all_pred_arr - all_true_arr)**2) /
#     np.sum((all_true_arr - all_true_arr.mean())**2)
# )

# print(f"\n  Mean train RMSE : {train_rmse:.4f}")
# print(f"  Mean val RMSE   : {val_rmse:.4f}")
# print(f"  Overall RMSE    : {overall_rmse:.4f}")
# print(f"  Overall R²      : {overall_r2:.4f}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 7: PLOTS
# # ══════════════════════════════════════════════════════════════════════════

# # ── Plot A: Training curves ────────────────────────────────────────────────
# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# epochs_r  = range(1, len(train_losses) + 1)
# axes[0].plot(epochs_r, train_losses, lw=1.5,
#              color="steelblue", label="Train loss")
# axes[0].plot(epochs_r, val_losses, lw=1.5,
#              color="darkorange", label="Val loss")
# axes[0].axvline(best_epoch, color="red", lw=1, linestyle="--",
#                 label=f"Best epoch {best_epoch}")
# axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
# axes[0].set_title("Training Loss (log scale)")
# axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.4)
# axes[0].set_yscale("log")

# start_zoom = max(0, len(train_losses) - len(train_losses) // 5)
# axes[1].plot(list(epochs_r)[start_zoom:],
#              train_losses[start_zoom:], lw=1.5,
#              color="steelblue", label="Train loss")
# axes[1].plot(list(epochs_r)[start_zoom:],
#              val_losses[start_zoom:], lw=1.5,
#              color="darkorange", label="Val loss")
# axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
# axes[1].set_title(f"Last {len(train_losses) // 5} epochs (zoomed)")
# axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.4)
# plt.suptitle(f"Transformer Pretraining v4  |  "
#              f"Best val={best_val:.6f} at epoch {best_epoch}  |  "
#              f"LR={LR}  dropout={DROPOUT}  wd={WEIGHT_DECAY}",
#              fontsize=11, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "transformer_training_curves.png"),
#             dpi=150, bbox_inches="tight")
# plt.close()

# # ── Plot B: SOH predictions per cell ──────────────────────────────────────
# pred_cache = {}
# with torch.no_grad():
#     for seq in cell_sequences:
#         X = torch.tensor(seq["features"],
#                           dtype=torch.float32).to(DEVICE)
#         pred_cache[seq["cell_id"]] = model(X).cpu().numpy().squeeze()

# ncols = 4
# nrows = int(np.ceil(len(cell_sequences) / ncols))
# fig, axes = plt.subplots(nrows, ncols,
#                           figsize=(ncols * 5, nrows * 4))
# axes = axes.flatten()

# for i, seq in enumerate(cell_sequences):
#     y_true = seq["soh"]
#     y_pred = pred_cache[seq["cell_id"]]
#     rmse   = float(np.sqrt(np.mean((y_pred - y_true)**2)))
#     split  = "val" if seq["cell_id"] in \
#              [s["cell_id"] for s in val_seqs] else "train"
#     color  = "darkorange" if split == "val" else "steelblue"
#     cycles = np.arange(1, len(y_true) + 1)
#     axes[i].plot(cycles, y_true, "k-", lw=1.5, label="True")
#     axes[i].plot(cycles, y_pred, "--", lw=1.5,
#                  color=color, label=f"Pred [{split}]")
#     axes[i].fill_between(cycles, y_true, y_pred,
#                          alpha=0.15, color=color)
#     axes[i].set_title(f"{seq['cell_id']} ({seq['dataset']})\n"
#                       f"RMSE={rmse:.4f}", fontsize=8)
#     axes[i].set_ylim(0.70, 1.05)
#     axes[i].legend(fontsize=6); axes[i].grid(True, alpha=0.3)

# for j in range(len(cell_sequences), len(axes)):
#     axes[j].set_visible(False)

# plt.suptitle(f"Transformer v4 Predictions -- "
#              f"train RMSE={train_rmse:.4f}  val RMSE={val_rmse:.4f}",
#              fontsize=12, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "transformer_predictions.png"),
#             dpi=150, bbox_inches="tight")
# plt.close()

# # ── Plot C: Predicted vs True scatter ─────────────────────────────────────
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.scatter(all_true_arr, all_pred_arr, s=2, alpha=0.3,
#            color="steelblue")
# lims = [all_true_arr.min(), all_true_arr.max()]
# ax.plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
# ax.set_xlabel("True SOH"); ax.set_ylabel("Predicted SOH")
# ax.set_title(f"Predicted vs True SOH\n"
#              f"RMSE={overall_rmse:.4f}  R²={overall_r2:.4f}")
# ax.legend(); ax.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "transformer_scatter.png"),
#             dpi=150, bbox_inches="tight")
# plt.close()

# print("Saved -> transformer_training_curves.png")
# print("Saved -> transformer_predictions.png")
# print("Saved -> transformer_scatter.png")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 8: SUMMARY
# # ══════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("TRANSFORMER PRETRAINING SUMMARY (v6)")
# print("=" * 60)
# print(f"  Architecture   : Linear({FUSED_DIM}->{D_MODEL}) -> "
#       f"Transformer(x{NUM_LAYERS}, ff={DIM_FEEDFORWARD}) -> "
#       f"fc(48)->fc(32)->fc(1)")
# print(f"  Parameters     : {n_params:,}")
# print(f"  Dropout        : {DROPOUT}")
# print(f"  Weight decay   : {WEIGHT_DECAY}")
# print(f"  LR             : {LR}")
# print(f"  Epochs trained : {len(train_losses)}")
# print(f"  Best epoch     : {best_epoch}")
# print(f"  Best val loss  : {best_val:.6f}")
# print(f"  Mean train RMSE: {train_rmse:.4f}")
# print(f"  Mean val RMSE  : {val_rmse:.4f}")
# print(f"  Overall RMSE   : {overall_rmse:.4f}")
# print(f"  Overall R²     : {overall_r2:.4f}")
# print(f"\n  Version comparison:")
# print(f"    LSTM v1     : train=0.0127  val=0.0248  R²=0.9445")
# print(f"    Transf. v2  : train=0.0178  val=0.0242  R²=0.9047"
#       f"  (best=18,  overfit,   lr=1e-3, do=0.10, wd=1e-4)")
# print(f"    Transf. v3  : train=0.0441  val=0.0371  R²=0.4857"
#       f"  (best=24,  underfit,  lr=5e-4, do=0.25, wd=1e-3)")
# print(f"    Transf. v6  : train={train_rmse:.4f}  "
#       f"val={val_rmse:.4f}  R²={overall_r2:.4f}"
#       f"  (best={best_epoch}, "
#       f"lr={LR}, do={DROPOUT}, wd={WEIGHT_DECAY})")
# print(f"\n  Next step: fine-tune on drone pack data")
# print("\nDone.")

"""
pretrain_transformer.py (FIXED v6 + VISUALS)
============================================
Includes:
- Fixed training loop
- Correct masking
- Noise augmentation
- Full evaluation
- Full plotting suite (curves, predictions, scatter)
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

# ═════════ CONFIG ═════════
SEQ_DIR  = "../01a_cell_input/processed"
OUT_DIR  = "00_processed"
os.makedirs(OUT_DIR, exist_ok=True)

FUSED_DIM       = 38
D_MODEL         = 64
NHEAD           = 4
NUM_LAYERS      = 2
DIM_FEEDFORWARD = 128
DROPOUT         = 0.10

LAMBDA1      = 0.0
EPOCHS       = 3000
LR           = 2e-4
WEIGHT_DECAY = 2e-4
NOISE_STD    = 0.003

SEED         = 42
LR_PATIENCE  = 50
LR_FACTOR    = 0.5
LR_MIN       = 1e-6
ES_PATIENCE  = 300

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(SEED)
np.random.seed(SEED)

print("Device:", DEVICE)


# ═════════ MODEL ═════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()

        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_proj = nn.Linear(FUSED_DIM, D_MODEL)
        self.pos_enc = PositionalEncoding(D_MODEL, dropout=DROPOUT)

        enc = nn.TransformerEncoderLayer(
            d_model=D_MODEL,
            nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(enc, NUM_LAYERS)

        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = self.input_proj(x)
        x = self.pos_enc(x)

        T = x.size(1)
        causal = torch.triu(torch.ones(T, T, device=x.device), 1).bool()

        x = self.transformer(x, mask=causal)

        return self.head(x).squeeze(-1)


# ═════════ LOSS ═════════
def composite_loss(pred, target):
    err = (pred - target) ** 2
    mse = err.mean()
    emse = err[err > mse.detach()].mean() if (err > mse).any() else err.mean()*0
    return mse + LAMBDA1 * emse


# ═════════ DATA ═════════
with open(os.path.join(SEQ_DIR, "cell_sequences.pkl"), "rb") as f:
    cell_sequences = pickle.load(f)


val_idx = []
for ds in ["NASA", "CALCE"]:
    idx = [i for i,s in enumerate(cell_sequences) if s["dataset"] == ds]
    if idx:
        val_idx.append(idx[-1])

train_seqs = [s for i,s in enumerate(cell_sequences) if i not in val_idx]
val_seqs   = [s for i,s in enumerate(cell_sequences) if i in val_idx]


# ═════════ BATCHING ═════════
def collate(seqs):
    T = max(len(s["features"]) for s in seqs)

    X = torch.zeros(len(seqs), T, FUSED_DIM)
    Y = torch.zeros(len(seqs), T)
    mask = torch.zeros(len(seqs), T, dtype=torch.bool)

    for i,s in enumerate(seqs):
        L = len(s["features"])
        X[i,:L] = torch.tensor(s["features"])
        Y[i,:L] = torch.tensor(s["soh"], dtype=torch.float32)
        mask[i,:L] = 1

    return X, Y, mask


# ═════════ INIT ═════════
model = TransformerModel().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
sched = ReduceLROnPlateau(opt, patience=LR_PATIENCE,
                          factor=LR_FACTOR, min_lr=LR_MIN)


# ═════════ TRAIN ═════════
train_losses, val_losses = [], []
best_val = 1e9
best_w = None
es = 0

for epoch in range(EPOCHS):

    model.train()
    np.random.shuffle(train_seqs)

    total = 0
    n_batches = 0

    for i in range(0, len(train_seqs), 4):
        batch = train_seqs[i:i+4]

        X,Y,mask = collate(batch)
        X,Y,mask = X.to(DEVICE), Y.to(DEVICE), mask.to(DEVICE)

        X = X + NOISE_STD * torch.randn_like(X)

        pred_full = model(X)

        # per-cell loss: each cell contributes equally regardless of length
        cell_losses = []
        for k in range(len(batch)):
            L = int(mask[k].sum())
            cell_losses.append(composite_loss(pred_full[k, :L], Y[k, :L]))
        loss = torch.stack(cell_losses).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total += loss.item()
        n_batches += 1

    train_loss = total / max(1, n_batches)


    # VALIDATION
    model.eval()
    val_total = 0

    with torch.no_grad():
        for i in range(0, len(val_seqs), 4):
            batch = val_seqs[i:i+4]

            X,Y,mask = collate(batch)
            X,Y,mask = X.to(DEVICE), Y.to(DEVICE), mask.to(DEVICE)

            pred_full = model(X)

            cell_losses = []
            for k in range(len(batch)):
                L = int(mask[k].sum())
                cell_losses.append(composite_loss(pred_full[k, :L], Y[k, :L]))
            val_total += torch.stack(cell_losses).mean().item()

    val_loss = val_total / max(1, len(val_seqs))

    sched.step(val_loss)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val:
        best_val = val_loss
        best_w = {k:v.cpu() for k,v in model.state_dict().items()}
        es = 0
    else:
        es += 1

    if epoch % 50 == 0:
        print(epoch, train_loss, val_loss)

    if es > ES_PATIENCE:
        break

model.load_state_dict(best_w)


# ═════════ SAVE ═════════
torch.save(model.state_dict(),
           os.path.join(OUT_DIR, "transformer_pretrained.pt"))


# ═════════════════════════════════════════════════════════════
# 📊 PLOTTING SECTION (FULL VISUAL OUTPUT)
# ═════════════════════════════════════════════════════════════

# ── 1. Training curves ───────────────────────────────────────
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Val")
plt.yscale("log")
plt.title("Transformer Loss Curves")
plt.legend()
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "loss_curves.png"))
plt.close()


# ── 2. Per-cell prediction plots ─────────────────────────────
model.eval()
pred_cache = {}

with torch.no_grad():
    for s in cell_sequences:
        X = torch.tensor(s["features"]).float().to(DEVICE)
        pred_cache[s["cell_id"]] = model(X).cpu().numpy().squeeze()

n = len(cell_sequences)
cols = 4
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(16, rows*3))
axes = axes.flatten()

for i,s in enumerate(cell_sequences):
    y = s["soh"]
    p = pred_cache[s["cell_id"]]
    x = np.arange(len(y))

    axes[i].plot(x, y, label="True")
    axes[i].plot(x, p, label="Pred")
    axes[i].set_title(s["cell_id"])
    axes[i].legend()
    axes[i].grid()

for j in range(i+1, len(axes)):
    axes[j].axis("off")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "cell_predictions.png"))
plt.close()


# ── 3. Scatter plot ──────────────────────────────────────────
true_all, pred_all = [], []

for s in cell_sequences:
    y = s["soh"]
    p = pred_cache[s["cell_id"]]
    true_all.extend(y)
    pred_all.extend(p)

true_all = np.array(true_all)
pred_all = np.array(pred_all)

plt.figure()
plt.scatter(true_all, pred_all, s=2)
lims = [true_all.min(), true_all.max()]
plt.plot(lims, lims, "r--")
plt.title("Predicted vs True SOH")
plt.grid()
plt.savefig(os.path.join(OUT_DIR, "scatter.png"))
plt.close()


print("Saved all plots:")
print(" - loss_curves.png")
print(" - cell_predictions.png")
print(" - scatter.png")
print(" - transformer_pretrained.pt")