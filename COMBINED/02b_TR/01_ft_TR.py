# """
# finetune_transformer.py
# =======================
# Fine-tunes the pretrained Transformer on drone pack data (target domain)
# and evaluates on the held-out test set.

# Identical to finetune_lstm.py except:
#     - Loads transformer_pretrained.pt instead of lstm_pretrained.pt
#     - Uses TransformerModel instead of LSTMModel

# Everything else is identical for fair comparison:
#     - Same fine-tune data: finetune_fused.npy (35 cycles, 10%)
#     - Same test data: test_fused.npy (315 cycles, 90%)
#     - Same loss function: L_mse + lambda1 * L_emse
#     - Same learning rate: 1e-4
#     - Same early stopping patience: 80

# Reads (from ../processed/):
#     finetune_fused.npy     (35, 38)
#     finetune_soh.npy       (35,)
#     test_fused.npy         (315, 38)
#     test_soh.npy           (315,)
#     drone_index.csv

# Reads (from ../04_pretrain_transformer/processed/):
#     transformer_pretrained.pt

# Writes (to processed/):
#     transformer_finetuned.pt
#     transformer_finetune_log.csv
#     transformer_results.csv
#     transformer_finetune_results.png

# Run from COMBINED/05_finetune_transformer/ directory:
#     python finetune_transformer.py
# """

# import os
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # ══════════════════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════
# FEAT_DIR     = "../02a_LSTM/01_processed"
# PRETRAIN_DIR = "00_processed"
# OUT_DIR      = "01_processed"
# os.makedirs(OUT_DIR, exist_ok=True)

# # Architecture (must match pretraining exactly)
# FUSED_DIM       = 38
# D_MODEL         = 64
# NHEAD           = 4
# NUM_LAYERS      = 2
# DIM_FEEDFORWARD = 128
# DROPOUT         = 0.1

# # Fine-tuning (identical to LSTM fine-tuning)
# EPOCHS      = 300
# LR          = 1e-5
# LAMBDA1     = 1.0
# LR_PATIENCE = 30
# LR_FACTOR   = 0.5
# LR_MIN      = 1e-7
# ES_PATIENCE = 50
# SEED        = 42

# torch.manual_seed(SEED)
# np.random.seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {DEVICE}")

# # ══════════════════════════════════════════════════════════════════════════
# # MODEL DEFINITION (identical to pretraining)
# # ══════════════════════════════════════════════════════════════════════════
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe     = torch.zeros(max_len, d_model)
#         pos    = torch.arange(0, max_len).unsqueeze(1).float()
#         div    = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-math.log(10000.0) / d_model)
#         )
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         pe = pe.unsqueeze(0)   # (1, max_len, d_model)
#         self.register_buffer("pe", pe)

#     def forward(self, x):
#         """x: (seq_len, d_model)"""
#         x = x + self.pe[0, :x.size(0), :]
#         return self.dropout(x)


# class TransformerModel(nn.Module):
#     def __init__(self,
#                  input_dim=38,
#                  d_model=64,
#                  nhead=4,
#                  num_layers=2,
#                  dim_feedforward=128,
#                  dropout=0.1):
#         super().__init__()
#         self.input_proj = nn.Linear(input_dim, d_model)
#         self.pos_enc    = PositionalEncoding(d_model, dropout=dropout)
#         encoder_layer   = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout,
#             batch_first=False,
#             activation="relu",
#         )
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers)
#         self.head = nn.Sequential(
#             nn.Linear(d_model, 8), nn.ReLU(),
#             nn.Linear(8,       1),
#         )

#     def forward(self, x):
#         """x: (seq_len, 38) -> (seq_len, 1)"""
#         x = self.input_proj(x)
#         x = self.pos_enc(x)
#         x = x.unsqueeze(1)
#         x = self.transformer(x)
#         x = x.squeeze(1)
#         return self.head(x)


# # ══════════════════════════════════════════════════════════════════════════
# # LOSS FUNCTION (identical to pretraining and LSTM fine-tuning)
# # ══════════════════════════════════════════════════════════════════════════
# def composite_loss(pred, target, lambda1=LAMBDA1):
#     errors_sq  = (pred - target) ** 2
#     l_mse      = errors_sq.mean()
#     large_mask = errors_sq > l_mse.detach()
#     if large_mask.sum() > 0:
#         l_emse = errors_sq[large_mask].mean()
#     else:
#         l_emse = torch.tensor(0.0, device=pred.device)
#     return l_mse + lambda1 * l_emse, l_mse, l_emse


# # ══════════════════════════════════════════════════════════════════════════
# # STEP 1: LOAD DATA
# # ══════════════════════════════════════════════════════════════════════════
# print("=" * 60)
# print("Transformer Fine-tuning on Drone Pack Data")
# print("=" * 60)

# ft_fused = np.load(os.path.join(FEAT_DIR, "finetune_fused.npy"))
# ft_soh   = np.load(os.path.join(FEAT_DIR, "finetune_soh.npy"))
# te_fused = np.load(os.path.join(FEAT_DIR, "test_fused.npy"))
# te_soh   = np.load(os.path.join(FEAT_DIR, "test_soh.npy"))
# index    = pd.read_csv(os.path.join(FEAT_DIR, "drone_index.csv"))

# print(f"\nLoaded:")
# print(f"  Fine-tune : {ft_fused.shape}  "
#       f"SOH {ft_soh.min():.3f}-{ft_soh.max():.3f}")
# print(f"  Test      : {te_fused.shape}  "
#       f"SOH {te_soh.min():.3f}-{te_soh.max():.3f}")

# ft_cycles = index["cycle"].values[:len(ft_soh)]
# te_cycles = index["cycle"].values[len(ft_soh):]

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 2: LOAD PRETRAINED MODEL
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nLoading pretrained Transformer weights ...")

# model = TransformerModel(
#     input_dim=FUSED_DIM,
#     d_model=D_MODEL,
#     nhead=NHEAD,
#     num_layers=NUM_LAYERS,
#     dim_feedforward=DIM_FEEDFORWARD,
#     dropout=DROPOUT,
# ).to(DEVICE)

# model.load_state_dict(
#     torch.load(os.path.join(PRETRAIN_DIR, "transformer_pretrained.pt"),
#                map_location=DEVICE))

# n_params = sum(p.numel() for p in model.parameters()
#                if p.requires_grad)
# print(f"  Parameters loaded : {n_params:,}")

# # ── Freeze encoder -- only fine-tune output head ──────────────────────────
# for name, param in model.named_parameters():
#     if "head" not in name:
#         param.requires_grad = False

# n_trainable = sum(p.numel() for p in model.parameters()
#                   if p.requires_grad)
# n_frozen    = sum(p.numel() for p in model.parameters()
#                   if not p.requires_grad)
# print(f"  Trainable params : {n_trainable:,}  (head only)")
# print(f"  Frozen params    : {n_frozen:,}  (encoder + pos enc)")

# # Evaluate BEFORE fine-tuning
# model.eval()
# with torch.no_grad():
#     X_te   = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
#     y_pre  = model(X_te).cpu().numpy().squeeze()
# rmse_pre = float(np.sqrt(np.mean((y_pre - te_soh)**2)))
# mae_pre  = float(np.mean(np.abs(y_pre - te_soh)))
# print(f"\n  Performance BEFORE fine-tuning (on test set):")
# print(f"    RMSE = {rmse_pre:.4f}")
# print(f"    MAE  = {mae_pre:.4f}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 3: FINE-TUNE
# # ══════════════════════════════════════════════════════════════════════════
# optimizer = torch.optim.Adam(model.parameters(), lr=LR)
# scheduler = ReduceLROnPlateau(optimizer, mode="min",
#                                patience=LR_PATIENCE,
#                                factor=LR_FACTOR,
#                                min_lr=LR_MIN)

# print(f"\nFine-tuning (max {EPOCHS} epochs, "
#       f"lr={LR}, early stopping patience={ES_PATIENCE}) ...")
# print(f"{'Epoch':>6}  {'Train Loss':>12}  {'LR':>10}")
# print("-" * 35)

# X_ft   = torch.tensor(ft_fused, dtype=torch.float32).to(DEVICE)
# y_ft   = torch.tensor(ft_soh,   dtype=torch.float32).unsqueeze(1).to(DEVICE)

# train_losses = []
# best_loss    = float("inf")
# best_epoch   = 0
# es_counter   = 0
# best_weights = None

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     optimizer.zero_grad()
#     y_pred     = model(X_ft)
#     loss, _, _ = composite_loss(y_pred, y_ft)
#     loss.backward()
#     nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#     optimizer.step()

#     train_loss = loss.item()
#     scheduler.step(train_loss)
#     train_losses.append(train_loss)
#     current_lr = optimizer.param_groups[0]["lr"]

#     if train_loss < best_loss:
#         best_loss    = train_loss
#         best_epoch   = epoch
#         es_counter   = 0
#         best_weights = {k: v.cpu().clone()
#                         for k, v in model.state_dict().items()}
#         marker = " <-- best"
#     else:
#         es_counter += 1
#         marker = ""

#     if epoch % 50 == 0 or epoch == 1 or marker:
#         print(f"{epoch:>6}  {train_loss:>12.6f}  "
#               f"{current_lr:>10.2e}{marker}")

#     if es_counter >= ES_PATIENCE:
#         print(f"\nEarly stopping at epoch {epoch} "
#               f"(best at epoch {best_epoch})")
#         break

# model.load_state_dict(best_weights)
# print(f"\nRestored best weights from epoch {best_epoch}")
# print(f"Best train loss : {best_loss:.6f}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 4: EVALUATE ON TEST SET
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nEvaluating on test set ({len(te_soh)} cycles) ...")

# model.eval()
# with torch.no_grad():
#     X_te   = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
#     y_post = model(X_te).cpu().numpy().squeeze()

# rmse_post = float(np.sqrt(np.mean((y_post - te_soh)**2)))
# mae_post  = float(np.mean(np.abs(y_post - te_soh)))
# r2_post   = float(1 - np.sum((y_post - te_soh)**2) /
#                   np.sum((te_soh - te_soh.mean())**2))

# print(f"\n  Performance AFTER fine-tuning (on test set):")
# print(f"    RMSE = {rmse_post:.4f}  (was {rmse_pre:.4f} before)")
# print(f"    MAE  = {mae_post:.4f}  (was {mae_pre:.4f} before)")
# print(f"    R²   = {r2_post:.4f}")
# print(f"    RMSE improvement: "
#       f"{(rmse_pre - rmse_post)/rmse_pre * 100:.1f}%")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 5: SAVE
# # ══════════════════════════════════════════════════════════════════════════
# torch.save(model.state_dict(),
#            os.path.join(OUT_DIR, "transformer_finetuned.pt"))

# log_df = pd.DataFrame({
#     "epoch"     : range(1, len(train_losses) + 1),
#     "train_loss": train_losses,
# })
# log_df.to_csv(os.path.join(OUT_DIR,
#               "transformer_finetune_log.csv"), index=False)

# results_df = pd.DataFrame({
#     "cycle"    : te_cycles,
#     "soh_true" : te_soh,
#     "soh_pred" : y_post,
#     "error"    : y_post - te_soh,
#     "abs_error": np.abs(y_post - te_soh),
# })
# results_df.to_csv(os.path.join(OUT_DIR,
#                   "transformer_results.csv"), index=False)

# print(f"\nSaved -> transformer_finetuned.pt")
# print(f"Saved -> transformer_finetune_log.csv")
# print(f"Saved -> transformer_results.csv")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 6: PLOTS
# # ══════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # ── Plot 1: Fine-tuning loss ───────────────────────────────────────────────
# axes[0,0].plot(range(1, len(train_losses)+1), train_losses,
#                lw=1.5, color="steelblue", label="Fine-tune loss")
# axes[0,0].axvline(best_epoch, color="red", lw=1, linestyle="--",
#                   label=f"Best epoch {best_epoch}")
# axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
# axes[0,0].set_title("Fine-tuning Loss")
# axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.4)
# axes[0,0].set_yscale("log")

# # ── Plot 2: SOH prediction on full drone lifecycle ────────────────────────
# all_cycles = np.concatenate([ft_cycles, te_cycles])
# all_soh    = np.concatenate([ft_soh, te_soh])

# model.eval()
# with torch.no_grad():
#     y_ft_pred = model(X_ft).cpu().numpy().squeeze()

# axes[0,1].plot(all_cycles, all_soh, "k-", lw=2,
#                label="True SOH", zorder=3)
# axes[0,1].plot(te_cycles, y_post, "r--", lw=1.5,
#                label=f"Predicted (test)  RMSE={rmse_post:.4f}")
# axes[0,1].plot(ft_cycles, y_ft_pred, "g--", lw=1.5,
#                label="Predicted (fine-tune)")
# axes[0,1].axvspan(ft_cycles[0], ft_cycles[-1],
#                   alpha=0.15, color="green",
#                   label=f"Fine-tune region (10%)")
# axes[0,1].axhline(0.80, color="red", lw=1, linestyle=":",
#                   label="EOL 80%")
# axes[0,1].set_xlabel("Cycle"); axes[0,1].set_ylabel("SOH")
# axes[0,1].set_title(f"SOH Prediction -- Transformer\n"
#                     f"RMSE={rmse_post:.4f}  MAE={mae_post:.4f}  "
#                     f"R²={r2_post:.4f}")
# axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.4)
# axes[0,1].set_ylim(0.74, 1.02)

# # ── Plot 3: Error distribution ────────────────────────────────────────────
# errors = y_post - te_soh
# axes[1,0].hist(errors, bins=30, color="steelblue",
#                alpha=0.8, edgecolor="black", lw=0.5)
# axes[1,0].axvline(0, color="red", lw=1.5, linestyle="--",
#                   label="Zero error")
# axes[1,0].axvline(errors.mean(), color="orange", lw=1.5,
#                   linestyle="--",
#                   label=f"Mean={errors.mean():.4f}")
# axes[1,0].set_xlabel("Prediction error (pred - true)")
# axes[1,0].set_ylabel("Count")
# axes[1,0].set_title("Error Distribution on Test Set")
# axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.4)

# # ── Plot 4: Scatter ────────────────────────────────────────────────────────
# axes[1,1].scatter(te_soh, y_post, s=10, alpha=0.6,
#                   color="steelblue")
# lims = [min(te_soh.min(), y_post.min()),
#         max(te_soh.max(), y_post.max())]
# axes[1,1].plot(lims, lims, "r--", lw=1.5,
#                label="Perfect prediction")
# axes[1,1].set_xlabel("True SOH")
# axes[1,1].set_ylabel("Predicted SOH")
# axes[1,1].set_title(f"Predicted vs True SOH (test set)\n"
#                     f"RMSE={rmse_post:.4f}  R²={r2_post:.4f}")
# axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.4)

# plt.suptitle(f"Transformer Fine-tuning Results -- Drone Pack Data\n"
#              f"Pretrained on 4083 source cycles, "
#              f"fine-tuned on {len(ft_soh)} drone cycles (10%), "
#              f"tested on {len(te_soh)} cycles (90%)",
#              fontsize=12, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR,
#             "transformer_finetune_results.png"),
#             dpi=150, bbox_inches="tight")
# plt.close()
# print("Saved -> transformer_finetune_results.png")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 7: FINAL SUMMARY
# # ══════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("TRANSFORMER FINE-TUNING SUMMARY")
# print("=" * 60)
# print(f"  Fine-tune cycles  : {len(ft_soh)} (10% of drone data)")
# print(f"  Test cycles       : {len(te_soh)} (90% of drone data)")
# print(f"  Fine-tune epochs  : {len(train_losses)}")
# print(f"  Best epoch        : {best_epoch}")
# print(f"\n  Results on test set:")
# print(f"    RMSE before FT  : {rmse_pre:.4f}")
# print(f"    RMSE after FT   : {rmse_post:.4f}")
# print(f"    MAE  after FT   : {mae_post:.4f}")
# print(f"    R²   after FT   : {r2_post:.4f}")
# print(f"    Improvement     : "
#       f"{(rmse_pre - rmse_post)/rmse_pre * 100:.1f}%")
# print(f"\n  Comparison with LSTM fine-tuning:")
# print(f"    LSTM       : RMSE=0.0200  MAE=0.0150  R²=0.7746")
# print(f"    Transformer: RMSE={rmse_post:.4f}  "
#       f"MAE={mae_post:.4f}  R²={r2_post:.4f}")
# print(f"\n  Output files:")
# print(f"    transformer_finetuned.pt")
# print(f"    transformer_finetune_log.csv")
# print(f"    transformer_results.csv")
# print(f"    transformer_finetune_results.png")
# print(f"\n  Next step:")
# print(f"    Implement GRU pretraining + fine-tuning")
# print(f"    Final comparison: LSTM vs Transformer vs GRU")
# print("\nDone.")
# """
# finetune_transformer.py
# =======================
# Fine-tunes the pretrained Transformer on drone pack data.

# Updates from v1:
#     - Weight decay in Adam optimizer (L2 regularisation)
#     - Encoder frozen -- only output head is fine-tuned
#       Prevents catastrophic forgetting on 35 cycles

# Architecture matches pretrain_transformer.py exactly.

# Run from COMBINED/05_finetune_transformer/ directory:
#     python finetune_transformer.py
# """

# import os
# import math
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # ══════════════════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════
# FEAT_DIR     = "../01b_pack_input/processed"
# PRETRAIN_DIR = "00_processed"
# OUT_DIR      = "01_processed"
# os.makedirs(OUT_DIR, exist_ok=True)

# FUSED_DIM       = 38
# D_MODEL         = 64
# NHEAD           = 4
# NUM_LAYERS      = 2
# DIM_FEEDFORWARD = 128
# DROPOUT         = 0.1

# EPOCHS       = 500
# LR           = 1e-4
# WEIGHT_DECAY = 1e-4
# LAMBDA1      = 1.0
# LR_PATIENCE  = 30
# LR_FACTOR    = 0.5
# LR_MIN       = 1e-7
# ES_PATIENCE  = 80
# SEED         = 42

# torch.manual_seed(SEED)
# np.random.seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {DEVICE}")

# # ══════════════════════════════════════════════════════════════════════════
# # MODEL (identical to pretraining)
# # ══════════════════════════════════════════════════════════════════════════
# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=2000, dropout=0.1):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#         pe  = torch.zeros(max_len, d_model)
#         pos = torch.arange(0, max_len).unsqueeze(1).float()
#         div = torch.exp(
#             torch.arange(0, d_model, 2).float()
#             * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(pos * div)
#         pe[:, 1::2] = torch.cos(pos * div)
#         self.register_buffer("pe", pe.unsqueeze(0))

#     def forward(self, x):
#         x = x + self.pe[0, :x.size(0), :]
#         return self.dropout(x)


# class TransformerModel(nn.Module):
#     def __init__(self, input_dim=38, d_model=64, nhead=4,
#                  num_layers=2, dim_feedforward=128, dropout=0.1):
#         super().__init__()
#         self.input_proj  = nn.Linear(input_dim, d_model)
#         self.pos_enc     = PositionalEncoding(d_model, dropout=dropout)
#         encoder_layer    = nn.TransformerEncoderLayer(
#             d_model=d_model, nhead=nhead,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout, batch_first=False, activation="relu")
#         self.transformer = nn.TransformerEncoder(
#             encoder_layer, num_layers=num_layers)
#         self.head = nn.Sequential(
#             nn.Linear(d_model, 64), nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(64, 32),      nn.ReLU(),
#             nn.Linear(32, 1),
#         )

#     def forward(self, x):
#         x = self.input_proj(x)
#         x = self.pos_enc(x)
#         x = self.transformer(x.unsqueeze(1)).squeeze(1)
#         return self.head(x)


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
# # STEP 1: LOAD DATA
# # ══════════════════════════════════════════════════════════════════════════
# print("=" * 60)
# print("Transformer Fine-tuning v2 -- Frozen Encoder + Weight Decay")
# print("=" * 60)

# ft_fused = np.load(os.path.join(FEAT_DIR, "finetune_fused.npy"))
# ft_soh   = np.load(os.path.join(FEAT_DIR, "finetune_soh.npy"))
# te_fused = np.load(os.path.join(FEAT_DIR, "test_fused.npy"))
# te_soh   = np.load(os.path.join(FEAT_DIR, "test_soh.npy"))
# index    = pd.read_csv(os.path.join(FEAT_DIR, "drone_index.csv"))

# print(f"\nLoaded:")
# print(f"  Fine-tune : {ft_fused.shape}  "
#       f"SOH {ft_soh.min():.3f}-{ft_soh.max():.3f}")
# print(f"  Test      : {te_fused.shape}  "
#       f"SOH {te_soh.min():.3f}-{te_soh.max():.3f}")

# ft_cycles = index["cycle"].values[:len(ft_soh)]
# te_cycles = index["cycle"].values[len(ft_soh):]

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 2: LOAD PRETRAINED MODEL + FREEZE ENCODER
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nLoading pretrained Transformer weights ...")

# model = TransformerModel(
#     input_dim=FUSED_DIM, d_model=D_MODEL, nhead=NHEAD,
#     num_layers=NUM_LAYERS, dim_feedforward=DIM_FEEDFORWARD,
#     dropout=DROPOUT).to(DEVICE)

# model.load_state_dict(
#     torch.load(os.path.join(PRETRAIN_DIR, "transformer_pretrained.pt"),
#                map_location=DEVICE))

# n_params = sum(p.numel() for p in model.parameters()
#                if p.requires_grad)
# print(f"  Parameters loaded : {n_params:,}")

# # Freeze encoder -- only fine-tune the output head
# for name, param in model.named_parameters():
#     if "head" not in name:
#         param.requires_grad = False

# n_trainable = sum(p.numel() for p in model.parameters()
#                   if p.requires_grad)
# n_frozen    = sum(p.numel() for p in model.parameters()
#                   if not p.requires_grad)
# print(f"  Trainable params  : {n_trainable:,}  (head only)")
# print(f"  Frozen params     : {n_frozen:,}  (encoder frozen)")

# # Evaluate BEFORE fine-tuning
# model.eval()
# with torch.no_grad():
#     X_te  = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
#     y_pre = model(X_te).cpu().numpy().squeeze()
# rmse_pre = float(np.sqrt(np.mean((y_pre - te_soh)**2)))
# mae_pre  = float(np.mean(np.abs(y_pre - te_soh)))
# print(f"\n  Performance BEFORE fine-tuning:")
# print(f"    RMSE = {rmse_pre:.4f}")
# print(f"    MAE  = {mae_pre:.4f}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 3: FINE-TUNE (only trainable params updated)
# # ══════════════════════════════════════════════════════════════════════════
# # Only pass trainable parameters to optimizer
# optimizer = torch.optim.Adam(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=LR, weight_decay=WEIGHT_DECAY)
# scheduler = ReduceLROnPlateau(optimizer, mode="min",
#                                patience=LR_PATIENCE,
#                                factor=LR_FACTOR, min_lr=LR_MIN)

# print(f"\nFine-tuning head only (max {EPOCHS} epochs, "
#       f"lr={LR}, wd={WEIGHT_DECAY}) ...")
# print(f"{'Epoch':>6}  {'Train Loss':>12}  {'LR':>10}")
# print("-" * 35)

# X_ft = torch.tensor(ft_fused, dtype=torch.float32).to(DEVICE)
# y_ft = torch.tensor(ft_soh,   dtype=torch.float32).unsqueeze(1).to(DEVICE)

# train_losses = []
# best_loss    = float("inf")
# best_epoch   = 0
# es_counter   = 0
# best_weights = None

# for epoch in range(1, EPOCHS + 1):
#     model.train()
#     optimizer.zero_grad()
#     loss, _, _ = composite_loss(model(X_ft), y_ft)
#     loss.backward()
#     nn.utils.clip_grad_norm_(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         max_norm=1.0)
#     optimizer.step()

#     train_loss = loss.item()
#     scheduler.step(train_loss)
#     train_losses.append(train_loss)
#     current_lr = optimizer.param_groups[0]["lr"]

#     if train_loss < best_loss:
#         best_loss    = train_loss
#         best_epoch   = epoch
#         es_counter   = 0
#         best_weights = {k: v.cpu().clone()
#                         for k, v in model.state_dict().items()}
#         marker = " <-- best"
#     else:
#         es_counter += 1
#         marker = ""

#     if epoch % 50 == 0 or epoch == 1 or marker:
#         print(f"{epoch:>6}  {train_loss:>12.6f}  "
#               f"{current_lr:>10.2e}{marker}")

#     if es_counter >= ES_PATIENCE:
#         print(f"\nEarly stopping at epoch {epoch} "
#               f"(best at epoch {best_epoch})")
#         break

# model.load_state_dict(best_weights)
# print(f"\nRestored best weights from epoch {best_epoch}")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 4: EVALUATE
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nEvaluating on test set ({len(te_soh)} cycles) ...")

# model.eval()
# with torch.no_grad():
#     X_te   = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
#     y_post = model(X_te).cpu().numpy().squeeze()

# rmse_post = float(np.sqrt(np.mean((y_post - te_soh)**2)))
# mae_post  = float(np.mean(np.abs(y_post - te_soh)))
# r2_post   = float(1 - np.sum((y_post - te_soh)**2) /
#                   np.sum((te_soh - te_soh.mean())**2))

# print(f"\n  Performance AFTER fine-tuning:")
# print(f"    RMSE = {rmse_post:.4f}  (was {rmse_pre:.4f} before)")
# print(f"    MAE  = {mae_post:.4f}  (was {mae_pre:.4f} before)")
# print(f"    R²   = {r2_post:.4f}")
# print(f"    RMSE improvement: "
#       f"{(rmse_pre - rmse_post)/rmse_pre * 100:.1f}%")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 5: SAVE
# # ══════════════════════════════════════════════════════════════════════════
# torch.save(model.state_dict(),
#            os.path.join(OUT_DIR, "transformer_finetuned.pt"))
# pd.DataFrame({
#     "epoch": range(1, len(train_losses)+1),
#     "train_loss": train_losses,
# }).to_csv(os.path.join(OUT_DIR, "transformer_finetune_log.csv"),
#           index=False)
# pd.DataFrame({
#     "cycle": te_cycles, "soh_true": te_soh,
#     "soh_pred": y_post, "error": y_post - te_soh,
#     "abs_error": np.abs(y_post - te_soh),
# }).to_csv(os.path.join(OUT_DIR, "transformer_results.csv"), index=False)

# print(f"\nSaved -> transformer_finetuned.pt")
# print(f"Saved -> transformer_finetune_log.csv")
# print(f"Saved -> transformer_results.csv")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 6: PLOTS
# # ══════════════════════════════════════════════════════════════════════════
# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# axes[0,0].plot(range(1, len(train_losses)+1), train_losses,
#                lw=1.5, color="steelblue", label="Fine-tune loss")
# axes[0,0].axvline(best_epoch, color="red", lw=1, linestyle="--",
#                   label=f"Best epoch {best_epoch}")
# axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
# axes[0,0].set_title("Fine-tuning Loss (head only)")
# axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.4)
# axes[0,0].set_yscale("log")

# all_cycles = np.concatenate([ft_cycles, te_cycles])
# all_soh    = np.concatenate([ft_soh, te_soh])
# model.eval()
# with torch.no_grad():
#     y_ft_pred = model(X_ft).cpu().numpy().squeeze()

# axes[0,1].plot(all_cycles, all_soh, "k-", lw=2,
#                label="True SOH", zorder=3)
# axes[0,1].plot(te_cycles, y_post, "r--", lw=1.5,
#                label=f"Predicted (test) RMSE={rmse_post:.4f}")
# axes[0,1].plot(ft_cycles, y_ft_pred, "g--", lw=1.5,
#                label="Predicted (fine-tune)")
# axes[0,1].axvspan(ft_cycles[0], ft_cycles[-1],
#                   alpha=0.15, color="green",
#                   label="Fine-tune region (10%)")
# axes[0,1].axhline(0.80, color="red", lw=1, linestyle=":")
# axes[0,1].set_xlabel("Cycle"); axes[0,1].set_ylabel("SOH")
# axes[0,1].set_title(f"SOH Prediction -- Transformer v2\n"
#                     f"RMSE={rmse_post:.4f}  MAE={mae_post:.4f}  "
#                     f"R²={r2_post:.4f}")
# axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.4)
# axes[0,1].set_ylim(0.74, 1.02)

# errors = y_post - te_soh
# axes[1,0].hist(errors, bins=30, color="steelblue",
#                alpha=0.8, edgecolor="black", lw=0.5)
# axes[1,0].axvline(0, color="red", lw=1.5, linestyle="--",
#                   label="Zero error")
# axes[1,0].axvline(errors.mean(), color="orange", lw=1.5,
#                   linestyle="--",
#                   label=f"Mean={errors.mean():.4f}")
# axes[1,0].set_xlabel("Prediction error")
# axes[1,0].set_ylabel("Count")
# axes[1,0].set_title("Error Distribution")
# axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.4)

# axes[1,1].scatter(te_soh, y_post, s=10, alpha=0.6,
#                   color="steelblue")
# lims = [min(te_soh.min(), y_post.min()),
#         max(te_soh.max(), y_post.max())]
# axes[1,1].plot(lims, lims, "r--", lw=1.5,
#                label="Perfect prediction")
# axes[1,1].set_xlabel("True SOH")
# axes[1,1].set_ylabel("Predicted SOH")
# axes[1,1].set_title(f"Predicted vs True\n"
#                     f"RMSE={rmse_post:.4f}  R²={r2_post:.4f}")
# axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.4)

# plt.suptitle(f"Transformer v2 Fine-tuning -- Drone Pack\n"
#              f"Frozen encoder, head-only fine-tuning, "
#              f"weight_decay={WEIGHT_DECAY}",
#              fontsize=12, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "transformer_finetune_results.png"),
#             dpi=150, bbox_inches="tight")
# plt.close()
# print("Saved -> transformer_finetune_results.png")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 7: SUMMARY
# # ══════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("TRANSFORMER FINE-TUNING SUMMARY (v2)")
# print("=" * 60)
# print(f"  Fine-tune cycles  : {len(ft_soh)} (10%)")
# print(f"  Test cycles       : {len(te_soh)} (90%)")
# print(f"  Trainable params  : {n_trainable:,} (head only)")
# print(f"  Fine-tune epochs  : {len(train_losses)}")
# print(f"  Best epoch        : {best_epoch}")
# print(f"\n  Results on test set:")
# print(f"    RMSE before FT  : {rmse_pre:.4f}")
# print(f"    RMSE after FT   : {rmse_post:.4f}")
# print(f"    MAE  after FT   : {mae_post:.4f}")
# print(f"    R²   after FT   : {r2_post:.4f}")
# print(f"    Improvement     : "
#       f"{(rmse_pre - rmse_post)/rmse_pre * 100:.1f}%")
# print(f"\n  Comparison with LSTM fine-tuning:")
# print(f"    LSTM        : RMSE=0.0200  MAE=0.0150  R²=0.7746")
# print(f"    Transformer : RMSE={rmse_post:.4f}  "
#       f"MAE={mae_post:.4f}  R²={r2_post:.4f}")
# print(f"\n  Output files:")
# print(f"    transformer_finetuned.pt")
# print(f"    transformer_finetune_log.csv")
# print(f"    transformer_results.csv")
# print(f"    transformer_finetune_results.png")
# print(f"\n  Next step: GRU pretraining + fine-tuning")
# print("\nDone.")

"""
finetune_transformer.py
=======================
Fine-tunes the pretrained Transformer on drone pack data.
Implements continual backpropagation (Fan et al. 2025, Section 3.4)
adapted from LSTM to Transformer architecture.

Continual Backpropagation (Fan et al. 2025, Eq. 9):
    Tracks a utility score per neuron in every linear layer:
        u[i] = eta * u[i] + (1-eta) * |h[i]| * sum_k(|W[i,k]|)
    where:
        h[i]     = neuron activation at current step
        W[i,k]   = outgoing weights from neuron i to next layer
        eta      = 0.01  (decay rate, from paper)
        rho      = 1e-4  (lowered from paper's 3e-4: transformer has
                          more linear layers than LSTM, so the utility
                          threshold fires more at the original value)
    maturity = 30 epochs (raised from 10: allow adaptation before reinit)

    Applied to all linear layers in the Transformer:
        input_proj, feedforward layers inside encoder, and head.
    Paper uses it on LSTM hidden units; we extend to Transformer
    linear layers as the equivalent hidden unit representation.

Fine-tuning strategy (v9):
    - Full model unfrozen, LR=5e-5 (halved: 1e-4 caused spiky oscillating loss)
    - L2-SP loss: lambda=0.05 * ||theta - theta_pretrain||^2
      Prevents fine-tune SOH range (0.946-0.969) from overwriting
      pretrained knowledge of the full 0.75-1.0 degradation range.
    - ES on training loss, patience=100 (avoids noisy 7-sample val signal)
    - Gaussian noise augmentation on features (std=0.002)
    - All 35 fine-tune cycles used for training (no val split)
    - Causal mask (must match v6 pretraining exactly)

Architecture matches pretrain_transformer.py v4 EXACTLY:
    Linear projection (38 -> 48)
    2 x TransformerEncoderLayer (nhead=4, ff=96, dropout=0.10)
    fc(48) -> Dropout(0.10) -> fc(32) -> ReLU -> fc(1)

Run from COMBINED/05_finetune_transformer/ directory:
    python finetune_transformer.py
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# =====================================================================
# CONFIGURATION
# =====================================================================
FEAT_DIR     = "../01b_pack_input/processed"
PRETRAIN_DIR = "00_processed"
OUT_DIR      = "01_processed"
os.makedirs(OUT_DIR, exist_ok=True)

# Must match pretrain_transformer.py v5 exactly
FUSED_DIM       = 38
D_MODEL         = 64
NHEAD           = 4
NUM_LAYERS      = 2
DIM_FEEDFORWARD = 128
DROPOUT         = 0.10

# Fine-tuning hyperparameters
EPOCHS       = 1500
LR           = 5e-5
WEIGHT_DECAY = 0.0
LAMBDA1      = 0.0
LR_PATIENCE  = 80
LR_FACTOR    = 0.5
LR_MIN       = 1e-8
ES_PATIENCE  = 250
SEED         = 42

LAMBDA_L2SP  = 0.05

CBP_ETA      = 0.01
CBP_RHO      = 3e-4
CBP_MATURITY = 10
CBP_INTERVAL = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# =====================================================================
# MODEL
# =====================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class TransformerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(FUSED_DIM, D_MODEL)
        self.pos_enc     = PositionalEncoding(D_MODEL, dropout=DROPOUT)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=NHEAD,
            dim_feedforward=DIM_FEEDFORWARD,
            dropout=DROPOUT, batch_first=True, activation="relu")
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=NUM_LAYERS)
        self.head = nn.Sequential(
            nn.Linear(D_MODEL, 32), nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 16),      nn.ReLU(),
            nn.Linear(16, 1),
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


# =====================================================================
# LOSS
# =====================================================================
def composite_loss(pred, target, lambda1=LAMBDA1):
    errors_sq  = (pred - target) ** 2
    l_mse      = errors_sq.mean()
    large_mask = errors_sq > l_mse.detach()
    l_emse     = errors_sq[large_mask].mean() \
                 if large_mask.sum() > 0 \
                 else torch.tensor(0.0, device=pred.device)
    return l_mse + lambda1 * l_emse, l_mse, l_emse


# =====================================================================
# CONTINUAL BACKPROPAGATION (Fan et al. 2025, Eq. 9)
# Adapted from LSTM to Transformer linear layers
# =====================================================================
class ContinualBackprop:
    """
    Utility per neuron i in layer l (Fan et al. Eq. 9):
        u[i] = eta*u[i] + (1-eta)*|h[i]|*sum_k(|W[i,k]|)

    Reinitialisation condition:
        - neuron age >= maturity_threshold
        - utility < rho * mean_utility_of_layer

    On reinitialisation:
        - weight row reset via kaiming_uniform_
        - bias reset to zero
        - age and utility reset to 0
    """

    def __init__(self, model, eta=CBP_ETA, rho=CBP_RHO,
                 maturity=CBP_MATURITY):
        self.model    = model
        self.eta      = eta
        self.rho      = rho
        self.maturity = maturity

        self.linear_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                n_out = module.out_features
                self.linear_layers.append({
                    "name"   : name,
                    "module" : module,
                    "utility": torch.zeros(n_out).flatten(),
                    "age"    : torch.zeros(n_out, dtype=torch.long).flatten(),
                })

        self.total_reinit   = 0
        self.reinit_history = []

        print(f"\n  Continual Backprop initialised on "
              f"{len(self.linear_layers)} linear layers:")
        for info in self.linear_layers:
            m = info["module"]
            print(f"    {info['name']:45s}  "
                  f"({m.out_features}, {m.in_features})")

    def update(self, activations_dict, epoch):
        epoch_reinit = 0
        for info in self.linear_layers:
            name   = info["name"]
            module = info["module"]
            u      = info["utility"]
            age    = info["age"]

            if name not in activations_dict:
                info["age"] = age + 1
                continue

            h     = activations_dict[name].detach().cpu()
            w_sum = module.weight.detach().cpu().abs().sum(dim=1)

            # Fan et al. Eq. 9
            u_new = (self.eta * u + (1 - self.eta) * h.abs() * w_sum).flatten()
            info["utility"] = u_new
            info["age"]     = (age + 1).flatten()

            mean_u    = u_new.mean().item()
            threshold = self.rho * mean_u if mean_u > 0 else 0.0

            mature_mask = info["age"] >= self.maturity
            low_u_mask  = u_new < threshold
            reinit_mask = mature_mask & low_u_mask
            n_reinit    = reinit_mask.sum().item()

            if n_reinit > 0:
                idx = reinit_mask.nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    nn.init.kaiming_uniform_(
                        module.weight[idx], a=math.sqrt(5))
                    if module.bias is not None:
                        module.bias[idx].zero_()
                info["age"][reinit_mask]     = 0
                info["utility"][reinit_mask] = 0.0
                epoch_reinit      += n_reinit
                self.total_reinit += n_reinit
                self.reinit_history.append((epoch, name, int(n_reinit)))

        return epoch_reinit

    def mean_utility(self):
        all_u = torch.cat([info["utility"].flatten()
                           for info in self.linear_layers])
        return float(all_u.mean())


def collect_activations(model, X):
    activations = {}
    hooks       = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def make_hook(n, out_features):
                def hook(mod, inp, out):
                    # out can be (seq, out_features) or
                    # (batch, seq, out_features) or (N, out_features)
                    # Always reduce to (out_features,) by flattening
                    # all dims except the last then taking mean
                    o = out.detach()
                    activations[n] = o.reshape(-1, out_features).mean(dim=0)
                return hook
            hooks.append(module.register_forward_hook(
                make_hook(name, module.out_features)))
    with torch.no_grad():
        model(X)
    for h in hooks:
        h.remove()
    return activations


# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
print("=" * 60)
print("Transformer Fine-tuning v33 -- old pretrain restored + L2SP=0.05 + LR=5e-5")
print("(Fan et al. 2025, Section 3.4, adapted for Transformer)")
print("=" * 60)

ft_fused = np.load(os.path.join(FEAT_DIR, "finetune_fused.npy"))
ft_soh   = np.load(os.path.join(FEAT_DIR, "finetune_soh.npy"))
te_fused = np.load(os.path.join(FEAT_DIR, "test_fused.npy"))
te_soh   = np.load(os.path.join(FEAT_DIR, "test_soh.npy"))
index    = pd.read_csv(os.path.join(FEAT_DIR, "drone_index.csv"))

print(f"\nLoaded:")
print(f"  Fine-tune : {ft_fused.shape}  "
      f"SOH {ft_soh.min():.3f}-{ft_soh.max():.3f}")
print(f"  Test      : {te_fused.shape}  "
      f"SOH {te_soh.min():.3f}-{te_soh.max():.3f}")

ft_cycles = index["cycle"].values[:len(ft_soh)]
te_cycles = index["cycle"].values[len(ft_soh):]

# Use all fine-tune cycles for training (same as LSTM).
# A 20% val split gives only 7 samples -- too few for reliable ES.
ft_train_fused = ft_fused
ft_train_soh   = ft_soh

print(f"\n  Fine-tune: {ft_train_fused.shape}  "
      f"SOH {ft_train_soh.min():.3f}-{ft_train_soh.max():.3f}")
print(f"  ES patience={ES_PATIENCE}, LR={LR}, LAMBDA1={LAMBDA1}")

# =====================================================================
# STEP 2: LOAD PRETRAINED MODEL
# =====================================================================
print(f"\nLoading pretrained Transformer v6 weights ...")

model = TransformerModel().to(DEVICE)

model.load_state_dict(
    torch.load(os.path.join(PRETRAIN_DIR, "transformer_pretrained.pt"),
               map_location=DEVICE))

n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"  Total parameters  : {n_params:,}  (all unfrozen)")
print(f"  Strategy          : single-phase LR={LR}, "
      f"L2SP={LAMBDA_L2SP}, CBP eta={CBP_ETA}, rho={CBP_RHO}")

# Snapshot pretrained weights for L2-SP penalty
pretrain_snapshot = {name: param.clone().detach()
                     for name, param in model.named_parameters()}

model.eval()
with torch.no_grad():
    X_te  = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
    y_pre = model(X_te).cpu().numpy().squeeze()
rmse_pre = float(np.sqrt(np.mean((y_pre - te_soh)**2)))
mae_pre  = float(np.mean(np.abs(y_pre - te_soh)))
print(f"\n  Performance BEFORE fine-tuning:")
print(f"    RMSE = {rmse_pre:.4f}")
print(f"    MAE  = {mae_pre:.4f}")

# =====================================================================
# STEP 3: INITIALISE OPTIMISER + CONTINUAL BACKPROP
# =====================================================================
optimizer = torch.optim.AdamW(model.parameters(),
                               lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="min",
                               patience=LR_PATIENCE,
                               factor=LR_FACTOR, min_lr=LR_MIN)

cbp = ContinualBackprop(model, eta=CBP_ETA,
                         rho=CBP_RHO, maturity=CBP_MATURITY)

X_ft_tr = torch.tensor(ft_train_fused, dtype=torch.float32).to(DEVICE)
y_ft_tr = torch.tensor(ft_train_soh,   dtype=torch.float32).to(DEVICE)

print(f"\nFine-tuning (max {EPOCHS} epochs, ES patience={ES_PATIENCE}) ...")
print(f"{'Epoch':>6}  {'Train':>10}  {'LR':>9}  {'Reinit':>6}  {'MeanU':>8}  {'ES':>4}")
print("-" * 50)

# =====================================================================
# STEP 4: TRAINING LOOP WITH CONTINUAL BACKPROPAGATION
# =====================================================================
train_losses    = []
reinit_history  = []
utility_history = []
best_loss       = float("inf")
best_epoch      = 0
es_counter      = 0
best_weights    = None

NOISE_STD = 0.002

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimizer.zero_grad()
    X_noisy = X_ft_tr + NOISE_STD * torch.randn_like(X_ft_tr)
    loss, _, _ = composite_loss(model(X_noisy), y_ft_tr)
    l2sp = sum(
        ((p - pretrain_snapshot[n].to(DEVICE)) ** 2).sum()
        for n, p in model.named_parameters()
    )
    loss = loss + LAMBDA_L2SP * l2sp
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss = loss.item()

    scheduler.step(train_loss)
    train_losses.append(train_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    epoch_reinit = 0
    if epoch % CBP_INTERVAL == 0:
        model.eval()
        acts = collect_activations(model, X_ft_tr)
        model.train()
        epoch_reinit = cbp.update(acts, epoch)

    mean_u = cbp.mean_utility()
    reinit_history.append(epoch_reinit)
    utility_history.append(mean_u)

    if train_loss < best_loss:
        best_loss    = train_loss
        best_epoch   = epoch
        es_counter   = 0
        best_weights = {k: v.cpu().clone()
                        for k, v in model.state_dict().items()}
        marker = " <-- best"
    else:
        es_counter += 1
        marker = ""

    if epoch % 50 == 0 or epoch == 1 or marker:
        print(f"{epoch:>6}  {train_loss:>10.6f}  "
              f"{current_lr:>9.2e}  {epoch_reinit:>6}  "
              f"{mean_u:>8.5f}  {es_counter:>4}{marker}")

    if es_counter >= ES_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(best at epoch {best_epoch})")
        break

model.load_state_dict(best_weights)
print(f"\nRestored best weights from epoch {best_epoch}")
print(f"Best train loss      : {best_loss:.6f}")
print(f"Total reinitialised  : {cbp.total_reinit} neurons")

# =====================================================================
# STEP 5: EVALUATE
# =====================================================================
print(f"\nEvaluating on test set ({len(te_soh)} cycles) ...")

model.eval()
with torch.no_grad():
    X_te   = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
    y_post = model(X_te).cpu().numpy().squeeze()

rmse_post   = float(np.sqrt(np.mean((y_post - te_soh)**2)))
mae_post    = float(np.mean(np.abs(y_post - te_soh)))
r2_post     = float(1 - np.sum((y_post - te_soh)**2) /
                    np.sum((te_soh - te_soh.mean())**2))
improvement = (rmse_pre - rmse_post) / rmse_pre * 100

print(f"\n  Performance AFTER fine-tuning:")
print(f"    RMSE = {rmse_post:.4f}  (was {rmse_pre:.4f} before)")
print(f"    MAE  = {mae_post:.4f}  (was {mae_pre:.4f} before)")
print(f"    R2   = {r2_post:.4f}")
print(f"    RMSE improvement: {improvement:.1f}%")

# =====================================================================
# STEP 6: SAVE
# =====================================================================
torch.save(model.state_dict(),
           os.path.join(OUT_DIR, "transformer_finetuned.pt"))
pd.DataFrame({
    "epoch"       : range(1, len(train_losses) + 1),
    "train_loss"  : train_losses,
    "reinit"      : reinit_history,
    "mean_utility": utility_history,
}).to_csv(os.path.join(OUT_DIR, "transformer_finetune_log.csv"),
          index=False)
pd.DataFrame({
    "cycle"    : te_cycles,
    "soh_true" : te_soh,
    "soh_pred" : y_post,
    "error"    : y_post - te_soh,
    "abs_error": np.abs(y_post - te_soh),
}).to_csv(os.path.join(OUT_DIR, "transformer_results.csv"), index=False)

if cbp.reinit_history:
    pd.DataFrame(cbp.reinit_history,
                 columns=["epoch", "layer", "n_reinit"]
                 ).to_csv(os.path.join(OUT_DIR,
                          "cbp_reinit_history.csv"), index=False)

print(f"\nSaved -> transformer_finetuned.pt")
print(f"Saved -> transformer_finetune_log.csv")
print(f"Saved -> transformer_results.csv")
print(f"Saved -> cbp_reinit_history.csv")

# =====================================================================
# STEP 7: PLOTS
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
epochs_r  = range(1, len(train_losses) + 1)

axes[0, 0].plot(epochs_r, train_losses, lw=1.5,
                color="steelblue", label="Train loss")
axes[0, 0].axvline(best_epoch, color="red", lw=1, linestyle="--",
                   label=f"Best epoch {best_epoch}")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Fine-tuning Loss (log scale)")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.4)
axes[0, 0].set_yscale("log")

axes[0, 1].plot(epochs_r, utility_history, lw=1.5, color="purple")
axes[0, 1].axvline(best_epoch, color="red", lw=1, linestyle="--")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Mean utility")
axes[0, 1].set_title("CBP Mean Neuron Utility")
axes[0, 1].grid(True, alpha=0.4)

axes[0, 2].bar(list(epochs_r), reinit_history,
               color="crimson", alpha=0.7, width=1.0)
axes[0, 2].set_xlabel("Epoch")
axes[0, 2].set_ylabel("Neurons reinitialised")
axes[0, 2].set_title(f"CBP Reinitialisation Events\n"
                     f"Total: {cbp.total_reinit}")
axes[0, 2].grid(True, alpha=0.4)

all_cycles = np.concatenate([ft_cycles, te_cycles])
all_soh    = np.concatenate([ft_soh, te_soh])
model.eval()
with torch.no_grad():
    y_ft_pred = model(
        torch.tensor(ft_fused, dtype=torch.float32).to(DEVICE)
    ).cpu().numpy().squeeze()

axes[1, 0].plot(all_cycles, all_soh, "k-", lw=2, label="True SOH")
axes[1, 0].plot(te_cycles, y_post, "r--", lw=1.5,
                label=f"Predicted (test) RMSE={rmse_post:.4f}")
axes[1, 0].plot(ft_cycles, y_ft_pred, "g--", lw=1.5,
                label="Predicted (fine-tune)")
axes[1, 0].axvspan(ft_cycles[0], ft_cycles[-1],
                   alpha=0.15, color="green",
                   label="Fine-tune region (10%)")
axes[1, 0].axhline(0.80, color="red", lw=1, linestyle=":")
axes[1, 0].set_xlabel("Cycle"); axes[1, 0].set_ylabel("SOH")
axes[1, 0].set_title(f"SOH Prediction\n"
                     f"RMSE={rmse_post:.4f}  MAE={mae_post:.4f}  "
                     f"R2={r2_post:.4f}")
axes[1, 0].legend(fontsize=7); axes[1, 0].grid(True, alpha=0.4)
axes[1, 0].set_ylim(0.74, 1.02)

errors = y_post - te_soh
axes[1, 1].hist(errors, bins=30, color="steelblue",
                alpha=0.8, edgecolor="black", lw=0.5)
axes[1, 1].axvline(0, color="red", lw=1.5, linestyle="--",
                   label="Zero error")
axes[1, 1].axvline(errors.mean(), color="orange", lw=1.5,
                   linestyle="--",
                   label=f"Mean={errors.mean():.4f}")
axes[1, 1].set_xlabel("Prediction error"); axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Error Distribution")
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.4)

axes[1, 2].scatter(te_soh, y_post, s=10, alpha=0.6, color="steelblue")
lims = [min(te_soh.min(), y_post.min()),
        max(te_soh.max(), y_post.max())]
axes[1, 2].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
axes[1, 2].set_xlabel("True SOH"); axes[1, 2].set_ylabel("Predicted SOH")
axes[1, 2].set_title(f"Predicted vs True\n"
                     f"RMSE={rmse_post:.4f}  R2={r2_post:.4f}")
axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.4)

plt.suptitle(
    f"Transformer Fine-tuning with Continual Backpropagation\n"
    f"eta={CBP_ETA}  rho={CBP_RHO}  maturity={CBP_MATURITY}  "
    f"total_reinit={cbp.total_reinit}  LR={LR}",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "transformer_finetune_results.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> transformer_finetune_results.png")

# =====================================================================
# STEP 8: SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("TRANSFORMER FINE-TUNING SUMMARY (v9 -- Causal Mask + CBP + L2-SP)")
print("=" * 60)
print(f"  Architecture      : Linear({FUSED_DIM}->{D_MODEL}) -> "
      f"Transformer(x{NUM_LAYERS}, ff={DIM_FEEDFORWARD}) -> "
      f"fc(48)->fc(32)->fc(1)")
print(f"  Total parameters  : {n_params:,}")
print(f"  Fine-tune epochs  : {len(train_losses)}")
print(f"  Best epoch        : {best_epoch}")
print(f"\n  Continual Backpropagation (Fan et al. 2025, Eq. 9):")
print(f"    Decay rate eta  : {CBP_ETA}")
print(f"    Replace rate rho: {CBP_RHO}")
print(f"    Maturity thresh : {CBP_MATURITY} epochs")
print(f"    Layers tracked  : {len(cbp.linear_layers)}")
print(f"    Total reinit    : {cbp.total_reinit} neurons")
print(f"\n  Results on test set:")
print(f"    RMSE before FT  : {rmse_pre:.4f}")
print(f"    RMSE after FT   : {rmse_post:.4f}")
print(f"    MAE  after FT   : {mae_post:.4f}")
print(f"    R2   after FT   : {r2_post:.4f}")
print(f"    Improvement     : {improvement:.1f}%")
print(f"\n  Version comparison:")
print(f"    LSTM            : RMSE=0.0200  MAE=0.0150  R2=0.7746")
print(f"    Transf. v4 (FT) : RMSE=0.0486  MAE=0.0374  R2=-0.326"
      f"  (head-only)")
print(f"    Transf. v9 (FT) : RMSE={rmse_post:.4f}  "
      f"MAE={mae_post:.4f}  R2={r2_post:.4f}"
      f"  (causal + CBP + L2-SP, LR=5e-5)")
print(f"\n  Output files:")
print(f"    transformer_finetuned.pt")
print(f"    transformer_finetune_log.csv  (reinit + utility included)")
print(f"    transformer_results.csv")
print(f"    cbp_reinit_history.csv")
print(f"    transformer_finetune_results.png")
print(f"\nDone.")