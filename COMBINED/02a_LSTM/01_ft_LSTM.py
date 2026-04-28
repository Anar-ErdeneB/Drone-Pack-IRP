# """
# finetune_lstm.py
# ================
# Fine-tunes the pretrained LSTM on drone pack data (target domain)
# and evaluates on the held-out test set.

# Following Fan et al. (2025):
#     - Fine-tune on first 10% of drone cycles (35 cycles)
#     - Evaluate on remaining 90% (315 cycles)
#     - Lower learning rate than pretraining (1e-4 vs 1e-3)
#     - Same loss function: L_mse + lambda1 * L_emse

# Architecture (Fan et al. 2025, Table 3):
#     fc(38) -> lstm(128) -> fc(64) -> fc(8) -> fc(1)

# During fine-tuning the fused vector changes from source domain:
#     Source:  [latent(24) | T_mean(1) | zeros(13)]
#     Drone:   [latent(24) | T_mean(1) | pack_feats(13)]
# The model sees real pack inconsistency features for the first time.

# Reads (from ../processed/):
#     finetune_fused.npy     (35, 38)   fine-tuning input
#     finetune_soh.npy       (35,)      fine-tuning labels
#     test_fused.npy         (315, 38)  test input
#     test_soh.npy           (315,)     test labels
#     drone_index.csv        (350, *)   cycle metadata

# Reads (from ../02_LSTM/processed/):
#     lstm_pretrained.pt     pretrained LSTM weights

# Writes (to processed/):
#     lstm_finetuned.pt          fine-tuned weights
#     lstm_finetune_log.csv      loss per epoch
#     lstm_finetune_curves.png   training curves
#     lstm_test_predictions.png  SOH predictions vs true
#     lstm_results.csv           per-cycle predictions and errors

# Run from COMBINED/03_finetune_lstm/ directory:
#     python finetune_lstm.py
# """

# import os
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import torch
# import torch.nn as nn
# from torch.optim.lr_scheduler import ReduceLROnPlateau

# # ══════════════════════════════════════════════════════════════════════════
# # CONFIGURATION
# # ══════════════════════════════════════════════════════════════════════════
# FEAT_DIR    = "../01b_pack_input/processed"              # drone fused features
# PRETRAIN_DIR= "00_processed"      # pretrained LSTM weights
# OUT_DIR     = "01_processed"
# os.makedirs(OUT_DIR, exist_ok=True)

# # Architecture (must match pretraining exactly)
# FUSED_DIM  = 38
# HIDDEN_DIM = 128

# # Fine-tuning hyperparameters
# EPOCHS      = 700   # fewer epochs than pretraining
# LR          = 1e-4       # lower LR to preserve pretrained knowledge
# LAMBDA1     = 1.0
# LR_PATIENCE = 30
# LR_FACTOR   = 0.5
# LR_MIN      = 1e-7
# ES_PATIENCE = 100
# SEED        = 42

# torch.manual_seed(SEED)
# np.random.seed(SEED)

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Device: {DEVICE}")

# # ══════════════════════════════════════════════════════════════════════════
# # MODEL DEFINITION (identical to pretraining)
# # ══════════════════════════════════════════════════════════════════════════
# class LSTMModel(nn.Module):
#     def __init__(self, input_dim=38, hidden_dim=128):
#         super().__init__()
#         self.input_fc = nn.Linear(input_dim, input_dim)
#         self.lstm     = nn.LSTM(input_size=input_dim,
#                                 hidden_size=hidden_dim,
#                                 num_layers=1,
#                                 batch_first=True)
#         self.head     = nn.Sequential(
#             nn.Linear(hidden_dim, 64), nn.ReLU(),
#             nn.Linear(64,          8), nn.ReLU(),
#             nn.Linear(8,           1),
#         )

#     def forward(self, x):
#         x      = torch.relu(self.input_fc(x))
#         out, _ = self.lstm(x.unsqueeze(0))
#         return self.head(out.squeeze(0))


# # ══════════════════════════════════════════════════════════════════════════
# # LOSS FUNCTION (identical to pretraining)
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
# print("LSTM Fine-tuning on Drone Pack Data")
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

# # Sort index to get cycle numbers
# ft_cycles = index["cycle"].values[:len(ft_soh)]
# te_cycles = index["cycle"].values[len(ft_soh):]

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 2: LOAD PRETRAINED MODEL
# # ══════════════════════════════════════════════════════════════════════════
# print(f"\nLoading pretrained LSTM weights ...")

# model = LSTMModel(FUSED_DIM, HIDDEN_DIM).to(DEVICE)
# model.load_state_dict(
#     torch.load(os.path.join(PRETRAIN_DIR, "lstm_pretrained.pt"),
#                map_location=DEVICE))

# n_params = sum(p.numel() for p in model.parameters()
#                if p.requires_grad)
# print(f"  Parameters loaded : {n_params:,}")
# print(f"  Architecture      : fc({FUSED_DIM}) -> lstm({HIDDEN_DIM})"
#       f" -> fc(64) -> fc(8) -> fc(1)")

# # Evaluate pretrained model BEFORE fine-tuning
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
#     y_pred        = model(X_ft)
#     loss, _, _    = composite_loss(y_pred, y_ft)
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
#            os.path.join(OUT_DIR, "lstm_finetuned.pt"))

# log_df = pd.DataFrame({
#     "epoch"     : range(1, len(train_losses) + 1),
#     "train_loss": train_losses,
# })
# log_df.to_csv(os.path.join(OUT_DIR, "lstm_finetune_log.csv"), index=False)

# # Save per-cycle predictions
# results_df = pd.DataFrame({
#     "cycle"    : te_cycles,
#     "soh_true" : te_soh,
#     "soh_pred" : y_post,
#     "error"    : y_post - te_soh,
#     "abs_error": np.abs(y_post - te_soh),
# })
# results_df.to_csv(os.path.join(OUT_DIR, "lstm_results.csv"), index=False)

# print(f"\nSaved -> lstm_finetuned.pt")
# print(f"Saved -> lstm_finetune_log.csv")
# print(f"Saved -> lstm_results.csv")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 6: PLOTS
# # ══════════════════════════════════════════════════════════════════════════

# fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# # ── Plot 1: Fine-tuning loss curve ────────────────────────────────────────
# axes[0,0].plot(range(1, len(train_losses)+1), train_losses,
#                lw=1.5, color="steelblue", label="Fine-tune loss")
# axes[0,0].axvline(best_epoch, color="red", lw=1, linestyle="--",
#                   label=f"Best epoch {best_epoch}")
# axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("Loss")
# axes[0,0].set_title("Fine-tuning Loss")
# axes[0,0].legend(fontsize=8); axes[0,0].grid(True, alpha=0.4)
# axes[0,0].set_yscale("log")

# # ── Plot 2: SOH prediction on test set ────────────────────────────────────
# all_cycles = np.concatenate([ft_cycles, te_cycles])
# all_soh    = np.concatenate([ft_soh, te_soh])

# # Get fine-tune predictions too
# model.eval()
# with torch.no_grad():
#     y_ft_pred = model(X_ft).cpu().numpy().squeeze()

# axes[0,1].plot(all_cycles, all_soh, "k-", lw=2, label="True SOH",
#                zorder=3)
# axes[0,1].plot(te_cycles, y_post, "r--", lw=1.5,
#                label=f"Predicted (test)  RMSE={rmse_post:.4f}")
# axes[0,1].plot(ft_cycles, y_ft_pred, "g--", lw=1.5,
#                label=f"Predicted (fine-tune)")
# axes[0,1].axvspan(ft_cycles[0], ft_cycles[-1], alpha=0.15,
#                   color="green", label="Fine-tune region (10%)")
# axes[0,1].axhline(0.80, color="red", lw=1, linestyle=":",
#                   label="EOL 80%")
# axes[0,1].set_xlabel("Cycle"); axes[0,1].set_ylabel("SOH")
# axes[0,1].set_title(f"SOH Prediction -- LSTM\n"
#                     f"RMSE={rmse_post:.4f}  MAE={mae_post:.4f}  "
#                     f"R²={r2_post:.4f}")
# axes[0,1].legend(fontsize=7); axes[0,1].grid(True, alpha=0.4)
# axes[0,1].set_ylim(0.74, 1.02)

# # ── Plot 3: Error distribution ────────────────────────────────────────────
# errors = y_post - te_soh
# axes[1,0].hist(errors, bins=30, color="steelblue", alpha=0.8,
#                edgecolor="black", lw=0.5)
# axes[1,0].axvline(0,            color="red",  lw=1.5,
#                   linestyle="--", label="Zero error")
# axes[1,0].axvline(errors.mean(), color="orange", lw=1.5,
#                   linestyle="--",
#                   label=f"Mean error={errors.mean():.4f}")
# axes[1,0].set_xlabel("Prediction error (pred - true)")
# axes[1,0].set_ylabel("Count")
# axes[1,0].set_title("Error Distribution on Test Set")
# axes[1,0].legend(fontsize=8); axes[1,0].grid(True, alpha=0.4)

# # ── Plot 4: Predicted vs True scatter ─────────────────────────────────────
# axes[1,1].scatter(te_soh, y_post, s=10, alpha=0.6,
#                   color="steelblue")
# lims = [min(te_soh.min(), y_post.min()),
#         max(te_soh.max(), y_post.max())]
# axes[1,1].plot(lims, lims, "r--", lw=1.5, label="Perfect prediction")
# axes[1,1].set_xlabel("True SOH"); axes[1,1].set_ylabel("Predicted SOH")
# axes[1,1].set_title(f"Predicted vs True SOH (test set)\n"
#                     f"RMSE={rmse_post:.4f}  R²={r2_post:.4f}")
# axes[1,1].legend(fontsize=8); axes[1,1].grid(True, alpha=0.4)

# plt.suptitle(f"LSTM Fine-tuning Results -- Drone Pack Data\n"
#              f"Pretrained on {4083} source cycles, "
#              f"fine-tuned on {len(ft_soh)} drone cycles (10%), "
#              f"tested on {len(te_soh)} cycles (90%)",
#              fontsize=12, fontweight="bold")
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, "lstm_finetune_results.png"),
#             dpi=150, bbox_inches="tight")
# plt.close()
# print("Saved -> lstm_finetune_results.png")

# # ══════════════════════════════════════════════════════════════════════════
# # STEP 7: FINAL SUMMARY
# # ══════════════════════════════════════════════════════════════════════════
# print("\n" + "=" * 60)
# print("LSTM FINE-TUNING SUMMARY")
# print("=" * 60)
# print(f"  Architecture      : fc({FUSED_DIM}) -> lstm({HIDDEN_DIM})"
#       f" -> fc(64) -> fc(8) -> fc(1)")
# print(f"  Fine-tune cycles  : {len(ft_soh)} (10% of drone data)")
# print(f"  Test cycles       : {len(te_soh)} (90% of drone data)")
# print(f"  Fine-tune epochs  : {len(train_losses)}")
# print(f"  Best epoch        : {best_epoch}")
# print(f"\n  Results on test set:")
# print(f"    RMSE before FT  : {rmse_pre:.4f}")
# print(f"    RMSE after FT   : {rmse_post:.4f}")
# print(f"    MAE after FT    : {mae_post:.4f}")
# print(f"    R² after FT     : {r2_post:.4f}")
# print(f"    Improvement     : "
#       f"{(rmse_pre - rmse_post)/rmse_pre * 100:.1f}%")
# print(f"\n  Output files:")
# print(f"    lstm_finetuned.pt")
# print(f"    lstm_finetune_log.csv")
# print(f"    lstm_results.csv")
# print(f"    lstm_finetune_results.png")
# print(f"\n  Next step:")
# print(f"    Implement Transformer pretraining with same pipeline")
# print(f"    Compare results on same test set")
# print("\nDone.")

"""
finetune_lstm.py
================
Fine-tunes the pretrained LSTM on drone pack data (target domain)
and evaluates on the held-out test set.

Implements continual backpropagation (Fan et al. 2025, Section 3.4)
as originally designed for LSTM networks.

CBP applied to:
    - input_fc (linear layer)
    - LSTM hidden units (via lstm.weight_hh_l0 outgoing weights)
    - head fc layers (linear layers)

Parameters (Fan et al. 2025, Table 4 -- fine-tuning):
    eta = 0.01    decay rate
    rho = 3e-4    replacement rate
    maturity = 10 epochs before a unit can be replaced

Following Fan et al. (2025):
    - Fine-tune on first 10% of drone cycles (35 cycles)
    - Evaluate on remaining 90% (315 cycles)
    - Lower learning rate than pretraining (1e-4 vs 1e-3)
    - Same loss function: L_mse + lambda1 * L_emse

Architecture (Fan et al. 2025, Table 3):
    fc(38) -> lstm(128) -> fc(64) -> fc(8) -> fc(1)

Run from COMBINED/03_finetune_lstm/ directory:
    python finetune_lstm.py
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

FUSED_DIM  = 38
HIDDEN_DIM = 128

EPOCHS      = 670
LR          = 1e-4
LAMBDA1     = 1.0
LR_PATIENCE = 30
LR_FACTOR   = 0.5
LR_MIN      = 1e-7
ES_PATIENCE = 100
SEED        = 42

# Continual backpropagation (Fan et al. 2025, Table 4)
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
class LSTMModel(nn.Module):
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
        x      = torch.relu(self.input_fc(x))
        out, _ = self.lstm(x.unsqueeze(0))
        return self.head(out.squeeze(0))


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
# Original design for LSTM -- tracks both linear layers and LSTM
# hidden units. This is the most faithful implementation to the paper.
# =====================================================================
class ContinualBackpropLSTM:
    """
    Utility per unit i (Fan et al. Eq. 9):
        u[i] = eta*u[i] + (1-eta)*|h[i]|*sum_k(|W[i,k]|)

    For nn.Linear layers:
        h[i]   = output activation of neuron i
        W[i,k] = outgoing weights (module.weight rows)

    For LSTM hidden units:
        h[i]   = hidden state h_t for unit i (mean over sequence)
        W[i,k] = rows of weight_hh_l0 (hidden-to-hidden weights)
                 These are the outgoing connections from hidden unit i
                 in the recurrent loop -- most faithful to paper intent

    Reinitialisation:
        - Linear: reset weight row + bias via kaiming_uniform_
        - LSTM  : reset corresponding rows in weight_ih_l0,
                  weight_hh_l0, bias_ih_l0, bias_hh_l0
                  (all 4 gate slices affected for that hidden unit)
    """

    def __init__(self, model, eta=CBP_ETA, rho=CBP_RHO,
                 maturity=CBP_MATURITY):
        self.model    = model
        self.eta      = eta
        self.rho      = rho
        self.maturity = maturity
        self.layers   = []

        # -- input_fc and head linear layers --
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                n = module.out_features
                self.layers.append({
                    "type"   : "linear",
                    "name"   : name,
                    "module" : module,
                    "utility": torch.zeros(n),
                    "age"    : torch.zeros(n, dtype=torch.long),
                })

        # -- LSTM hidden units --
        # weight_hh_l0 shape: (4*hidden, hidden)
        # rows correspond to [input, forget, cell, output] gates
        # We track utility per hidden unit (hidden_dim units total)
        lstm = model.lstm
        self.layers.append({
            "type"   : "lstm",
            "name"   : "lstm.hidden",
            "module" : lstm,
            "utility": torch.zeros(HIDDEN_DIM),
            "age"    : torch.zeros(HIDDEN_DIM, dtype=torch.long),
        })

        self.total_reinit   = 0
        self.reinit_history = []

        print(f"\n  Continual Backprop (LSTM) initialised on "
              f"{len(self.layers)} unit groups:")
        for info in self.layers:
            if info["type"] == "linear":
                m = info["module"]
                print(f"    [linear] {info['name']:30s}  "
                      f"({m.out_features}, {m.in_features})")
            else:
                print(f"    [lstm  ] {info['name']:30s}  "
                      f"hidden_dim={HIDDEN_DIM}")

    def update(self, activations_dict, epoch):
        """
        activations_dict: {name: activation tensor (n_units,)}
        """
        epoch_reinit = 0

        for info in self.layers:
            name = info["name"]
            u    = info["utility"]
            age  = info["age"]

            if name not in activations_dict:
                info["age"] = (age + 1).flatten()
                continue

            h = activations_dict[name].detach().cpu().flatten()

            if info["type"] == "linear":
                module = info["module"]
                # Outgoing weights: rows of weight matrix (out, in)
                w_sum = module.weight.detach().cpu().abs().sum(dim=1).flatten()

            else:
                # LSTM: use weight_hh_l0 (hidden->hidden)
                # shape (4*hidden, hidden) -- take abs row sum
                # per hidden unit, average across all 4 gates
                lstm   = info["module"]
                w_hh   = lstm.weight_hh_l0.detach().cpu()  # (4H, H)
                # Reshape to (4, H, H): 4 gates, H output, H input
                w_hh_g = w_hh.view(4, HIDDEN_DIM, HIDDEN_DIM)
                # Sum abs weights per output hidden unit, avg over gates
                w_sum  = w_hh_g.abs().sum(dim=2).mean(dim=0).flatten()

            # Fan et al. Eq. 9
            u_new = (self.eta * u +
                     (1 - self.eta) * h.abs() * w_sum).flatten()
            info["utility"] = u_new
            info["age"]     = (age + 1).flatten()

            mean_u    = u_new.mean().item()
            threshold = self.rho * mean_u if mean_u > 0 else 0.0
            mature    = info["age"] >= self.maturity
            low_u     = u_new < threshold
            reinit    = mature & low_u
            n_reinit  = reinit.sum().item()

            if n_reinit > 0:
                idx = reinit.nonzero(as_tuple=True)[0]
                with torch.no_grad():
                    if info["type"] == "linear":
                        module = info["module"]
                        nn.init.kaiming_uniform_(
                            module.weight[idx], a=math.sqrt(5))
                        if module.bias is not None:
                            module.bias[idx].zero_()
                    else:
                        # LSTM reinit: reset all 4 gate rows for each
                        # affected hidden unit in weight_ih and weight_hh
                        lstm = info["module"]
                        for gate in range(4):
                            offset = gate * HIDDEN_DIM
                            rows   = idx + offset
                            nn.init.kaiming_uniform_(
                                lstm.weight_ih_l0[rows],
                                a=math.sqrt(5))
                            nn.init.kaiming_uniform_(
                                lstm.weight_hh_l0[rows],
                                a=math.sqrt(5))
                            lstm.bias_ih_l0[rows].zero_()
                            lstm.bias_hh_l0[rows].zero_()

                info["age"][reinit]     = 0
                info["utility"][reinit] = 0.0
                epoch_reinit      += n_reinit
                self.total_reinit += n_reinit
                self.reinit_history.append(
                    (epoch, name, int(n_reinit)))

        return epoch_reinit

    def mean_utility(self):
        all_u = torch.cat([info["utility"].flatten()
                           for info in self.layers])
        return float(all_u.mean())


def collect_activations_lstm(model, X):
    """
    Collect mean activation per unit for linear layers and LSTM
    hidden states. Returns dict keyed by layer name.
    """
    activations = {}
    hooks       = []

    # Hook linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            def make_hook(n, out_f):
                def hook(mod, inp, out):
                    activations[n] = out.detach().reshape(
                        -1, out_f).mean(dim=0)
                return hook
            hooks.append(module.register_forward_hook(
                make_hook(name, module.out_features)))

    # Hook LSTM to capture hidden states
    def lstm_hook(mod, inp, out):
        # out[0] is output tensor (1, seq_len, hidden)
        # mean over sequence to get (hidden,)
        activations["lstm.hidden"] = out[0].detach().squeeze(0).mean(dim=0)
    hooks.append(model.lstm.register_forward_hook(lstm_hook))

    with torch.no_grad():
        model(X)

    for h in hooks:
        h.remove()

    return activations


# =====================================================================
# STEP 1: LOAD DATA
# =====================================================================
print("=" * 60)
print("LSTM Fine-tuning with Continual Backpropagation")
print("(Fan et al. 2025, Section 3.4 -- original LSTM design)")
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

# =====================================================================
# STEP 2: LOAD PRETRAINED MODEL
# =====================================================================
print(f"\nLoading pretrained LSTM weights ...")

model = LSTMModel(FUSED_DIM, HIDDEN_DIM).to(DEVICE)
model.load_state_dict(
    torch.load(os.path.join(PRETRAIN_DIR, "lstm_pretrained.pt"),
               map_location=DEVICE))

n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"  Parameters : {n_params:,}")
print(f"  Architecture: fc({FUSED_DIM}) -> lstm({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")

model.eval()
with torch.no_grad():
    X_te  = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
    y_pre = model(X_te).cpu().numpy().squeeze()
rmse_pre = float(np.sqrt(np.mean((y_pre - te_soh)**2)))
mae_pre  = float(np.mean(np.abs(y_pre - te_soh)))

r2_pre = float(1 - np.sum((y_pre - te_soh)**2) /
               np.sum((te_soh - te_soh.mean())**2))

print(f"\n  Performance BEFORE fine-tuning:")
print(f"    RMSE = {rmse_pre:.4f}")
print(f"    MAE  = {mae_pre:.4f}")

# =====================================================================
# PRETRAINED MODEL PREDICTION PLOT (BEFORE FINE-TUNING)
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(te_cycles, te_soh, "k-", lw=2, label="True SOH")
ax.plot(te_cycles, y_pre, "b--", lw=1.5,
        label=f"Pretrained prediction\n"
              f"RMSE={rmse_pre:.4f}, MAE={mae_pre:.4f}, R2={r2_pre:.4f}")

ax.axhline(0.80, color="red", lw=1, linestyle=":", label="EOL 80%")

ax.set_xlabel("Cycle")
ax.set_ylabel("SOH")
ax.set_title("Pretrained LSTM Prediction (Before Fine-tuning)")
ax.legend()
ax.grid(True, alpha=0.4)
ax.set_ylim(0.74, 1.02)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lstm_pretrained_prediction.png"),
            dpi=150, bbox_inches="tight")
plt.close()

print("Saved -> lstm_pretrained_prediction.png")

# =====================================================================
# STEP 3: INITIALISE OPTIMISER + CBP
# =====================================================================
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = ReduceLROnPlateau(optimizer, mode="min",
                               patience=LR_PATIENCE,
                               factor=LR_FACTOR, min_lr=LR_MIN)

cbp = ContinualBackpropLSTM(model, eta=CBP_ETA,
                              rho=CBP_RHO, maturity=CBP_MATURITY)

X_ft = torch.tensor(ft_fused, dtype=torch.float32).to(DEVICE)
y_ft = torch.tensor(ft_soh,   dtype=torch.float32).unsqueeze(1).to(DEVICE)

print(f"\nFine-tuning with continual backprop "
      f"(max {EPOCHS} epochs, lr={LR}, "
      f"ES patience={ES_PATIENCE}) ...")
print(f"  CBP: eta={CBP_ETA}, rho={CBP_RHO}, "
      f"maturity={CBP_MATURITY}")
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'LR':>10}  "
      f"{'Reinit':>6}  {'MeanU':>8}")
print("-" * 52)

# =====================================================================
# STEP 4: TRAINING LOOP WITH CBP
# =====================================================================
train_losses    = []
reinit_history  = []
utility_history = []
best_loss       = float("inf")
best_epoch      = 0
es_counter      = 0
best_weights    = None

for epoch in range(1, EPOCHS + 1):

    # Train
    model.train()
    optimizer.zero_grad()
    loss, _, _ = composite_loss(model(X_ft), y_ft)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss = loss.item()

    scheduler.step(train_loss)
    train_losses.append(train_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # Continual backpropagation
    epoch_reinit = 0
    if epoch % CBP_INTERVAL == 0:
        model.eval()
        acts = collect_activations_lstm(model, X_ft)
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
        print(f"{epoch:>6}  {train_loss:>12.6f}  "
              f"{current_lr:>10.2e}  {epoch_reinit:>6}  "
              f"{mean_u:>8.5f}{marker}")

    if es_counter >= ES_PATIENCE:
        print(f"\nEarly stopping at epoch {epoch} "
              f"(best at epoch {best_epoch})")
        break

model.load_state_dict(best_weights)
print(f"\nRestored best weights from epoch {best_epoch}")
print(f"Best train loss      : {best_loss:.6f}")
print(f"Total reinitialised  : {cbp.total_reinit} units")

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
           os.path.join(OUT_DIR, "lstm_finetuned.pt"))
pd.DataFrame({
    "epoch"       : range(1, len(train_losses) + 1),
    "train_loss"  : train_losses,
    "reinit"      : reinit_history,
    "mean_utility": utility_history,
}).to_csv(os.path.join(OUT_DIR, "lstm_finetune_log.csv"), index=False)
pd.DataFrame({
    "cycle"    : te_cycles,
    "soh_true" : te_soh,
    "soh_pred" : y_post,
    "error"    : y_post - te_soh,
    "abs_error": np.abs(y_post - te_soh),
}).to_csv(os.path.join(OUT_DIR, "lstm_results.csv"), index=False)

if cbp.reinit_history:
    pd.DataFrame(cbp.reinit_history,
                 columns=["epoch", "layer", "n_reinit"]
                 ).to_csv(os.path.join(OUT_DIR,
                          "cbp_reinit_history.csv"), index=False)

print(f"\nSaved -> lstm_finetuned.pt")
print(f"Saved -> lstm_finetune_log.csv  (reinit + utility included)")
print(f"Saved -> lstm_results.csv")
print(f"Saved -> cbp_reinit_history.csv")

# =====================================================================
# STEP 7: PLOTS
# =====================================================================
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
epochs_r  = range(1, len(train_losses) + 1)

# Plot 1: Training loss
axes[0, 0].plot(epochs_r, train_losses, lw=1.5,
                color="steelblue", label="Fine-tune loss")
axes[0, 0].axvline(best_epoch, color="red", lw=1, linestyle="--",
                   label=f"Best epoch {best_epoch}")
axes[0, 0].set_xlabel("Epoch"); axes[0, 0].set_ylabel("Loss")
axes[0, 0].set_title("Fine-tuning Loss (log scale)")
axes[0, 0].legend(fontsize=8); axes[0, 0].grid(True, alpha=0.4)
axes[0, 0].set_yscale("log")

# Plot 2: Mean utility
axes[0, 1].plot(epochs_r, utility_history, lw=1.5, color="purple")
axes[0, 1].axvline(best_epoch, color="red", lw=1, linestyle="--",
                   label=f"Best epoch {best_epoch}")
axes[0, 1].set_xlabel("Epoch"); axes[0, 1].set_ylabel("Mean utility")
axes[0, 1].set_title("CBP Mean Neuron Utility\n"
                     "(higher = more active units)")
axes[0, 1].legend(fontsize=8); axes[0, 1].grid(True, alpha=0.4)

# Plot 3: Reinit events
axes[0, 2].bar(list(epochs_r), reinit_history,
               color="crimson", alpha=0.7, width=1.0)
axes[0, 2].set_xlabel("Epoch")
axes[0, 2].set_ylabel("Units reinitialised")
axes[0, 2].set_title(f"CBP Reinitialisation Events\n"
                     f"Total: {cbp.total_reinit}")
axes[0, 2].grid(True, alpha=0.4)

# Plot 4: SOH trajectory
all_cycles = np.concatenate([ft_cycles, te_cycles])
all_soh    = np.concatenate([ft_soh, te_soh])
model.eval()
with torch.no_grad():
    y_ft_pred = model(X_ft).cpu().numpy().squeeze()

axes[1, 0].plot(all_cycles, all_soh, "k-", lw=2,
                label="True SOH", zorder=3)
axes[1, 0].plot(te_cycles, y_post, "r--", lw=1.5,
                label=f"Predicted (test) RMSE={rmse_post:.4f}")
axes[1, 0].plot(ft_cycles, y_ft_pred, "g--", lw=1.5,
                label="Predicted (fine-tune)")
axes[1, 0].axvspan(ft_cycles[0], ft_cycles[-1],
                   alpha=0.15, color="green",
                   label="Fine-tune region (10%)")
axes[1, 0].axhline(0.80, color="red", lw=1, linestyle=":",
                   label="EOL 80%")
axes[1, 0].set_xlabel("Cycle"); axes[1, 0].set_ylabel("SOH")
axes[1, 0].set_title(f"SOH Prediction -- LSTM + CBP\n"
                     f"RMSE={rmse_post:.4f}  MAE={mae_post:.4f}  "
                     f"R2={r2_post:.4f}")
axes[1, 0].legend(fontsize=7); axes[1, 0].grid(True, alpha=0.4)
axes[1, 0].set_ylim(0.74, 1.02)

# Plot 5: Error distribution
errors = y_post - te_soh
axes[1, 1].hist(errors, bins=30, color="steelblue",
                alpha=0.8, edgecolor="black", lw=0.5)
axes[1, 1].axvline(0, color="red", lw=1.5, linestyle="--",
                   label="Zero error")
axes[1, 1].axvline(errors.mean(), color="orange", lw=1.5,
                   linestyle="--",
                   label=f"Mean={errors.mean():.4f}")
axes[1, 1].set_xlabel("Prediction error")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Error Distribution")
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.4)

# Plot 6: Scatter
axes[1, 2].scatter(te_soh, y_post, s=10, alpha=0.6,
                   color="steelblue")
lims = [min(te_soh.min(), y_post.min()),
        max(te_soh.max(), y_post.max())]
axes[1, 2].plot(lims, lims, "r--", lw=1.5,
                label="Perfect prediction")
axes[1, 2].set_xlabel("True SOH")
axes[1, 2].set_ylabel("Predicted SOH")
axes[1, 2].set_title(f"Predicted vs True\n"
                     f"RMSE={rmse_post:.4f}  R2={r2_post:.4f}")
axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.4)

plt.suptitle(
    f"LSTM Fine-tuning with Continual Backpropagation\n"
    f"eta={CBP_ETA}  rho={CBP_RHO}  maturity={CBP_MATURITY}  "
    f"total_reinit={cbp.total_reinit}  LR={LR}",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "lstm_finetune_results.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> lstm_finetune_results.png")

# =====================================================================
# STEP 8: SUMMARY
# =====================================================================
print("\n" + "=" * 60)
print("LSTM FINE-TUNING SUMMARY (with Continual Backpropagation)")
print("=" * 60)
print(f"  Architecture      : fc({FUSED_DIM}) -> lstm({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")
print(f"  Parameters        : {n_params:,}")
print(f"  Fine-tune cycles  : {len(ft_soh)} (10%)")
print(f"  Test cycles       : {len(te_soh)} (90%)")
print(f"  Fine-tune epochs  : {len(train_losses)}")
print(f"  Best epoch        : {best_epoch}")
print(f"\n  Continual Backpropagation (Fan et al. 2025, Eq. 9):")
print(f"    Decay rate eta  : {CBP_ETA}")
print(f"    Replace rate rho: {CBP_RHO}")
print(f"    Maturity thresh : {CBP_MATURITY} epochs")
print(f"    Unit groups     : {len(cbp.layers)}")
print(f"    Total reinit    : {cbp.total_reinit} units")
print(f"\n  Results on test set:")
print(f"    RMSE before FT  : {rmse_pre:.4f}")
print(f"    RMSE after FT   : {rmse_post:.4f}")
print(f"    MAE  after FT   : {mae_post:.4f}")
print(f"    R2   after FT   : {r2_post:.4f}")
print(f"    Improvement     : {improvement:.1f}%")
print(f"\n  Output files:")
print(f"    lstm_finetuned.pt")
print(f"    lstm_finetune_log.csv  (reinit + utility included)")
print(f"    lstm_results.csv")
print(f"    cbp_reinit_history.csv")
print(f"    lstm_finetune_results.png")
print(f"\nDone.")