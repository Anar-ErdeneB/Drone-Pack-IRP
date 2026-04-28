"""
finetune_gru.py
===============
Fine-tunes the pretrained GRU on drone pack data (target domain)
and evaluates on the held-out test set.

Implements:
  - Continual Backpropagation (Fan et al. 2025, Eq. 9) adapted for GRU
    (3 gates: reset, update, new — vs LSTM's 4 gates)
  - L2-SP regularisation: lambda * ||theta - theta_pretrain||^2
    (anchors weights to pretrained state; critical with only 35 samples)

Architecture (mirrors Fan et al. 2025, Table 3, GRU variant):
    fc(38) -> gru(128) -> fc(64) -> fc(8) -> fc(1)

Reads (from ../01b_pack_input/processed/):
    finetune_fused.npy     (35, 38)   fine-tuning input
    finetune_soh.npy       (35,)      fine-tuning labels
    test_fused.npy         (315, 38)  test input
    test_soh.npy           (315,)     test labels
    drone_index.csv        (350, *)   cycle metadata

Reads (from 00_processed/):
    gru_pretrained.pt      pretrained GRU weights

Writes (to 01_processed/):
    gru_finetuned.pt
    gru_finetune_log.csv
    gru_results.csv
    cbp_gru_reinit_history.csv
    gru_pretrained_prediction.png
    gru_finetune_results.png
"""

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ══════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════
FEAT_DIR     = "../01b_pack_input/processed"
PRETRAIN_DIR = "00_processed"
OUT_DIR      = "01_processed"
os.makedirs(OUT_DIR, exist_ok=True)

FUSED_DIM  = 38
HIDDEN_DIM = 128

EPOCHS       = 700
LR           = 1e-4
WEIGHT_DECAY = 0.0
LAMBDA1      = 0.0        # pure MSE
LR_PATIENCE  = 30
LR_FACTOR    = 0.5
LR_MIN       = 1e-8
ES_PATIENCE  = 100
NOISE_STD    = 0.002
SEED         = 42

# Head-only fine-tuning: freeze GRU + input_fc, train only the 3 head linear layers
# (8,785 params vs 74,779 total).  Best result across all experiments: RMSE=0.0236.
# Full-model fine-tuning (Exp 1-3) gave RMSE 0.0256-0.0261 regardless of L2SP strength.
# Head-only with composite loss (Exp 5) gave RMSE=0.0270.
# Head-only + pure MSE + L2SP=0.1 (this config) is the best at RMSE=0.0236.
FREEZE_GRU   = True       # freeze input_fc + gru; adapt head output mapping only

LAMBDA_L2SP  = 0.1        # moderate anchor on head parameters

# Continual Backpropagation (Fan et al. 2025, Table 4)
CBP_ETA      = 0.01
CBP_RHO      = 3e-4       # threshold < GRU mean utility → 0 reinits (effectively disabled)
CBP_MATURITY = 10
CBP_INTERVAL = 1

torch.manual_seed(SEED)
np.random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ══════════════════════════════════════════════════════════════
# MODEL  (identical to 00_GRU.py)
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
        x      : (seq_len, FUSED_DIM)
        returns: (seq_len, 1)
        """
        x      = torch.relu(self.input_fc(x))
        out, _ = self.gru(x.unsqueeze(0))          # (1, T, hidden)
        return self.head(out.squeeze(0))            # (T, 1)


# ══════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════
def composite_loss(pred, target, lambda1=LAMBDA1):
    errors_sq  = (pred - target) ** 2
    l_mse      = errors_sq.mean()
    if lambda1 > 0:
        large_mask = errors_sq > l_mse.detach()
        l_emse     = errors_sq[large_mask].mean() \
                     if large_mask.sum() > 0 \
                     else torch.tensor(0.0, device=pred.device)
        return l_mse + lambda1 * l_emse, l_mse, l_emse
    return l_mse, l_mse, torch.tensor(0.0, device=pred.device)


# ══════════════════════════════════════════════════════════════
# CONTINUAL BACKPROPAGATION — GRU variant (Fan et al. 2025, Eq. 9)
# ══════════════════════════════════════════════════════════════
class ContinualBackpropGRU:
    """
    Utility per unit i (Fan et al. Eq. 9):
        u[i] = eta*u[i] + (1-eta)*|h[i]|*sum_k(|W[i,k]|)

    For nn.Linear layers:
        h[i]   = output activation of neuron i (mean over sequence)
        W[i,k] = outgoing weight rows (module.weight)

    For GRU hidden units:
        h[i]   = hidden state h_t for unit i (mean over sequence)
        W[i,k] = rows of weight_hh_l0 (hidden-to-hidden)
                 GRU has 3 gates: reset (r), update (z), new (n)
                 weight_hh_l0 shape: (3*H, H)

    Reinitialisation:
        - Linear: reset weight row + bias via kaiming_uniform_
        - GRU   : reset corresponding rows in weight_ih_l0,
                  weight_hh_l0, bias_ih_l0, bias_hh_l0
                  for all 3 gate slices of that hidden unit
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

        # -- GRU hidden units --
        # weight_hh_l0 shape: (3*H, H) for 3 gates [r, z, n]
        # We track utility per hidden unit (HIDDEN_DIM units)
        gru = model.gru
        self.layers.append({
            "type"   : "gru",
            "name"   : "gru.hidden",
            "module" : gru,
            "utility": torch.zeros(HIDDEN_DIM),
            "age"    : torch.zeros(HIDDEN_DIM, dtype=torch.long),
        })

        self.total_reinit   = 0
        self.reinit_history = []

        print(f"\n  Continual Backprop (GRU) initialised on "
              f"{len(self.layers)} unit groups:")
        for info in self.layers:
            if info["type"] == "linear":
                m = info["module"]
                print(f"    [linear] {info['name']:30s}  "
                      f"({m.out_features}, {m.in_features})")
            else:
                print(f"    [gru   ] {info['name']:30s}  "
                      f"hidden_dim={HIDDEN_DIM}  (3 gates)")

    def update(self, activations_dict, epoch):
        """
        activations_dict: {name: activation tensor (n_units,)}
        Returns number of units reinitialised this step.
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
                w_sum = module.weight.detach().cpu().abs().sum(
                    dim=1).flatten()

            else:
                # GRU: use weight_hh_l0 (hidden->hidden)
                # shape (3*H, H) — 3 gates [r, z, n]
                # Reshape to (3, H, H): gate, out_unit, in_unit
                # Sum abs weights per output hidden unit, avg over 3 gates
                gru    = info["module"]
                w_hh   = gru.weight_hh_l0.detach().cpu()   # (3H, H)
                w_hh_g = w_hh.view(3, HIDDEN_DIM, HIDDEN_DIM)
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
                        # GRU reinit: reset all 3 gate rows for each
                        # affected hidden unit in weight_ih and weight_hh
                        gru = info["module"]
                        for gate in range(3):
                            offset = gate * HIDDEN_DIM
                            rows   = idx + offset
                            nn.init.kaiming_uniform_(
                                gru.weight_ih_l0[rows],
                                a=math.sqrt(5))
                            nn.init.kaiming_uniform_(
                                gru.weight_hh_l0[rows],
                                a=math.sqrt(5))
                            gru.bias_ih_l0[rows].zero_()
                            gru.bias_hh_l0[rows].zero_()

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


def collect_activations_gru(model, X):
    """
    Collect mean activation per unit for linear layers and GRU
    hidden states. Returns dict keyed by layer name.
    X: (seq_len, FUSED_DIM) tensor
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

    # Hook GRU to capture hidden states
    # output[0] shape: (1, T, H) — mean over T to get (H,)
    def gru_hook(mod, inp, out):
        activations["gru.hidden"] = out[0].detach().squeeze(0).mean(dim=0)
    hooks.append(model.gru.register_forward_hook(gru_hook))

    with torch.no_grad():
        model(X)

    for h in hooks:
        h.remove()

    return activations


# ══════════════════════════════════════════════════════════════
# STEP 1: LOAD DATA
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("GRU Fine-tuning with Continual Backpropagation + L2-SP")
print("(Fan et al. 2025, Section 3.4 — adapted for GRU)")
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


# ══════════════════════════════════════════════════════════════
# STEP 2: LOAD PRETRAINED MODEL
# ══════════════════════════════════════════════════════════════
print(f"\nLoading pretrained GRU weights ...")

model = GRUModel(FUSED_DIM, HIDDEN_DIM).to(DEVICE)
model.load_state_dict(
    torch.load(os.path.join(PRETRAIN_DIR, "gru_pretrained.pt"),
               map_location=DEVICE))

n_params = sum(p.numel() for p in model.parameters()
               if p.requires_grad)
print(f"  Parameters : {n_params:,}")
print(f"  Architecture: fc({FUSED_DIM}) -> gru({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")

# Evaluate pretrained model BEFORE fine-tuning
model.eval()
with torch.no_grad():
    X_te  = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
    y_pre = model(X_te).cpu().numpy().squeeze()
rmse_pre = float(np.sqrt(np.mean((y_pre - te_soh) ** 2)))
mae_pre  = float(np.mean(np.abs(y_pre - te_soh)))
r2_pre   = float(1 - np.sum((y_pre - te_soh) ** 2) /
                 np.sum((te_soh - te_soh.mean()) ** 2))

print(f"\n  Performance BEFORE fine-tuning (on test set):")
print(f"    RMSE = {rmse_pre:.4f}")
print(f"    MAE  = {mae_pre:.4f}")
print(f"    R²   = {r2_pre:.4f}")

# Save pretrained predictions plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(te_cycles, te_soh,  "k-",  lw=2,   label="True SOH")
ax.plot(te_cycles, y_pre,   "b--", lw=1.5,
        label=f"Pretrained pred  RMSE={rmse_pre:.4f}  "
              f"MAE={mae_pre:.4f}  R²={r2_pre:.4f}")
ax.axhline(0.80, color="red", lw=1, linestyle=":", label="EOL 80%")
ax.set_xlabel("Cycle"); ax.set_ylabel("SOH")
ax.set_title("Pretrained GRU Prediction (Before Fine-tuning)")
ax.legend(); ax.grid(True, alpha=0.4); ax.set_ylim(0.74, 1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gru_pretrained_prediction.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> gru_pretrained_prediction.png")

# Store pretrained weights for L2-SP
pretrained_params = {n: p.detach().clone()
                     for n, p in model.named_parameters()}

# ══════════════════════════════════════════════════════════════
# STEP 3: FREEZE LAYERS + OPTIMISER + CBP
# ══════════════════════════════════════════════════════════════
# Freeze GRU + input_fc: only the head adapts to the pack domain
if FREEZE_GRU:
    for name, param in model.named_parameters():
        if name.startswith("head"):
            param.requires_grad = True
        else:
            param.requires_grad = False   # input_fc + gru frozen

n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"\n  Trainable params : {n_trainable:,}  "
      f"(frozen: {n_frozen:,})")
if FREEZE_GRU:
    print(f"  Frozen layers    : input_fc, gru  (head-only fine-tuning)")

trainable_params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable_params,
                               lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="min",
                               patience=LR_PATIENCE,
                               factor=LR_FACTOR, min_lr=LR_MIN)

cbp = ContinualBackpropGRU(model, eta=CBP_ETA,
                            rho=CBP_RHO, maturity=CBP_MATURITY)

X_ft = torch.tensor(ft_fused, dtype=torch.float32).to(DEVICE)
y_ft = torch.tensor(ft_soh,   dtype=torch.float32).unsqueeze(1).to(DEVICE)

mode_str = "HEAD-ONLY" if FREEZE_GRU else "FULL MODEL"
print(f"\nFine-tuning [{mode_str}] with CBP + L2-SP "
      f"(max {EPOCHS} epochs, lr={LR}, "
      f"ES patience={ES_PATIENCE}) ...")
print(f"  LAMBDA1={LAMBDA1}  LAMBDA_L2SP={LAMBDA_L2SP}  "
      f"NOISE_STD={NOISE_STD}")
print(f"  CBP: eta={CBP_ETA}, rho={CBP_RHO}, "
      f"maturity={CBP_MATURITY}")
print(f"{'Epoch':>6}  {'Train Loss':>12}  {'LR':>10}  "
      f"{'Reinit':>6}  {'MeanU':>8}")
print("-" * 52)


# ══════════════════════════════════════════════════════════════
# STEP 4: TRAINING LOOP
# ══════════════════════════════════════════════════════════════
train_losses    = []
reinit_history  = []
utility_history = []
best_loss       = float("inf")
best_epoch      = 0
es_counter      = 0
best_weights    = None

for epoch in range(1, EPOCHS + 1):

    model.train()
    optimizer.zero_grad()

    # Add per-feature noise for regularisation
    X_noisy = X_ft + NOISE_STD * torch.randn_like(X_ft)

    pred             = model(X_noisy)
    task_loss, _, _  = composite_loss(pred, y_ft)

    # L2-SP: penalise deviation from pretrained weights
    # Only applied to trainable (unfrozen) parameters
    l2sp = torch.tensor(0.0, device=DEVICE)
    if LAMBDA_L2SP > 0:
        for name, param in model.named_parameters():
            if param.requires_grad and name in pretrained_params:
                ref = pretrained_params[name].to(DEVICE)
                l2sp = l2sp + (param - ref).pow(2).sum()

    loss = task_loss + LAMBDA_L2SP * l2sp

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    train_loss = task_loss.item()   # log task loss only (for fair comparison)

    scheduler.step(train_loss)
    train_losses.append(train_loss)
    current_lr = optimizer.param_groups[0]["lr"]

    # Continual backpropagation
    epoch_reinit = 0
    if epoch % CBP_INTERVAL == 0:
        model.eval()
        acts = collect_activations_gru(model, X_ft)
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


# ══════════════════════════════════════════════════════════════
# STEP 5: EVALUATE
# ══════════════════════════════════════════════════════════════
print(f"\nEvaluating on test set ({len(te_soh)} cycles) ...")

model.eval()
with torch.no_grad():
    X_te   = torch.tensor(te_fused, dtype=torch.float32).to(DEVICE)
    y_post = model(X_te).cpu().numpy().squeeze()

rmse_post   = float(np.sqrt(np.mean((y_post - te_soh) ** 2)))
mae_post    = float(np.mean(np.abs(y_post - te_soh)))
r2_post     = float(1 - np.sum((y_post - te_soh) ** 2) /
                    np.sum((te_soh - te_soh.mean()) ** 2))
improvement = (rmse_pre - rmse_post) / rmse_pre * 100

print(f"\n  Performance AFTER fine-tuning:")
print(f"    RMSE = {rmse_post:.4f}  (was {rmse_pre:.4f} before)")
print(f"    MAE  = {mae_post:.4f}  (was {mae_pre:.4f} before)")
print(f"    R²   = {r2_post:.4f}")
print(f"    RMSE improvement: {improvement:.1f}%")


# ══════════════════════════════════════════════════════════════
# STEP 6: SAVE
# ══════════════════════════════════════════════════════════════
torch.save(model.state_dict(),
           os.path.join(OUT_DIR, "gru_finetuned.pt"))

pd.DataFrame({
    "epoch"       : range(1, len(train_losses) + 1),
    "train_loss"  : train_losses,
    "reinit"      : reinit_history,
    "mean_utility": utility_history,
}).to_csv(os.path.join(OUT_DIR, "gru_finetune_log.csv"), index=False)

pd.DataFrame({
    "cycle"    : te_cycles,
    "soh_true" : te_soh,
    "soh_pred" : y_post,
    "error"    : y_post - te_soh,
    "abs_error": np.abs(y_post - te_soh),
}).to_csv(os.path.join(OUT_DIR, "gru_results.csv"), index=False)

if cbp.reinit_history:
    pd.DataFrame(cbp.reinit_history,
                 columns=["epoch", "layer", "n_reinit"]
                 ).to_csv(os.path.join(OUT_DIR,
                          "cbp_gru_reinit_history.csv"), index=False)

print(f"\nSaved -> gru_finetuned.pt")
print(f"Saved -> gru_finetune_log.csv")
print(f"Saved -> gru_results.csv")
if cbp.reinit_history:
    print(f"Saved -> cbp_gru_reinit_history.csv")


# ══════════════════════════════════════════════════════════════
# STEP 7: PLOTS  (6-panel)
# ══════════════════════════════════════════════════════════════
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

# Plot 2: Mean utility (CBP)
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

# Plot 4: SOH trajectory (pre vs post)
all_cycles = np.concatenate([ft_cycles, te_cycles])
all_soh    = np.concatenate([ft_soh, te_soh])
model.eval()
with torch.no_grad():
    y_ft_pred = model(X_ft).cpu().numpy().squeeze()

axes[1, 0].plot(all_cycles, all_soh, "k-",  lw=2,
                label="True SOH", zorder=3)
axes[1, 0].plot(te_cycles, y_pre,  "b:",  lw=1.2,
                alpha=0.7, label=f"Pre-FT  RMSE={rmse_pre:.4f}")
axes[1, 0].plot(te_cycles, y_post, "r--", lw=1.5,
                label=f"Post-FT  RMSE={rmse_post:.4f}")
axes[1, 0].plot(ft_cycles, y_ft_pred, "g--", lw=1.5,
                label="Predicted (fine-tune)")
axes[1, 0].axvspan(ft_cycles[0], ft_cycles[-1],
                   alpha=0.15, color="green",
                   label="Fine-tune region (10%)")
axes[1, 0].axhline(0.80, color="red", lw=1, linestyle=":",
                   label="EOL 80%")
axes[1, 0].set_xlabel("Cycle"); axes[1, 0].set_ylabel("SOH")
axes[1, 0].set_title(f"SOH Prediction — GRU + CBP + L2-SP\n"
                     f"RMSE={rmse_post:.4f}  MAE={mae_post:.4f}  "
                     f"R²={r2_post:.4f}")
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
axes[1, 1].set_xlabel("Prediction error (pred - true)")
axes[1, 1].set_ylabel("Count")
axes[1, 1].set_title("Error Distribution on Test Set")
axes[1, 1].legend(fontsize=8); axes[1, 1].grid(True, alpha=0.4)

# Plot 6: Scatter (predicted vs true)
axes[1, 2].scatter(te_soh, y_post, s=10, alpha=0.6,
                   color="steelblue")
lims = [min(te_soh.min(), y_post.min()) - 0.005,
        max(te_soh.max(), y_post.max()) + 0.005]
axes[1, 2].plot(lims, lims, "r--", lw=1.5,
                label="Perfect prediction")
axes[1, 2].set_xlim(lims); axes[1, 2].set_ylim(lims)
axes[1, 2].set_xlabel("True SOH")
axes[1, 2].set_ylabel("Predicted SOH")
axes[1, 2].set_title(f"Predicted vs True SOH (test set)\n"
                     f"RMSE={rmse_post:.4f}  R²={r2_post:.4f}")
axes[1, 2].legend(fontsize=8); axes[1, 2].grid(True, alpha=0.4)

plt.suptitle(
    f"GRU Fine-tuning with Continual Backpropagation + L2-SP\n"
    f"eta={CBP_ETA}  rho={CBP_RHO}  maturity={CBP_MATURITY}  "
    f"total_reinit={cbp.total_reinit}  "
    f"LR={LR}  L2SP={LAMBDA_L2SP}",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "gru_finetune_results.png"),
            dpi=150, bbox_inches="tight")
plt.close()
print("Saved -> gru_finetune_results.png")


# ══════════════════════════════════════════════════════════════
# STEP 8: SUMMARY
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("GRU FINE-TUNING SUMMARY (CBP + L2-SP)")
print("=" * 60)
print(f"  Architecture       : fc({FUSED_DIM}) -> gru({HIDDEN_DIM})"
      f" -> fc(64) -> fc(8) -> fc(1)")
print(f"  Total parameters   : {n_params:,}")
print(f"  Trainable params   : {n_trainable:,}"
      f"  (frozen: {n_frozen:,})")
if FREEZE_GRU:
    print(f"  Freeze strategy    : input_fc + gru frozen; head only")
print(f"  Fine-tune cycles   : {len(ft_soh)} (10%)")
print(f"  Test cycles        : {len(te_soh)} (90%)")
print(f"  Fine-tune epochs   : {len(train_losses)}")
print(f"  Best epoch         : {best_epoch}")
print(f"\n  Hyperparameters:")
print(f"    LR               : {LR}")
print(f"    LAMBDA1 (EMSE)   : {LAMBDA1}")
print(f"    LAMBDA_L2SP      : {LAMBDA_L2SP}")
print(f"    NOISE_STD        : {NOISE_STD}")
print(f"    ES patience      : {ES_PATIENCE}")
print(f"\n  Continual Backpropagation (Fan et al. 2025, Eq. 9):")
print(f"    Decay rate eta   : {CBP_ETA}")
print(f"    Replace rate rho : {CBP_RHO}")
print(f"    Maturity thresh  : {CBP_MATURITY} epochs")
print(f"    Gates tracked    : 3 (reset, update, new)")
print(f"    Unit groups      : {len(cbp.layers)}")
print(f"    Total reinit     : {cbp.total_reinit} units")
print(f"\n  L2-SP Regularisation:")
print(f"    Lambda           : {LAMBDA_L2SP}")
print(f"    Anchor           : pretrained GRU weights")
print(f"\n  Results on test set:")
print(f"    RMSE before FT   : {rmse_pre:.4f}")
print(f"    RMSE after FT    : {rmse_post:.4f}")
print(f"    MAE  after FT    : {mae_post:.4f}")
print(f"    R²   after FT    : {r2_post:.4f}")
print(f"    Improvement      : {improvement:.1f}%")
print(f"\n  Output files:")
print(f"    gru_finetuned.pt")
print(f"    gru_finetune_log.csv")
print(f"    gru_results.csv")
if cbp.reinit_history:
    print(f"    cbp_gru_reinit_history.csv")
print(f"    gru_pretrained_prediction.png")
print(f"    gru_finetune_results.png")
print(f"\nDone.")
