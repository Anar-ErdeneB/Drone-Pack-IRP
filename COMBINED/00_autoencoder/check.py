import numpy as np
import torch
import torch.nn as nn

# Load the saved model and check reconstruction error
curves = np.load("../processed/source_curves.npy")

# Check a few curves manually
print("First curve stats:")
print(f"  min={curves[0].min():.4f}  max={curves[0].max():.4f}  "
      f"mean={curves[0].mean():.4f}  std={curves[0].std():.4f}")

print(f"\nAll curves stats:")
print(f"  min={curves.min():.4f}  max={curves.max():.4f}")
print(f"  mean={curves.mean():.4f}  std={curves.std():.4f}")
print(f"  std across curves: {curves.std(axis=0).mean():.4f}")

# MSE of 0.001 in raw voltage means:
mse = 0.001045
rmse = np.sqrt(mse)
v_range = curves.max() - curves.min()
print(f"\nMSE interpretation:")
print(f"  MSE  = {mse:.6f}")
print(f"  RMSE = {rmse:.4f} V")
print(f"  V range = {v_range:.4f} V")
print(f"  RMSE as % of V range = {rmse/v_range*100:.2f}%")