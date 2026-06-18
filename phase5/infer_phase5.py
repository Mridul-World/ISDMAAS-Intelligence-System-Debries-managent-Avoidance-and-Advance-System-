import torch
import numpy as np

from phase5_TRAINER import ResidualOrbitTransformer

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

# ============================================
# LOAD DATA
# ============================================

X_val = torch.load(
    "../phase4/X_val.pt"
)

y_val = torch.load(
    "../phase4/y_val.pt"
)

baseline_val = torch.load(
    "baseline_val.pt"
)

# choose sample
idx = 0

x = X_val[idx:idx+1].to(DEVICE)

truth = y_val[idx]
baseline = baseline_val[idx]

# ============================================
# LOAD MODEL
# ============================================

model = ResidualOrbitTransformer(
    n_features=23,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_ff=512,
    dropout=0.1,
    n_targets=6
).to(DEVICE)

model.load_state_dict(
    torch.load(
        "model/residual_transformer_pure_ml.pt",
        map_location=DEVICE
    )
)

model.eval()

# ============================================
# PREDICT
# ============================================

with torch.no_grad():

    residual = model(x).cpu()[0]

prediction = baseline + residual

# ============================================
# POSITION ERRORS
# ============================================

truth_pos = truth[:3]
baseline_pos = baseline[:3]
pred_pos = prediction[:3]

baseline_error = np.linalg.norm(
    baseline_pos.numpy() - truth_pos.numpy()
)

prediction_error = np.linalg.norm(
    pred_pos.numpy() - truth_pos.numpy()
)

# ============================================
# OUTPUT
# ============================================

print("\nTruth Position:")
print(truth_pos)

print("\nSGP4 Baseline Position:")
print(baseline_pos)

print("\nResidual Prediction:")
print(pred_pos)

print("\nSGP4 Error (km):")
print(baseline_error)

print("\nResidual Model Error (km):")
print(prediction_error)