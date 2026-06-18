import os
import sys
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt

# =====================================================
# PATHS
# =====================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.append(PROJECT_ROOT)

# =====================================================
# IMPORTS
# =====================================================

from phase4.data import OrbitalTransformer

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

# =====================================================
# LOAD SCALER
# =====================================================

scaler = joblib.load(
    "../phase4/feature_scaler.pkl"
)

print("✓ Scaler loaded")

# =====================================================
# LOAD DATA
# =====================================================

X_val = torch.load(
    "../phase4/X_val.pt"
).to(DEVICE)

y_val = torch.load(
    "../phase4/y_val.pt"
).to(DEVICE)

print("Validation samples:", len(X_val))

# =====================================================
# MODEL CONFIG
# =====================================================

model_args = dict(
    n_features=15,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_ff=512,
    dropout=0.1,
    n_targets=6,
)

# =====================================================
# LOAD PHASE 4 MODEL
# =====================================================

phase4_model = OrbitalTransformer(
    **model_args
).to(DEVICE)

phase4_model.load_state_dict(
    torch.load(
        "../phase4/idsmass_data/models/phase4_transformer.pt",
        map_location=DEVICE
    )
)

phase4_model.eval()

print("✓ Phase4 loaded")

# =====================================================
# LOAD PHASE 5 MODEL
# =====================================================

phase5_model = OrbitalTransformer(
    **model_args
).to(DEVICE)

phase5_model.load_state_dict(
    torch.load(
        "model/physics_transformer.pt",
        map_location=DEVICE
    )
)

phase5_model.eval()

print("✓ Phase5 loaded")

# =====================================================
# PREDICTIONS
# =====================================================

N = 1000

with torch.no_grad():

    pred4 = phase4_model(
        X_val[:N]
    )

    pred5 = phase5_model(
        X_val[:N]
    )

truth = y_val[:N].cpu().numpy()
pred4 = pred4.cpu().numpy()
pred5 = pred5.cpu().numpy()

# =====================================================
# INVERSE NORMALIZATION
# =====================================================

def inverse_state(states):

    n = states.shape[0]

    dummy = np.zeros(
        (n, 15),
        dtype=np.float32
    )

    dummy[:, 0:6] = states

    real = scaler.inverse_transform(
        dummy
    )

    return real[:, 0:6]

truth_real = inverse_state(truth)
pred4_real = inverse_state(pred4)
pred5_real = inverse_state(pred5)

print("✓ Converted back to km")

# =====================================================
# PLOT
# =====================================================

os.makedirs(
    "plots",
    exist_ok=True
)

plt.figure(figsize=(10, 8))

plt.plot(
    truth_real[:, 0],
    truth_real[:, 1],
    label="Truth",
    linewidth=2,
)

plt.plot(
    pred4_real[:, 0],
    pred4_real[:, 1],
    label="Phase4",
    linewidth=1,
)

plt.plot(
    pred5_real[:, 0],
    pred5_real[:, 1],
    label="Phase5",
    linewidth=2,
)

plt.xlabel("X (km)")
plt.ylabel("Y (km)")

plt.title(
    "Orbit Comparison (Real Units)"
)

plt.legend()

plt.grid(True)

plt.axis("equal")

plt.tight_layout()

save_path = (
    "plots/orbit_comparison.png"
)

plt.savefig(
    save_path,
    dpi=300
)

print(
    f"✓ Saved: {save_path}"
)