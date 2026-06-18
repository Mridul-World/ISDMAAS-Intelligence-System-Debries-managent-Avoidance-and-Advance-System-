import os
import sys
import json
import torch

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(PROJECT_ROOT)

from phase4.data import OrbitalTransformer
from validate_phase5 import compute_metrics

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("Device:", DEVICE)

# =====================================================
# LOAD DATA
# =====================================================

X_val = torch.load(
    "../phase4/X_val.pt"
)

y_val = torch.load(
    "../phase4/y_val.pt"
)

X_val = X_val.to(DEVICE)
y_val = y_val.to(DEVICE)

# =====================================================
# MODEL CONFIG
# =====================================================

model_args = dict(
    n_features=23,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_ff=512,
    dropout=0.1,
    n_targets=6,
)

# =====================================================
# PHASE4 MODEL
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

# =====================================================
# PHASE5 MODEL
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

print("Models loaded")

# =====================================================
# PREDICT
# =====================================================

with torch.no_grad():

    pred4 = phase4_model(X_val)

    pred5 = phase5_model(X_val)

# =====================================================
# METRICS
# =====================================================

phase4_metrics = compute_metrics(
    pred4,
    y_val
)

phase5_metrics = compute_metrics(
    pred5,
    y_val
)

results = {
    "phase4": phase4_metrics,
    "phase5": phase5_metrics,
}

print(
    json.dumps(
        results,
        indent=4
    )
)

os.makedirs(
    "reports",
    exist_ok=True
)

with open(
    "reports/final_validation.json",
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )

print(
    "\nSaved:"
)

print(
    "reports/final_validation.json"
)