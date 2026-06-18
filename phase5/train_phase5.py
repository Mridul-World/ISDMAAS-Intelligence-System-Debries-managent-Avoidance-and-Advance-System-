import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =====================================================
# PROJECT ROOT
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
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {DEVICE}")

# =====================================================
# LOAD DATA
# =====================================================

print("\nLoading tensors...")

X_train = torch.load("../phase4/X_train.pt")
y_train = torch.load("../phase4/y_train.pt")

X_val = torch.load("../phase4/X_val.pt")
y_val = torch.load("../phase4/y_val.pt")

baseline_train = torch.load(
    "../phase4/baseline_train.pt"
)

baseline_val = torch.load(
    "../phase4/baseline_val.pt"
)

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("baseline_train:", baseline_train.shape)
print("baseline_val:", baseline_val.shape)

# =====================================================
# DATALOADERS
# =====================================================

train_loader = DataLoader(
    TensorDataset(
        X_train,
        y_train,
        baseline_train
    ),
    batch_size=64,
    shuffle=True
)

val_loader = DataLoader(
    TensorDataset(
        X_val,
        y_val,
        baseline_val
    ),
    batch_size=64,
    shuffle=False
)

# =====================================================
# MODEL
# =====================================================

model = OrbitalTransformer(
    n_features=23,
    d_model=128,
    nhead=8,
    num_layers=4,
    dim_ff=512,
    dropout=0.1,
    n_targets=6,
).to(DEVICE)

# =====================================================
# LOAD PHASE 4 WEIGHTS
# =====================================================

print("\nLoading Phase 4 model...")

model.load_state_dict(
    torch.load(
        "../phase4/idsmass_data/models/phase4_transformer.pt",
        map_location=DEVICE,
    )
)

print("✓ Phase 4 weights loaded")

# =====================================================
# OPTIMIZER
# =====================================================

optimizer = optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5
)

# =====================================================
# TRAINING
# =====================================================

EPOCHS = 30

best_val_loss = float("inf")

os.makedirs(
    "model",
    exist_ok=True
)

print("\n====================================")
print("STARTING RESIDUAL PHASE 5 TRAINING")
print("====================================")

for epoch in range(EPOCHS):

    # ====================================
    # TRAIN
    # ====================================

    model.train()

    train_loss = 0.0

    for x, y, baseline in train_loader:

        x = x.to(DEVICE)
        y = y.to(DEVICE)
        baseline = baseline.to(DEVICE)

        optimizer.zero_grad()

        delta = model(x)

        target_delta = (
            y - baseline
        )

        loss = torch.nn.functional.mse_loss(
            delta,
            target_delta
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = (
        train_loss / len(train_loader)
    )

    # ====================================
    # VALIDATION
    # ====================================

    model.eval()

    val_loss = 0.0

    with torch.no_grad():

        for x, y, baseline in val_loader:

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            baseline = baseline.to(DEVICE)

            delta = model(x)

            target_delta = (
                y - baseline
            )

            loss = torch.nn.functional.mse_loss(
                delta,
                target_delta
            )

            val_loss += loss.item()

    avg_val_loss = (
        val_loss / len(val_loader)
    )

    print(
        f"Epoch {epoch+1:02d}/{EPOCHS} | "
        f"Train={avg_train_loss:.6f} | "
        f"Val={avg_val_loss:.6f}"
    )

    # ====================================
    # SAVE BEST MODEL
    # ====================================

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(
            model.state_dict(),
            "model/physics_transformer.pt"
        )

        print(
            "✓ Best model saved"
        )

print("\n====================================")
print("RESIDUAL PHASE 5 TRAINING COMPLETE")
print("====================================")

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    "\nSaved model:"
)

print(
    "model/physics_transformer.pt"
)