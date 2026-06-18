import os
import joblib
import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sgp4.api import Satrec
from sgp4.api import jday

from datetime import datetime
from datetime import timezone
# =========================================================
# PATHS
# =========================================================

DATASET_PATH = (
    "idsmass_data/processed/historical_trajectory_dataset_v2.csv"
)

SAVE_DIR = "."

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# SETTINGS
# =========================================================

SEQUENCE_LENGTH = 30
PREDICTION_HORIZON = 1

# =========================================================
# FEATURES (23)
# =========================================================

FEATURE_COLS = [

    # Cartesian State
    "x_km",
    "y_km",
    "z_km",
    "vx_kms",
    "vy_kms",
    "vz_kms",

    # Orbital Elements
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
    "true_anomaly",

    # Physics Features
    "specific_energy",
    "angular_momentum_mag",

    # Environment
    "bstar",
    "f107",
    "kp",

    # Time Features
    "sin_doy",
    "cos_doy",
    "sin_tod",
    "cos_tod",

    # Satellite Properties
    "mass_kg",
    "cross_section_m2",
]

# =========================================================
# TARGETS
# =========================================================

TARGET_COLS = [
    "x_km",
    "y_km",
    "z_km",
    "vx_kms",
    "vy_kms",
    "vz_kms",
]

# =========================================================
# LOAD DATASET
# =========================================================

print("\nLoading dataset...")

df = pd.read_csv(DATASET_PATH)

print("Dataset shape:", df.shape)

# =========================================================
# SORT BY TIME
# =========================================================

if "timestamp" in df.columns:
    df = df.sort_values("timestamp")

# =========================================================
# REMOVE NaNs
# =========================================================

df = df.dropna(
    subset=FEATURE_COLS + TARGET_COLS
)

print(
    "Dataset after cleaning:",
    df.shape
)

# =========================================================
# SAVE RAW COPY
# =========================================================

raw_df = df.copy()

# =========================================================
# NORMALIZE FEATURES ONLY
# =========================================================

print("\nFitting feature scaler...")

scaler = StandardScaler()

df[FEATURE_COLS] = scaler.fit_transform(
    df[FEATURE_COLS]
)

# =========================================================
# SAVE SCALER
# =========================================================

joblib.dump(
    scaler,
    "feature_scaler.pkl"
)

print(
    "✓ Scaler saved: feature_scaler.pkl"
)

# =========================================================
# BUILD SEQUENCES
# =========================================================
print("\nBuilding sequences...")

X = []
y = []
baseline = []

# Inputs = normalized
features = df[FEATURE_COLS].values

# Targets = REAL units
targets = raw_df[TARGET_COLS].values

total_samples = (
    len(df)
    - SEQUENCE_LENGTH
    - PREDICTION_HORIZON
)

for i in range(total_samples):

    x_seq = features[
        i : i + SEQUENCE_LENGTH
    ]

    y_target = targets[
        i + SEQUENCE_LENGTH +
        PREDICTION_HORIZON - 1
    ]

    sgp4_base = targets[
        i + SEQUENCE_LENGTH - 1
    ]

    X.append(x_seq)
    y.append(y_target)
    baseline.append(sgp4_base)

# =========================================================
# NUMPY
# =========================================================

X = np.array(
    X,
    dtype=np.float32
)

y = np.array(
    y,
    dtype=np.float32
)

baseline = np.array(
    baseline,
    dtype=np.float32
)

print("\nSequence generation complete")

print("X shape:", X.shape)
print("y shape:", y.shape)
print("baseline shape:", baseline.shape)

print("\nTarget range check:")

print(
    "Position min:",
    np.min(y[:, :3])
)

print(
    "Position max:",
    np.max(y[:, :3])
)

# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

print("\nSplitting train/validation...")

split_idx = int(
    len(X) * 0.8
)

X_train = X[:split_idx]
X_val = X[split_idx:]

y_train = y[:split_idx]
y_val = y[split_idx:]

baseline_train = baseline[:split_idx]
baseline_val = baseline[split_idx:]

print(
    "Train shape:",
    X_train.shape
)

print(
    "Validation shape:",
    X_val.shape
)

# =========================================================
# TORCH
# =========================================================

X_train = torch.tensor(
    X_train,
    dtype=torch.float32
)

y_train = torch.tensor(
    y_train,
    dtype=torch.float32
)

X_val = torch.tensor(
    X_val,
    dtype=torch.float32
)

y_val = torch.tensor(
    y_val,
    dtype=torch.float32
)

baseline_train = torch.tensor(
    baseline_train,
    dtype=torch.float32
)

baseline_val = torch.tensor(
    baseline_val,
    dtype=torch.float32
)

# =========================================================
# SAVE TENSORS
# =========================================================

print("\nSaving tensors...")

torch.save(
    X_train,
    "X_train.pt"
)

torch.save(
    y_train,
    "y_train.pt"
)

torch.save(
    X_val,
    "X_val.pt"
)

torch.save(
    y_val,
    "y_val.pt"
)

torch.save(
    baseline_train,
    "baseline_train.pt"
)

torch.save(
    baseline_val,
    "baseline_val.pt"
)

# =========================================================
# SUMMARY
# =========================================================

print("\n===================================")
print("PHASE 5 TENSORS EXPORTED")
print("===================================")

print("\nSaved:")
print("✓ X_train.pt")
print("✓ y_train.pt")
print("✓ X_val.pt")
print("✓ y_val.pt")
print("✓ baseline_train.pt")
print("✓ baseline_val.pt")
print("✓ feature_scaler.pkl")

print(
    "\nSequence length:",
    SEQUENCE_LENGTH
)

print(
    "Prediction horizon:",
    PREDICTION_HORIZON
)

print(
    "Feature count:",
    len(FEATURE_COLS)
)

print(
    "Target count:",
    len(TARGET_COLS)
)