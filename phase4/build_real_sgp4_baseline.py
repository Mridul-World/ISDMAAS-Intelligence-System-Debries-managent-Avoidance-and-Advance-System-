import pandas as pd
import numpy as np
import torch

from datetime import datetime
from datetime import timedelta
from datetime import timezone

from sgp4.api import Satrec
from sgp4.api import jday

# =====================================================
# CONFIG
# =====================================================

SEQ_LEN = 30
HORIZON = 1

# =====================================================
# TLE EPOCH PARSER
# =====================================================

def tle_epoch_to_datetime(tle1):

    epoch_str = tle1[18:32].strip()

    yy = int(epoch_str[:2])

    if yy < 57:
        year = 2000 + yy
    else:
        year = 1900 + yy

    day = float(epoch_str[2:])

    return (
        datetime(
            year,
            1,
            1,
            tzinfo=timezone.utc
        )
        + timedelta(days=day - 1)
    )

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "idsmass_data/processed/historical_trajectory_dataset_v2.csv"
)

tle_df = pd.read_csv(
    "idsmass_data/raw/spacetrack/historical_tles.csv"
)

# FIXED TIMESTAMP PARSING
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

print("\nDataset shape:")
print(df.shape)

print("\nTLE shape:")
print(tle_df.shape)

# =====================================================
# BUILD TLE CACHE
# =====================================================

print("\nBuilding TLE cache...")

tle_df["epoch"] = tle_df["tle1"].apply(
    tle_epoch_to_datetime
)

cache = {}

for norad, group in tle_df.groupby("norad_id"):

    cache[int(norad)] = group.copy()

print("Satellites in cache:", len(cache))

# =====================================================
# BUILD BASELINES
# =====================================================

print("\nGenerating real SGP4 baselines...")

baseline = []

total = len(df) - SEQ_LEN - HORIZON

for i in range(total):

    if i % 5000 == 0:
        print(f"{i:,} / {total:,}")

    input_idx = (
    i + SEQ_LEN - 1
)

    target_idx = (
    i + SEQ_LEN + HORIZON - 1
)

    input_row = df.iloc[input_idx]
    target_row = df.iloc[target_idx]
    norad = int(
    input_row["norad_id"]
)
    input_time = input_row["timestamp"]
    target_time = target_row["timestamp"]
    # -------------------------------
    # SATELLITE EXISTS?
    # -------------------------------

    if norad not in cache:

        baseline.append(
            np.zeros(6, dtype=np.float32)
        )

        continue

    sat_tles = cache[norad]

    # -------------------------------
    # FIND CLOSEST TLE
    # -------------------------------

    diffs = (
        sat_tles["epoch"] - input_time
    ).abs()

    best_idx = diffs.idxmin()

    if pd.isna(best_idx):

        baseline.append(
            np.zeros(6, dtype=np.float32)
        )

        continue

    best_tle = sat_tles.loc[best_idx]

    # -------------------------------
    # PROPAGATE WITH SGP4
    # -------------------------------

    try:

        sat = Satrec.twoline2rv(
            best_tle["tle1"],
            best_tle["tle2"]
        )

        jd, fr = jday(
    target_time.year,
    target_time.month,
    target_time.day,
    target_time.hour,
    target_time.minute,
    target_time.second
)

        err, r, v = sat.sgp4(
            jd,
            fr
        )

        if err != 0:

            baseline.append(
                np.zeros(6, dtype=np.float32)
            )

            continue

        baseline.append(
            np.array(
                [
                    r[0],
                    r[1],
                    r[2],
                    v[0],
                    v[1],
                    v[2]
                ],
                dtype=np.float32
            )
        )

    except Exception:

        baseline.append(
            np.zeros(6, dtype=np.float32)
        )

# =====================================================
# CONVERT
# =====================================================

baseline = np.array(
    baseline,
    dtype=np.float32
)

print("\nBaseline shape:")
print(baseline.shape)

# =====================================================
# SPLIT
# =====================================================

N = len(baseline)

split = int(
    N * 0.8
)

baseline_train = baseline[:split]

baseline_val = baseline[split:]

print("\nTrain baseline:")
print(baseline_train.shape)

print("\nValidation baseline:")
print(baseline_val.shape)

# =====================================================
# SAVE
# =====================================================

torch.save(
    torch.tensor(
        baseline_train
    ),
    "baseline_train.pt"
)

torch.save(
    torch.tensor(
        baseline_val
    ),
    "baseline_val.pt"
)

print("\nSaved:")
print("baseline_train.pt")
print("baseline_val.pt")

print("\nDONE")