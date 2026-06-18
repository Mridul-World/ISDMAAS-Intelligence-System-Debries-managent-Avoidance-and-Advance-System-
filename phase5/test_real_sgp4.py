import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec
from sgp4.api import jday

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

    return datetime(
        year,
        1,
        1,
        tzinfo=timezone.utc
    ) + timedelta(days=day - 1)

# =====================================================
# LOAD DATA
# =====================================================

print("Loading dataset...")

df = pd.read_csv(
    "../phase4/idsmass_data/processed/historical_trajectory_dataset.csv"
)

tle_df = pd.read_csv(
    "../phase4/idsmass_data/raw/spacetrack/historical_tles.csv"
)

print("Dataset:", df.shape)
print("TLEs:", tle_df.shape)

# =====================================================
# SAMPLE
# =====================================================

sample = df.iloc[1000]

sample_time = pd.to_datetime(
    sample["timestamp"],
    utc=True
)

sample_time = sample_time.tz_convert(None)

sample_time = sample_time.to_pydatetime()

norad_id = sample["norad_id"]

print("\nNORAD:", norad_id)
print("Sample time:", sample_time)

# =====================================================
# FILTER SAME SATELLITE
# =====================================================

sat_tles = tle_df[
    tle_df["norad_id"] == norad_id
].copy()

sat_tles["epoch"] = sat_tles["tle1"].apply(
    tle_epoch_to_datetime
)

sat_tles["epoch"] = pd.to_datetime(
    sat_tles["epoch"],
    utc=True
)

sat_tles["epoch"] = sat_tles["epoch"].dt.tz_convert(None)

# =====================================================
# DEBUG
# =====================================================

print(
    "\nSample tz:",
    sample_time.tzinfo
)

print(
    "Epoch tz:",
    sat_tles["epoch"].iloc[0].tzinfo
)

# =====================================================
# FIND CLOSEST TLE
# =====================================================

sat_tles["time_diff"] = (
    sat_tles["epoch"] - sample_time
).abs()

best_tle = sat_tles.loc[
    sat_tles["time_diff"].idxmin()
]

print("\nClosest TLE epoch:")
print(best_tle["epoch"])

print(
    "\nTime difference:",
    best_tle["time_diff"]
)

# =====================================================
# PROPAGATE
# =====================================================

sat = Satrec.twoline2rv(
    best_tle["tle1"],
    best_tle["tle2"]
)

jd, fr = jday(
    sample_time.year,
    sample_time.month,
    sample_time.day,
    sample_time.hour,
    sample_time.minute,
    sample_time.second
)

error, r, v = sat.sgp4(
    jd,
    fr
)

print("\nSGP4 error code:", error)

truth = np.array([
    sample["x_km"],
    sample["y_km"],
    sample["z_km"]
])

pred = np.array(r)

position_error = np.linalg.norm(
    pred - truth
)

velocity_truth = np.array([
    sample["vx_kms"],
    sample["vy_kms"],
    sample["vz_kms"]
])

velocity_pred = np.array(v)

velocity_error = np.linalg.norm(
    velocity_pred - velocity_truth
)

print("\nTruth Position:")
print(truth)

print("\nPredicted Position:")
print(pred)

print("\nTruth Velocity:")
print(velocity_truth)

print("\nPredicted Velocity:")
print(velocity_pred)

print("\nPosition Error (km):")
print(position_error)

print("\nVelocity Error (km/s):")
print(velocity_error)