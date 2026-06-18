import pandas as pd
import numpy as np

from datetime import datetime, timedelta, timezone

from sgp4.api import Satrec, jday

SEQ_LEN = 30
HORIZON = 1

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

print("Loading...")

df = pd.read_csv(
    "idsmass_data/processed/historical_trajectory_dataset_v2.csv"
)

tle_df = pd.read_csv(
    "idsmass_data/raw/spacetrack/historical_tles.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

# sample
i = 1000

input_idx = i + SEQ_LEN - 1
target_idx = i + SEQ_LEN + HORIZON - 1

input_row = df.iloc[input_idx]
target_row = df.iloc[target_idx]

norad = int(input_row["norad_id"])

input_time = input_row["timestamp"]
target_time = target_row["timestamp"]

print("\nInput time:")
print(input_time)

print("\nTarget time:")
print(target_time)

sat_tles = tle_df[
    tle_df["norad_id"] == norad
].copy()

sat_tles["epoch"] = sat_tles["tle1"].apply(
    tle_epoch_to_datetime
)

sat_tles["time_diff"] = (
    sat_tles["epoch"] - input_time
).abs()

best_tle = sat_tles.loc[
    sat_tles["time_diff"].idxmin()
]

print("\nChosen TLE epoch:")
print(best_tle["epoch"])

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

err, r, v = sat.sgp4(jd, fr)

truth = np.array([
    target_row["x_km"],
    target_row["y_km"],
    target_row["z_km"]
])

pred = np.array(r)

pos_error = np.linalg.norm(
    pred - truth
)

print("\nPosition Error (km):")
print(pos_error)