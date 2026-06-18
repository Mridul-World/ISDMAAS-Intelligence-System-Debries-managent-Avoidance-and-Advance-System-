import pandas as pd
from datetime import datetime, timedelta, timezone

# ==========================================
# TLE EPOCH PARSER
# ==========================================

def tle_epoch_to_datetime(tle1):

    epoch_str = tle1[18:32].strip()

    yy = int(epoch_str[:2]

)
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

# ==========================================
# LOAD
# ==========================================

df = pd.read_csv(
    "idsmass_data/processed/historical_trajectory_dataset_v2.csv"
)

tle_df = pd.read_csv(
    "idsmass_data/raw/spacetrack/historical_tles.csv"
)

# ==========================================
# SAMPLE
# ==========================================

row = df.iloc[1000]

norad = row["norad_id"]

ts = pd.to_datetime(
    row["timestamp"],
    utc=True
)

sat_tles = tle_df[
    tle_df["norad_id"] == norad
].copy()

sat_tles["epoch"] = sat_tles["tle1"].apply(
    tle_epoch_to_datetime
)

sat_tles["time_diff"] = (
    sat_tles["epoch"] - ts
).abs()

best = sat_tles.loc[
    sat_tles["time_diff"].idxmin()
]

print("\nNORAD:")
print(norad)

print("\nTimestamp:")
print(ts)

print("\nClosest TLE Epoch:")
print(best["epoch"])

print("\nDifference:")
print(best["time_diff"])

print("\nTLE1:")
print(best["tle1"])

print("\nTLE2:")
print(best["tle2"])