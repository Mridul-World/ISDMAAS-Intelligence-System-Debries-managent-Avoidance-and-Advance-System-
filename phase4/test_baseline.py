import pandas as pd

tle_df = pd.read_csv(
    "idsmass_data/raw/spacetrack/historical_tles.csv"
)

data = pd.read_csv(
    "idsmass_data/processed/historical_trajectory_dataset_v2.csv"
)

print(data.columns)

print(data.head())

print(tle_df.head())