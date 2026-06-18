import pandas as pd

tle_df = pd.read_csv(
    "../phase4/idsmass_data/raw/spacetrack/historical_tles.csv"
)

print(tle_df.head())
print(tle_df["tle1"].iloc[0])