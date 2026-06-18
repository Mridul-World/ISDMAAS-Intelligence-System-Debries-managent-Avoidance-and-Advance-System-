import pandas as pd

df = pd.read_csv(
    "../phase4/idsmass_data/raw/spacetrack/historical_tles.csv"
)

print(df.head())
print(df.columns)
print(df.shape)