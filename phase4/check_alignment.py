import pandas as pd

df = pd.read_csv(
    "C:\Work_place\projects\isdmaas\phase4\idsmass_data\processed\historical_trajectory_dataset_v2.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

split = int(
    120244 * 0.8
)

print(df.iloc[split + 29][[
    "timestamp",
    "norad_id"
]])

print(df.iloc[split + 30][[
    "timestamp",
    "norad_id"
]])