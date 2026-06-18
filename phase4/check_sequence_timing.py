import pandas as pd

SEQ_LEN = 30
HORIZON = 1

df = pd.read_csv(
    "idsmass_data/processed/historical_trajectory_dataset_v2.csv"
)

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="mixed",
    utc=True
)

i = 1000

input_end = df.iloc[
    i + SEQ_LEN - 1
]

target = df.iloc[
    i + SEQ_LEN + HORIZON - 1
]

print("Input end:")
print(input_end["timestamp"])

print("\nTarget:")
print(target["timestamp"])

print("\nDifference:")
print(
    target["timestamp"]
    -
    input_end["timestamp"]
)