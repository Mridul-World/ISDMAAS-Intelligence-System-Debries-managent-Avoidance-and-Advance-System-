import pandas as pd

from orbital_elements import (
    rv_to_orbital_elements
)

INPUT = (
    "idsmass_data/processed/"
    "historical_trajectory_dataset.csv"
)

OUTPUT = (
    "idsmass_data/processed/"
    "historical_trajectory_dataset_v2.csv"
)

df = pd.read_csv(INPUT)

new_cols = []

for _, row in df.iterrows():

    r = [
        row["x_km"],
        row["y_km"],
        row["z_km"]
    ]

    v = [
        row["vx_kms"],
        row["vy_kms"],
        row["vz_kms"]
    ]

    new_cols.append(
        rv_to_orbital_elements(
            r,
            v
        )
    )

orb_df = pd.DataFrame(new_cols)

df = pd.concat(
    [df, orb_df],
    axis=1
)

df.to_csv(
    OUTPUT,
    index=False
)

print(
    "Saved:",
    OUTPUT
)