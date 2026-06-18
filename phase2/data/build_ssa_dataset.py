import os
import requests
import pandas as pd
import numpy as np
import time
from io import StringIO

# ======================
# CONFIG
# ======================

DISCOS_TOKEN = "IjJhN2U4ZjQ0LTYzYWUtNDVhYy05ZDc5LWEwMzBjODFjYjExZCI.Kf3nkpVjdN9GvWqA5CqI-XN-roQ"

SPACETRACK_USER = "mridul1735@gmail.com"
SPACETRACK_PASS = "RxR74SUEqXh9Nh-"

DATA_DIR = "ssa_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ======================
# DOWNLOAD CELESTRAK
# ======================

def download_celestrak():

    print("Downloading CelesTrak...")

    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=csv"

    r = requests.get(url)

    if r.status_code != 200:
        raise Exception("Failed to download CelesTrak")

    path = f"{DATA_DIR}/celestrak.csv"

    with open(path, "wb") as f:
        f.write(r.content)

    print("CelesTrak ready")

# ======================
# DOWNLOAD SPACETRACK
# ======================

from io import StringIO

def download_spacetrack():

    print("Downloading SpaceTrack...")

    session = requests.Session()

    login_url = "https://www.space-track.org/ajaxauth/login"

    payload = {
        "identity": SPACETRACK_USER,
        "password": SPACETRACK_PASS
    }

    r = session.post(login_url, data=payload)

    # Debug output
    print("Login response status:", r.status_code)

    if r.status_code != 200:
        raise Exception("SpaceTrack login request failed")

    # Now query data
    url = "https://www.space-track.org/basicspacedata/query/class/gp/format/json"

    r = session.get(url)

    if r.status_code != 200:
        raise Exception("SpaceTrack query failed")

    # If HTML returned, login failed
    if "<html" in r.text.lower():
        raise Exception("SpaceTrack returned HTML (login likely failed)")

    df = pd.read_json(StringIO(r.text))

    df.to_csv(f"{DATA_DIR}/spacetrack.csv", index=False)

    print("SpaceTrack ready:", len(df))

# ======================
# DOWNLOAD DISCOS
# ======================

def download_discos():

    print("Downloading DISCOS...")

    headers = {
        "Authorization": f"Bearer {DISCOS_TOKEN}",
        "DiscosWeb-Api-Version": "2"
    }

    objects = []
    page = 1

    while True:

        params = {
            "page[size]": 100,
            "page[number]": page
        }

        r = requests.get(
            "https://discosweb.esoc.esa.int/api/objects",
            headers=headers,
            params=params
        )

        if r.status_code != 200:
            raise Exception("DISCOS request failed")

        data = r.json()["data"]

        if not data:
            break

        for obj in data:

            attrs = obj["attributes"]
            attrs["NORAD_ID"] = attrs.get("satno")

            objects.append(attrs)

        page += 1
        time.sleep(0.2)

    df = pd.DataFrame(objects)

    df.to_csv(f"{DATA_DIR}/discos.csv", index=False)

    print("DISCOS ready:", len(df))

# ======================
# DOWNLOAD SOCRATES
# ======================

def download_socrates():

    print("Downloading SOCRATES...")

    url = "https://celestrak.org/SOCRATES/socrates.csv"

    r = requests.get(url)

    with open(f"{DATA_DIR}/socrates.csv", "wb") as f:
        f.write(r.content)

    print("SOCRATES ready")

# ======================
# DOWNLOAD SPACE WEATHER
# ======================

def download_space_weather():

    print("Downloading space weather...")

    url = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"

    data = requests.get(url).json()

    df = pd.DataFrame(data)

    df.to_csv(f"{DATA_DIR}/space_weather.csv", index=False)

    print("Space weather ready")

# ======================
# FEATURE ENGINEERING
# ======================

def compute_orbital_features(df):

    mu = 398600.4418

    if "MEAN_MOTION" in df.columns:

        df["orbital_period"] = 1440 / df["MEAN_MOTION"]

        df["semi_major_axis"] = (mu / ((df["MEAN_MOTION"] * 2 * np.pi / 1440) ** 2)) ** (1/3)

        df["altitude"] = df["semi_major_axis"] - 6378

    return df

# ======================
# MERGE DATASETS
# ======================

def combine():

    print("Combining datasets...")

    discos = pd.read_csv(f"{DATA_DIR}/discos.csv", low_memory=False)
    spacetrack = pd.read_csv(f"{DATA_DIR}/spacetrack.csv", low_memory=False)
    celes = pd.read_csv(f"{DATA_DIR}/celestrak.csv", low_memory=False)

    # ---------- Ensure NORAD column exists ----------

    if "satno" in discos.columns:
        discos["NORAD_ID"] = discos["satno"]

    if "NORAD_CAT_ID" in spacetrack.columns:
        spacetrack["NORAD_ID"] = spacetrack["NORAD_CAT_ID"]

    if "NORAD_CAT_ID" in celes.columns:
        celes["NORAD_ID"] = celes["NORAD_CAT_ID"]

    # ---------- Convert to numeric ----------

    discos["NORAD_ID"] = pd.to_numeric(discos["NORAD_ID"], errors="coerce")
    spacetrack["NORAD_ID"] = pd.to_numeric(spacetrack["NORAD_ID"], errors="coerce")
    celes["NORAD_ID"] = pd.to_numeric(celes["NORAD_ID"], errors="coerce")

    # ---------- Merge ----------

    df = discos.merge(spacetrack, on="NORAD_ID", how="outer")
    df = df.merge(celes, on="NORAD_ID", how="outer")

    # ---------- Remove duplicates ----------

    df = df.drop_duplicates(subset=["NORAD_ID"])

    # ---------- Save ----------

    output_path = f"{DATA_DIR}/SSA_MASTER_DATASET.csv"

    df.to_csv(output_path, index=False)

    print("Master SSA dataset ready:", len(df))

# ======================
# RUN PIPELINE
# ======================

if __name__ == "__main__":

    download_celestrak()
    download_spacetrack()
    download_discos()
    download_socrates()
    download_space_weather()
    combine()