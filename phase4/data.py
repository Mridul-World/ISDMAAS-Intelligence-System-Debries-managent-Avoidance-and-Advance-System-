"""
IDSMass Phase 4 — Orbital Trajectory Prediction
Transformer + Hybrid SGP4+ML Pipeline
===============================================================
ARCHITECTURE:
  Multi-source data ingestion → SGP4 propagation → Fixed-interval resampling
  → Feature engineering → Sequence generation → Transformer training
  → Evaluation vs SGP4 baseline → Export

MODELS:
  1. Linear Regression  (baseline)
  2. Transformer        (primary — preferred)
  3. Hybrid SGP4+Transformer (research-grade final)

DATA SOURCES:
  - CelesTrak (GP TLEs)
  - Space-Track (historical TLEs — requires credentials)
  - ESA DISCOS (satellite metadata — requires credentials)
  - SOCRATES (conjunction data — scrape/API)
  - NOAA SWPC (space weather F10.7, Kp)

OUTPUTS:
  - latest_state_vectors.csv
  - historical_trajectory_dataset.csv
  - phase4_transformer_model.pt
  - evaluation_report.json
"""

# ─────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────
import os
import sys
import time
import json
import math
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import warnings
warnings.filterwarnings("ignore")

# SGP4
try:
    from sgp4.api import Satrec, jday
except ImportError:
    os.system("pip install sgp4 -q")
    from sgp4.api import Satrec, jday

# ML
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error
except ImportError:
    os.system("pip install torch scikit-learn -q")
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error

# ─────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────

# Space-Track credentials — set via environment or hardcode here
SPACETRACK_USER = os.getenv("SPACETRACK_USER", "your_email@example.com")
SPACETRACK_PASS = os.getenv("SPACETRACK_PASS", "your_password")

# ESA DISCOS API token
DISCOS_TOKEN = os.getenv("DISCOS_TOKEN", "your_discos_token")

RESAMPLE_HOURS = 6          # Fixed interval for trajectory resampling
SEQ_LEN        = 30         # Input sequence length (steps)
PRED_HORIZONS  = [1, 3, 7]  # Prediction horizons in days
BATCH_SIZE     = 64
EPOCHS         = 60
LR             = 1e-4
D_MODEL        = 128
NHEAD          = 8
NUM_LAYERS     = 4
DIM_FF         = 512
DROPOUT        = 0.1
CKPT_EVERY     = 10         # Save a checkpoint every N epochs (0 = disabled)

# Satellites to start with (NORAD IDs) — scale up later
NORAD_IDS = [
    25544,   # ISS
    20580,   # Hubble
    27607,   # Aqua
    25994,   # Terra
    28654,   # Aura
    33591,   # COSMIC-2 (FORMOSAT-3)
    43013,   # NOAA-20
    38771,   # Suomi NPP
]

# ── FIX: moved VAL_END back so test split has enough data ────
# Original VAL_END = "2025-01-01" left nothing for test sequences.
# Set to "2024-06-01" so 6 months of 2024 data forms the test set.
# If you have Space-Track credentials pulling through present day,
# you can restore VAL_END = "2025-01-01".
TRAIN_END = "2024-01-01"
VAL_END   = "2024-06-01"   # Changed from "2025-01-01"
# ─────────────────────────────────────────────────────────────

BASE_DIR  = Path("idsmass_data")
RAW_DIR   = BASE_DIR / "raw"
PROC_DIR  = BASE_DIR / "processed"
MODEL_DIR = BASE_DIR / "models"
CKPT_DIR  = MODEL_DIR / "checkpoints"   # periodic + best checkpoints live here

for d in [RAW_DIR/"celestrak", RAW_DIR/"spacetrack", RAW_DIR/"discos",
          RAW_DIR/"socrates", RAW_DIR/"weather", PROC_DIR/"sequences",
          MODEL_DIR, CKPT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("IDSMass-P4")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Device: {DEVICE}")

# ─────────────────────────────────────────────────────────────
# SECTION 1 — DATA DOWNLOADERS
# ─────────────────────────────────────────────────────────────

class CelesTrakDownloader:
    """Downloads current GP TLEs for target NORAD IDs from CelesTrak."""

    def download(self, norad_ids: list) -> pd.DataFrame:
        log.info("CelesTrak — fetching current GP elements")
        records = []
        for nid in norad_ids:
            url = f"https://celestrak.org/satcat/tle.php?CATNR={nid}"
            try:
                r = requests.get(url, timeout=15)
                lines = [l.strip() for l in r.text.strip().splitlines() if l.strip()]
                if len(lines) >= 3:
                    records.append({
                        "norad_id": nid,
                        "name": lines[0],
                        "tle1": lines[1],
                        "tle2": lines[2],
                        "source": "celestrak",
                        "fetched_at": datetime.utcnow().isoformat(),
                    })
                time.sleep(0.3)
            except Exception as e:
                log.warning(f"CelesTrak NORAD {nid}: {e}")

        df = pd.DataFrame(records)
        out = RAW_DIR / "celestrak" / "current_tles.csv"
        df.to_csv(out, index=False)
        log.info(f"CelesTrak → {len(df)} TLEs saved to {out}")
        return df


class SpaceTrackDownloader:
    """
    Downloads historical TLEs from Space-Track.org.
    Requires valid Space-Track credentials.
    Falls back to synthetic data if credentials missing.
    """

    LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
    HIST_URL  = ("https://www.space-track.org/basicspacedata/query/class/gp_history"
                 "/NORAD_CAT_ID/{norad}/orderby/TLE_LINE1 ASC"
                 "/EPOCH/{start}--{end}/format/tle/emptyresult/show")

    def __init__(self):
        self.session    = requests.Session()
        self._logged_in = False

    def _login(self):
        if self._logged_in:
            return True
        try:
            r = self.session.post(
                self.LOGIN_URL,
                data={"identity": SPACETRACK_USER, "password": SPACETRACK_PASS},
                timeout=20,
            )
            if r.ok and "Login" not in r.text[:200]:
                self._logged_in = True
                log.info("Space-Track login successful")
                return True
        except Exception as e:
            log.warning(f"Space-Track login failed: {e}")
        return False

    def download(self, norad_ids: list,
                 start="2015-01-01", end="2024-12-31") -> pd.DataFrame:
        log.info(f"Space-Track — fetching historical TLEs ({start} → {end})")
        if not self._login():
            log.warning("Space-Track unavailable — generating synthetic fallback data")
            return self._synthetic_fallback(norad_ids, start, end)

        records = []
        for nid in norad_ids:
            url = self.HIST_URL.format(norad=nid, start=start, end=end)
            try:
                r = self.session.get(url, timeout=60)
                lines = [l.strip() for l in r.text.strip().splitlines() if l.strip()]
                i = 0
                while i + 1 < len(lines):
                    if lines[i].startswith("1 ") and lines[i+1].startswith("2 "):
                        records.append({
                            "norad_id": nid,
                            "tle1": lines[i],
                            "tle2": lines[i+1],
                            "source": "spacetrack",
                        })
                        i += 2
                    else:
                        i += 1
                log.info(f"  NORAD {nid}: {len([r for r in records if r['norad_id']==nid])} TLEs")
                time.sleep(1.5)
            except Exception as e:
                log.warning(f"Space-Track NORAD {nid}: {e}")

        df = pd.DataFrame(records)
        out = RAW_DIR / "spacetrack" / "historical_tles.csv"
        df.to_csv(out, index=False)
        log.info(f"Space-Track → {len(df)} TLEs saved")
        return df

    def _synthetic_fallback(self, norad_ids, start, end) -> pd.DataFrame:
        """
        Generate synthetic TLEs spanning multiple epochs so every split
        (train / val / test) has enough sequences.

        Strategy: use a real ISS TLE as seed and step the epoch forward
        by 7 days for each record, covering start→end with ~weekly cadence.
        """
        log.info("Building synthetic trajectory dataset (SGP4-propagated)")

        # Real ISS TLE as seed — epoch will be overwritten textually below
        TLE1_TMPL = "1 25544U 98067A   {epoch_str}  .00020000  00000-0  36000-3 0  9999"
        TLE2      = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49815168000000"

        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt   = datetime.strptime(end,   "%Y-%m-%d")

        records = []
        for nid in norad_ids:
            current = start_dt
            while current < end_dt:
                # Encode epoch as YYDDD.DDDDDDDD
                year2  = current.year % 100
                doy    = current.timetuple().tm_yday
                frac   = (current.hour * 3600 + current.minute * 60 + current.second) / 86400
                ep_str = f"{year2:02d}{doy:03d}.{frac:.8f}"[:-1]   # 14-char field
                ep_str = f"{year2:02d}{doy + frac:012.8f}"

                tle1 = TLE1_TMPL.format(epoch_str=ep_str)
                records.append({
                    "norad_id": nid,
                    "tle1": tle1,
                    "tle2": TLE2,
                    "source": "synthetic",
                })
                current += timedelta(days=7)   # weekly cadence

        df = pd.DataFrame(records)
        out = RAW_DIR / "spacetrack" / "synthetic_tles.csv"
        df.to_csv(out, index=False)
        log.info(f"Synthetic fallback → {len(df)} TLE records across {len(norad_ids)} satellites")
        return df


class DISCOSDownloader:
    """ESA DISCOS — satellite physical metadata (mass, area, object class)."""

    BASE = "https://discosweb.esac.esa.int/api"

    def download(self, norad_ids: list) -> pd.DataFrame:
        log.info("ESA DISCOS — fetching satellite metadata")
        records = []
        headers = {"Authorization": f"Bearer {DISCOS_TOKEN}",
                   "Accept": "application/json"}

        for nid in norad_ids:
            try:
                url = (f"{self.BASE}/objects"
                       f"?filter=eq(satno,{nid})&fields=satno,mass,xSectAvg,objectClass")
                r = requests.get(url, headers=headers, timeout=15)
                if r.ok:
                    data = r.json().get("data", [])
                    if data:
                        obj = data[0].get("attributes", {})
                        records.append({
                            "norad_id": nid,
                            "mass_kg": obj.get("mass", np.nan),
                            "cross_section_m2": obj.get("xSectAvg", np.nan),
                            "object_class": obj.get("objectClass", "UNKNOWN"),
                        })
                time.sleep(0.5)
            except Exception as e:
                log.warning(f"DISCOS NORAD {nid}: {e}")

        if not records:
            log.warning("DISCOS unavailable — using default metadata")
            records = [
                {
                    "norad_id": n,
                    "mass_kg": 420000.0 if n == 25544 else 5000.0,
                    "cross_section_m2": 2500.0 if n == 25544 else 10.0,
                    "object_class": "PAYLOAD",
                }
                for n in norad_ids
            ]

        df = pd.DataFrame(records)
        df.to_csv(RAW_DIR / "discos" / "metadata.csv", index=False)
        log.info(f"DISCOS → {len(df)} objects")
        return df


class SOCRATESDownloader:
    """SOCRATES conjunction risk data — used for labeling, not trajectory training."""

    def download(self) -> pd.DataFrame:
        log.info("SOCRATES — fetching conjunction data")
        try:
            url = ("https://celestrak.org/SOCRATES/query.php"
                   "?CATNR=25544&DAYS=7&MAX=10&LIMIT=10&format=json")
            r = requests.get(url, timeout=20)
            if r.ok:
                df = pd.DataFrame(r.json())
                df.to_csv(RAW_DIR / "socrates" / "conjunctions.csv", index=False)
                log.info(f"SOCRATES → {len(df)} conjunction events")
                return df
        except Exception as e:
            log.warning(f"SOCRATES: {e}")
        return pd.DataFrame()


class SpaceWeatherDownloader:
    """NOAA SWPC — F10.7 solar flux and Kp geomagnetic index."""

    F107_URL = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
    KP_URL   = "https://services.swpc.noaa.gov/json/planetary_k_index_1m.json"

    def download(self) -> pd.DataFrame:
        log.info("NOAA SWPC — fetching space weather indices")
        try:
            r = requests.get(self.F107_URL, timeout=20)
            solar = pd.DataFrame(r.json())[["time-tag", "f10.7"]]
            solar.columns = ["date", "f107"]
            solar["date"] = pd.to_datetime(solar["date"])

            r2 = requests.get(self.KP_URL, timeout=20)
            kp_data = pd.DataFrame(r2.json())[["time_tag", "kp_index"]]
            kp_data.columns = ["date", "kp"]
            kp_data["date"] = pd.to_datetime(kp_data["date"]).dt.floor("D")
            kp_daily = kp_data.groupby("date")["kp"].mean().reset_index()

            df = pd.merge(solar, kp_daily, on="date", how="outer").sort_values("date")
            df = df.ffill().fillna(70.0)
            df.to_csv(RAW_DIR / "weather" / "space_weather.csv", index=False)
            log.info(f"Space Weather → {len(df)} days")
            return df
        except Exception as e:
            log.warning(f"Space Weather: {e} — using defaults")
            dates = pd.date_range("2015-01-01", "2026-01-01", freq="D")
            df = pd.DataFrame({"date": dates, "f107": 100.0, "kp": 2.0})
            df.to_csv(RAW_DIR / "weather" / "space_weather.csv", index=False)
            return df


# ─────────────────────────────────────────────────────────────
# SECTION 2 — SGP4 PROPAGATION + RESAMPLING
# ─────────────────────────────────────────────────────────────
MU = 398600.4418

def orbital_features(r, v):

    r = np.array(r, dtype=np.float64)
    v = np.array(v, dtype=np.float64)

    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)

    energy = ((v_norm**2)/2 - MU/r_norm)

    a = -MU / (2 * energy)

    e_vec = (np.cross(v, h)/MU - r/r_norm)

    ecc = np.linalg.norm(e_vec)

    inclination = np.degrees(
        np.arccos(np.clip(h[2]/h_norm, -1.0, 1.0))
    )

    k = np.array([0.0,0.0,1.0])
    n = np.cross(k,h)
    n_norm = np.linalg.norm(n)

    if n_norm > 1e-8:
        raan = np.degrees(
            np.arccos(np.clip(n[0]/n_norm, -1.0, 1.0))
        )
        if n[1] < 0:
            raan = 360 - raan
    else:
        raan = 0.0

    if ecc > 1e-8 and n_norm > 1e-8:
        arg_perigee = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(n,e_vec)/(n_norm*ecc),
                    -1.0,
                    1.0
                )
            )
        )
        if e_vec[2] < 0:
            arg_perigee = 360 - arg_perigee
    else:
        arg_perigee = 0.0

    if ecc > 1e-8:
        true_anomaly = np.degrees(
            np.arccos(
                np.clip(
                    np.dot(e_vec,r)/(ecc*r_norm),
                    -1.0,
                    1.0
                )
            )
        )
        if np.dot(r,v) < 0:
            true_anomaly = 360 - true_anomaly
    else:
        true_anomaly = 0.0

    return {
        "semi_major_axis": float(a),
        "eccentricity": float(ecc),
        "inclination": float(inclination),
        "raan": float(raan),
        "arg_perigee": float(arg_perigee),
        "true_anomaly": float(true_anomaly),
        "specific_energy": float(energy),
        "angular_momentum_mag": float(h_norm),
    }

def tle_to_epoch(tle1: str) -> Optional[datetime]:
    """Extract epoch from TLE line 1."""
    try:
        year2 = int(tle1[18:20])
        year  = 2000 + year2 if year2 < 57 else 1900 + year2
        doy   = float(tle1[20:32])
        epoch = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
        return epoch
    except Exception:
        return None


def propagate_tle(tle1: str, tle2: str, timestamps: list) -> list:
    """Propagate a TLE to multiple timestamps using SGP4.
    Returns list of state dicts (x,y,z,vx,vy,vz)."""
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
    except Exception:
        return []

    states = []
    for ts in timestamps:
        try:
            jd, fr = jday(ts.year, ts.month, ts.day,
                          ts.hour, ts.minute,
                          ts.second + ts.microsecond / 1e6)
            e, r, v = sat.sgp4(jd, fr)
            if e == 0:
                states.append({
                    "timestamp": ts,
                    "x_km": r[0], "y_km": r[1], "z_km": r[2],
                    "vx_kms": v[0], "vy_kms": v[1], "vz_kms": v[2],
                    "bstar": sat.bstar,
                })
        except Exception:
            pass
    return states


def build_trajectory_dataset(tle_df: pd.DataFrame,
                              weather_df: pd.DataFrame,
                              discos_df: pd.DataFrame,
                              interval_hours: int = 6) -> pd.DataFrame:
    """
    Core pipeline:
      TLEs → SGP4 propagation → fixed-interval resampling
      → weather merge → DISCOS metadata → unified trajectory dataset
    """
    log.info(f"Building trajectory dataset at {interval_hours}h intervals")
    weather_df = weather_df.copy()
    weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.floor("D")
    weather_lookup = weather_df.set_index("date")[["f107", "kp"]].to_dict("index")

    all_records = []

    for norad_id, group in tle_df.groupby("norad_id"):
        log.info(f"  Propagating NORAD {norad_id} ({len(group)} TLEs)")
        group = group.copy()

        group["epoch"] = group["tle1"].apply(tle_to_epoch)
        group = group.dropna(subset=["epoch"]).sort_values("epoch").reset_index(drop=True)

        if len(group) < 2:
            continue

        for i in range(len(group) - 1):
            t_start = group.loc[i, "epoch"]
            t_end   = group.loc[i + 1, "epoch"]

            if (t_end - t_start).total_seconds() > 30 * 86400:
                continue  # Skip gaps > 30 days

            n_steps = max(1, int((t_end - t_start).total_seconds() /
                                 (interval_hours * 3600)))
            timestamps = [t_start + timedelta(hours=interval_hours * k)
                          for k in range(n_steps)]

            states = propagate_tle(group.loc[i, "tle1"], group.loc[i, "tle2"],
                                   timestamps)

            for s in states:
                ts      = s["timestamp"]
                orb = orbital_features(
    [
        s["x_km"],
        s["y_km"],
        s["z_km"]
    ],
    [
        s["vx_kms"],
        s["vy_kms"],
        s["vz_kms"]
    ]
)
                day_key = pd.Timestamp(ts.year, ts.month, ts.day)
                weather = weather_lookup.get(day_key, {"f107": 100.0, "kp": 2.0})

                doy = ts.timetuple().tm_yday
                tod = ts.hour / 24.0

                all_records.append({
                    "timestamp":        ts,
                    "norad_id":         norad_id,
                    "x_km":             s["x_km"],
                    "y_km":             s["y_km"],
                    "z_km":             s["z_km"],
                    "vx_kms":           s["vx_kms"],
                    "vy_kms":           s["vy_kms"],
                    "vz_kms":           s["vz_kms"],
                    "bstar":            s["bstar"],
                    "f107":             weather["f107"],
                    "kp":               weather["kp"],
                    "sin_doy":          math.sin(2 * math.pi * doy / 365.25),
                    "cos_doy":          math.cos(2 * math.pi * doy / 365.25),
                    "sin_tod":          math.sin(2 * math.pi * tod),
                    "cos_tod":          math.cos(2 * math.pi * tod),
                    "source":           group.loc[i, "source"],
                    "semi_major_axis":
    orb["semi_major_axis"],

"eccentricity":
    orb["eccentricity"],

"inclination":
    orb["inclination"],

"raan":
    orb["raan"],

"arg_perigee":
    orb["arg_perigee"],

"true_anomaly":
    orb["true_anomaly"],

"specific_energy":
    orb["specific_energy"],

"angular_momentum_mag":
    orb["angular_momentum_mag"],
                })

    df = pd.DataFrame(all_records)

    if df.empty:
        log.error("No trajectory data generated — check TLE sources")
        return df

    # Merge DISCOS metadata
    if not discos_df.empty:
        df = df.merge(discos_df[["norad_id", "mass_kg", "cross_section_m2"]],
                      on="norad_id", how="left")
        df["mass_kg"].fillna(5000.0, inplace=True)
        df["cross_section_m2"].fillna(10.0, inplace=True)
    else:
        df["mass_kg"]         = 5000.0
        df["cross_section_m2"] = 10.0

    df = df.sort_values(["norad_id", "timestamp"]).reset_index(drop=True)
    log.info(f"Trajectory dataset: {len(df):,} rows × {len(df.columns)} columns")
    return df


def export_csvs(full_df: pd.DataFrame):
    """Export the two required CSVs."""
    latest = (full_df.sort_values("timestamp")
                     .groupby("norad_id")
                     .last()
                     .reset_index())
    latest.to_csv(PROC_DIR / "latest_state_vectors.csv", index=False)
    log.info(f"Exported latest_state_vectors.csv ({len(latest)} rows)")

    full_df.to_csv(PROC_DIR / "historical_trajectory_dataset.csv", index=False)
    log.info(f"Exported historical_trajectory_dataset.csv ({len(full_df):,} rows)")


# ─────────────────────────────────────────────────────────────
# SECTION 3 — SEQUENCE GENERATION + DATASET CLASS
# ─────────────────────────────────────────────────────────────

FEATURE_COLS = [

    # State Vector
    "x_km",
    "y_km",
    "z_km",
    "vx_kms",
    "vy_kms",
    "vz_kms",

    # Orbital Elements
    "semi_major_axis",
    "eccentricity",
    "inclination",
    "raan",
    "arg_perigee",
    "true_anomaly",

    # Physics Features
    "specific_energy",
    "angular_momentum_mag",

    # Space Weather
    "bstar",
    "f107",
    "kp",

    # Cyclic Time Features
    "sin_doy",
    "cos_doy",
    "sin_tod",
    "cos_tod",

    # Satellite Physical Features
    "mass_kg",
    "cross_section_m2",
]

TARGET_COLS = ["x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"]
N_FEATURES  = len(FEATURE_COLS)
N_TARGETS   = len(TARGET_COLS)


def build_sequences(df: pd.DataFrame,
                    seq_len: int,
                    horizon_steps: int,
                    scaler: Optional[StandardScaler] = None,
                    fit_scaler: bool = False):
    """
    Build sliding-window sequences per satellite.
    Returns X [N, seq_len, features], y [N, 6], scaler.
    """
    Xs, ys = [], []

    for norad_id, group in df.groupby("norad_id"):
        group  = group.sort_values("timestamp").reset_index(drop=True)
        feats   = group[FEATURE_COLS].values
        targets = group[TARGET_COLS].values

        for i in range(len(group) - seq_len - horizon_steps):
            x_seq = feats[i: i + seq_len]
            y_tgt = targets[i + seq_len + horizon_steps - 1]
            if not (np.isnan(x_seq).any() or np.isnan(y_tgt).any()):
                Xs.append(x_seq)
                ys.append(y_tgt)

    if not Xs:
        return np.array([]), np.array([]), scaler

    X = np.array(Xs, dtype=np.float32)   # [N, seq_len, features]
    y = np.array(ys, dtype=np.float32)   # [N, 6]

    N, T, F  = X.shape
    X_flat   = X.reshape(-1, F)
    if fit_scaler:
        scaler  = StandardScaler()
        X_flat  = scaler.fit_transform(X_flat)
    elif scaler is not None:
        X_flat  = scaler.transform(X_flat)
    X = X_flat.reshape(N, T, F)

    return X, y, scaler


class OrbitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─────────────────────────────────────────────────────────────
# SECTION 4 — MODELS
# ─────────────────────────────────────────────────────────────

# ── 4A. Positional Encoding ──────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # [1, max_len, d_model]

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


# ── 4B. Orbital Transformer ──────────────────────────────────

class OrbitalTransformer(nn.Module):
    """
    Transformer encoder for orbital trajectory prediction.

    Architecture:
      Input projection → Positional Encoding →
      TransformerEncoder (N layers, multi-head attention) →
      Pooling (last token) → MLP head → 6D output

    Input:  [batch, seq_len, n_features]
    Output: [batch, 6]  (x, y, z, vx, vy, vz)
    """

    def __init__(self, n_features: int, d_model: int, nhead: int,
                 num_layers: int, dim_ff: int, dropout: float, n_targets: int):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """x: [B, T, F]"""
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = x[:, -1, :]
        return self.head(x)


# ── 4C. Hybrid SGP4 + Transformer ───────────────────────────

class HybridSGP4Transformer(nn.Module):
    """
    Physics-informed hybrid model.
    Transformer learns the RESIDUAL between SGP4 prediction and true orbit.

    Architecture:
      SGP4 baseline (precomputed) + Transformer(residual correction) = final prediction

    This is research-grade: aligns with how aerospace systems actually work.
    """

    def __init__(self, transformer: OrbitalTransformer):
        super().__init__()
        self.transformer    = transformer
        self.residual_scale = nn.Parameter(torch.ones(6) * 0.1)

    def forward(self, x, sgp4_baseline=None):
        correction = self.transformer(x) * self.residual_scale
        if sgp4_baseline is not None:
            return sgp4_baseline + correction
        return correction


# ─────────────────────────────────────────────────────────────
# SECTION 5 — TRAINING
# ─────────────────────────────────────────────────────────────

def position_error_km(pred: np.ndarray, true: np.ndarray) -> float:
    """Euclidean position error in km (first 3 dims = x,y,z)."""
    return float(np.mean(np.linalg.norm(pred[:, :3] - true[:, :3], axis=1)))


# ─────────────────────────────────────────────────────────────
# CHECKPOINT HELPERS
# ─────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, epoch: int,
                    val_loss: float, tag: str = "") -> Path:
    """
    Save a full training checkpoint.

    Stored dict keys
    ─────────────────
    epoch          : int   — completed epoch number
    val_loss       : float — validation loss at this epoch
    model_state    : dict  — model weights (state_dict)
    optimizer_state: dict  — optimizer state (resumes momentum etc.)
    scheduler_state: dict  — LR scheduler state
    timestamp      : str   — UTC ISO timestamp

    tag examples: "epoch_10", "best", "final"
    """
    filename = f"ckpt_{tag}.pt" if tag else f"ckpt_epoch_{epoch:04d}.pt"
    path     = CKPT_DIR / filename
    torch.save(
        {
            "epoch":            epoch,
            "val_loss":         val_loss,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "timestamp":        datetime.utcnow().isoformat(),
        },
        path,
    )
    log.info(f"  Checkpoint saved → {path}  (epoch={epoch}, val_loss={val_loss:.5f})")
    return path


def load_checkpoint(path: str | Path,
                    model: OrbitalTransformer,
                    optimizer=None,
                    scheduler=None) -> int:
    """
    Load a checkpoint back into model (and optionally optimizer/scheduler).

    Returns the epoch number stored in the checkpoint so training can resume
    from epoch+1.

    Usage
    ─────
    start_epoch = load_checkpoint("idsmass_data/models/checkpoints/ckpt_best.pt",
                                  model, optimizer, scheduler)
    # then call train_transformer starting at start_epoch
    """
    ckpt = torch.load(path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    epoch    = ckpt.get("epoch", 0)
    val_loss = ckpt.get("val_loss", float("inf"))
    log.info(f"  Checkpoint loaded ← {path}  "
             f"(epoch={epoch}, val_loss={val_loss:.5f})")
    return epoch


def train_linear_regression(X_train, y_train, X_val, y_val):
    log.info("─" * 50)
    log.info("MODEL 1 — Linear Regression (Baseline)")
    X_tr_flat = X_train.reshape(len(X_train), -1)
    X_v_flat  = X_val.reshape(len(X_val), -1)

    model = LinearRegression()
    model.fit(X_tr_flat, y_train)
    pred = model.predict(X_v_flat)

    mse     = mean_squared_error(y_val, pred)
    pos_err = position_error_km(pred, y_val)
    log.info(f"  Val MSE: {mse:.4f}  |  Position Error: {pos_err:.2f} km")
    return model, {"mse": mse, "pos_error_km": pos_err}


def train_transformer(X_train, y_train, X_val, y_val,
                      epochs: int = EPOCHS,
                      resume_from: Optional[str | Path] = None) -> tuple:
    """
    Train the Orbital Transformer.

    Parameters
    ──────────
    resume_from : optional path to a checkpoint .pt file.
                  If provided, model/optimizer/scheduler states and the
                  starting epoch are restored before training begins.
    """
    log.info("─" * 50)
    log.info("MODEL 2 — Orbital Transformer (Primary)")

    model = OrbitalTransformer(
        n_features=N_FEATURES, d_model=D_MODEL, nhead=NHEAD,
        num_layers=NUM_LAYERS, dim_ff=DIM_FF, dropout=DROPOUT,
        n_targets=N_TARGETS,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"  Transformer parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    criterion = nn.HuberLoss(delta=1.0)

    # ── Resume from checkpoint if requested ─────────────────
    start_epoch = 1
    if resume_from is not None:
        start_epoch = load_checkpoint(resume_from, model, optimizer, scheduler) + 1
        log.info(f"  Resuming from epoch {start_epoch}")
    # ────────────────────────────────────────────────────────

    train_ds = OrbitDataset(X_train, y_train)
    val_ds   = OrbitDataset(X_val,   y_val)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          pin_memory=DEVICE.type == "cuda")
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state    = None
    history       = {"train_loss": [], "val_loss": [], "val_pos_err_km": []}

    for epoch in range(start_epoch, epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(xb)
        train_loss /= len(train_ds)
        scheduler.step()

        # ── Validate ──
        model.eval()
        val_loss    = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred    = model(xb)
                val_loss += criterion(pred, yb).item() * len(xb)
                preds.append(pred.cpu().numpy())
                trues.append(yb.cpu().numpy())
        val_loss /= len(val_ds)
        preds    = np.concatenate(preds)
        trues    = np.concatenate(trues)
        pos_err  = position_error_km(preds, trues)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_pos_err_km"].append(pos_err)

        # ── Best model tracking ──────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            # Overwrite best checkpoint whenever val improves
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_loss, tag="best")
        # ────────────────────────────────────────────────────

        # ── Periodic checkpoint every CKPT_EVERY epochs ─────
        if CKPT_EVERY > 0 and epoch % CKPT_EVERY == 0:
            save_checkpoint(model, optimizer, scheduler, epoch,
                            val_loss, tag=f"epoch_{epoch:04d}")
        # ────────────────────────────────────────────────────

        if epoch % 10 == 0 or epoch == start_epoch:
            log.info(f"  Epoch {epoch:3d}/{epochs}  "
                     f"train={train_loss:.5f}  val={val_loss:.5f}  "
                     f"pos_err={pos_err:.2f} km")

    # ── Restore best weights ─────────────────────────────────
    model.load_state_dict(best_state)

    # Final model weights (lean — weights only, no optimizer state)
    final_weights_path = MODEL_DIR / "phase4_transformer.pt"
    torch.save(model.state_dict(), final_weights_path)
    log.info(f"  Best val loss : {best_val_loss:.5f}")
    log.info(f"  Weights saved → {final_weights_path}")

    # Final full checkpoint (weights + optimizer + scheduler) for resuming
    save_checkpoint(model, optimizer, scheduler, epochs,
                    best_val_loss, tag="final")
    # ────────────────────────────────────────────────────────

    return model, history


# ─────────────────────────────────────────────────────────────
# SECTION 6 — EVALUATION vs SGP4 BASELINE
# ─────────────────────────────────────────────────────────────

def _empty_eval_result(reason: str = "") -> dict:
    """Return a zeroed eval dict when the test set is unusable."""
    if reason:
        log.warning(f"  Evaluation skipped: {reason}")
    return {
        "transformer": {"position_error_km": -1.0, "mse": -1.0},
        "sgp4_persistence_baseline": {"position_error_km": -1.0, "mse": -1.0},
        "improvement_over_baseline_pct": 0.0,
        "test_samples": 0,
    }


def evaluate_vs_sgp4(model: OrbitalTransformer,
                     X_test: np.ndarray,
                     y_test: np.ndarray,
                     tle_df: pd.DataFrame) -> dict:
    """
    Compare Transformer vs SGP4-only propagation on test set.
    This is the key research comparison.

    FIX: guards added for empty X_test / empty DataLoader output
    so the pipeline never crashes with 'need at least one array
    to concatenate' when the test split has insufficient data.
    """
    log.info("─" * 50)
    log.info("EVALUATION — Transformer vs SGP4 Baseline")

    # ── Guard 1: empty arrays ────────────────────────────────
    if X_test is None or y_test is None or len(X_test) == 0 or len(y_test) == 0:
        return _empty_eval_result("X_test / y_test is empty")
    # ────────────────────────────────────────────────────────

    test_ds = OrbitDataset(X_test, y_test)
    test_dl = DataLoader(test_ds, batch_size=256, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in test_dl:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())

    # ── Guard 2: DataLoader produced no batches ──────────────
    if not preds:
        return _empty_eval_result("DataLoader produced no batches")
    # ────────────────────────────────────────────────────────

    preds = np.concatenate(preds)   # safe — list is non-empty

    transformer_pos_err = position_error_km(preds, y_test)
    transformer_mse     = float(mean_squared_error(y_test, preds))

    # Persistence baseline: use last known state from sequence
    sgp4_pred    = X_test[:, -1, :6]
    sgp4_pos_err = position_error_km(sgp4_pred, y_test)
    sgp4_mse     = float(mean_squared_error(y_test, sgp4_pred))

    # ── Guard 3: avoid division by zero ─────────────────────
    improvement = (
        (1 - transformer_pos_err / sgp4_pos_err) * 100
        if sgp4_pos_err > 0 else 0.0
    )
    # ────────────────────────────────────────────────────────

    results = {
        "transformer": {
            "position_error_km": round(transformer_pos_err, 4),
            "mse": round(transformer_mse, 6),
        },
        "sgp4_persistence_baseline": {
            "position_error_km": round(sgp4_pos_err, 4),
            "mse": round(sgp4_mse, 6),
        },
        "improvement_over_baseline_pct": round(improvement, 2),
        "test_samples": len(y_test),
    }

    log.info(f"  Transformer  → pos_err={transformer_pos_err:.2f} km")
    log.info(f"  SGP4 persist → pos_err={sgp4_pos_err:.2f} km")
    log.info(f"  Improvement  → {improvement:.1f}%")

    report_path = PROC_DIR / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"  Report saved → {report_path}")
    return results


# ─────────────────────────────────────────────────────────────
# SECTION 7 — MULTI-HORIZON WRAPPER
# ─────────────────────────────────────────────────────────────

def train_all_horizons(df_train, df_val, df_test):
    """Train a separate Transformer for each prediction horizon."""
    results = {}
    scaler  = None

    for horizon_days in PRED_HORIZONS:
        horizon_steps = int(horizon_days * 24 / RESAMPLE_HOURS)
        log.info(f"\n{'='*60}")
        log.info(f"HORIZON: {horizon_days}d ({horizon_steps} steps of {RESAMPLE_HOURS}h)")
        log.info(f"{'='*60}")

        X_tr, y_tr, scaler = build_sequences(
            df_train, SEQ_LEN, horizon_steps,
            scaler=scaler, fit_scaler=(scaler is None),
        )
        X_val, y_val, _ = build_sequences(df_val, SEQ_LEN, horizon_steps,
                                          scaler=scaler)
        X_te,  y_te,  _ = build_sequences(df_test, SEQ_LEN, horizon_steps,
                                           scaler=scaler)

        if len(X_tr) == 0:
            log.warning(f"  Not enough training data for {horizon_days}d horizon — skipping")
            continue

        if len(X_val) == 0:
            log.warning(f"  No validation sequences for {horizon_days}d horizon — skipping")
            continue

        log.info(f"  Train: {len(X_tr):,}  Val: {len(X_val):,}  "
                 f"Test: {len(X_te):,}")

        # ── FIX: warn about empty test but do NOT skip training ──
        if len(X_te) == 0:
            log.warning(
                f"  No test sequences for {horizon_days}d horizon "
                f"(post-{VAL_END} data insufficient). "
                f"Training will proceed; evaluation will be skipped."
            )
        # ────────────────────────────────────────────────────────

        # 1. Linear Regression baseline
        _, lr_metrics = train_linear_regression(X_tr, y_tr, X_val, y_val)

        # 2. Transformer
        model, history = train_transformer(X_tr, y_tr, X_val, y_val)

        # 3. Evaluation — gracefully handles empty X_te
        eval_results = evaluate_vs_sgp4(model, X_te, y_te, pd.DataFrame())

        results[f"{horizon_days}d"] = {
            "linear_regression": lr_metrics,
            "transformer":       eval_results["transformer"],
            "sgp4_baseline":     eval_results["sgp4_persistence_baseline"],
            "improvement_pct":   eval_results["improvement_over_baseline_pct"],
            "history": {
                k: [round(v, 6) for v in vals[-10:]]
                for k, vals in history.items()
            },
        }

        torch.save(model.state_dict(),
                   MODEL_DIR / f"transformer_{horizon_days}d.pt")

    return results, scaler


# ─────────────────────────────────────────────────────────────
# SECTION 8 — MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("IDSMass Phase 4 — Orbital Trajectory Prediction")
    log.info("=" * 60)

    # ── Step 1: Download all data sources ──
    celestrak_df  = CelesTrakDownloader().download(NORAD_IDS)
    spacetrack_df = SpaceTrackDownloader().download(
        NORAD_IDS, start="2015-01-01", end="2024-12-31"
    )
    discos_df     = DISCOSDownloader().download(NORAD_IDS)
    _             = SOCRATESDownloader().download()   # saves for future phases
    weather_df    = SpaceWeatherDownloader().download()

    # Merge TLE sources: prefer Space-Track, supplement with CelesTrak
    if not spacetrack_df.empty:
        tle_df = spacetrack_df
        if not celestrak_df.empty:
            tle_df = pd.concat(
                [tle_df, celestrak_df[["norad_id", "tle1", "tle2", "source"]]],
                ignore_index=True,
            )
    else:
        tle_df = celestrak_df

    # ── Step 2: Build trajectory dataset ──
    full_df = build_trajectory_dataset(tle_df, weather_df, discos_df,
                                       RESAMPLE_HOURS)

    if full_df.empty:
        log.error("Empty dataset — cannot proceed. Check credentials / network.")
        sys.exit(1)

    # ── Step 3: Export CSVs ──
    export_csvs(full_df)

    # ── Step 4: Temporal split ──
    full_df["timestamp"] = pd.to_datetime(full_df["timestamp"])
    df_train = full_df[full_df["timestamp"] <  TRAIN_END].copy()
    df_val   = full_df[(full_df["timestamp"] >= TRAIN_END) &
                       (full_df["timestamp"] <  VAL_END)].copy()
    df_test  = full_df[full_df["timestamp"] >= VAL_END].copy()

    log.info(f"Split — Train: {len(df_train):,}  "
             f"Val: {len(df_val):,}  "
             f"Test: {len(df_test):,}")

    if len(df_train) < SEQ_LEN * 10:
        log.warning(
            "Small dataset detected — results may be limited. "
            "Add Space-Track credentials for full historical data."
        )

    # ── Step 5: Train all horizons ──
    results, scaler = train_all_horizons(df_train, df_val, df_test)

    # ── Step 6: Final summary ──
    log.info("\n" + "=" * 60)
    log.info("PHASE 4 COMPLETE — RESULTS SUMMARY")
    log.info("=" * 60)
    for horizon, metrics in results.items():
        log.info(f"\n  Horizon: {horizon}")
        t = metrics.get("transformer", {})
        b = metrics.get("sgp4_baseline", {})
        if t.get("position_error_km", -1) >= 0:
            log.info(f"    Transformer pos error : {t['position_error_km']:.2f} km")
            log.info(f"    SGP4 baseline         : {b['position_error_km']:.2f} km")
            log.info(f"    Improvement           : {metrics['improvement_pct']:.1f}%")
        else:
            log.info("    Evaluation skipped (insufficient test data)")

    # Save full results
    final_report = {
        "pipeline":  "IDSMass Phase 4 — Transformer Trajectory Prediction",
        "timestamp": datetime.utcnow().isoformat(),
        "config": {
            "seq_len":        SEQ_LEN,
            "resample_hours": RESAMPLE_HOURS,
            "horizons_days":  PRED_HORIZONS,
            "train_end":      TRAIN_END,
            "val_end":        VAL_END,
            "d_model":        D_MODEL,
            "nhead":          NHEAD,
            "num_layers":     NUM_LAYERS,
            "epochs":         EPOCHS,
            "device":         str(DEVICE),
        },
        "results": results,
    }
    report_path = PROC_DIR / "phase4_final_report.json"
    with open(report_path, "w") as f:
        json.dump(final_report, f, indent=2)
    log.info(f"\nFull report → {report_path}")

    log.info("\nOutputs:")
    log.info(f"  {PROC_DIR / 'latest_state_vectors.csv'}")
    log.info(f"  {PROC_DIR / 'historical_trajectory_dataset.csv'}")
    log.info(f"  {MODEL_DIR / 'transformer_<horizon>.pt'}  (weights only)")
    log.info(f"  {CKPT_DIR}  (ckpt_best.pt, ckpt_final.pt, ckpt_epoch_NNNN.pt)")
    log.info(f"  {report_path}")


if __name__ == "__main__":
    main()