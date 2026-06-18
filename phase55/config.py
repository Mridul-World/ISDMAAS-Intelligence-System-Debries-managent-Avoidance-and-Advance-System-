"""
Phase 5.5 — Real Truth Validation — INDUSTRY CONFIGURATION (multi-satellite).

Truth  : ESA Copernicus POD precise ephemerides (AUX_POEORB .EOF, ~5 cm, ITRF)
Inputs : TLE/SGP4-derived features ONLY (deployable — at inference time you
         never need precise ephemerides, only public TLEs).

Run once: downloads + dataset + tensors are persisted; later phases reuse them.
"""
from datetime import datetime, timezone

# ---------------------------------------------------------------- fleet
# CDSE AUX_POEORB availability is queried per satellite; satellites whose
# query returns nothing are reported and skipped gracefully.
SATELLITES = {
    39634: {"name": "SENTINEL-1A", "prefix": "S1A", "mass_kg": 2300.0, "area_m2": 35.0},
    40697: {"name": "SENTINEL-2A", "prefix": "S2A", "mass_kg": 1140.0, "area_m2": 18.0},
    42063: {"name": "SENTINEL-2B", "prefix": "S2B", "mass_kg": 1140.0, "area_m2": 18.0},
    41335: {"name": "SENTINEL-3A", "prefix": "S3A", "mass_kg": 1250.0, "area_m2": 21.0},
    43437: {"name": "SENTINEL-3B", "prefix": "S3B", "mass_kg": 1250.0, "area_m2": 21.0},
}

CFG = {
    # ---- data window (1 year default; widen for more solar-cycle coverage) ----
    "DATE_START": "2024-01-01",
    "DATE_END":   "2024-12-31",

    # ---- paths (relative to phase55/) ----
    "TLE_DIR":        "data/tle",                 # tle_<norad>.csv
    "POE_DIR":        "data/precise_orbit",       # <PREFIX>/*.EOF
    "SW_CSV":         "data/space_weather_daily.csv",
    "EPH_DIR":        "data",                     # ephemeris_<norad>.csv (TEME truth)
    "DATASET_CSV":    "data/phase55_dataset.csv", # combined, all satellites
    "TENSOR_DIR":     ".",
    "MODEL_DIR":      "model",
    "REPORT_DIR":     "reports",

    # ---- dataset geometry (Phase-5 contract, frozen) ----
    "SEQ": 30,
    "N_FEATURES": 23,
    "FEATURE_STEP_S": 600,        # 10-min grid
    "HORIZON_S": 259200,           # 1-day prediction horizon
    "TRAIN_FRACTION": 0.8,        # chronological split PER SATELLITE
    "MAX_WINDOWS_PER_SAT": 20000, # uniform subsample cap (memory control)
    "SEED": 42,

    # ---- credentials read from environment ----
    # SPACETRACK_USER / SPACETRACK_PASS   (https://www.space-track.org)
    # CDSE_USER / CDSE_PASS               (https://dataspace.copernicus.eu)
}

# 23 features — IDENTICAL order to Phase 4/5 (frozen, do not reorder)
FEATURE_COLS = [
    "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms",
    "semi_major_axis", "eccentricity", "inclination",
    "raan", "arg_perigee", "true_anomaly",
    "specific_energy", "angular_momentum_mag",
    "bstar", "f107", "kp",
    "sin_doy", "cos_doy", "sin_tod", "cos_tod",
    "mass_kg", "cross_section_m2",
]

MU = 398600.4418   # km^3/s^2
RE = 6378.137      # km


def parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def tle_csv(norad: int) -> str:
    return f"{CFG['TLE_DIR']}/tle_{norad}.csv"


def eph_csv(norad: int) -> str:
    return f"{CFG['EPH_DIR']}/ephemeris_{norad}.csv"
