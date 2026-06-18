"""
================================================================================
PHASE 4 (CONSOLIDATED)  —  SENTINEL-X / ISDMAAS  Orbit Trajectory Tensors
================================================================================
Replaces BOTH:
    - export_phase5_tensors.py   (was: global time-sort -> cross-satellite mixing)
    - build_real_sgp4_baseline.py (was: unsorted + propagated wrong sat to wrong time)

WHAT WAS BROKEN
---------------
1. ALIGNMENT BUG (the 9123 km):
   export_phase5_tensors.py called df.sort_values("timestamp"). The raw dataset is
   GROUPED BY SATELLITE; time-sorting interleaves all 8 satellites (~90% of adjacent
   rows belong to DIFFERENT satellites). Every 30-step window therefore mixed ~8
   satellites and the "baseline"/"target" were unrelated orbits -> ~9000-13000 km.

2. CIRCULAR-BASELINE BUG (the deeper one):
   The dataset ground-truth (x_km..vz_kms) is itself SGP4 output of each satellite's
   TLE valid at that time (see data.py build_trajectory_dataset). If the baseline uses
   the TLE valid at the *target* time, SGP4 "predicts" SGP4 -> ~0.0 km error. There is
   then NOTHING for the residual model to learn and NO degradation to beat. Verified
   empirically (sat 25544):
        baseline = TLE-at-target  -> 0.00 km at every horizon  (circular, useless)
        baseline = TLE-at-INPUT   -> 0.9 km @6h, 8 km @1d, 34 km @3d, 195 km @7d
   The second is the real, patent-defensible SGP4 degradation curve.

THE FIX (this file)
-------------------
- Build sequences PER SATELLITE (never cross a norad_id boundary).
- X / y / baseline are produced in ONE pass from the same index -> aligned by build.
- BASELINE = SGP4 of the most recent TLE available AT INPUT TIME (epoch <= input_time),
  propagated forward to the TARGET time. This is what an operator actually has.
- Multi-day PREDICTION_HORIZON_DAYS so SGP4 genuinely drifts and the ML model has a
  real residual to correct.
- Gap guard: a sample is kept only if (target_time - input_time) is within tolerance of
  the requested horizon, so resampling gaps don't create fake long-horizon pairs.
- Chronological per-satellite split (no shuffle) -> no train/val leakage.

OUTPUTS (./ by default): X_train.pt y_train.pt X_val.pt y_val.pt
                         baseline_train.pt baseline_val.pt feature_scaler.pkl
                         phase4_meta.json
================================================================================
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import torch

from datetime import datetime, timedelta, timezone
from sklearn.preprocessing import StandardScaler
from sgp4.api import Satrec, jday


# =========================================================
# AUTO PATH RESOLUTION  (so it works from ANY directory)
# =========================================================
def find_data_root():
    """Locate the folder that contains idsmass_data/. Searches the script dir,
    the current working dir, their parents, and a sibling 'phase4' folder."""
    here = os.path.dirname(os.path.abspath(__file__))
    seeds = [here, os.getcwd(),
             os.path.join(here, "phase4"),
             os.path.join(here, "..", "phase4"),
             os.path.join(os.getcwd(), "phase4"),
             os.path.join(os.getcwd(), "..", "phase4")]
    # also walk up a few parents from script dir and cwd
    for start in (here, os.getcwd()):
        p = start
        for _ in range(4):
            seeds.append(p)
            p = os.path.dirname(p)
    for c in seeds:
        if c and os.path.isdir(os.path.join(c, "idsmass_data")):
            return os.path.abspath(c)
    raise FileNotFoundError(
        "Could not find 'idsmass_data/'. Put phase4_build_tensors.py in (or next to) "
        "the folder that contains idsmass_data/, or run it from there.")


DATA_ROOT = find_data_root()

# =========================================================
# CONFIG
# =========================================================
CONFIG = {
    # paths are resolved automatically relative to idsmass_data/
    "DATASET_PATH": os.path.join(DATA_ROOT, "idsmass_data", "processed",
                                 "historical_trajectory_dataset_v2.csv"),
    "TLE_PATH":     os.path.join(DATA_ROOT, "idsmass_data", "raw", "spacetrack",
                                 "historical_tles.csv"),
    "SAVE_DIR":     DATA_ROOT,      # tensors are written next to idsmass_data/

    "SEQUENCE_LENGTH": 30,          # input steps (dataset is 6h-resampled -> 30 = 7.5 d)
    "RESAMPLE_HOURS":  6,           # must match data.py RESAMPLE_HOURS
    "PREDICTION_HORIZON_DAYS": 21,   # forecast horizon; SGP4 degrades over multi-day
    "HORIZON_TOL_HOURS": 24,        # accept target within +/- this of the exact horizon

    "TRAIN_FRACTION": 0.8,          # chronological split, applied PER SATELLITE
    "MIN_VALID_BASELINE_FRAC": 0.5, # sanity threshold for warning
}

FEATURE_COLS = [
    "x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms",
    "semi_major_axis", "eccentricity", "inclination",
    "raan", "arg_perigee", "true_anomaly",
    "specific_energy", "angular_momentum_mag",
    "bstar", "f107", "kp",
    "sin_doy", "cos_doy", "sin_tod", "cos_tod",
    "mass_kg", "cross_section_m2",
]
TARGET_COLS = ["x_km", "y_km", "z_km", "vx_kms", "vy_kms", "vz_kms"]


# =========================================================
# HELPERS
# =========================================================
def tle_epoch_to_datetime(tle1: str) -> datetime:
    s = tle1[18:32].strip()
    yy = int(s[:2])
    year = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(s[2:]) - 1)


def sgp4_propagate(tle1: str, tle2: str, t) -> np.ndarray:
    """Propagate one TLE to datetime t. Returns [x,y,z,vx,vy,vz] or None on error."""
    try:
        sat = Satrec.twoline2rv(tle1, tle2)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                      t.second + t.microsecond / 1e6)
        e, r, v = sat.sgp4(jd, fr)
        if e != 0:
            return None
        return np.array([r[0], r[1], r[2], v[0], v[1], v[2]], dtype=np.float32)
    except Exception:
        return None


def build_tle_index(tle_df: pd.DataFrame):
    """norad_id -> dict(epochs=np.array[datetime64], tle1=list, tle2=list) sorted by epoch."""
    tle_df = tle_df.copy()
    tle_df["epoch"] = tle_df["tle1"].apply(tle_epoch_to_datetime)
    idx = {}
    for norad, g in tle_df.groupby("norad_id"):
        g = g.sort_values("epoch").reset_index(drop=True)
        idx[int(norad)] = {
            "epochs": g["epoch"].values.astype("datetime64[ns]"),
            "tle1": g["tle1"].tolist(),
            "tle2": g["tle2"].tolist(),
        }
    return idx


def latest_tle_at_or_before(tle_entry, t_np64):
    """Index of the most recent TLE with epoch <= t. Falls back to earliest if none."""
    epochs = tle_entry["epochs"]
    pos = np.searchsorted(epochs, t_np64, side="right") - 1
    if pos < 0:
        pos = 0          # input precedes first TLE -> use earliest available
    return pos


# =========================================================
# MAIN
# =========================================================
def main():
    cfg = CONFIG
    os.makedirs(cfg["SAVE_DIR"], exist_ok=True)

    seq_len = cfg["SEQUENCE_LENGTH"]
    horizon_steps = int(round(cfg["PREDICTION_HORIZON_DAYS"] * 24 / cfg["RESAMPLE_HOURS"]))
    horizon_td = pd.Timedelta(days=cfg["PREDICTION_HORIZON_DAYS"])
    tol_td = pd.Timedelta(hours=cfg["HORIZON_TOL_HOURS"])

    print("=" * 64)
    print("PHASE 4 — CONSOLIDATED TENSOR BUILD")
    print("=" * 64)
    print(f"data root (auto)     : {DATA_ROOT}")
    print(f"tensors will save to : {cfg['SAVE_DIR']}")
    print(f"sequence_length      : {seq_len} steps "
          f"({seq_len * cfg['RESAMPLE_HOURS'] / 24:.1f} d of input)")
    print(f"prediction_horizon   : {cfg['PREDICTION_HORIZON_DAYS']} d "
          f"= {horizon_steps} steps")
    print(f"horizon tolerance    : +/- {cfg['HORIZON_TOL_HOURS']} h")

    # ---- load ----
    df = pd.read_csv(cfg["DATASET_PATH"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=True)
    df = df.dropna(subset=FEATURE_COLS + TARGET_COLS + ["norad_id", "timestamp"])
    print(f"\ndataset rows (clean) : {len(df):,}")
    print(f"satellites           : {df['norad_id'].nunique()}")

    tle_df = pd.read_csv(cfg["TLE_PATH"])
    tle_idx = build_tle_index(tle_df)
    print(f"TLE satellites cached: {len(tle_idx)}")

    # ---- fit scaler GLOBALLY on features (targets stay in real km / km-s) ----
    # Fit on the full cleaned set so train & val share one transform.
    scaler = StandardScaler().fit(df[FEATURE_COLS].values)
    joblib.dump(scaler, os.path.join(cfg["SAVE_DIR"], "feature_scaler.pkl"))
    print("scaler fitted & saved: feature_scaler.pkl")

    # ---- build per-satellite, chronological ----
    X_tr, y_tr, b_tr, age_tr = [], [], [], []
    X_va, y_va, b_va, age_va = [], [], [], []

    baseline_err_log = []     # (split, pos_err_km) for verification
    n_bad_baseline = 0
    n_gap_skipped = 0

    for norad, sub in df.groupby("norad_id"):
        norad = int(norad)
        sub = sub.sort_values("timestamp").reset_index(drop=True)

        feats_raw = sub[FEATURE_COLS].values
        feats = scaler.transform(feats_raw).astype(np.float32)   # normalized inputs
        targs = sub[TARGET_COLS].values.astype(np.float32)        # real-unit targets
        times = sub["timestamp"].values.astype("datetime64[ns]")

        n = len(sub)
        if norad not in tle_idx:
            continue
        # need room for at least the input window; target is found by TIME below
        last_start = n - seq_len
        if last_start <= 0:
            continue

        # per-satellite chronological split point (computed on valid range)
        split_start = int(last_start * cfg["TRAIN_FRACTION"])
        tol_ns = np.timedelta64(tol_td)
        hor_ns = np.timedelta64(horizon_td)

        for i in range(last_start):
            in_idx = i + seq_len - 1                 # last observed step
            in_t = times[in_idx]

            # TIME-BASED target: row whose timestamp is closest to in_t + horizon.
            # Robust to the non-uniform 6h grid (it resets at each TLE epoch).
            desired_t = in_t + hor_ns
            tg_idx = int(np.searchsorted(times, desired_t))
            # pick the nearer of the two bracketing rows
            cands = [c for c in (tg_idx - 1, tg_idx) if 0 <= c < n]
            if not cands:
                n_gap_skipped += 1
                continue
            tg_idx = min(cands, key=lambda c: abs(times[c] - desired_t))
            tg_t = times[tg_idx]

            if tg_idx <= in_idx:                     # target must be in the future
                n_gap_skipped += 1
                continue
            # gap guard: matched time must be within tolerance of the exact horizon
            if abs((tg_t - in_t) - hor_ns) > tol_ns:
                n_gap_skipped += 1
                continue

            # SGP4 baseline: latest TLE known AT INPUT TIME, propagated to target time
            te = tle_idx[norad]
            k = latest_tle_at_or_before(te, in_t)
            tg_dt = pd.Timestamp(tg_t).to_pydatetime()
            base = sgp4_propagate(te["tle1"][k], te["tle2"][k], tg_dt)
            if base is None:
                n_bad_baseline += 1
                continue

            x_seq = feats[i: i + seq_len]
            y_t = targs[tg_idx]
            pos_err = float(np.linalg.norm(base[:3] - y_t[:3]))

            # TLE staleness = how far SGP4 propagated from its TLE epoch to the target.
            # This is the key stratifier: SGP4 degrades with propagation age.
            age_days = float((tg_t - te["epochs"][k]) / np.timedelta64(1, "D"))

            if i < split_start:
                X_tr.append(x_seq); y_tr.append(y_t); b_tr.append(base); age_tr.append(age_days)
                baseline_err_log.append(("train", pos_err))
            else:
                X_va.append(x_seq); y_va.append(y_t); b_va.append(base); age_va.append(age_days)
                baseline_err_log.append(("val", pos_err))

        print(f"  sat {norad}: usable={last_start:,}  "
              f"train={sum(1 for s,_ in baseline_err_log if s=='train')-0:,} (cum)")

    def stack(a):
        return torch.tensor(np.array(a, dtype=np.float32))

    X_train, y_train, baseline_train = stack(X_tr), stack(y_tr), stack(b_tr)
    X_val,   y_val,   baseline_val   = stack(X_va), stack(y_va), stack(b_va)
    age_train, age_val = stack(age_tr), stack(age_va)

    # ---- save ----
    sd = cfg["SAVE_DIR"]
    torch.save(X_train, os.path.join(sd, "X_train.pt"))
    torch.save(y_train, os.path.join(sd, "y_train.pt"))
    torch.save(X_val,   os.path.join(sd, "X_val.pt"))
    torch.save(y_val,   os.path.join(sd, "y_val.pt"))
    torch.save(baseline_train, os.path.join(sd, "baseline_train.pt"))
    torch.save(age_train, os.path.join(sd, "age_train.pt"))
    torch.save(age_val,   os.path.join(sd, "age_val.pt"))
    torch.save(baseline_val,   os.path.join(sd, "baseline_val.pt"))

    # ---- verification ----
    tr_err = np.array([e for s, e in baseline_err_log if s == "train"])
    va_err = np.array([e for s, e in baseline_err_log if s == "val"])

    def rmse(a, b):
        return float(np.sqrt(np.mean(np.sum((a[:, :3] - b[:, :3]) ** 2, axis=1))))

    meta = {
        "config": cfg,
        "horizon_steps": horizon_steps,
        "shapes": {
            "X_train": list(X_train.shape), "y_train": list(y_train.shape),
            "X_val": list(X_val.shape), "y_val": list(y_val.shape),
            "baseline_train": list(baseline_train.shape),
            "baseline_val": list(baseline_val.shape),
        },
        "baseline_pos_error_km": {
            "train_mean": float(tr_err.mean()) if len(tr_err) else None,
            "train_median": float(np.median(tr_err)) if len(tr_err) else None,
            "val_mean": float(va_err.mean()) if len(va_err) else None,
            "val_median": float(np.median(va_err)) if len(va_err) else None,
            "val_rmse": rmse(baseline_val.numpy(), y_val.numpy()) if len(y_val) else None,
        },
        "n_gap_skipped": n_gap_skipped,
        "n_bad_baseline": n_bad_baseline,
    }
    with open(os.path.join(sd, "phase4_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print("\n" + "=" * 64)
    print("PHASE 4 TENSORS EXPORTED")
    print("=" * 64)
    for k, v in meta["shapes"].items():
        print(f"  {k:16s}: {v}")
    print(f"\n  baseline pos error (val)  mean   = "
          f"{meta['baseline_pos_error_km']['val_mean']:.2f} km")
    print(f"  baseline pos error (val)  median = "
          f"{meta['baseline_pos_error_km']['val_median']:.2f} km")
    print(f"  baseline pos error (val)  RMSE   = "
          f"{meta['baseline_pos_error_km']['val_rmse']:.2f} km")
    print(f"  gap-skipped samples       = {n_gap_skipped:,}")
    print(f"  bad-baseline samples      = {n_bad_baseline:,}")

    # alignment self-check
    if len(y_val):
        i0_b = baseline_val[0, :3].numpy()
        i0_y = y_val[0, :3].numpy()
        print(f"\n  first val sample baseline = {i0_b}")
        print(f"  first val sample target   = {i0_y}")
        print(f"  first val sample err      = "
              f"{np.linalg.norm(i0_b - i0_y):.2f} km")

    print("\nDONE.")


if __name__ == "__main__":
    main()