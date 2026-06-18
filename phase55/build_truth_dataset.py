"""
Phase 5.5 — build the REAL residual dataset for the whole fleet.

DEPLOYABILITY DESIGN (this is what makes the final model usable in production):
  - INPUT FEATURES at every grid time t come from SGP4 states derived from the
    freshest TLE with epoch <= t (public data, always available at inference).
  - TRUTH y(t) comes from POE precise ephemeris — used ONLY as training target.
  - BASELINE(t) = SGP4 from the freshest TLE with epoch <= t - HORIZON
    (only what is available when the prediction is issued — no leakage).
  - residual_rtn = RTN_baseline(baseline_pos - truth_pos)   [Phase-5 contract]

Output: data/phase55_dataset.csv (all satellites, norad_id column).
"""
import os
import numpy as np
import pandas as pd
from sgp4.api import Satrec
from config import CFG, SATELLITES, FEATURE_COLS, tle_csv, eph_csv
from phase55_utils import (rv_to_elements_vec, eci_to_rtn_np, load_tles,
                           hermite_interp, sgp4_at, unix_seconds)


def load_space_weather():
    if os.path.exists(CFG["SW_CSV"]):
        sw = pd.read_csv(CFG["SW_CSV"])
        sw["date"] = pd.to_datetime(sw["date"], utc=True, format="mixed")
        sw = sw.set_index(sw["date"].dt.normalize()).sort_index()
        print(f"space weather: {CFG['SW_CSV']} ({len(sw)} rows)")
        return sw
    print("WARN: no space-weather file — using constants f107=120, kp=2.")
    return None


def sw_lookup(sw, days):
    if sw is None:
        return np.full(len(days), 120.0), np.full(len(days), 2.0)
    f107 = sw["f107"].reindex(days, method="ffill").to_numpy()
    kp = sw["kp"].reindex(days, method="ffill").to_numpy()
    f107 = np.where(np.isfinite(f107) & (f107 > 0), f107, 120.0)
    kp = np.where(np.isfinite(kp) & (kp >= 0), kp, 2.0)
    return f107, kp


def build_satellite(norad, info, sw):
    if not (os.path.exists(eph_csv(norad)) and os.path.exists(tle_csv(norad))):
        print(f"[{info['name']}] missing ephemeris or TLEs — skipped"); return None

    eph = pd.read_csv(eph_csv(norad))
    eph["timestamp"] = pd.to_datetime(eph["timestamp"], utc=True, format="mixed")
    eph = eph.sort_values("timestamp").reset_index(drop=True)
    t_eph = unix_seconds(eph["timestamp"])
    Pt_all = eph[["true_x_km", "true_y_km", "true_z_km"]].to_numpy()
    Vt_all = eph[["true_vx_kms", "true_vy_kms", "true_vz_kms"]].to_numpy()

    tles = load_tles(tle_csv(norad))
    tle_t = unix_seconds(tles["epoch"])
    sats = [Satrec.twoline2rv(r.tle1, r.tle2) for r in tles.itertuples()]
    bstars = np.array([s.bstar for s in sats])

    step, H = CFG["FEATURE_STEP_S"], CFG["HORIZON_S"]
    assert H % step == 0
    grid = np.arange(np.ceil(t_eph[0] / step) * step,
                     np.floor(t_eph[-1] / step) * step + 1, step)
    grid = grid[grid - H >= tle_t[0]]          # baseline TLE must exist
    if len(grid) < CFG["SEQ"] + H // step + 1:
        print(f"[{info['name']}] window too short — skipped"); return None

    # truth at grid (Hermite from 10 s OSVs; sub-mm interpolation error)
    Pt, Vt = hermite_interp(grid, t_eph, Pt_all, Vt_all)

    # FEATURE states: SGP4 from freshest TLE <= t   (deployable input)
    kf = np.searchsorted(tle_t, grid, side="right") - 1
    Pf, Vf, okf = sgp4_at(sats, kf, grid)

    # BASELINE states: SGP4 from freshest TLE <= t - H   (no leakage)
    kb = np.searchsorted(tle_t, grid - H, side="right") - 1
    Pb, Vb, okb = sgp4_at(sats, kb, grid)

    ok = okf & okb
    grid, Pt, Vt, Pf, Vf, Pb, Vb, kf, kb = (a[ok] for a in
        (grid, Pt, Vt, Pf, Vf, Pb, Vb, kf, kb))
    age = (grid - tle_t[kb]) / 86400.0

    err = np.linalg.norm(Pb - Pt, axis=1)
    print(f"[{info['name']}] rows={len(grid)}  SGP4-vs-REAL-truth: "
          f"median={np.median(err):.3f} p90={np.percentile(err,90):.3f} "
          f"rmse={np.sqrt((err**2).mean()):.3f} km")

    res_pos, res_vel = Pt - Pb, Vt - Vb
    res_rtn = eci_to_rtn_np(Pb - Pt, Pb, Vb)

    elems = rv_to_elements_vec(Pf, Vf)         # features from TLE-derived state
    ts = pd.to_datetime(grid, unit="s", utc=True)
    doy = ts.dayofyear.to_numpy()
    tod = (ts.hour * 3600 + ts.minute * 60 + ts.second).to_numpy()
    f107, kp = sw_lookup(sw, ts.normalize())

    return pd.DataFrame({"timestamp": ts, "norad_id": norad,
        "x_km": Pf[:, 0], "y_km": Pf[:, 1], "z_km": Pf[:, 2],
        "vx_kms": Vf[:, 0], "vy_kms": Vf[:, 1], "vz_kms": Vf[:, 2],
        **elems,
        "bstar": bstars[kf], "f107": f107, "kp": kp,
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "sin_tod": np.sin(2 * np.pi * tod / 86400.0),
        "cos_tod": np.cos(2 * np.pi * tod / 86400.0),
        "mass_kg": info["mass_kg"], "cross_section_m2": info["area_m2"],
        "tle_age_days": age,
        "sgp4_x": Pb[:, 0], "sgp4_y": Pb[:, 1], "sgp4_z": Pb[:, 2],
        "sgp4_vx": Vb[:, 0], "sgp4_vy": Vb[:, 1], "sgp4_vz": Vb[:, 2],
        "true_x": Pt[:, 0], "true_y": Pt[:, 1], "true_z": Pt[:, 2],
        "true_vx": Vt[:, 0], "true_vy": Vt[:, 1], "true_vz": Vt[:, 2],
        "residual_x": res_pos[:, 0], "residual_y": res_pos[:, 1],
        "residual_z": res_pos[:, 2],
        "residual_vx": res_vel[:, 0], "residual_vy": res_vel[:, 1],
        "residual_vz": res_vel[:, 2],
        "residual_r": res_rtn[:, 0], "residual_t": res_rtn[:, 1],
        "residual_n": res_rtn[:, 2]})


def main():
    sw = load_space_weather()
    frames = []
    for norad, info in SATELLITES.items():
        df = build_satellite(norad, info, sw)
        if df is not None:
            frames.append(df)
    if not frames:
        raise SystemExit("No satellite produced data.")
    out = pd.concat(frames, ignore_index=True)
    assert all(c in out.columns for c in FEATURE_COLS)
    out.to_csv(CFG["DATASET_CSV"], index=False)
    print(f"\nSaved {len(out)} rows ({out.norad_id.nunique()} satellites) "
          f"-> {CFG['DATASET_CSV']}")


if __name__ == "__main__":
    main()
