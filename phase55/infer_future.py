"""
Phase 5.5 — OPERATIONAL INFERENCE: Trusted AI prediction of a FUTURE orbit
state from public TLEs only. This is the end-product of the phase.

    python infer_future.py <NORAD_ID> [hours_ahead]

    python infer_future.py 39634          # Sentinel-1A, now + 24 h
    python infer_future.py 39634 24

How it works (mirrors training exactly):
  1. Build the 30-step feature history from SGP4 states of the freshest TLE
     available at each history time (TLE/SGP4-derived features — exactly the
     training distribution; no precise ephemeris needed).
  2. SGP4 baseline at target T from the freshest TLE (epoch <= T - horizon
     in training; at runtime the latest TLE plays this role since T is in
     the future).
  3. Model predicts the RTN residual; corrected state =
         baseline_pos - RTN_to_ECI(residual, baseline frame)
  4. Output: corrected TEME state, plus geodetic lat/lon/alt for readability.

NOTE: the model was trained at HORIZON_S; accuracy is characterized at that
horizon. Other horizons run but are extrapolation — flag accordingly.
"""
import os, sys
import numpy as np
import pandas as pd
import torch, joblib
from datetime import datetime, timedelta, timezone
from sgp4.api import Satrec
from config import CFG, SATELLITES, FEATURE_COLS, tle_csv
from phase55_utils import (rv_to_elements_vec, load_tles, unix_seconds, sgp4_at)
from train_phase55_rtn import RTNTransformer, rtn_to_eci, CFG as TCFG


def latest_sw():
    if os.path.exists(CFG["SW_CSV"]):
        sw = pd.read_csv(CFG["SW_CSV"]).dropna()
        return float(sw["f107"].iloc[-1]), float(sw["kp"].iloc[-1])
    return 120.0, 2.0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    norad = int(sys.argv[1])
    hours = float(sys.argv[2]) if len(sys.argv) > 2 else CFG["HORIZON_S"] / 3600
    info = SATELLITES.get(norad)
    if info is None:
        sys.exit(f"NORAD {norad} not in config SATELLITES — add it (name/mass/area).")
    if abs(hours * 3600 - CFG["HORIZON_S"]) > 1:
        print(f"WARNING: model trained at {CFG['HORIZON_S']/3600:.0f} h horizon; "
              f"{hours:.0f} h is extrapolation.")

    tles = load_tles(tle_csv(norad))
    tle_t = unix_seconds(tles["epoch"])
    sats = [Satrec.twoline2rv(r.tle1, r.tle2) for r in tles.itertuples()]
    bstars = np.array([s.bstar for s in sats])

    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=hours)
    t_tgt = target.timestamp()
    step, seq = CFG["FEATURE_STEP_S"], CFG["SEQ"]

    # 30-step history ending now (TLE-derived = training distribution)
    hist_t = np.array([now.timestamp() - (seq - 1 - i) * step for i in range(seq)])
    kf = np.clip(np.searchsorted(tle_t, hist_t, side="right") - 1, 0, len(sats) - 1)
    Pf, Vf, okf = sgp4_at(sats, kf, hist_t)
    if not okf.all():
        sys.exit("SGP4 failed on history — check TLE file freshness.")

    elems = rv_to_elements_vec(Pf, Vf)
    ts = pd.to_datetime(hist_t, unit="s", utc=True)
    doy = ts.dayofyear.to_numpy()
    tod = (ts.hour * 3600 + ts.minute * 60 + ts.second).to_numpy()
    f107, kp = latest_sw()
    feat = pd.DataFrame({
        "x_km": Pf[:, 0], "y_km": Pf[:, 1], "z_km": Pf[:, 2],
        "vx_kms": Vf[:, 0], "vy_kms": Vf[:, 1], "vz_kms": Vf[:, 2],
        **elems, "bstar": bstars[kf], "f107": f107, "kp": kp,
        "sin_doy": np.sin(2 * np.pi * doy / 365.25),
        "cos_doy": np.cos(2 * np.pi * doy / 365.25),
        "sin_tod": np.sin(2 * np.pi * tod / 86400.0),
        "cos_tod": np.cos(2 * np.pi * tod / 86400.0),
        "mass_kg": info["mass_kg"], "cross_section_m2": info["area_m2"],
    })[FEATURE_COLS].to_numpy(np.float32)

    sc = joblib.load("feature_scaler.pkl")
    X = torch.tensor(sc.transform(feat).astype(np.float32)).unsqueeze(0)

    # baseline at target from freshest TLE
    kb = np.array([np.searchsorted(tle_t, now.timestamp(), side="right") - 1])
    Pb, Vb, okb = sgp4_at(sats, kb, np.array([t_tgt]))
    if not okb.all():
        sys.exit("SGP4 failed at target time.")
    b = torch.tensor(np.concatenate([Pb[0], Vb[0]]).astype(np.float32)).unsqueeze(0)
    tle_age = (t_tgt - tle_t[kb[0]]) / 86400.0

    norm = torch.load(f"{TCFG['MODEL_DIR']}/residual_norm_phase55.pt", weights_only=False)
    net = RTNTransformer(X.shape[2], TCFG["D_MODEL"], TCFG["NHEAD"],
                         TCFG["NUM_LAYERS"], TCFG["DIM_FF"], TCFG["DROPOUT"])
    net.load_state_dict(torch.load(f"{TCFG['MODEL_DIR']}/rtn_physics_ml_phase55.pt",
                                   map_location="cpu"))
    net.eval()
    with torch.no_grad():
        pred_rtn = net(X) * norm["rs"] + norm["rm"]
        corrected = b.clone()
        corrected[:, :3] = b[:, :3] - rtn_to_eci(pred_rtn, b)

    print("=" * 64)
    print(f"satellite      : {info['name']} (NORAD {norad})")
    print(f"target epoch   : {target.isoformat()}  (now + {hours:.1f} h)")
    print(f"TLE age at tgt : {tle_age:.2f} days")
    print(f"pred RTN resid : R={pred_rtn[0,0]:.4f}  T={pred_rtn[0,1]:.4f}  "
          f"N={pred_rtn[0,2]:.4f} km")
    print(f"SGP4 state     : pos {Pb[0].round(3).tolist()} km")
    print(f"                 vel {Vb[0].round(6).tolist()} km/s")
    print(f"AI-corrected   : pos {corrected[0,:3].numpy().round(3).tolist()} km (TEME)")
    print(f"correction mag : {float(np.linalg.norm(pred_rtn[0].numpy())):.4f} km")
    print("=" * 64)


if __name__ == "__main__":
    main()
