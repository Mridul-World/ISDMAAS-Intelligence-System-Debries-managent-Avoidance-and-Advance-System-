"""
PHASE 6 — operational inference with uncertainty + RTN/ECI covariance export.

Produces, for a future epoch from public TLEs only:
  - AI-corrected position (per-channel SNR gated)
  - per-axis 1-sigma in RTN (temperature-calibrated to match validation)
  - the 3x3 covariance rotated into ECI (input for Phase 7 collision probability)

    python infer_uncertain.py <NORAD> [hours] [--mc 20] [--alpha 1.0]

Reuses the feature-building logic of infer_future.py (TLE-derived history).
Loads model/sigma_temperature_phase6.pt so inference sigma matches the
calibrated sigma used in Phase-6 validation.
"""
import os, sys
import numpy as np
import torch, joblib
from datetime import datetime, timedelta, timezone
from sgp4.api import Satrec
from config import CFG, SATELLITES, FEATURE_COLS, tle_csv
from phase55_utils import rv_to_elements_vec, load_tles, unix_seconds, sgp4_at
from train_phase55_rtn import rtn_to_eci, rtn_basis, CFG as TCFG
from train_phase6_uncertainty import UncertaintyRTN


def latest_sw():
    if os.path.exists(CFG["SW_CSV"]):
        import pandas as pd
        sw = pd.read_csv(CFG["SW_CSV"]).dropna()
        return float(sw["f107"].iloc[-1]), float(sw["kp"].iloc[-1])
    return 120.0, 2.0


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    norad = int(sys.argv[1])
    hours = float(sys.argv[2]) if len(sys.argv) > 2 and not sys.argv[2].startswith("--") \
        else CFG["HORIZON_S"] / 3600
    mc = int(sys.argv[sys.argv.index("--mc") + 1]) if "--mc" in sys.argv else 20
    alpha = float(sys.argv[sys.argv.index("--alpha") + 1]) if "--alpha" in sys.argv else 1.0
    info = SATELLITES[norad]

    tles = load_tles(tle_csv(norad))
    tle_t = unix_seconds(tles["epoch"])
    sats = [Satrec.twoline2rv(r.tle1, r.tle2) for r in tles.itertuples()]
    bstars = np.array([s.bstar for s in sats])

    now = datetime.now(timezone.utc)
    target = now + timedelta(hours=hours)
    seq, step = CFG["SEQ"], CFG["FEATURE_STEP_S"]
    hist_t = np.array([now.timestamp() - (seq - 1 - i) * step for i in range(seq)])
    kf = np.clip(np.searchsorted(tle_t, hist_t, "right") - 1, 0, len(sats) - 1)
    Pf, Vf, okf = sgp4_at(sats, kf, hist_t)

    import pandas as pd
    elems = rv_to_elements_vec(Pf, Vf)
    ts = pd.to_datetime(hist_t, unit="s", utc=True)
    doy = ts.dayofyear.to_numpy(); tod = (ts.hour*3600+ts.minute*60+ts.second).to_numpy()
    f107, kp = latest_sw()
    feat = pd.DataFrame({
        "x_km":Pf[:,0],"y_km":Pf[:,1],"z_km":Pf[:,2],
        "vx_kms":Vf[:,0],"vy_kms":Vf[:,1],"vz_kms":Vf[:,2], **elems,
        "bstar":bstars[kf],"f107":f107,"kp":kp,
        "sin_doy":np.sin(2*np.pi*doy/365.25),"cos_doy":np.cos(2*np.pi*doy/365.25),
        "sin_tod":np.sin(2*np.pi*tod/86400.),"cos_tod":np.cos(2*np.pi*tod/86400.),
        "mass_kg":info["mass_kg"],"cross_section_m2":info["area_m2"],
    })[FEATURE_COLS].to_numpy(np.float32)

    sc = joblib.load("feature_scaler.pkl")
    X = torch.tensor(sc.transform(feat).astype(np.float32)).unsqueeze(0)

    kb = np.array([np.searchsorted(tle_t, now.timestamp(), "right") - 1])
    Pb, Vb, _ = sgp4_at(sats, kb, np.array([target.timestamp()]))
    b = torch.tensor(np.concatenate([Pb[0], Vb[0]]).astype(np.float32)).unsqueeze(0)

    norm = torch.load("model/residual_norm_phase6.pt", weights_only=False)
    rm, rs = norm["rm"], norm["rs"]
    net = UncertaintyRTN(X.shape[2], TCFG["D_MODEL"], TCFG["NHEAD"],
                         TCFG["NUM_LAYERS"], TCFG["DIM_FF"], TCFG["DROPOUT"])
    net.load_state_dict(torch.load("model/rtn_uncertainty_phase6.pt", map_location="cpu"))

    net.eval()
    with torch.no_grad():
        mu_n, lsg = net(X); sig_n = torch.exp(lsg)
    if mc:
        net.train()
        with torch.no_grad():
            S = torch.stack([net(X)[0] for _ in range(mc)])
            sig_n = torch.sqrt(sig_n**2 + S.var(0))
        net.eval()

    mu_km = (mu_n * rs + rm)[0]                 # (3,) RTN correction
    sig_km = (sig_n * rs)[0]                    # (3,) RTN 1-sigma (raw)

    # apply Phase-6 temperature calibration so sigma matches validated coverage
    tpath = "model/sigma_temperature_phase6.pt"
    if os.path.exists(tpath):
        temp = torch.load(tpath, weights_only=False)["temp"][0]   # (3,)
        sig_km = sig_km * temp
        print(f"applied sigma temperature scaling: "
              f"R={float(temp[0]):.3f} T={float(temp[1]):.3f} N={float(temp[2]):.3f}")
    else:
        print("WARNING: sigma_temperature_phase6.pt not found — sigma is UNCALIBRATED")

    snr = (mu_km.abs() / sig_km.clamp_min(1e-9))
    gate = (snr > alpha).float()
    mu_gated = mu_km * gate

    corrected = b.clone()
    corrected[:, :3] = b[:, :3] - rtn_to_eci(mu_gated.unsqueeze(0), b)

    # RTN diagonal covariance -> ECI covariance:  C_eci = Rt @ diag(sig^2) @ Rt^T
    R, S_, W = rtn_basis(b)
    Rt = torch.stack([R[0], S_[0], W[0]], dim=1)            # columns = RTN axes in ECI
    C_rtn = torch.diag(sig_km**2)
    C_eci = (Rt @ C_rtn @ Rt.T).numpy()

    print("="*64)
    print(f"satellite      : {info['name']} (NORAD {norad})")
    print(f"target epoch   : {target.isoformat()}  (now + {hours:.1f} h)")
    print(f"RTN correction : R={mu_km[0]:+.4f} T={mu_km[1]:+.4f} N={mu_km[2]:+.4f} km")
    print(f"RTN 1-sigma    : R={sig_km[0]:.4f} T={sig_km[1]:.4f} N={sig_km[2]:.4f} km (calibrated)")
    print(f"channel SNR    : R={snr[0]:.2f} T={snr[1]:.2f} N={snr[2]:.2f}  (gate alpha={alpha}) "
          f"applied={gate.int().tolist()}")
    print(f"SGP4 position  : {Pb[0].round(3).tolist()} km (TEME)")
    print(f"AI-corrected   : {corrected[0,:3].numpy().round(3).tolist()} km (TEME)")
    print("ECI covariance (km^2), for Phase 7 Pc:")
    for row in C_eci:
        print("   " + "  ".join(f"{v:+.5f}" for v in row))
    print("="*64)

    # ---- emit machine-readable artifact for Phase 7 ----
    import json
    out = {
        "norad_id": norad, "name": info["name"],
        "epoch_utc": target.isoformat(),
        "position_teme_km": corrected[0, :3].numpy().tolist(),
        "velocity_teme_kms": Vb[0].tolist(),
        "rtn_sigma_km": sig_km.numpy().tolist(),
        "covariance_eci_km2": C_eci.tolist(),
        "gate_alpha": alpha, "channels_applied": gate.int().tolist(),
    }
    os.makedirs("reports", exist_ok=True)
    fp = f"reports/state_{norad}.json"
    json.dump(out, open(fp, "w"), indent=2)
    print(f"saved Phase-7 input: {fp}")


if __name__ == "__main__":
    main()