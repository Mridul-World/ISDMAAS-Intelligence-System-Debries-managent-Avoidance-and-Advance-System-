"""
Phase 5.5 — export tensors with the EXACT Phase-5 contract (multi-satellite).

  X_train/X_val.pt              (N,30,23) scaled features (TLE/SGP4-derived)
  y_train/y_val.pt              (N,6)     REAL POE truth at t+H (TEME)
  baseline_train/baseline_val.pt(N,6)     SGP4 at t+H (TLE <= t)
  residual_rtn_train/_val.pt    (N,3)     RTN(baseline_pos - truth_pos)
  age_train/age_val.pt          (N,)      TLE age (days)
  sat_train/sat_val.pt          (N,)      NORAD id (for per-satellite reporting)
  feature_scaler.pkl, phase55_meta.json

OUTLIER FILTER: windows whose SGP4-vs-truth error exceeds OUTLIER_KM are
excluded. These correspond to station-keeping maneuvers and anomalous TLE
element sets — corrupted labels, not learnable SGP4 physics error. This is
standard practice in TLE-correction literature and operations. (<~1% of data;
p99 of the clean error distribution is ~18 km, so 25 km keeps the honest tail.)

Split: chronological 80/20 PER SATELLITE (no temporal leakage, every
satellite represented in both splits).

LOSO MODE: if ISDMAAS_HOLDOUT_NORAD is set, that satellite contributes ZERO
windows to training and becomes the ENTIRE validation set (leave-one-satellite-
out); the other satellites contribute all their windows to training. Without the
env var, behavior is identical to the standard chronological split.
"""
import os, json, joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from config import CFG, FEATURE_COLS
from phase55_utils import unix_seconds

OUTLIER_KM = 25.0   # maneuver / bad-TLE exclusion threshold (position error at t+H)


def main():
    c = CFG
    np.random.seed(c["SEED"])
    df = pd.read_csv(c["DATASET_CSV"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="mixed")

    seq, step = c["SEQ"], c["FEATURE_STEP_S"]
    off = c["HORIZON_S"] // step
    Xs, ys, bs, rs, ags, sat, ends = [], [], [], [], [], [], []

    for norad, sub in df.groupby("norad_id"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        F = sub[FEATURE_COLS].to_numpy(np.float32)
        Yt = sub[["true_x","true_y","true_z","true_vx","true_vy","true_vz"]].to_numpy(np.float32)
        Bl = sub[["sgp4_x","sgp4_y","sgp4_z","sgp4_vx","sgp4_vy","sgp4_vz"]].to_numpy(np.float32)
        Rr = sub[["residual_r","residual_t","residual_n"]].to_numpy(np.float32)
        Age = sub["tle_age_days"].to_numpy(np.float32)
        tsec = unix_seconds(sub["timestamp"])

        cand = []
        for j in range(seq - 1, len(sub) - off):
            tgt = j + off
            if not np.isclose(tsec[tgt] - tsec[j], c["HORIZON_S"], atol=1.0):
                continue                              # data gap
            if not np.isclose(tsec[j] - tsec[j - seq + 1], (seq - 1) * step, atol=1.0):
                continue
            cand.append((j, tgt))
        if len(cand) > c["MAX_WINDOWS_PER_SAT"]:      # uniform subsample
            sel = np.linspace(0, len(cand) - 1, c["MAX_WINDOWS_PER_SAT"]).astype(int)
            cand = [cand[i] for i in sel]
        for j, tgt in cand:
            Xs.append(F[j - seq + 1: j + 1]); ys.append(Yt[tgt]); bs.append(Bl[tgt])
            rs.append(Rr[tgt]); ags.append(Age[tgt]); sat.append(norad)
            ends.append(tsec[tgt])
        print(f"norad {norad}: {len(cand)} windows")

    Xs = np.array(Xs, np.float32); ys = np.array(ys, np.float32)
    bs = np.array(bs, np.float32); rs = np.array(rs, np.float32)
    ags = np.array(ags, np.float32); sat = np.array(sat, np.int64)
    ends = np.array(ends)
    print(f"total windows: {len(Xs)}  ({len(np.unique(sat))} satellites)")

    # ---- outlier filter: maneuvers / anomalous TLEs are not learnable physics ----
    err_all = np.linalg.norm(bs[:, :3] - ys[:, :3], axis=1)
    keep = err_all < OUTLIER_KM
    n_drop = int((~keep).sum())
    print(f"outlier filter: dropped {n_drop}/{len(keep)} windows "
          f"({100*n_drop/len(keep):.2f}%) with SGP4 error > {OUTLIER_KM} km "
          f"(maneuver / bad-TLE windows)")
    per_sat_drop = {int(s): int(((~keep) & (sat == s)).sum()) for s in np.unique(sat)}
    print(f"  dropped per satellite: {per_sat_drop}")
    Xs, ys, bs, rs = Xs[keep], ys[keep], bs[keep], rs[keep]
    ags, sat, ends = ags[keep], sat[keep], ends[keep]

    # chronological split PER satellite — with optional LOSO holdout.
    holdout = os.environ.get("ISDMAAS_HOLDOUT_NORAD")
    holdout = int(holdout) if holdout else None
    tr = np.zeros(len(Xs), bool)
    for s in np.unique(sat):
        m = np.where(sat == s)[0]
        if holdout is not None and int(s) == holdout:
            continue                                   # held-out sat -> all validation
        m = m[np.argsort(ends[m])]
        if holdout is not None:
            tr[m] = True                               # use ALL of the 4 training sats
        else:
            tr[m[:int(c["TRAIN_FRACTION"] * len(m))]] = True
    va = ~tr
    if holdout is not None:
        va = (sat == holdout)                          # validation = held-out sat only
        tr = ~va & tr
        print(f"LOSO holdout NORAD {holdout}: train {tr.sum()} (4 sats)  "
              f"val {va.sum()} (held-out only)")
    else:
        print(f"train {tr.sum()}  val {va.sum()}")

    sc = StandardScaler().fit(Xs[tr].reshape(-1, c["N_FEATURES"]))
    joblib.dump(sc, os.path.join(c["TENSOR_DIR"], "feature_scaler.pkl"))
    nrm = lambda a: sc.transform(a.reshape(-1, c["N_FEATURES"])).reshape(a.shape).astype(np.float32)
    sv = lambda a, f: torch.save(torch.tensor(a), os.path.join(c["TENSOR_DIR"], f))

    sv(nrm(Xs[tr]), "X_train.pt"); sv(nrm(Xs[va]), "X_val.pt")
    sv(ys[tr], "y_train.pt");      sv(ys[va], "y_val.pt")
    sv(bs[tr], "baseline_train.pt"); sv(bs[va], "baseline_val.pt")
    sv(rs[tr], "residual_rtn_train.pt"); sv(rs[va], "residual_rtn_val.pt")
    sv(ags[tr], "age_train.pt");   sv(ags[va], "age_val.pt")
    sv(sat[tr].astype(np.float32), "sat_train.pt")
    sv(sat[va].astype(np.float32), "sat_val.pt")

    err = np.linalg.norm(bs[va, :3] - ys[va, :3], axis=1)
    meta = {"config": dict(c), "n_train": int(tr.sum()), "n_val": int(va.sum()),
            "satellites": [int(s) for s in np.unique(sat)],
            "holdout_norad": holdout,
            "outlier_filter_km": OUTLIER_KM,
            "n_outliers_dropped": n_drop,
            "outliers_dropped_per_satellite": per_sat_drop,
            "truth_source": "ESA POD AUX_POEORB (real precise ephemeris, TEME)",
            "feature_source": "TLE/SGP4-derived states (deployable)",
            "sgp4_vs_truth_km_val": {
                "median": float(np.median(err)), "mean": float(err.mean()),
                "p90": float(np.percentile(err, 90)),
                "p99": float(np.percentile(err, 99)),
                "rmse": float(np.sqrt((err ** 2).mean()))}}
    json.dump(meta, open(os.path.join(c["TENSOR_DIR"], "phase55_meta.json"), "w"),
              indent=2, default=str)
    print("SGP4 vs REAL truth (val):", meta["sgp4_vs_truth_km_val"])
    print("tensors saved -> run train_phase55_rtn.py")


if __name__ == "__main__":
    main()