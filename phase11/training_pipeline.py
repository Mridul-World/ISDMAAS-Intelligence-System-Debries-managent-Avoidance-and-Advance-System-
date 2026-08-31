"""
training_pipeline.py — periodic truth-check + training-sample generation + retrain hook.
==========================================================================================
WHAT IT DOES (each run):
  1. SCAN   truth_data/<NORAD>/*.csv for precise-truth ephemerides
            (columns: epoch_utc,x_km,y_km,z_km — same layout as the validation
            harness). New files are registered in the ops database.
  2. BUILD  for every registered satellite that HAS truth: pair each truth point
            with the newest stored TLE preceding it, propagate SGP4 to that
            epoch, and compute the RTN-frame residual (truth − SGP4). Each row
            is one labeled training sample — the exact target the
            residual-correction Transformer learns.
  3. RETRAIN  hand the sample file to your trainer (plug-in point below).

HONEST CONSTRAINT (by design): this pipeline converts truth data that EXISTS.
If no truth files are present, it reports that and exits — it never fabricates
labels. Truth arrives from ESA POD downloads (Sentinels) or an operator pilot.

Run manually:    python training_pipeline.py
Run on schedule: see the scheduling steps in the delivery notes (Task Scheduler).
"""
import os, csv, glob
import numpy as np
from datetime import datetime, timezone

import ops_db

TRUTH_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "truth_data")
SAMPLES_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "training_samples.csv")

# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _parse_dt(s):
    s = s.strip().replace("Z", "")
    # strip a trailing UTC offset like +00:00 (fromisoformat handles most forms)
    try:
        dt = datetime.fromisoformat(s)
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    raise ValueError(f"unparseable epoch: {s}")

def _load_truth(path):
    times, pos = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ep = row.get("epoch_utc") or row.get("epoch") or row.get("time")
            x, y, z = row.get("x_km"), row.get("y_km"), row.get("z_km")
            if None in (ep, x, y, z):
                continue
            times.append(_parse_dt(ep))
            pos.append([float(x), float(y), float(z)])
    return times, np.array(pos)

def _tle_epoch_dt(tle1):
    """TLE epoch (col 19-32, YYDDD.DDDDDDDD) -> datetime UTC."""
    f = tle1[18:32].strip()
    yy = int(f[:2]); doy = float(f[2:])
    year = 2000 + yy if yy < 57 else 1900 + yy
    base = datetime(year, 1, 1, tzinfo=timezone.utc)
    from datetime import timedelta
    return base + timedelta(days=doy - 1)

def _sgp4_at(tle1, tle2, when):
    from sgp4.api import Satrec, jday
    sat = Satrec.twoline2rv(tle1, tle2)
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond * 1e-6)
    e, r, v = sat.sgp4(jd, fr)
    return (None, None) if e != 0 else (np.array(r), np.array(v))

def _eci_to_rtn(vec, r, v):
    R = r / np.linalg.norm(r)
    N = np.cross(r, v); N = N / np.linalg.norm(N)
    T = np.cross(N, R)
    return np.array([np.dot(vec, R), np.dot(vec, T), np.dot(vec, N)])

# ----------------------------------------------------------------------------
# step 1: scan for truth data
# ----------------------------------------------------------------------------
def scan_truth():
    found = []
    os.makedirs(TRUTH_DIR, exist_ok=True)
    for satdir in sorted(glob.glob(os.path.join(TRUTH_DIR, "*"))):
        if not os.path.isdir(satdir):
            continue
        norad = os.path.basename(satdir)
        for path in sorted(glob.glob(os.path.join(satdir, "*.csv"))):
            try:
                times, pos = _load_truth(path)
            except Exception as e:
                print(f"  [skip] {path}: {e}")
                continue
            if len(times) < 5:
                continue
            ops_db.register_truth(norad, path, len(times),
                                  times[0].isoformat(), times[-1].isoformat())
            found.append((norad, path, len(times)))
    return found

# ----------------------------------------------------------------------------
# step 2: build labeled samples (TLE history × truth)
# ----------------------------------------------------------------------------
def build_samples():
    rows = []
    sats_used = set()
    for norad in ops_db.all_norads():
        truth_entries = ops_db.truth_for(norad)
        if not truth_entries:
            continue
        history = ops_db.tle_history(norad)
        if not history:
            continue
        # precompute TLE epochs for nearest-preceding lookup
        tles = []
        for h in history:
            try:
                tles.append((_tle_epoch_dt(h["tle1"]), h["tle1"], h["tle2"]))
            except Exception:
                continue
        tles.sort(key=lambda t: t[0])
        if not tles:
            continue
        for path, _, _, _ in truth_entries:
            try:
                times, pos = _load_truth(path)
            except Exception:
                continue
            for t, p_truth in zip(times, pos, strict=True):
                # newest TLE at or before t (fallback: earliest TLE)
                prior = [x for x in tles if x[0] <= t]
                ep, l1, l2 = (prior[-1] if prior else tles[0])
                r_sgp4, v_sgp4 = _sgp4_at(l1, l2, t)
                if r_sgp4 is None:
                    continue
                resid_eci = p_truth - r_sgp4
                resid_rtn = _eci_to_rtn(resid_eci, r_sgp4, v_sgp4)
                prop_h = (t - ep).total_seconds() / 3600.0
                rows.append({
                    "norad": norad, "epoch_utc": t.isoformat(),
                    "prop_hours_since_tle": round(prop_h, 3),
                    "sgp4_x_km": r_sgp4[0], "sgp4_y_km": r_sgp4[1], "sgp4_z_km": r_sgp4[2],
                    "sgp4_vx": v_sgp4[0], "sgp4_vy": v_sgp4[1], "sgp4_vz": v_sgp4[2],
                    "resid_R_km": resid_rtn[0], "resid_T_km": resid_rtn[1],
                    "resid_N_km": resid_rtn[2],
                    "resid_3d_km": float(np.linalg.norm(resid_eci)),
                })
                sats_used.add(norad)
    if rows:
        with open(SAMPLES_OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    return rows, sorted(sats_used)

# ----------------------------------------------------------------------------
# step 3: retrain hook (plug-in point)
# ----------------------------------------------------------------------------
def retrain(samples_csv):
    """
    PLUG-IN POINT: connect your phase55 trainer here.
    The samples file has the exact learning target (RTN residual vs SGP4
    features). Typical wiring:

        from phase55.train import train_from_samples
        train_from_samples(samples_csv, checkpoint_out="models/residual_transformer/")

    Until wired, this reports what WOULD be trained — it does not pretend to.
    """
    print(f"  [retrain hook] samples ready at {samples_csv}")
    print("  [retrain hook] no trainer wired — connect your phase55 training "
          "entry point here to retrain automatically.")
    return False

# ----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("ISDMAAS TRAINING PIPELINE —", datetime.now(timezone.utc).isoformat()[:19] + "Z")
    print("=" * 60)
    print(f"\n[1/3] Scanning {TRUTH_DIR}\\ for precise truth data...")
    found = scan_truth()
    if not found:
        print("  No truth files found.")
        print("  Truth arrives from: ESA POD downloads (Sentinels) or an operator")
        print("  pilot. Drop files as truth_data\\<NORAD>\\*.csv")
        print("  (columns: epoch_utc,x_km,y_km,z_km) and re-run.")
        print("\n  Nothing to train on — exiting honestly (no labels fabricated).")
        return
    for norad, path, n in found:
        print(f"  truth: NORAD {norad} · {os.path.basename(path)} · {n} points")

    print("\n[2/3] Building labeled samples (TLE history × truth)...")
    rows, sats = build_samples()
    if not rows:
        print("  Truth exists but no matching TLE history in the ops DB — upload")
        print("  TLEs for those NORADs via the console first.")
        return
    print(f"  {len(rows)} samples across {len(sats)} satellite(s) -> {SAMPLES_OUT}")
    med = float(np.median([r['resid_3d_km'] for r in rows]))
    print(f"  median SGP4 residual in this set: {med:.3f} km (the signal the model learns)")

    print("\n[3/3] Retrain...")
    trained = retrain(SAMPLES_OUT)
    ops_db.record_training_run(len(rows), sats,
                               "retrained" if trained else "samples generated; trainer not wired")
    print("\nDone. DB status:", ops_db.status())

if __name__ == "__main__":
    main()
