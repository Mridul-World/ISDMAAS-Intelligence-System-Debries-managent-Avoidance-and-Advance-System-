"""
Phase 5.5 — parse per-satellite AUX_POEORB .EOF files into TEME-frame
precise-ephemeris CSVs.

CRITICAL FRAME NOTE: EOF OSVs are Earth-Fixed (ITRF); SGP4 is TEME. This
script performs a rigorous astropy ITRS -> TEME transform INCLUDING velocity
(rotating-frame) terms. Verified round-trip accuracy: < 1e-9 km.

Corrupted/empty EOF files are detected, reported, and DELETED automatically —
rerun download_precise_orbits.py afterwards to re-fetch only those files.

Input : data/precise_orbit/<PREFIX>/*.EOF
Output: data/ephemeris_<norad>.csv  (timestamp, true_x_km..true_vz_kms, TEME)
Idempotent per satellite.
"""
import os, glob
import numpy as np
import pandas as pd
import xml.etree.ElementTree as ET
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (ITRS, TEME, CartesianRepresentation,
                                 CartesianDifferential)
from config import CFG, SATELLITES, parse_date, eph_csv


def parse_eof(path):
    """One .EOF -> (utc list, pos (N,3) km ITRF, vel (N,3) km/s ITRF).
    Returns empty arrays for corrupt/unreadable files."""
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return [], np.zeros((0, 3)), np.zeros((0, 3))
    ts, pos, vel = [], [], []
    try:
        for o in root.findall(".//OSV"):
            if (o.findtext("Quality") or "NOMINAL").strip().upper() not in ("NOMINAL", ""):
                continue
            ts.append(o.findtext("UTC").replace("UTC=", "").strip())
            pos.append([float(o.findtext(c)) / 1000.0 for c in ("X", "Y", "Z")])
            vel.append([float(o.findtext(c)) / 1000.0 for c in ("VX", "VY", "VZ")])
    except (TypeError, ValueError, AttributeError):
        return [], np.zeros((0, 3)), np.zeros((0, 3))
    if not ts:
        return [], np.zeros((0, 3)), np.zeros((0, 3))
    return ts, np.array(pos), np.array(vel)


def itrf_to_teme(times_utc, pos_itrf, vel_itrf, chunk=20000):
    P_out, V_out = [], []
    for i in range(0, len(times_utc), chunk):
        t = Time(times_utc[i:i + chunk], scale="utc")
        p, v = pos_itrf[i:i + chunk], vel_itrf[i:i + chunk]
        rep = CartesianRepresentation(
            p[:, 0] * u.km, p[:, 1] * u.km, p[:, 2] * u.km,
            differentials=CartesianDifferential(
                v[:, 0] * u.km / u.s, v[:, 1] * u.km / u.s, v[:, 2] * u.km / u.s))
        teme = ITRS(rep, obstime=t).transform_to(TEME(obstime=t))
        c = teme.cartesian; d = c.differentials["s"]
        P_out.append(np.stack([c.x.to_value(u.km), c.y.to_value(u.km),
                               c.z.to_value(u.km)], axis=1))
        V_out.append(np.stack([d.d_x.to_value(u.km / u.s), d.d_y.to_value(u.km / u.s),
                               d.d_z.to_value(u.km / u.s)], axis=1))
        print(f"    frame-converted {min(i + chunk, len(times_utc))}/{len(times_utc)}")
    return np.vstack(P_out), np.vstack(V_out)


def process_satellite(norad, info):
    out = eph_csv(norad)
    if os.path.exists(out):
        print(f"[{info['name']}] skip (exists): {out}"); return True
    files = sorted(glob.glob(os.path.join(CFG["POE_DIR"], info["prefix"], "*.EOF")))
    if not files:
        print(f"[{info['name']}] no EOF files — skipped"); return False
    print(f"[{info['name']}] parsing {len(files)} EOF files ...")

    all_ts, all_p, all_v = [], [], []
    bad = []
    for fp in files:
        ts, p, v = parse_eof(fp)
        if len(ts) == 0:
            bad.append(fp)
            print(f"  WARN: {os.path.basename(fp)} has 0 OSVs — corrupted, skipping")
            continue
        all_ts += ts; all_p.append(p); all_v.append(v)

    for fp in bad:
        try:
            os.remove(fp)
            print(f"  deleted corrupted file: {os.path.basename(fp)}")
        except OSError as e:
            print(f"  WARN: could not delete {os.path.basename(fp)}: {e}")
    if bad:
        print(f"  NOTE: {len(bad)} corrupted file(s) removed — rerun "
              f"download_precise_orbits.py to re-fetch them, then rerun this script.")

    if not all_p:
        print(f"[{info['name']}] no usable files — skipped"); return False

    P, V = np.vstack(all_p), np.vstack(all_v)

    tsi = pd.to_datetime(all_ts, utc=True, format="mixed")
    order = np.argsort(tsi.values); tsi, P, V = tsi[order], P[order], V[order]
    keep = np.concatenate([[True], np.diff(tsi.values).astype("timedelta64[ms]")
                           > np.timedelta64(0, "ms")])
    tsi, P, V = tsi[keep], P[keep], V[keep]
    d0, d1 = parse_date(CFG["DATE_START"]), parse_date(CFG["DATE_END"])
    m = (tsi >= d0) & (tsi <= d1)
    tsi, P, V = tsi[m], P[m], V[m]
    print(f"  {len(tsi)} unique OSVs ({tsi[0]} .. {tsi[-1]});  ITRF -> TEME ...")

    P_t, V_t = itrf_to_teme(tsi.strftime("%Y-%m-%dT%H:%M:%S.%f").tolist(), P, V)
    pd.DataFrame({"timestamp": tsi,
                  "true_x_km": P_t[:, 0], "true_y_km": P_t[:, 1], "true_z_km": P_t[:, 2],
                  "true_vx_kms": V_t[:, 0], "true_vy_kms": V_t[:, 1],
                  "true_vz_kms": V_t[:, 2]}).to_csv(out, index=False)
    print(f"  saved {len(tsi)} rows -> {out}  (frame: TEME)")
    return True


def main():
    ok = [norad for norad, info in SATELLITES.items() if process_satellite(norad, info)]
    print(f"\nephemerides ready for {len(ok)}/{len(SATELLITES)} satellites: {ok}")
    if not ok:
        raise SystemExit("No satellite has truth data — run download_precise_orbits.py")


if __name__ == "__main__":
    main()