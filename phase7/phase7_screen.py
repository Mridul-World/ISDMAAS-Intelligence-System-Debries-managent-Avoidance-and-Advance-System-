"""
PHASE 7.1 — runnable conjunction assessment + catalog screening.

MODE A — one pair (primary from Phase-6 state file, secondary by NORAD):
    python phase7_screen.py pair 39634 <secondary_norad>

MODE B — screen primary against the active catalog:
    python phase7_screen.py screen 39634

MODE C — demo: synthetic threat on a collision course with the REAL primary,
         to exercise the HIGH/CRITICAL path in the production code (miss in km):
    python phase7_screen.py demo-critical 39634 2.0
    python phase7_screen.py demo-critical 39634 0.5
    python phase7_screen.py demo-critical 39634 0.1

Two-stage screening gate:
  1. SHELL gate: reject objects whose orbital radius never overlaps the
     primary's altitude band (+/- COARSE_KM). Removes ~99% of catalog cheaply.
  2. TCA search: BOTH objects propagated with SGP4 across the window to find
     the real close approach, then full Pc is computed there.

Primary state: ../phase55/reports/state_<norad>.json (AI-corrected mean + the
Phase-6 calibrated covariance). The primary's TLE (from the catalog) propagates
its trajectory; the Phase-6 covariance is attached at the close approach.

Outputs CDM-style JSON per conjunction in reports/conjunctions/.
"""
import os, sys, json
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday
from phase7_collision import (assess_conjunction, secondary_covariance_rtn,
                              rtn_to_eci_cov)
from phase7_collision import pc_for_safety, pc_text

CELESTRAK_ACTIVE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
HBR_DEFAULT_KM = 0.020          # 20 m combined hard-body radius
SCREEN_WINDOW_S = 259200        # 3 days (matches model horizon)
COARSE_KM = 200.0               # proximity gate (lower to 50 for stricter screening)
N_TCA_SAMPLES = 288             # ~15 min spacing over 3 days


def load_primary(norad):
    fp = f"../phase55/reports/state_{norad}.json"
    if not os.path.exists(fp):
        sys.exit(f"{fp} not found — in phase55 run: "
                 f"python infer_uncertain.py {norad} 72 --mc 20 --alpha 1.0")
    d = json.load(open(fp))
    r = np.array(d["position_teme_km"], float)
    v = np.array(d["velocity_teme_kms"], float)
    C = np.array(d["covariance_eci_km2"], float)
    epoch = datetime.fromisoformat(d["epoch_utc"])
    return r, v, C, epoch, d["name"]


def sgp4_state(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return np.array(r), np.array(v)


def fetch_catalog():
    cache = "data/catalog_active.tle"
    if os.path.exists(cache) and (datetime.now().timestamp() -
                                  os.path.getmtime(cache)) < 86400:
        text = open(cache).read()
    else:
        print("downloading active catalog from CelesTrak ...")
        r = requests.get(CELESTRAK_ACTIVE, timeout=120); r.raise_for_status()
        text = r.text
        os.makedirs("data", exist_ok=True); open(cache, "w").write(text)
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    sats = {}
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        try:
            norad = int(l2[2:7])
            sats[norad] = (name.strip(), Satrec.twoline2rv(l1, l2))
        except Exception:
            continue
    return sats


def assess_pair(rp, vp, Cp, when, sat2, name2, hbr=HBR_DEFAULT_KM):
    r2, v2 = sgp4_state(sat2, when)
    if r2 is None:
        return None
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(SCREEN_WINDOW_S / 2), r2, v2)
    res = assess_conjunction(rp, vp, Cp, r2, v2, C2, hbr, SCREEN_WINDOW_S)
    res["secondary_name"] = name2
    return res


def tca_search(psat, sat2, epoch):
    """Both objects via SGP4; return (min_dist_km, t_min_s)."""
    min_d, t_min = 1e9, 0.0
    for s in range(N_TCA_SAMPLES):
        dt = s * (SCREEN_WINDOW_S / N_TCA_SAMPLES)
        when = epoch + timedelta(seconds=dt)
        rp_t, _ = sgp4_state(psat, when)
        r2, _ = sgp4_state(sat2, when)
        if rp_t is None or r2 is None:
            continue
        d = np.linalg.norm(rp_t - r2)
        if d < min_d:
            min_d, t_min = d, dt
    return min_d, t_min


def write_cdm(primary_name, primary_norad, secondary_norad, epoch, res):
    os.makedirs("reports/conjunctions", exist_ok=True)
    cdm = {"created_utc": datetime.now(timezone.utc).isoformat(),
           "primary": {"norad": primary_norad, "name": primary_name},
           "secondary": {"norad": secondary_norad, "name": res.get("secondary_name")},
           "screening_epoch_utc": epoch.isoformat(), **res}
    fp = f"reports/conjunctions/cdm_{primary_norad}_{secondary_norad}.json"
    json.dump(cdm, open(fp, "w"), indent=2)
    return fp


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    mode, norad = sys.argv[1], int(sys.argv[2])
    rp, vp, Cp, epoch, pname = load_primary(norad)
    cat = fetch_catalog()
    psat = cat.get(norad, (None, None))[1]
    if psat is None:
        sys.exit(f"primary {norad} not in active catalog TLEs (needed to propagate it)")

    if mode == "pair":
        sec = int(sys.argv[3])
        if sec not in cat:
            sys.exit(f"secondary {sec} not in active catalog")
        name2, sat2 = cat[sec]
        min_d, t_min = tca_search(psat, sat2, epoch)
        when = epoch + timedelta(seconds=t_min)
        rp_t, vp_t = sgp4_state(psat, when)
        res = assess_pair(rp_t, vp_t, Cp, when, sat2, name2)
        fp = write_cdm(pname, norad, sec, epoch, res)
        print(json.dumps({k: res[k] for k in
              ("tca_s_from_epoch", "miss_distance_km", "pc",
               "pc_methods_agree", "mahalanobis", "risk_level")}, indent=2))
        print(f"close approach found at t+{t_min/3600:.2f} h, sep {min_d:.3f} km")
        print("saved:", fp)

    elif mode == "demo-critical":
        # Construct a secondary on a near-collision course with the REAL primary
        # to exercise the HIGH/CRITICAL path through the production assess code.
        miss = float(sys.argv[3]) if len(sys.argv) > 3 else 0.2     # desired miss (km)
        rp_t, vp_t = sgp4_state(psat, epoch)
        speed = float(np.linalg.norm(vp_t))
        # offset the secondary by `miss` along the primary's radial direction,
        # send it crossing perpendicular to the primary velocity at orbital speed
        rhat = rp_t / np.linalg.norm(rp_t)
        r2 = rp_t + miss * rhat
        # perpendicular velocity: cross product of radial and primary velocity dir
        vhat = vp_t / speed
        cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
        v2 = cross * speed
        C2 = rtn_to_eci_cov(secondary_covariance_rtn(SCREEN_WINDOW_S / 2), r2, v2)
        res = assess_conjunction(rp_t, vp_t, Cp, r2, v2, C2, HBR_DEFAULT_KM, 600)
        res["secondary_name"] = f"SYNTHETIC_THREAT_{miss}km"
        print(f"SYNTHETIC THREAT vs {pname} — target miss {miss} km")
        print(json.dumps({k: res[k] for k in
              ("miss_distance_km", "pc", "pc_chan_crosscheck", "pc_methods_agree",
               "mahalanobis", "relative_speed_kms", "risk_level")}, indent=2))

    elif mode == "screen":
        print(f"screening {pname} ({norad}) vs {len(cat)} catalog objects ...")
        rp_norm = np.linalg.norm(rp)
        hits = []
        checked = 0
        for sec, (name2, sat2) in cat.items():
            if sec == norad:
                continue
            r2_now, _ = sgp4_state(sat2, epoch)
            if r2_now is None:
                continue
            if abs(np.linalg.norm(r2_now) - rp_norm) > COARSE_KM:
                continue
            checked += 1
            min_d, t_min = tca_search(psat, sat2, epoch)
            if min_d > COARSE_KM:
                continue
            when = epoch + timedelta(seconds=t_min)
            rp_t, vp_t = sgp4_state(psat, when)
            res = assess_pair(rp_t, vp_t, Cp, when, sat2, name2)
            if res and res["miss_distance_km"] < COARSE_KM:
                hits.append((sec, res))
        print(f"(shell-gated; ran TCA search on {checked} same-altitude objects)")
        # An unavailable Pc sorts to the top rather than raising a TypeError:
        # a conjunction we could not assess is the first one an operator should
        # look at, not one that silently drops out of the ranking.
        hits.sort(key=lambda x: -pc_for_safety(x[1]["pc"]))
        print(f"\n{len(hits)} conjunctions within {COARSE_KM} km:")
        print(f"{'NORAD':>8}{'miss_km':>10}{'Pc':>12}{'risk':>10}  name")
        for sec, res in hits[:30]:
            write_cdm(pname, norad, sec, epoch, res)
            print(f"{sec:>8}{res['miss_distance_km']:>10.3f}"
                  f"{pc_text(res['pc']):>12}"
                  f"{res['risk_level']:>10}  {res['secondary_name']}")
        print(f"\nCDMs saved in reports/conjunctions/")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()