"""
PHASE 10 — end-to-end safety-validated maneuver pipeline (demo).

Runs the FULL decision chain on a synthetic CRITICAL conjunction against the
real primary, including the live-catalog re-screen for new conjunctions:

  conjunction -> plan maneuver (Phase 8) -> validate (Phase 10) -> verdict

    python phase10_run.py 39634 [miss_km] [tca_h]

Locates the Phase-6 state file automatically. If your phase55 folder is in a
non-standard place, set an environment variable before running:
    set PHASE55_DIR=C:\path\to\isdmaas\phase55      (Windows CMD)
    $env:PHASE55_DIR="C:\path\to\isdmaas\phase55"   (PowerShell)

Produces reports/validated_maneuver.json.
"""
import os, sys, json
import numpy as np
import requests
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn, assess_conjunction
from phase7_collision import pc_text
from phase8_maneuver import plan_maneuver, fuel_kg
from phase10_safety import validate_maneuver, orbital_elements

CELESTRAK_ACTIVE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"


def find_state_file(norad):
    """Search common locations for the Phase-6 state_<norad>.json."""
    fname = f"state_{norad}.json"
    candidates = []
    # 1. explicit override
    if os.environ.get("PHASE55_DIR"):
        candidates.append(os.path.join(os.environ["PHASE55_DIR"], "reports", fname))
    # 2. sibling phase55 (phase10 next to phase55)
    candidates.append(os.path.join("..", "phase55", "reports", fname))
    # 3. local reports (if you copied the state file into phase10)
    candidates.append(os.path.join("reports", fname))
    candidates.append(fname)
    # 4. common absolute fallback (edit if your tree differs)
    candidates.append(rf"C:\Work_place\projects\isdmaas\phase55\reports\{fname}")
    for fp in candidates:
        if os.path.exists(fp):
            return fp
    sys.exit("state_%d.json not found. Searched:\n  %s\n"
             "Fix: in phase55 run  python infer_uncertain.py %d 72 --mc 20 --alpha 1.0\n"
             "then either put phase10 next to phase55, copy the state file into "
             "phase10\\reports\\, or set PHASE55_DIR." %
             (norad, "\n  ".join(candidates), norad))


def load_primary(norad):
    fp = find_state_file(norad)
    print(f"using primary state: {fp}")
    d = json.load(open(fp))
    return (np.array(d["position_teme_km"]), np.array(d["velocity_teme_kms"]),
            np.array(d["covariance_eci_km2"]), datetime.fromisoformat(d["epoch_utc"]),
            d["name"])


def fetch_catalog():
    cache = "data/catalog_active.tle"
    if os.path.exists(cache) and (datetime.now().timestamp() - os.path.getmtime(cache)) < 86400:
        text = open(cache).read()
    else:
        print("downloading active catalog ...")
        headers = {"User-Agent": "Mozilla/5.0 (ISDMAAS research; contact me@example.com)"}
        r = requests.get(CELESTRAK_ACTIVE, headers=headers, timeout=120)
        r.raise_for_status()
                         
        
        text = r.text; os.makedirs("data", exist_ok=True); open(cache, "w").write(text)
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    sats = {}
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i+1], lines[i+2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        try:
            sats[int(l2[2:7])] = (name.strip(), Satrec.twoline2rv(l1, l2))
        except Exception:
            continue
    return sats


def main():
    norad = int(sys.argv[1]) if len(sys.argv) > 1 else 39634
    miss_km = float(sys.argv[2]) if len(sys.argv) > 2 else 0.04
    tca_h = float(sys.argv[3]) if len(sys.argv) > 3 else 8.0

    rp, vp, Cp, epoch, pname = load_primary(norad)
    catalog = fetch_catalog()

    # construct a synthetic CRITICAL threat on a crossing course
    rhat = rp / np.linalg.norm(rp); speed = np.linalg.norm(vp); vhat = vp / speed
    cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
    r2 = rp + miss_km * rhat
    v2 = cross * speed
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(259200 / 2), r2, v2)
    tca_s = tca_h * 3600.0
    mission_sma, _, _ = orbital_elements(rp, vp)

    # Phase 8: plan
    plan = plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr=0.02, tca_s=tca_s, sat_mass_kg=2300)
    print("=" * 66)
    print(f"PRE-MANEUVER: miss {plan['pre_maneuver']['miss_distance_km']:.4f} km  "
          f"Pc {pc_text(plan['pre_maneuver']['pc'])}  "
          f"{plan['pre_maneuver']['risk_level']}")
    if not isinstance(plan["recommendation"], dict):
        print("no maneuver:", plan["recommendation"]); return
    rec = plan["recommendation"]
    sign = 1.0 if rec["direction"] == "prograde" else -1.0
    print(f"PLANNED BURN: {rec['dv_magnitude_ms']:.4f} m/s {rec['direction']}, "
          f"{rec['burn_lead_time_h']:.1f}h before TCA, fuel {rec['fuel_kg']:.4f} kg")
    print("-" * 66)

    # Phase 10: validate (with live-catalog re-screen)
    print("running safety validation (incl. catalog re-screen) ...")
    report = validate_maneuver(
        rp, vp, Cp, r2, v2, C2, 0.02, tca_s,
        rec["dv_magnitude_ms"], sign, rec["burn_lead_time_h"], rec["fuel_kg"],
        fuel_available_kg=5.0, mission_sma_km=mission_sma,
        Cp_rtn_diag=[0.02, 0.05, 0.02], catalog=catalog, epoch=epoch,
        primary_norad=norad)

    print("-" * 66)
    for c in report["checks"]:
        mark = "PASS" if c["pass"] else "FAIL"
        print(f"  [{mark}] {c['check']:<26} {c['reason']}")
    print("-" * 66)
    print(f"VERDICT: {report['verdict']}")
    if report["failed_checks"]:
        print(f"  failed: {report['failed_checks']}")
    print("=" * 66)

    os.makedirs("reports", exist_ok=True)
    json.dump({"primary": pname, "plan": plan, "validation": report},
              open("reports/validated_maneuver.json", "w"), indent=2, default=str)
    print("saved: reports/validated_maneuver.json")


if __name__ == "__main__":
    main()