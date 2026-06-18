"""
PHASE 10 — self-test for the safety/validation layer.

Checks each gate triggers correctly:
  1. fuel check passes with margin, fails when insufficient
  2. orbit-safety passes for a nominal burn, fails for a deorbit-sized burn
  3. threat-resolved passes after a real avoidance burn
  4. full validate_maneuver returns APPROVED for a good burn,
     REJECTED for an over-budget burn
(no_new_conjunction needs a catalog; tested structurally without network)
"""
import numpy as np
from phase7_collision import rtn_to_eci_cov, assess_conjunction
from phase8_maneuver import plan_maneuver, fuel_kg, apply_along_track_dv
from phase10_safety import (check_fuel, check_orbit_safety, check_threat_resolved,
                            validate_maneuver, orbital_elements, MU)

ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("Phase 10 self-test")
print("-" * 64)

# setup: a CRITICAL conjunction with tight covariances
rp = np.array([7078.0, 0.0, 0.0]); vp = np.array([0.0, 7.5, 0.0])
speed = np.linalg.norm(vp)
rhat = rp / np.linalg.norm(rp); vhat = vp / speed
cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
Cp = rtn_to_eci_cov(np.diag([0.02**2, 0.05**2, 0.02**2]), rp, vp)
r2 = rp + 0.03 * rhat
v2 = cross * speed
C2 = rtn_to_eci_cov(np.diag([0.03**2, 0.1**2, 0.03**2]), r2, v2)
tca_s = 8 * 3600.0
mission_sma, _, _ = orbital_elements(rp, vp)

plan = plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr=0.02, tca_s=tca_s, sat_mass_kg=2300)
rec = plan["recommendation"]
dv_ms = rec["dv_magnitude_ms"]
sign = 1.0 if rec["direction"] == "prograde" else -1.0
lead = rec["burn_lead_time_h"]
fuel_req = rec["fuel_kg"]

# 1. fuel checks
c = check_fuel(dv_ms, fuel_req, fuel_available_kg=5.0)
check("fuel passes with plenty available", c["pass"], f"req={fuel_req:.4f}kg")
c = check_fuel(dv_ms, fuel_req, fuel_available_kg=fuel_req * 1.1)
check("fuel fails below 30% margin", not c["pass"])

# 2. orbit-safety: nominal burn ok
rp_post, vp_post = apply_along_track_dv(rp, vp, sign * dv_ms / 1000.0, lead * 3600.0)
c = check_orbit_safety(rp_post, vp_post, mission_sma)
check("orbit safe after small burn", c["pass"],
      f"perigee={c['post_perigee_alt_km']:.1f}km")
# huge retrograde burn -> should drop perigee / break SMA band
rp_bad, vp_bad = apply_along_track_dv(rp, vp, -0.5, lead * 3600.0)   # 500 m/s
c = check_orbit_safety(rp_bad, vp_bad, mission_sma)
check("orbit-safety flags huge burn", not c["pass"], c["reason"][:50])

# 3. threat resolved after burn
c = check_threat_resolved(rp_post, vp_post, Cp, r2, v2, C2, 0.02)
check("threat resolved after burn", c["pass"], f"post Pc={c['post_maneuver_pc']:.2e}")

# 4. full validate (no catalog) -> APPROVED for good burn
rep = validate_maneuver(rp, vp, Cp, r2, v2, C2, 0.02, tca_s,
                        dv_ms, sign, lead, fuel_req,
                        fuel_available_kg=5.0, mission_sma_km=mission_sma,
                        Cp_rtn_diag=[0.02, 0.05, 0.02])
check("full validation APPROVED (good burn)", rep["verdict"] == "APPROVED",
      f"failed={rep['failed_checks']}")

# 5. full validate with no fuel -> REJECTED
rep = validate_maneuver(rp, vp, Cp, r2, v2, C2, 0.02, tca_s,
                        dv_ms, sign, lead, fuel_req,
                        fuel_available_kg=0.0001, mission_sma_km=mission_sma,
                        Cp_rtn_diag=[0.02, 0.05, 0.02])
check("full validation REJECTED (no fuel)", rep["verdict"] == "REJECTED",
      f"failed={rep['failed_checks']}")

print("-" * 64)
print("ALL TESTS PASS" if ok else "SOME TESTS FAILED")
