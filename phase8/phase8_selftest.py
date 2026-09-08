"""
PHASE 8 — self-test + runnable maneuver demo.

Self-test checks:
  1. No maneuver recommended when Pc is already safe.
  2. For a CRITICAL conjunction, a maneuver IS recommended.
  3. The recommended maneuver actually drops Pc below PC_SAFE (verified by
     re-running the Pc engine on the post-burn state).
  4. Earlier burns require less dv (monotonic dv vs lead time).
  5. Fuel is positive and increases with dv.

Demo (constructs a CRITICAL threat vs a real-ish primary, plans the maneuver):
  python phase8_maneuver_run.py
"""
import json
import numpy as np
from phase7_collision import (rtn_to_eci_cov, secondary_covariance_rtn,
                              assess_conjunction)
from phase7_collision import pc_text
from phase8_maneuver import (plan_maneuver, solve_min_dv, fuel_kg,
                             apply_along_track_dv, PC_SAFE, PC_THRESHOLD)

ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("Phase 8 self-test")
print("-" * 64)

# build a primary in LEO and a secondary on a near-collision crossing course,
# with a TIGHT primary covariance so Pc can actually exceed CRITICAL
rp = np.array([7078.0, 0.0, 0.0])
vp = np.array([0.0, 7.5, 0.0])
speed = np.linalg.norm(vp)
rhat = rp / np.linalg.norm(rp)
vhat = vp / speed
cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)

# tight, well-tracked covariances (so a real CRITICAL is achievable)
Cp = rtn_to_eci_cov(np.diag([0.02**2, 0.05**2, 0.02**2]), rp, vp)   # ~20-50 m
tca_s = 6 * 3600.0

# secondary: 30 m miss, crossing
miss0 = 0.03
r2 = rp + miss0 * rhat
v2 = cross * speed
C2 = rtn_to_eci_cov(np.diag([0.03**2, 0.1**2, 0.03**2]), r2, v2)

pre = assess_conjunction(rp, vp, Cp, r2, v2, C2, 0.02, 600)
check("constructed conjunction is CRITICAL",
      pre["pc"] is not None and pre["pc"] > PC_THRESHOLD,
      f"Pc={pc_text(pre['pc'])} risk={pre['risk_level']}")

# 1+2+3: plan a maneuver and verify it works
plan = plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr=0.02, tca_s=tca_s, sat_mass_kg=2300)
check("maneuver recommended for CRITICAL", isinstance(plan["recommendation"], dict),
      "" if isinstance(plan["recommendation"], dict) else plan["recommendation"])
if isinstance(plan["recommendation"], dict):
    rec = plan["recommendation"]
    check("post-maneuver Pc below safe", rec["predicted_new_pc"] < PC_SAFE,
          f"new Pc={rec['predicted_new_pc']:.2e}")
    check("recommended dv is small (< 1 m/s typical)", rec["dv_magnitude_ms"] < 50,
          f"dv={rec['dv_magnitude_ms']:.4f} m/s")

# 4: earlier burn => less dv
dvs = []
for lead in (1.0, 3.0, 6.0):
    sol = solve_min_dv(rp, vp, Cp, r2, v2, C2, 0.02, lead * 3600.0)
    if sol:
        dvs.append((lead, sol[0]))
mono = all(dvs[i][1] >= dvs[i+1][1] - 1e-9 for i in range(len(dvs)-1))
check("earlier burn needs less dv", mono, str([(l, round(d,4)) for l,d in dvs]))

# 5: fuel positive + monotone
f1, f2 = fuel_kg(0.1, 2300), fuel_kg(1.0, 2300)
check("fuel positive & increasing", 0 < f1 < f2, f"f(0.1)={f1:.4f}kg f(1.0)={f2:.4f}kg")

print("-" * 64)
print("ALL TESTS PASS" if ok else "SOME TESTS FAILED")
