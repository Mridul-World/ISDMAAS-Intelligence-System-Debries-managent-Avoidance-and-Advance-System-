"""
PHASE 8 — runnable maneuver-planning demo (the actionable OUTPUT 4).

Constructs a CRITICAL conjunction against a well-tracked primary, then prints
the full operator-facing maneuver recommendation: dv vector, timing, fuel,
and the verified before/after Pc.

    python phase8_maneuver_run.py [miss_km] [tca_hours] [sat_mass_kg]

In production this consumes a Phase-7 CDM (conjunction) instead of a synthetic
one; the planner logic is identical.
"""
import sys, json
import numpy as np
from phase7_collision import rtn_to_eci_cov, assess_conjunction
from phase7_collision import pc_text
from phase8_maneuver import plan_maneuver, PC_THRESHOLD, PC_SAFE


def main():
    miss_km = float(sys.argv[1]) if len(sys.argv) > 1 else 0.04
    tca_h = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
    mass = float(sys.argv[3]) if len(sys.argv) > 3 else 2300.0

    # primary (well-tracked LEO), secondary on crossing course at `miss_km`
    rp = np.array([7078.0, 0.0, 0.0]); vp = np.array([0.0, 7.5, 0.0])
    speed = np.linalg.norm(vp)
    rhat = rp / np.linalg.norm(rp); vhat = vp / speed
    cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)

    Cp = rtn_to_eci_cov(np.diag([0.02**2, 0.05**2, 0.02**2]), rp, vp)
    r2 = rp + miss_km * rhat
    v2 = cross * speed
    C2 = rtn_to_eci_cov(np.diag([0.03**2, 0.1**2, 0.03**2]), r2, v2)
    tca_s = tca_h * 3600.0

    plan = plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr=0.02, tca_s=tca_s,
                         sat_mass_kg=mass)

    print("=" * 66)
    print("CONJUNCTION  (pre-maneuver)")
    pm = plan["pre_maneuver"]
    print(f"  miss distance : {pm['miss_distance_km']:.4f} km")
    print(f"  Pc            : {pc_text(pm['pc'], '.3e')}")
    print(f"  risk          : {pm['risk_level']}")
    print(f"  TCA           : t+{tca_h:.1f} h")
    print(f"  action needed : {plan['action_required']}")
    print("-" * 66)

    if not isinstance(plan["recommendation"], dict):
        print("RECOMMENDATION:", plan["recommendation"])
        print("=" * 66); return

    r = plan["recommendation"]
    print("MANEUVER RECOMMENDATION  (OUTPUT 4)")
    print(f"  burn dv (RTN)     : [{r['dv_rtn_ms'][0]:.4f}, "
          f"{r['dv_rtn_ms'][1]:.4f}, {r['dv_rtn_ms'][2]:.4f}] m/s")
    print(f"  dv magnitude      : {r['dv_magnitude_ms']:.4f} m/s ({r['direction']})")
    print(f"  execute           : {r['burn_lead_time_h']:.2f} h before TCA")
    print(f"  fuel required     : {r['fuel_kg']:.4f} kg")
    print(f"  -> new miss       : {r['predicted_new_miss_km']:.4f} km")
    print(f"  -> new Pc         : {r['predicted_new_pc']:.3e}")
    print(f"  -> new risk       : {r['predicted_new_risk']}")
    print("-" * 66)
    print("LEAD-TIME TRADE (earlier = cheaper):")
    print(f"  {'lead(h)':>8}{'dv(m/s)':>10}{'fuel(kg)':>10}{'new Pc':>12}{'risk':>10}")
    for o in plan["options"]:
        print(f"  {o['burn_lead_time_h']:>8.2f}{o['dv_ms']:>10.4f}"
              f"{o['fuel_kg']:>10.4f}{o['new_pc']:>12.2e}{o['new_risk']:>10}")
    print("=" * 66)

    import os
    os.makedirs("reports", exist_ok=True)
    json.dump(plan, open("reports/maneuver_plan.json", "w"), indent=2, default=str)
    print("saved: reports/maneuver_plan.json")


if __name__ == "__main__":
    main()
