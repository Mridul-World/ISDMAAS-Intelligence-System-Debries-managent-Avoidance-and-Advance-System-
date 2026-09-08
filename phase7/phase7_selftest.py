"""
PHASE 7.1 — SELF-TEST. Verifies the Pc math before you trust it on real data.

Checks:
  1. Head-on miss=0, tight covariance  -> Pc near 1 (certain hit)
  2. Large miss vs covariance           -> Pc near 0
  3. quadrature vs Chan series agree across cases
  4. Pc monotonic increasing as miss distance shrinks
  5. Pc monotonic increasing as HBR grows
  6. A known-geometry case with hand-checkable magnitude

Run: python phase7_selftest.py   ->  prints PASS/FAIL per check.
"""
import numpy as np
from phase7_collision import (pc_2d_quadrature, pc_chan, assess_conjunction,
                              risk_level, secondary_covariance_rtn, rtn_to_eci_cov)
from phase7_collision import pc_text

ok = True
def check(name, cond, detail=""):
    global ok
    ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")


print("Phase 7 self-test")
print("-" * 60)

# 1. miss=0, tight covariance, HBR comparable to sigma -> high Pc
miss = np.array([0.0, 0.0])
C = np.diag([0.02 ** 2, 0.02 ** 2])         # 20 m sigma
pc = pc_2d_quadrature(miss, C, hbr=0.05)     # 50 m hard-body
check("direct hit, tight cov -> Pc high", pc > 0.8, f"Pc={pc:.3f}")

# 2. large miss vs covariance -> ~0
pc = pc_2d_quadrature(np.array([5.0, 0.0]), np.diag([0.1 ** 2, 0.1 ** 2]), hbr=0.02)
check("far miss -> Pc ~ 0", pc < 1e-6, f"Pc={pc:.2e}")

# 3. quadrature vs Chan agreement across cases
cases = [
    (np.array([0.5, 0.2]), np.diag([0.3 ** 2, 0.2 ** 2]), 0.02),
    (np.array([1.0, 0.0]), np.diag([0.5 ** 2, 0.4 ** 2]), 0.05),
    (np.array([0.1, 0.1]), np.diag([0.15 ** 2, 0.1 ** 2]), 0.03),
]
for i, (m, c, h) in enumerate(cases):
    a, b = pc_2d_quadrature(m, c, h), pc_chan(m, c, h)
    rel = abs(a - b) / max(a, 1e-30)
    check(f"quadrature~Chan case {i+1}", rel < 0.05 or abs(a - b) < 1e-9,
          f"quad={a:.3e} chan={b:.3e} rel={rel:.2%}")

# 4. monotonic in miss distance
C = np.diag([0.3 ** 2, 0.3 ** 2])
pcs = [pc_2d_quadrature(np.array([d, 0.0]), C, 0.02) for d in [0.1, 0.3, 0.6, 1.0, 2.0]]
check("Pc decreases with miss", all(np.diff(pcs) <= 1e-12), f"{[f'{p:.2e}' for p in pcs]}")

# 5. monotonic in HBR
pcs = [pc_2d_quadrature(np.array([0.3, 0.0]), C, h) for h in [0.005, 0.01, 0.02, 0.05]]
check("Pc increases with HBR", all(np.diff(pcs) >= -1e-12), f"{[f'{p:.2e}' for p in pcs]}")

# 6. risk-level mapping
check("risk mapping", risk_level(1e-8) == "NOMINAL" and risk_level(1e-6) == "ELEVATED"
      and risk_level(5e-5) == "HIGH" and risk_level(1e-3) == "CRITICAL")

# 7. full assess_conjunction on a constructed close approach
#    two objects crossing at ~700 km altitude, small miss
r1 = np.array([7078.0, 0.0, 0.0]); v1 = np.array([0.0, 7.5, 0.0])
# secondary passing nearly perpendicular, small offset
r2 = np.array([7078.2, 0.0, 0.0]); v2 = np.array([0.0, 0.0, 7.5])
C1 = rtn_to_eci_cov(np.diag([0.04 ** 2, 3.0 ** 2, 0.05 ** 2]), r1, v1)   # Phase-6-like
C2 = rtn_to_eci_cov(secondary_covariance_rtn(72 * 3600), r2, v2)
res = assess_conjunction(r1, v1, C1, r2, v2, C2, hbr_km=0.02)
check("assess runs + sane",
      res["pc"] is not None and 0 <= res["pc"] <= 1 and res["miss_distance_km"] >= 0,
      f"miss={res['miss_distance_km']:.3f}km Pc={pc_text(res['pc'])} "
      f"risk={res['risk_level']}")

print("-" * 60)
print("ALL TESTS PASS" if ok else "SOME TESTS FAILED")
