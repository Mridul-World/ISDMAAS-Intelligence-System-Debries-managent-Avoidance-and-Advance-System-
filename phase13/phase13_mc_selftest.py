"""
PHASE 13 — MONTE CARLO Pc — self-test + robustness demonstration.

    python phase13_mc_selftest.py

Checks:
  1. MC agrees with analytical Foster/Chan across a range of miss distances
     (the analytical Pc falls within the MC 95% confidence interval)
  2. MC Pc is monotonic decreasing with miss distance
  3. Confidence interval shrinks as sample count grows (statistical sanity)
  4. Robustness case: a highly anisotropic (cigar-shaped) covariance where the
     2-D projection is most stressed — MC provides the trustworthy cross-check
"""
import numpy as np
from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn
from phase13_montecarlo import compare_methods, monte_carlo_pc

ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("Monte Carlo Pc self-test")
print("-" * 68)

rp = np.array([7078.0, 0, 0]); vp = np.array([0, 7.5, 0]); speed = np.linalg.norm(vp)
rhat = rp/np.linalg.norm(rp); vhat = vp/speed
cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
Cp = rtn_to_eci_cov(np.diag([0.04**2, 0.2**2, 0.05**2]), rp, vp)

# 1. agreement across miss distances
all_agree = True
for miss in [0.02, 0.05, 0.1]:
    r2 = rp + miss*rhat; v2 = cross*speed
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(8*3600), r2, v2)
    c = compare_methods(rp, vp, Cp, r2, v2, C2, 0.02, n_samples=80000, seed=1)
    all_agree = all_agree and c["agree"]
check("MC agrees with analytical (close range)", all_agree)

# 2. monotonic decreasing
#
# The miss distances have to span the uncertainty for this to test anything. The
# combined radial 1-sigma here is 0.402 km, so the earlier range (0.02-0.25 km)
# sat entirely inside 0.6 sigma: Pc is genuinely flat across it, and the four
# values differed only by Monte Carlo noise. That range passed against the old
# over-tight debris covariance (radial sigma 83 m) and stopped meaning anything
# once secondary_covariance_rtn was recalibrated for TLE-tracked objects.
#
# 0.1 -> 1.0 km spans 0.25 to 2.5 sigma and separates the four Pc values by far
# more than the sampling noise. Verified monotonic for seeds 0-5 at both 60k and
# 200k samples; 200k is used for margin.
pcs = []
for miss in [0.1, 0.5, 0.75, 1.0]:
    r2 = rp + miss*rhat; v2 = cross*speed
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(8*3600), r2, v2)
    pcs.append(monte_carlo_pc(rp, vp, Cp, r2, v2, C2, 0.02, n_samples=200000, seed=2)["pc"])
check("MC Pc decreases with miss", all(np.diff(pcs) <= 1e-6), f"{[f'{p:.2e}' for p in pcs]}")

# 3. CI shrinks with more samples
r2 = rp + 0.05*rhat; v2 = cross*speed
C2 = rtn_to_eci_cov(secondary_covariance_rtn(8*3600), r2, v2)
w_small = monte_carlo_pc(rp, vp, Cp, r2, v2, C2, 0.02, n_samples=5000, seed=3)
w_big = monte_carlo_pc(rp, vp, Cp, r2, v2, C2, 0.02, n_samples=100000, seed=3)
width_s = w_small["ci_high"] - w_small["ci_low"]
width_b = w_big["ci_high"] - w_big["ci_low"]
check("CI shrinks with more samples", width_b < width_s,
      f"5k width={width_s:.2e}  100k width={width_b:.2e}")

# 4. robustness: anisotropic cigar covariance (10:1) — 2-D projection most stressed
print("-" * 68)
print("  ROBUSTNESS CASE (anisotropic 10:1 covariance):")
Cp_aniso = rtn_to_eci_cov(np.diag([0.02**2, 1.0**2, 0.02**2]), rp, vp)  # cigar along-track
r2 = rp + 0.15*rhat; v2 = cross*speed
C2_aniso = rtn_to_eci_cov(np.diag([0.02**2, 1.0**2, 0.02**2]), r2, v2)
c = compare_methods(rp, vp, Cp_aniso, r2, v2, C2_aniso, 0.02, n_samples=100000, seed=4)
print(f"    analytical Pc : {c['analytical_pc']:.3e}")
print(f"    Monte Carlo Pc: {c['monte_carlo_pc']:.3e}  CI=[{c['mc_ci'][0]:.2e}, {c['mc_ci'][1]:.2e}]")
print(f"    agreement     : {c['agree']}")
print(f"    -> MC provides an assumption-free cross-check of the analytical Pc")

print("-" * 68)
print("ALL TESTS PASS" if ok else "SOME TESTS FAILED")
