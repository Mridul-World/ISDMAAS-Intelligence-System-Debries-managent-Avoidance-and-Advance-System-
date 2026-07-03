"""
fix_covariance.py  —  fixes the unrealistic Pc (1e-10 instead of ~1e-4).

THE BUG (confirmed by diagnostic):
  secondary_covariance_rtn used base_sigma_km=(0.05, 0.5, 0.05) with radial/cross
  growth of only 0.1 km/day. At a 1.5-day propagation that gives radial/cross
  uncertainty of just ~0.2 km. The encounter plane uses the radial + cross-track
  components, so the in-plane uncertainty was ~0.2 km — far too tight for objects
  tracked only by TLEs (which really have ~km-scale uncertainty in all axes).
  Result: a 1 km miss looked like ~5 sigma -> Pc ~ 1e-10 (wrongly "safe").

THE FIX:
  Raise the radial/cross-track base to 0.3 km and their growth to 0.3 km/day, so
  in-plane uncertainty reaches realistic TLE scale (~0.5-0.8 km at 1-2 days). This
  makes Pc land near 1e-4 for a ~1 km miss — matching SOCRATES-scale screening.
  Along-track is left as-is (it was already realistic and mostly projects out of
  the encounter plane anyway).

Run in phase11\:   python fix_covariance.py
Backs up phase7_collision.py first; verifies the Pc after patching.
"""
import re, shutil, os, sys

FILE = "phase7_collision.py"
if not os.path.exists(FILE):
    sys.exit(f"ERROR: {FILE} not found. Run inside phase11\\.")

src = open(FILE, encoding="utf-8").read()
shutil.copy2(FILE, FILE + ".bak")
print(f"backed up {FILE} -> {FILE}.bak")

orig = src
changes = []

# 1. raise the radial/cross-track base sigma: (0.05, 0.5, 0.05) -> (0.3, 0.5, 0.3)
m = re.search(r"base_sigma_km\s*=\s*\(([^)]*)\)", src)
if m:
    old = m.group(0)
    src = src.replace(old, "base_sigma_km=(0.3, 0.5, 0.3)")
    changes.append(f"{old}  ->  base_sigma_km=(0.3, 0.5, 0.3)")

# 2. raise radial growth: 'sr = sr + 0.1 * days' -> 0.3
src2 = re.sub(r"(sr\s*=\s*sr\s*\+\s*)0\.1(\s*\*\s*days)", r"\g<1>0.3\g<2>", src)
if src2 != src:
    changes.append("radial growth 0.1 -> 0.3 km/day")
    src = src2

# 3. raise cross-track growth: 'sn = sn + 0.1 * days' -> 0.3
src3 = re.sub(r"(sn\s*=\s*sn\s*\+\s*)0\.1(\s*\*\s*days)", r"\g<1>0.3\g<2>", src)
if src3 != src:
    changes.append("cross-track growth 0.1 -> 0.3 km/day")
    src = src3

if src != orig:
    open(FILE, "w", encoding="utf-8").write(src)
    print("\nApplied changes:")
    for c in changes:
        print("  -", c)
else:
    print("\nNo changes made — the covariance function may have a different form.")
    print("Paste me the current secondary_covariance_rtn and I'll give an exact patch.")

# verify
print("\nVerifying Pc for a 1.09 km miss after the fix...")
try:
    import importlib, numpy as np
    sys.modules.pop("phase7_collision", None)
    import phase7_collision as p7
    importlib.reload(p7)
    miss = np.array([1.09, 0.0])
    C2 = p7.secondary_covariance_rtn(259200 / 2)
    C1 = np.diag([0.05, 0.05, 0.05]) ** 2
    pc = p7.pc_2d_quadrature(miss, (C1 + C2)[:2, :2], 0.020)
    print(f"  C2 diag (km): {np.sqrt(np.diag(C2))}")
    print(f"  Pc for 1.09 km miss: {pc:.3e}")
    if 1e-6 < pc < 1e-2:
        print("  SUCCESS — Pc is now in a realistic range (was ~2e-10).")
    else:
        print("  Pc still looks off — paste this output and we'll adjust the values.")
except Exception as e:
    print(f"  verify error: {type(e).__name__}: {e}")

print("\nDone. Restart uvicorn and re-test. Restore with: copy phase7_collision.py.bak phase7_collision.py")
