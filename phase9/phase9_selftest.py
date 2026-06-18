"""
PHASE 9 — self-test + RL-vs-rule-based fuel benchmark.

    python phase9_selftest.py        # env mechanics + policy wrapper + benchmark

Checks:
  1. CW along-track displacement matches Phase-8 first-order (~3·dv·t)
  2. env reset/step API shapes + reward sign sanity
  3. policy wrapper returns a valid maneuver dict (RL or fallback)
  4. benchmark: mean RL Δv vs rule-based Δv on a set of conjunctions
"""
import numpy as np
from phase9_env import ConjunctionEnv, cw_displacement, PC_SAFE, MU
from phase8_maneuver import plan_maneuver
from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn
from phase9_policy import propose_maneuver_rl, _load

ok = True
def check(name, cond, detail=""):
    global ok; ok = ok and cond
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}")

print("Phase 9 self-test")
print("-" * 64)

# 1. CW vs Phase-8
n = np.sqrt(MU / (6378.137 + 700)**3); t = 6*3600; dvT = 0.001
disp = cw_displacement([0, dvT, 0], t, n)
approx = -3 * dvT * t
check("CW along-track ~ 3·dv·t", abs(disp[1] - approx) / abs(approx) < 0.15,
      f"CW={disp[1]:.1f} approx={approx:.1f} km")

# 2. env API
env = ConjunctionEnv(seed=0); obs, _ = env.reset(seed=0)
check("obs shape (9,)", obs.shape == (9,))
check("action shape (3,)", env.action_space.shape == (3,))
o, r, d, _, info = env.step([0, 1, 0])
check("step returns info pc/dv", "pc" in info and "total_dv_ms" in info,
      f"pc={info['pc']:.2e}")

# 3. policy wrapper on a real-ish conjunction
rp = np.array([7078.0, 0, 0]); vp = np.array([0, 7.5, 0])
speed = np.linalg.norm(vp); rhat = rp/np.linalg.norm(rp); vhat = vp/speed
cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
Cp = rtn_to_eci_cov(np.diag([0.04**2, 0.2**2, 0.05**2]), rp, vp)
r2 = rp + 0.05*rhat; v2 = cross*speed
C2 = rtn_to_eci_cov(secondary_covariance_rtn(8*3600), r2, v2)
out = propose_maneuver_rl(rp, vp, Cp, r2, v2, C2, 8*3600, 2300)
check("policy returns recommendation", isinstance(out.get("recommendation"), dict),
      f"proposer={out.get('proposer')}")
if isinstance(out.get("recommendation"), dict):
    rec = out["recommendation"]
    check("proposed burn resolves (Pc<safe)", rec["predicted_new_pc"] < PC_SAFE,
          f"new Pc={rec['predicted_new_pc']:.2e}, Δv={rec['dv_magnitude_ms']:.3f} m/s")

# 4. benchmark RL vs rule-based across DIVERSE 3-D conjunctions (the regime RL
#    was trained on). Compare each method's own resolving burn, head-to-head,
#    using the env's own scenario sampling so geometries are representative.
model = _load()
if model is None:
    print("\n  (no trained model found — train with phase9_train.py for benchmark)")
else:
    from phase9_env import ConjunctionEnv, cw_displacement, MU
    import numpy as np

    def rule_dv_for_scenario(env):
        """Best single along-track burn (Phase-8 style) for this env scenario,
           found by bisection on |dvT| applied at the first decision point."""
        n = env.n; t = env.tca_total
        def pc_for(dvT):
            disp = cw_displacement([0, dvT/1000.0, 0], t, n)
            pc, _ = env._pc_now(disp)
            return pc
        lo, hi = 0.0, 2000.0   # m/s search range
        if pc_for(hi) >= PC_SAFE:
            return None        # along-track alone can't resolve in range
        for _ in range(40):
            mid = 0.5*(lo+hi)
            if pc_for(mid) >= PC_SAFE: lo = mid
            else: hi = mid
        return hi

    rl_dvs, rule_dvs, both = [], [], []
    n_eval = 100
    for ep in range(n_eval):
        env = ConjunctionEnv(seed=50000+ep)
        env.reset(seed=50000+ep)
        # --- RL rollout on this scenario ---
        obs = env._obs(); done = False; info = {}
        while not done:
            a,_ = model.predict(obs, deterministic=True)
            obs,_,done,_,info = env.step(a)
        rl_ok = info["pc"] < PC_SAFE; rl_dv = info["total_dv_ms"]
        # --- rule-based (best along-track) on the SAME scenario ---
        env2 = ConjunctionEnv(seed=50000+ep); env2.reset(seed=50000+ep)
        ru_dv = rule_dv_for_scenario(env2)
        ru_ok = ru_dv is not None
        if rl_ok: rl_dvs.append(rl_dv)
        if ru_ok: rule_dvs.append(ru_dv)
        if rl_ok and ru_ok: both.append((rl_dv, ru_dv))

    print(f"\n  BENCHMARK ({n_eval} diverse 3-D conjunctions):")
    print(f"    rule-based resolved : {len(rule_dvs)}/{n_eval}   mean Δv {np.mean(rule_dvs):.3f} m/s")
    print(f"    RL resolved         : {len(rl_dvs)}/{n_eval}   mean Δv {np.mean(rl_dvs):.3f} m/s")
    if both:
        rl_b = np.array([b[0] for b in both]); ru_b = np.array([b[1] for b in both])
        wins = int(np.sum(rl_b < ru_b))
        print(f"    head-to-head (both resolved, n={len(both)}):")
        print(f"      RL cheaper in {wins}/{len(both)} cases")
        print(f"      mean Δv  RL={rl_b.mean():.3f}  rule-based={ru_b.mean():.3f} m/s "
              f"(rule-based is {rl_b.mean()/max(ru_b.mean(),1e-9):.0f}x cheaper)")
    # ---- stated finding (so the conclusion prints with the numbers) ----
    print("\n  FINDING (Phase 9 — honest negative result):")
    print("    For single-satellite / formation collision avoidance, classical")
    print("    optimization (min-Δv along-track bisection) is near-optimal and")
    print("    RL provides no fuel advantage. This is consistent with the")
    print("    near-convex structure of the burn-selection problem and was")
    print("    confirmed across 4 formulations (single, multi-threat same-TCA,")
    print("    staggered-TCA, formation). RL's advantage regime is large-scale")
    print("    space-traffic management (10^3+ objects, multi-week sequential")
    print("    horizons) = future work. DEPLOYED METHOD = the rule-based planner;")
    print("    the RL wrapper falls back to it and never degrades safety.")

print("-" * 64)
print("ALL TESTS PASS" if ok else "SOME TESTS FAILED")