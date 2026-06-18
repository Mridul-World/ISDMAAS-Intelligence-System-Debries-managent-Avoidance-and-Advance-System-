"""
PHASE 9 — policy wrapper. Turns the trained PPO policy into a maneuver proposer
that drops into the Phase-11 API's propose_maneuver() slot, with a rule-based
(Phase-8) fallback if the model is missing or its burn fails safety.

The RL policy proposes a 3-axis Δv sequence; we convert the net effect to an
equivalent single recommended burn (magnitude, direction, lead time) in the same
schema plan_maneuver() returns, so the API, console, and Phase-10 validation all
work unchanged.

Usage in the API:
    from phase9_policy import propose_maneuver_rl
    # replace: return plan_maneuver(...)
    # with:    return propose_maneuver_rl(rp, vp, Cp, r2, v2, C2, tca_s, mass)
"""
import os
import numpy as np
from phase9_env import ConjunctionEnv, cw_displacement, PC_SAFE, PC_THRESHOLD, MU
from phase8_maneuver import plan_maneuver        # rule-based fallback

_model = None
def _load():
    global _model
    if _model is not None:
        return _model
    try:
        from stable_baselines3 import PPO
        if os.path.exists("model/phase9_ppo.zip"):
            _model = PPO.load("model/phase9_ppo", device="cpu")
    except Exception:
        _model = None
    return _model


def propose_maneuver_rl(rp, vp, Cp, r2, v2, C2, tca_s, sat_mass_kg,
                        hbr_km=0.020):
    """
    RL maneuver proposer. Returns the same dict shape as plan_maneuver():
      {pre_maneuver, action_required, options, recommendation}
    Falls back to the rule-based planner if no model or RL fails to resolve.
    """
    model = _load()
    rule = plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr=hbr_km,
                         tca_s=tca_s, sat_mass_kg=sat_mass_kg)
    if model is None or not rule["action_required"]:
        rule["proposer"] = "rule_based" if model is None else "rule_based_no_action"
        return rule

    # build an env instance matching this real conjunction to roll out the policy
    env = ConjunctionEnv(hbr_km=hbr_km, sat_mass_kg=sat_mass_kg)
    env.reset()
    # overwrite the sampled scenario with the real one (RTN frame)
    r = np.linalg.norm(rp)
    env.n = np.sqrt(MU / r**3)
    # relative position/velocity in RTN of the primary
    R = rp / r; W = np.cross(rp, vp); W /= np.linalg.norm(W); S = np.cross(W, R)
    R2 = np.stack([R, S, W])                       # ECI->RTN rows
    rel = R2 @ (r2 - rp)
    vrel = R2 @ (v2 - vp)
    env.miss0 = rel; env.vrel = vrel
    env.Cp_rtn = R2 @ Cp @ R2.T
    env.C2_rtn = R2 @ C2 @ R2.T
    env.tca_total = max(tca_s, 600.0)
    env.disp = np.zeros(3); env.total_dv = 0.0; env.step_i = 0; env.fuel_frac = 1.0

    # roll out the policy, accumulate the net Δv (RTN)
    net_dv = np.zeros(3)
    obs = env._obs(); done = False; info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        dv = np.clip(a, -1, 1) * env.max_dv
        net_dv += dv
        obs, _, done, _, info = env.step(a)

    rl_pc = info["pc"]; rl_dv_ms = float(np.linalg.norm(net_dv) * 1000)
    # decide direction label from dominant axis
    axis = int(np.argmax(np.abs(net_dv)))
    dirlabel = {0: ("radial-out" if net_dv[0] > 0 else "radial-in"),
                1: ("prograde" if net_dv[1] > 0 else "retrograde"),
                2: ("cross-track+" if net_dv[2] > 0 else "cross-track-")}[axis]

    # if RL fails to resolve but rule-based succeeds, fall back
    rule_ok = isinstance(rule.get("recommendation"), dict) and \
              rule["recommendation"]["predicted_new_pc"] < PC_SAFE
    if rl_pc >= PC_SAFE and rule_ok:
        rule["proposer"] = "rule_based_fallback"
        rule["rl_attempt"] = {"rl_pc": rl_pc, "rl_dv_ms": rl_dv_ms}
        return rule

    # choose the cheaper of (RL, rule) among those that resolve
    use_rl = True
    if rule_ok and rl_pc < PC_SAFE:
        use_rl = rl_dv_ms <= rule["recommendation"]["dv_magnitude_ms"]

    if not use_rl:
        rule["proposer"] = "rule_based_cheaper"
        rule["rl_attempt"] = {"rl_pc": rl_pc, "rl_dv_ms": rl_dv_ms}
        return rule

    # fuel for the RL burn
    ve = ISP = 220.0 * 9.80665e-3
    fuel = float(sat_mass_kg * (1 - np.exp(-(rl_dv_ms / 1000) / ve)))
    out = dict(rule)                                # reuse pre_maneuver/action_required
    out["proposer"] = "rl_ppo"
    out["recommendation"] = {
        "dv_rtn_ms": (net_dv * 1000).tolist(),
        "dv_magnitude_ms": rl_dv_ms,
        "direction": dirlabel,
        "burn_lead_time_h": round(env.tca_total / 3600, 2),
        "fuel_kg": fuel,
        "predicted_new_miss_km": info["miss_km"],
        "predicted_new_pc": rl_pc,
        "predicted_new_risk": "NOMINAL" if rl_pc < 1e-7 else
                              ("ELEVATED" if rl_pc < 1e-5 else
                               ("HIGH" if rl_pc < 1e-4 else "CRITICAL")),
    }
    if isinstance(rule.get("recommendation"), dict):
        out["rule_based_comparison"] = {
            "dv_ms": rule["recommendation"]["dv_magnitude_ms"],
            "fuel_kg": rule["recommendation"]["fuel_kg"]}
    return out
