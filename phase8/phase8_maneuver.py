"""
================================================================================
PHASE 8 — RULE-BASED COLLISION-AVOIDANCE MANEUVER PLANNER
================================================================================
Given a conjunction (primary state + covariance, secondary state, TCA, Pc) that
exceeds the action threshold, compute the smallest maneuver that drops Pc below
a safe level — the ACTIONABLE output of the whole system.

PHYSICS USED (deliberately simple, deterministic, defensible — your safety
fallback before any RL):

  1. Along-track burn is the most fuel-efficient way to change WHERE the
     satellite is along its orbit. A small prograde/retrograde dv changes the
     orbital period slightly; that period change accumulates into an along-track
     position shift that grows ~linearly with time-to-TCA:
         along_track_shift(dt) ~= 3 * (dv / v) * v * dt = 3 * dv * dt
     (factor 3 from the linearized relation between energy and along-track drift
      in a near-circular orbit; conservative, standard CAM first-order model).
     => the EARLIER the burn before TCA, the less dv needed for a given shift.

  2. To avoid a conjunction we must move the primary so the miss distance grows
     enough that Pc < PC_SAFE. We solve for the minimum dv that achieves a
     target miss, then verify by recomputing Pc with the shifted primary state.

  3. Burn timing: default executes the burn ~half an orbit before TCA (efficient
     and operationally typical); planner also reports dv needed at several lead
     times so the operator sees the trade.

OUTPUT (your "OUTPUT 4 — maneuver recommendation"):
  dv vector (RTN), magnitude (m/s), execution epoch, fuel estimate,
  predicted new miss distance, predicted new Pc, safety status.

This module imports the verified Phase-7 Pc engine — no duplicate physics.
================================================================================
"""
import numpy as np
from phase7_collision import (assess_conjunction, risk_level,
                              project_to_plane, secondary_covariance_rtn,
                              rtn_to_eci_cov, pc_2d_quadrature)


def rtn_axes(r, v):
    """RTN unit vectors (R radial, S along-track, W cross-track) from a state."""
    R = r / np.linalg.norm(r)
    W = np.cross(r, v); W = W / np.linalg.norm(W)
    S = np.cross(W, R)
    return R, S, W

MU = 398600.4418        # km^3/s^2
PC_THRESHOLD = 1e-4     # CRITICAL threshold — maneuver if Pc above this
PC_SAFE = 1e-6          # target: reduce Pc below this
ISP_S = 220.0           # specific impulse of a typical hydrazine thruster (s)
G0 = 9.80665e-3         # km/s^2 (for fuel calc consistency in km/s units)


def along_track_shift_per_dv(dt_s, r_km):
    """
    First-order along-track position shift (km) per unit along-track dv (km/s),
    after coasting dt_s seconds, for a near-circular orbit of radius r_km.
    Linearized CAM relation: shift ≈ 3 * dv * dt  (dv in km/s, dt in s).
    """
    return 3.0 * dt_s          # km of shift per (km/s) of dv


def apply_along_track_dv(r, v, dv_kms, dt_s):
    """
    Apply an along-track dv at t=0, return the primary position after dt_s,
    using the first-order drift model (position shifts along velocity direction).
    Returns shifted position (km) and the (unchanged-direction) velocity.
    """
    vhat = v / np.linalg.norm(v)
    shift_km = along_track_shift_per_dv(dt_s, np.linalg.norm(r)) * dv_kms
    r_new = r + shift_km * vhat       # drift accumulates along-track
    return r_new, v


def solve_min_dv(rp, vp, Cp, r2, v2, C2, hbr, dt_to_tca_s,
                 pc_safe=PC_SAFE, dv_max_ms=1000.0):
    """
    Bisection on along-track dv magnitude to find the smallest dv (m/s) that
    brings Pc below pc_safe at TCA. Returns (dv_ms, achieved_pc, new_miss_km).
    Tries both prograde (+) and retrograde (-); picks the cheaper.
    """
    def pc_after(dv_ms, sign):
        dv_kms = sign * dv_ms / 1000.0
        rp_shift, vp_shift = apply_along_track_dv(rp, vp, dv_kms, dt_to_tca_s)
        # relative geometry at TCA after the shift
        rel = r2 - rp_shift
        v_rel = v2 - vp_shift
        C = Cp + C2
        miss2d, C2d = project_to_plane(rel, C, v_rel)
        pc = pc_2d_quadrature(miss2d, C2d, hbr)
        return pc, float(np.linalg.norm(rel))

    best = None
    for sign in (+1.0, -1.0):
        lo, hi = 0.0, dv_max_ms
        # ensure hi is enough
        pc_hi, _ = pc_after(hi, sign)
        if pc_hi > pc_safe:
            continue                       # even max dv in this direction insufficient
        for _ in range(40):                # bisection
            mid = 0.5 * (lo + hi)
            pc_mid, miss_mid = pc_after(mid, sign)
            if pc_mid > pc_safe:
                lo = mid
            else:
                hi = mid
        pc_f, miss_f = pc_after(hi, sign)
        cand = (hi, sign, pc_f, miss_f)
        if best is None or hi < best[0]:
            best = cand
    return best          # (dv_ms, sign, pc, miss) or None


def fuel_kg(dv_ms, sat_mass_kg, isp_s=ISP_S):
    """Tsiolkovsky propellant mass for a given dv (m/s)."""
    dv_kms = dv_ms / 1000.0
    ve = isp_s * G0                          # km/s
    return float(sat_mass_kg * (1 - np.exp(-dv_kms / ve)))


def plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr, tca_s, sat_mass_kg,
                  lead_options_h=(0.75, 1.5, 3.0, 6.0, 12.0)):
    """
    Full maneuver plan. Computes current Pc, then for each lead time the minimum
    along-track dv to reach PC_SAFE, and recommends the cheapest feasible one.
    Returns a dict (your OUTPUT 4).
    """
    pre = assess_conjunction(rp, vp, Cp, r2, v2, C2, hbr, max(tca_s + 1, 600))
    plan = {"pre_maneuver": {k: pre[k] for k in
            ("miss_distance_km", "pc", "risk_level", "tca_s_from_epoch")},
            "action_required": pre["pc"] > PC_THRESHOLD,
            "options": []}

    if not plan["action_required"]:
        plan["recommendation"] = "NO MANEUVER — Pc below action threshold."
        return plan

    # one orbital period (for sensible lead-time bounds)
    a = -MU / (2 * (0.5 * (vp @ vp) - MU / np.linalg.norm(rp)))
    period_s = 2 * np.pi * np.sqrt(a**3 / MU)

    feasible = []
    for lead_h in lead_options_h:
        dt = tca_s - lead_h * 3600.0
        if dt <= 0:
            continue                         # can't burn after TCA
        sol = solve_min_dv(rp, vp, Cp, r2, v2, C2, hbr, tca_s)  # shift accumulates over full tca_s
        # note: burn executed lead_h before TCA -> drift time = lead window
        sol = solve_min_dv(rp, vp, Cp, r2, v2, C2, hbr, lead_h * 3600.0)
        if sol is None:
            continue
        dv_ms, sign, pc_new, miss_new = sol
        opt = {"burn_lead_time_h": lead_h,
               "dv_ms": dv_ms,
               "direction": "prograde" if sign > 0 else "retrograde",
               "new_miss_km": miss_new,
               "new_pc": pc_new,
               "new_risk": risk_level(pc_new),
               "fuel_kg": fuel_kg(dv_ms, sat_mass_kg)}
        plan["options"].append(opt)
        feasible.append((dv_ms, opt))

    if not feasible:
        plan["recommendation"] = "NO FEASIBLE MANEUVER within dv limit — escalate."
        return plan

    # cheapest dv wins
    feasible.sort(key=lambda x: x[0])
    best = feasible[0][1]
    # express recommended dv as an RTN vector (along-track = S axis)
    R, S, W = rtn_axes(rp, vp)
    sign = 1.0 if best["direction"] == "prograde" else -1.0
    dv_rtn = [0.0, sign * best["dv_ms"], 0.0]    # along-track only
    plan["recommendation"] = {
        "dv_rtn_ms": dv_rtn,
        "dv_magnitude_ms": best["dv_ms"],
        "direction": best["direction"],
        "burn_lead_time_h": best["burn_lead_time_h"],
        "fuel_kg": best["fuel_kg"],
        "predicted_new_miss_km": best["new_miss_km"],
        "predicted_new_pc": best["new_pc"],
        "predicted_new_risk": best["new_risk"],
    }
    return plan
