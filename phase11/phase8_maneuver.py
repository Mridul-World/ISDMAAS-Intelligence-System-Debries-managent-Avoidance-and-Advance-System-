"""
================================================================================
COLLISION-AVOIDANCE MANEUVER PLANNER  (deterministic, rule-based)
================================================================================
Given a conjunction (primary state + covariance, secondary state, TCA, Pc) that
exceeds the action threshold, compute the smallest maneuver that drops Pc below
a safe level — the actionable output of the whole system.

PHYSICS
-------
A tangential (along-track) burn is the cheapest way to change WHERE a satellite
is along its orbit. For a near-circular orbit, a tangential dv changes the
semi-major axis and therefore the period:

    da    = 2 dv / n                       (n = mean motion, rad/s)
    dn    = -3/2 * n * da / a = -3 dv / a
    ds(t) = a * dn * t = -3 dv t           (secular along-track drift)

Two consequences the earlier implementation had backwards or missing:

  * SIGN. A PROGRADE burn raises the orbit, which LENGTHENS the period, so the
    satellite falls BEHIND where it would otherwise have been. The drift is
    retrograde for a prograde burn. The old model displaced the satellite
    forwards for a positive dv. Because the search tried both signs the final
    number came out similar, but the reported burn DIRECTION was inverted — an
    operator following it would have burned the wrong way.

  * VELOCITY. The old model moved the position and returned the velocity
    unchanged. Every downstream orbit check then evaluated the post-burn orbit
    from a state that had not been burned, so the semi-major-axis and perigee
    checks were measuring round-off rather than the maneuver. The post-burn
    velocity now actually carries the dv.

  * RADIAL OFFSET. The raised orbit also sits ~da higher. That is roughly fifty
    times smaller than the along-track drift over a typical lead time, but it is
    free to include and it makes the predicted post-burn miss match a numerical
    propagation more closely.

OUTPUT: dv vector (RTN), magnitude (m/s), execution lead time, propellant,
predicted new miss distance, predicted new Pc, per-option trade table.

This module imports the verified Pc engine — no duplicate physics.
================================================================================
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np

from phase7_collision import (
    assess_conjunction,
    pc_2d_quadrature,
    project_to_plane,
    risk_level,
)

MU = 398600.4418          # km^3/s^2
PC_THRESHOLD = 1e-4       # maneuver if Pc is at or above this
PC_SAFE = 1e-6            # target: reduce Pc below this
ISP_S = 220.0             # specific impulse of a typical hydrazine thruster (s)
G0_KM_S2 = 9.80665e-3     # standard gravity in km/s^2
DEFAULT_LEAD_OPTIONS_H = (0.75, 1.5, 3.0, 6.0, 12.0)
DV_MAX_MS = 1000.0        # search ceiling; beyond this the answer is "escalate"
MIN_LEAD_TIME_H = 1.0 / 60.0   # one minute: below this no operator can act


def lead_time_options(tca_s: float, ladder=DEFAULT_LEAD_OPTIONS_H) -> list:
    """
    Burn lead times worth evaluating for a conjunction `tca_s` seconds away.

    The standard ladder starts at 45 minutes, so a conjunction detected inside
    that window used to produce no feasible option at all and the planner
    reported "escalate" — for a CRITICAL conjunction, at the moment an operator
    most needs a number. Short-notice conjunctions now also get fractions of the
    remaining time, so a burn eleven minutes out is planned as a burn eleven
    minutes out. It costs more propellant, and the trade table shows that.
    """
    tca_h = max(float(tca_s), 0.0) / 3600.0
    options = {round(h, 4) for h in ladder if MIN_LEAD_TIME_H <= h < tca_h}
    for fraction in (0.9, 0.75, 0.5, 0.25):
        lead = tca_h * fraction
        if lead >= MIN_LEAD_TIME_H:
            options.add(round(lead, 4))
    return sorted(options)


def rtn_axes(r: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RTN unit vectors (R radial, S along-track, W cross-track) from a state."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    R = r / np.linalg.norm(r)
    W = np.cross(r, v)
    W = W / np.linalg.norm(W)
    S = np.cross(W, R)
    return R, S, W


def mean_motion_rad_s(r: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """Osculating (semi-major axis km, mean motion rad/s) from a state."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    energy = 0.5 * float(v @ v) - MU / float(np.linalg.norm(r))
    if energy >= 0.0:
        raise ValueError("state is not on a bound orbit")
    a = -MU / (2.0 * energy)
    return a, float(np.sqrt(MU / a ** 3))


def along_track_shift_per_dv(dt_s: float, r_km: float = 0.0) -> float:
    """
    Magnitude of the secular along-track drift (km) per unit tangential dv
    (km/s) after coasting dt_s seconds: |ds| = 3 dv dt. `r_km` is accepted for
    call compatibility; the semi-major-axis dependence cancels at first order.
    """
    return 3.0 * float(dt_s)


def cw_impulse_response(
    dv_kms: float, dt_s: float, n_rad_s: float
) -> Tuple[float, float]:
    """
    Displacement from a tangential impulse, from the Clohessy-Wiltshire equations.

        radial(t)      = (2 dv / n) (1 - cos n t)
        along-track(t) = (4 dv / n) sin(n t) - 3 dv t

    Both terms matter, and getting the radial one wrong is not a small error. A
    constant `da = 2 dv / n` radial offset — the obvious-looking simplification —
    says the satellite jumps to its new altitude the instant the thruster fires.
    It does not: the burn point stays on the orbit and the altitude difference
    grows over the following half revolution. For a burn a few minutes before
    TCA that overstates the achieved separation by roughly the whole radial term,
    which showed up as every lead time in the trade table needing an identical
    dv — the giveaway that the answer had stopped depending on the lead time at
    all.

    Returns (radial_km, along_track_km), signed in the local orbital frame:
    radial positive outward, along-track positive in the direction of motion.
    """
    dt = float(dt_s)
    if n_rad_s <= 0.0:
        return 0.0, -3.0 * dv_kms * dt
    phase = n_rad_s * dt
    radial = (2.0 * dv_kms / n_rad_s) * (1.0 - math.cos(phase))
    along = (4.0 * dv_kms / n_rad_s) * math.sin(phase) - 3.0 * dv_kms * dt
    return radial, along


def apply_along_track_dv(
    r: np.ndarray, v: np.ndarray, dv_kms: float, dt_s: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    First-order effect of a tangential burn of `dv_kms` executed now, evaluated
    `dt_s` later. Returns the displaced position AND the post-burn velocity.
    """
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    speed = float(np.linalg.norm(v))
    if speed <= 0.0:
        return r.copy(), v.copy()

    vhat = v / speed
    rhat = r / float(np.linalg.norm(r))
    v_post = v + dv_kms * vhat

    try:
        _, n = mean_motion_rad_s(r, v)
    except ValueError:
        n = 0.0
    radial_km, along_km = cw_impulse_response(dv_kms, dt_s, n)
    return r + along_km * vhat + radial_km * rhat, v_post


def fuel_kg(dv_ms: float, sat_mass_kg: float, isp_s: float = ISP_S) -> float:
    """Tsiolkovsky propellant mass for a given dv (m/s)."""
    ve = isp_s * G0_KM_S2                            # km/s
    if ve <= 0.0:
        raise ValueError("specific impulse must be positive")
    return float(sat_mass_kg * (1.0 - np.exp(-(dv_ms / 1000.0) / ve)))


def _pc_after_burn(
    rp: np.ndarray,
    vp: np.ndarray,
    Cp: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    C2: np.ndarray,
    hbr: float,
    dv_ms: float,
    sign: float,
    drift_s: float,
) -> Tuple[float, float]:
    """Pc and miss distance at TCA after a burn of dv_ms in the given direction."""
    rp_post, vp_post = apply_along_track_dv(rp, vp, sign * dv_ms / 1000.0, drift_s)
    rel = np.asarray(r2, dtype=float) - rp_post
    v_rel = np.asarray(v2, dtype=float) - vp_post
    miss = float(np.linalg.norm(rel))
    if float(np.linalg.norm(v_rel)) <= 0.0:
        return 1.0, miss
    miss_2d, C_2d = project_to_plane(rel, np.asarray(Cp) + np.asarray(C2), v_rel)
    return pc_2d_quadrature(miss_2d, C_2d, hbr), miss


def solve_min_dv(
    rp: np.ndarray,
    vp: np.ndarray,
    Cp: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    C2: np.ndarray,
    hbr: float,
    dt_to_tca_s: float,
    pc_safe: float = PC_SAFE,
    dv_max_ms: float = DV_MAX_MS,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Smallest tangential dv (m/s) that brings Pc below `pc_safe` at TCA.

    Bisection on dv magnitude, tried in both directions; the cheaper feasible one
    wins. Pc is not strictly monotone in dv — moving the satellite through the
    secondary's position on the way out makes it rise before it falls — so the
    upper bracket is validated by an outward scan rather than a single probe at
    dv_max. Without that scan a direction whose Pc happens to peak at dv_max was
    silently discarded as infeasible.

    Returns (dv_ms, sign, achieved_pc, new_miss_km) or None.
    """
    best: Optional[Tuple[float, float, float, float]] = None

    for sign in (1.0, -1.0):
        # Find the smallest dv on a coarse ladder that already reaches safety;
        # that value is a valid upper bracket for the bisection.
        upper = None
        ladder = np.geomspace(1e-3, dv_max_ms, 40)
        for dv in ladder:
            pc, _ = _pc_after_burn(rp, vp, Cp, r2, v2, C2, hbr, dv, sign, dt_to_tca_s)
            if pc < pc_safe:
                upper = float(dv)
                break
        if upper is None:
            continue

        lo, hi = 0.0, upper
        for _ in range(50):
            mid = 0.5 * (lo + hi)
            pc_mid, _ = _pc_after_burn(
                rp, vp, Cp, r2, v2, C2, hbr, mid, sign, dt_to_tca_s
            )
            if pc_mid >= pc_safe:
                lo = mid
            else:
                hi = mid
        pc_final, miss_final = _pc_after_burn(
            rp, vp, Cp, r2, v2, C2, hbr, hi, sign, dt_to_tca_s
        )
        if best is None or hi < best[0]:
            best = (hi, sign, pc_final, miss_final)

    return best


def plan_maneuver(
    rp: np.ndarray,
    vp: np.ndarray,
    Cp: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    C2: np.ndarray,
    hbr: float,
    tca_s: float,
    sat_mass_kg: float,
    lead_options_h: Optional[Sequence[float]] = None,
    states_at_tca: bool = True,
) -> dict:
    """
    Full maneuver plan: current Pc, then for each lead time the minimum
    tangential dv that reaches PC_SAFE, recommending the cheapest feasible one.

    `states_at_tca` says the caller has already propagated both objects to the
    time of closest approach, which is what every path in this service does. In
    that case the pre-maneuver assessment uses the geometry as given instead of
    re-solving for a TCA it already knows.
    """
    pre = assess_conjunction(
        rp, vp, Cp, r2, v2, C2, hbr,
        t_window_s=max(abs(tca_s), 600.0),
        tca_already_refined=states_at_tca,
    )
    pre_pc = pre["pc"]
    plan = {
        "pre_maneuver": {
            k: pre[k]
            for k in ("miss_distance_km", "pc", "risk_level", "tca_s_from_epoch")
        },
        "action_required": bool(pre_pc is not None and pre_pc >= PC_THRESHOLD),
        "options": [],
    }

    if pre_pc is None:
        plan["recommendation"] = (
            "NO MANEUVER — Pc is not computable for this geometry "
            "(relative speed below the short-encounter limit); escalate for a "
            "Monte-Carlo assessment."
        )
        return plan

    if not plan["action_required"]:
        plan["recommendation"] = "NO MANEUVER — Pc below action threshold."
        return plan

    if lead_options_h is None:
        lead_options_h = lead_time_options(tca_s)

    feasible = []
    for lead_h in lead_options_h:
        drift_s = float(lead_h) * 3600.0
        if drift_s <= 0.0 or drift_s > abs(tca_s):
            # Cannot burn after TCA, or earlier than the conjunction is known.
            continue
        solution = solve_min_dv(rp, vp, Cp, r2, v2, C2, hbr, drift_s)
        if solution is None:
            continue
        dv_ms, sign, pc_new, miss_new = solution
        option = {
            "burn_lead_time_h": float(lead_h),
            "dv_ms": dv_ms,
            "direction": "prograde" if sign > 0 else "retrograde",
            "new_miss_km": miss_new,
            "new_pc": pc_new,
            "new_risk": risk_level(pc_new),
            "fuel_kg": fuel_kg(dv_ms, sat_mass_kg),
        }
        plan["options"].append(option)
        feasible.append((dv_ms, option))

    if not feasible:
        plan["recommendation"] = (
            f"NO FEASIBLE MANEUVER within the {DV_MAX_MS:.0f} m/s search limit at "
            "any available lead time — escalate."
        )
        return plan

    feasible.sort(key=lambda item: item[0])
    best = feasible[0][1]
    sign = 1.0 if best["direction"] == "prograde" else -1.0
    plan["recommendation"] = {
        # RTN vector: along-track component only (index 1 = S axis).
        "dv_rtn_ms": [0.0, sign * best["dv_ms"], 0.0],
        "dv_magnitude_ms": best["dv_ms"],
        "direction": best["direction"],
        "burn_lead_time_h": best["burn_lead_time_h"],
        "fuel_kg": best["fuel_kg"],
        "predicted_new_miss_km": best["new_miss_km"],
        "predicted_new_pc": best["new_pc"],
        "predicted_new_risk": best["new_risk"],
    }
    return plan
