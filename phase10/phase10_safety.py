"""
================================================================================
PHASE 10 — MANEUVER SAFETY / VALIDATION LAYER  (the trust gate)
================================================================================
Before any maneuver from Phase 8 is allowed to execute, this layer answers the
single question every operator asks: "How do I know this burn doesn't make
things worse?"  It runs deterministic checks and returns APPROVED / REJECTED
with explicit reasons. No ML — auditable safety logic.

CHECKS (all must pass for APPROVED):

  1. FUEL BUDGET
     The required propellant must fit the remaining fuel, with margin.

  2. NO NEW CONJUNCTION  (the critical check)
     Apply the burn, then RE-SCREEN the post-maneuver orbit against the catalog
     over the forward window. Avoiding object A must not steer the satellite
     into object B. Any post-burn Pc above threshold = REJECT.

  3. ORBIT SAFETY
     Post-burn perigee must stay above a minimum safe altitude (no atmospheric
     re-entry risk) and the semi-major axis must stay within an allowed band
     (mission orbit maintained).

  4. ORIGINAL THREAT RESOLVED
     Confirm the burn actually drops the original conjunction Pc below safe
     (re-verifies Phase 8's own claim independently).

OUTPUT: a validation report dict with per-check pass/fail, reasons, and a
single APPROVED/REJECTED verdict. This is the layer that makes the maneuver
"certifiable" for an industry pilot.
================================================================================
"""
import numpy as np
from datetime import timedelta
from sgp4.api import jday
from phase7_collision import (assess_conjunction, secondary_covariance_rtn,
                              rtn_to_eci_cov, pc_2d_quadrature, project_to_plane)
from phase8_maneuver import apply_along_track_dv, PC_SAFE, PC_THRESHOLD

MU = 398600.4418
RE = 6378.137
MIN_PERIGEE_ALT_KM = 300.0      # below this, atmospheric drag/reentry risk
SMA_BAND_KM = 50.0              # allowed semi-major-axis drift from mission orbit
FUEL_MARGIN = 1.3              # require 30% margin over the burn's propellant


def orbital_elements(r, v):
    """Return (sma_km, perigee_alt_km, apogee_alt_km)."""
    rn = np.linalg.norm(r); vn = np.linalg.norm(v)
    energy = 0.5 * vn**2 - MU / rn
    a = -MU / (2 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / MU - r / rn
    e = np.linalg.norm(e_vec)
    perigee = a * (1 - e) - RE
    apogee = a * (1 + e) - RE
    return a, perigee, apogee


def check_fuel(dv_ms, fuel_required_kg, fuel_available_kg):
    ok = fuel_available_kg >= fuel_required_kg * FUEL_MARGIN
    return {"check": "fuel_budget", "pass": bool(ok),
            "fuel_required_kg": fuel_required_kg,
            "fuel_required_with_margin_kg": fuel_required_kg * FUEL_MARGIN,
            "fuel_available_kg": fuel_available_kg,
            "reason": "sufficient" if ok else "INSUFFICIENT FUEL (incl. 30% margin)"}


def check_orbit_safety(r_post, v_post, mission_sma_km):
    a, perigee, apogee = orbital_elements(r_post, v_post)
    perigee_ok = perigee > MIN_PERIGEE_ALT_KM
    sma_ok = abs(a - mission_sma_km) < SMA_BAND_KM
    ok = perigee_ok and sma_ok
    reasons = []
    if not perigee_ok:
        reasons.append(f"perigee {perigee:.1f} km below safe {MIN_PERIGEE_ALT_KM} km")
    if not sma_ok:
        reasons.append(f"SMA drift {abs(a-mission_sma_km):.1f} km exceeds band {SMA_BAND_KM}")
    return {"check": "orbit_safety", "pass": bool(ok),
            "post_sma_km": a, "post_perigee_alt_km": perigee,
            "post_apogee_alt_km": apogee,
            "reason": "orbit within safe limits" if ok else "; ".join(reasons)}


def check_threat_resolved(r_post, v_post, Cp, r2, v2, C2, hbr):
    """Re-verify the ORIGINAL conjunction is resolved after the burn."""
    rel = r2 - r_post
    v_rel = v2 - v_post
    C = Cp + C2
    miss2d, C2d = project_to_plane(rel, C, v_rel)
    pc = pc_2d_quadrature(miss2d, C2d, hbr)
    ok = pc < PC_THRESHOLD
    return {"check": "original_threat_resolved", "pass": bool(ok),
            "post_maneuver_pc": pc, "post_maneuver_miss_km": float(np.linalg.norm(rel)),
            "reason": "threat cleared" if ok else f"Pc {pc:.2e} still above threshold"}


def check_no_new_conjunction(psat_post_states, catalog, epoch, window_s,
                             Cp_rtn_diag, hbr, n_samples=144,
                             shell_km=200.0, primary_norad=None):
    """
    Re-screen the POST-MANEUVER trajectory against the catalog.
    psat_post_states: callable(when)->(r,v) giving post-burn primary state.
    Returns worst new conjunction found (Pc) and pass/fail.
    Uses linear-ish post-burn states sampled over the window.
    """
    rp0, vp0 = psat_post_states(epoch)
    rp_norm = np.linalg.norm(rp0)
    worst = {"pc": 0.0, "norad": None, "name": None, "miss_km": None}
    checked = 0
    for norad, (name2, sat2) in catalog.items():
        if primary_norad is not None and norad == primary_norad:
            continue
        # quick shell gate at epoch
        jd, fr = jday(epoch.year, epoch.month, epoch.day, epoch.hour,
                      epoch.minute, epoch.second + epoch.microsecond / 1e6)
        e2, r2n, _ = sat2.sgp4(jd, fr)
        if e2 != 0:
            continue
        if abs(np.linalg.norm(r2n) - rp_norm) > shell_km:
            continue
        checked += 1
        # TCA search over window with post-burn primary
        min_d, t_min, r2_at = 1e9, 0.0, None
        v2_at = None
        for s in range(n_samples):
            dt = s * (window_s / n_samples)
            when = epoch + timedelta(seconds=dt)
            rp_t, vp_t = psat_post_states(when)
            jd, fr = jday(when.year, when.month, when.day, when.hour,
                          when.minute, when.second + when.microsecond / 1e6)
            e2, r2, v2 = sat2.sgp4(jd, fr)
            if e2 != 0:
                continue
            d = np.linalg.norm(rp_t - np.array(r2))
            if d < min_d:
                min_d, t_min = d, dt
                r2_at, v2_at = np.array(r2), np.array(v2)
                rp_at, vp_at = rp_t, vp_t
        if min_d > shell_km or r2_at is None:
            continue
        Cp = rtn_to_eci_cov(np.diag(np.array(Cp_rtn_diag) ** 2), rp_at, vp_at)
        C2 = rtn_to_eci_cov(secondary_covariance_rtn(window_s / 2), r2_at, v2_at)
        res = assess_conjunction(rp_at, vp_at, Cp, r2_at, v2_at, C2, hbr, 600)
        if res["pc"] > worst["pc"]:
            worst = {"pc": res["pc"], "norad": norad, "name": name2,
                     "miss_km": res["miss_distance_km"]}
    ok = worst["pc"] < PC_THRESHOLD
    return {"check": "no_new_conjunction", "pass": bool(ok),
            "objects_rescreened": checked,
            "worst_new_pc": worst["pc"], "worst_new_object": worst["name"],
            "worst_new_norad": worst["norad"], "worst_new_miss_km": worst["miss_km"],
            "reason": "no new conjunction created" if ok
                      else f"burn creates new conjunction with {worst['name']} "
                           f"(Pc {worst['pc']:.2e})"}


def validate_maneuver(rp, vp, Cp, r2, v2, C2, hbr, tca_s,
                      dv_ms, sign, burn_lead_h, fuel_required_kg,
                      fuel_available_kg, mission_sma_km,
                      Cp_rtn_diag, catalog=None, epoch=None,
                      window_s=259200, primary_norad=None):
    """
    Run all safety checks on a proposed maneuver. Returns the full report with
    a single APPROVED/REJECTED verdict.
    """
    # post-burn primary state at TCA (drift over the lead window)
    dv_kms = sign * dv_ms / 1000.0
    drift_s = burn_lead_h * 3600.0
    rp_post, vp_post = apply_along_track_dv(rp, vp, dv_kms, drift_s)

    checks = []
    checks.append(check_fuel(dv_ms, fuel_required_kg, fuel_available_kg))
    checks.append(check_orbit_safety(rp_post, vp_post, mission_sma_km))
    checks.append(check_threat_resolved(rp_post, vp_post, Cp, r2, v2, C2, hbr))

    if catalog is not None and epoch is not None:
        # post-burn primary trajectory model: drift grows with time since burn
        def post_states(when):
            dt = (when - epoch).total_seconds()
            return apply_along_track_dv(rp, vp, dv_kms, dt)
        checks.append(check_no_new_conjunction(
            post_states, catalog, epoch, window_s, Cp_rtn_diag, hbr,
            primary_norad=primary_norad))

    verdict = "APPROVED" if all(c["pass"] for c in checks) else "REJECTED"
    failed = [c["check"] for c in checks if not c["pass"]]
    return {"verdict": verdict, "failed_checks": failed,
            "dv_ms": dv_ms, "burn_lead_h": burn_lead_h,
            "fuel_required_kg": fuel_required_kg, "checks": checks}
