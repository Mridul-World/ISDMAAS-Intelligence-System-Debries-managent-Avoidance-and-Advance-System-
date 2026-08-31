"""
================================================================================
MANEUVER SAFETY / VALIDATION LAYER  (the trust gate)
================================================================================
Before any planned maneuver is allowed to execute, this layer answers the single
question every operator asks: "How do I know this burn doesn't make things
worse?" It runs deterministic checks and returns APPROVED / REJECTED with
explicit reasons. No ML — auditable safety logic.

CHECKS (all must pass for APPROVED)

  1. FUEL BUDGET
     The required propellant must fit the remaining fuel, with margin.

  2. NO NEW CONJUNCTION  (the critical check)
     Apply the burn, then RE-SCREEN the post-maneuver orbit against the catalog
     over the forward window. Avoiding object A must not steer the satellite
     into object B. Any post-burn Pc above threshold rejects the maneuver.

  3. ORBIT SAFETY
     Post-burn perigee must stay above a minimum safe altitude and the
     semi-major axis must stay within the mission band.

  4. ORIGINAL THREAT RESOLVED
     Confirm the burn actually drops the original conjunction Pc, independently
     of the planner's own claim.

WHAT CHANGED, AND WHY IT MATTERED
---------------------------------
* The post-burn trajectory used to be a straight-line drift away from a single
  state vector, extrapolated for up to three days. Three days of straight-line
  motion is not an orbit, so the re-screen was comparing real catalog objects
  against a trajectory that had left the planet. The post-burn orbit is now
  modelled as the primary's own SGP4 trajectory evaluated at a time offset — the
  along-track drift of a tangential burn is exactly equivalent to arriving early
  or late — plus the semi-major-axis change as a radial offset. That is a real
  orbit, and it is as accurate as SGP4 itself.

* The re-screen sampled 144 points across three days: one sample every half
  hour, during which two objects close 27 000 km. Nothing it reported was a
  conjunction. Screening now uses the vectorized coarse grid plus Brent
  refinement, so a post-burn conjunction is found to millisecond accuracy.

* The orbit-safety check evaluated the post-burn semi-major axis from a state
  whose velocity had never been changed by the burn, so it measured round-off.
  It now uses the actual post-burn velocity.
================================================================================
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from isdmaas_core import astrodynamics as astro
from phase7_collision import (
    assess_conjunction,
    pc_2d_quadrature,
    project_to_plane,
    rtn_to_eci_cov,
    secondary_covariance_rtn,
)
from phase8_maneuver import PC_THRESHOLD, apply_along_track_dv, cw_impulse_response

MU = 398600.4418
RE = 6378.137
MIN_PERIGEE_ALT_KM = 300.0      # below this, atmospheric drag / reentry risk
SMA_BAND_KM = 50.0              # allowed semi-major-axis drift from mission orbit
FUEL_MARGIN = 1.3               # require 30% margin over the burn's propellant
RESCREEN_GATE_KM = 25.0         # report gate for post-burn conjunctions


def orbital_elements(r: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
    """Return (sma_km, perigee_alt_km, apogee_alt_km)."""
    elements = astro.elements_from_state(r, v)
    return elements.sma_km, elements.perigee_alt_km, elements.apogee_alt_km


# ----------------------------------------------------------- post-burn orbit
def post_burn_state_fn(
    primary_sat, epoch: datetime, dv_kms: float, burn_offset_s: float
) -> astro.StateFn:
    """
    A state function for the primary AFTER a tangential burn.

    A tangential dv displaces the satellite along track by the Clohessy-Wiltshire
    response. Being displaced along track is the same as being at the position
    the unperturbed orbit reaches ds/|v| seconds later, so the post-burn
    trajectory is the primary's own SGP4 solution evaluated at a shifted time —
    a genuine orbit rather than a linear extrapolation from a single state — plus
    the CW radial term.
    """
    base = astro.satrec_state_fn(primary_sat, epoch)

    def state(offset_s: float):
        if offset_s <= burn_offset_s:
            return base(offset_s)
        r_ref, v_ref = base(offset_s)
        if r_ref is None:
            return None, None
        speed = float(np.linalg.norm(v_ref))
        if speed <= 0.0:
            return r_ref, v_ref
        try:
            n = 2.0 * math.pi / astro.elements_from_state(r_ref, v_ref).period_s
        except ValueError:
            n = 0.0
        radial_km, along_km = cw_impulse_response(
            dv_kms, offset_s - burn_offset_s, n
        )
        r, v = base(offset_s + along_km / speed)
        if r is None:
            return None, None
        rhat = r / float(np.linalg.norm(r))
        vhat = v / float(np.linalg.norm(v))
        return r + radial_km * rhat, v + dv_kms * vhat

    return state


def post_burn_track(
    primary_sat,
    epoch: datetime,
    offsets_s: np.ndarray,
    dv_kms: float,
    burn_offset_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version of `post_burn_state_fn` over a whole time grid.

    Used for the coarse re-screen, where evaluating the state function point by
    point across the catalog would dominate the runtime.
    """
    r_ref, v_ref, ok_ref = astro.propagate_series(primary_sat, epoch, offsets_s)
    if ok_ref.sum() < 3:
        return r_ref, ok_ref
    speed = float(np.nanmedian(np.linalg.norm(v_ref[ok_ref], axis=1)))
    radius = float(np.nanmedian(np.linalg.norm(r_ref[ok_ref], axis=1)))
    if speed <= 0.0 or radius <= 0.0:
        return r_ref, ok_ref

    # Mean motion from the near-circular relation n = v / r. The coarse screen
    # only needs the displacement to a few metres; the refinement stage uses the
    # exact per-sample elements.
    n = speed / radius
    drift = np.maximum(offsets_s - burn_offset_s, 0.0)
    phase = n * drift
    radial = (2.0 * dv_kms / n) * (1.0 - np.cos(phase))
    along = (4.0 * dv_kms / n) * np.sin(phase) - 3.0 * dv_kms * drift
    radial = np.where(drift > 0.0, radial, 0.0)
    along = np.where(drift > 0.0, along, 0.0)

    r, _, ok = astro.propagate_series(primary_sat, epoch, offsets_s + along / speed)
    with np.errstate(invalid="ignore", divide="ignore"):
        r_norm = np.linalg.norm(r, axis=1, keepdims=True)
        rhat = np.divide(r, r_norm, out=np.zeros_like(r), where=r_norm > 0)
    return r + radial[:, None] * rhat, ok


# --------------------------------------------------------------------- checks
def check_fuel(
    dv_ms: float, fuel_required_kg: float, fuel_available_kg: float
) -> Dict:
    ok = fuel_available_kg >= fuel_required_kg * FUEL_MARGIN
    return {
        "check": "fuel_budget",
        "pass": bool(ok),
        "fuel_required_kg": fuel_required_kg,
        "fuel_required_with_margin_kg": fuel_required_kg * FUEL_MARGIN,
        "fuel_available_kg": fuel_available_kg,
        "reason": "sufficient"
        if ok
        else f"insufficient fuel: need {fuel_required_kg * FUEL_MARGIN:.4f} kg "
             f"including the {int((FUEL_MARGIN - 1) * 100)}% margin, "
             f"{fuel_available_kg:.4f} kg available",
    }


def check_orbit_safety(
    r_post: np.ndarray, v_post: np.ndarray, mission_sma_km: float
) -> Dict:
    try:
        a, perigee, apogee = orbital_elements(r_post, v_post)
    except ValueError as exc:
        return {
            "check": "orbit_safety",
            "pass": False,
            "reason": f"post-burn state is not a bound orbit ({exc})",
        }
    perigee_ok = perigee > MIN_PERIGEE_ALT_KM
    sma_ok = abs(a - mission_sma_km) < SMA_BAND_KM
    reasons = []
    if not perigee_ok:
        reasons.append(
            f"perigee {perigee:.1f} km is below the safe floor {MIN_PERIGEE_ALT_KM} km"
        )
    if not sma_ok:
        reasons.append(
            f"semi-major axis moves {abs(a - mission_sma_km):.1f} km, outside the "
            f"{SMA_BAND_KM} km mission band"
        )
    return {
        "check": "orbit_safety",
        "pass": bool(perigee_ok and sma_ok),
        "post_sma_km": a,
        "post_perigee_alt_km": perigee,
        "post_apogee_alt_km": apogee,
        "reason": "orbit within safe limits" if not reasons else "; ".join(reasons),
    }


def check_threat_resolved(
    r_post: np.ndarray,
    v_post: np.ndarray,
    Cp: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    C2: np.ndarray,
    hbr: float,
) -> Dict:
    """Re-verify the ORIGINAL conjunction is resolved after the burn."""
    rel = np.asarray(r2, dtype=float) - np.asarray(r_post, dtype=float)
    v_rel = np.asarray(v2, dtype=float) - np.asarray(v_post, dtype=float)
    miss = float(np.linalg.norm(rel))
    if float(np.linalg.norm(v_rel)) <= 0.0:
        return {
            "check": "original_threat_resolved",
            "pass": False,
            "post_maneuver_miss_km": miss,
            "reason": "post-burn relative velocity is zero; Pc is not computable",
        }
    miss_2d, C_2d = project_to_plane(rel, np.asarray(Cp) + np.asarray(C2), v_rel)
    pc = pc_2d_quadrature(miss_2d, C_2d, hbr)
    ok = pc < PC_THRESHOLD
    return {
        "check": "original_threat_resolved",
        "pass": bool(ok),
        "post_maneuver_pc": pc,
        "post_maneuver_miss_km": miss,
        "reason": "threat cleared"
        if ok
        else f"Pc {pc:.2e} is still at or above the {PC_THRESHOLD:.0e} threshold",
    }


def check_no_new_conjunction(
    primary_sat,
    catalog: Dict[int, Tuple[str, object]],
    epoch: datetime,
    window_s: float,
    dv_kms: float,
    burn_offset_s: float,
    Cp_rtn_diag: Sequence[float],
    hbr: float,
    primary_norad: Optional[int] = None,
    gate_km: float = RESCREEN_GATE_KM,
    coarse_step_s: float = 30.0,
) -> Dict:
    """
    Re-screen the POST-MANEUVER trajectory against the catalog.

    The post-burn orbit is propagated with SGP4 (time-shifted, see
    post_burn_track), screened coarsely against every catalog object that
    survives the apogee/perigee filter, and every hit is refined to its exact
    time of closest approach before Pc is computed. A burn that solves one
    conjunction by creating another is the failure mode this check exists to
    catch, and it can only catch it if the screen is accurate.
    """
    entries = [
        (norad, name, sat)
        for norad, (name, sat) in catalog.items()
        if primary_norad is None or norad != primary_norad
    ]
    survivors, considered = astro.geometric_filter(primary_sat, entries, gate_km)
    if not survivors:
        return {
            "check": "no_new_conjunction",
            "pass": True,
            "objects_rescreened": 0,
            "objects_considered": considered,
            "worst_new_pc": 0.0,
            "reason": "no catalog object shares the post-burn altitude shell",
        }

    offsets = astro.coarse_grid(window_s, coarse_step_s)
    track, track_ok = post_burn_track(
        primary_sat, epoch, offsets, dv_kms, burn_offset_s
    )
    candidates = astro.screen_track(
        track, track_ok, survivors, epoch, offsets, gate_km
    )

    post_state = post_burn_state_fn(primary_sat, epoch, dv_kms, burn_offset_s)
    by_norad = {norad: sat for norad, _, sat in survivors}
    Cp_rtn = np.diag(np.asarray(Cp_rtn_diag, dtype=float) ** 2)

    worst = {"pc": 0.0, "norad": None, "name": None, "miss_km": None, "tca_utc": None}
    refined = 0
    for candidate in candidates:
        secondary = by_norad.get(candidate.norad)
        if secondary is None:
            continue
        approach = None
        for lo, hi in candidate.brackets or [
            (candidate.coarse_offset_s - 60.0, candidate.coarse_offset_s + 60.0)
        ]:
            found = astro.refine_between(
                post_state, astro.satrec_state_fn(secondary, epoch), epoch, lo, hi
            )
            if found is not None and (approach is None or found.miss_km < approach.miss_km):
                approach = found
        if approach is None or approach.miss_km > gate_km:
            continue
        refined += 1
        Cp = rtn_to_eci_cov(Cp_rtn, approach.r_primary, approach.v_primary)
        C2 = rtn_to_eci_cov(
            secondary_covariance_rtn(max(approach.tca_offset_s, 0.0)),
            approach.r_secondary,
            approach.v_secondary,
        )
        result = assess_conjunction(
            approach.r_primary, approach.v_primary, Cp,
            approach.r_secondary, approach.v_secondary, C2,
            hbr, tca_already_refined=True,
        )
        pc = result["pc"]
        if pc is not None and pc > worst["pc"]:
            worst = {
                "pc": pc,
                "norad": candidate.norad,
                "name": candidate.name,
                "miss_km": approach.miss_km,
                "tca_utc": approach.tca.isoformat(),
            }

    ok = worst["pc"] < PC_THRESHOLD
    return {
        "check": "no_new_conjunction",
        "pass": bool(ok),
        "objects_considered": considered,
        "objects_rescreened": len(survivors),
        "conjunctions_refined": refined,
        "worst_new_pc": worst["pc"],
        "worst_new_object": worst["name"],
        "worst_new_norad": worst["norad"],
        "worst_new_miss_km": worst["miss_km"],
        "worst_new_tca_utc": worst["tca_utc"],
        "reason": "no new conjunction created"
        if ok
        else f"the burn creates a new conjunction with {worst['name']} "
             f"(Pc {worst['pc']:.2e} at {worst['miss_km']:.3f} km)",
    }


# ------------------------------------------------------------------- top level
def validate_maneuver(
    rp: np.ndarray,
    vp: np.ndarray,
    Cp: np.ndarray,
    r2: np.ndarray,
    v2: np.ndarray,
    C2: np.ndarray,
    hbr: float,
    tca_s: float,
    dv_ms: float,
    sign: float,
    burn_lead_h: float,
    fuel_required_kg: float,
    fuel_available_kg: float,
    mission_sma_km: float,
    Cp_rtn_diag: Sequence[float],
    catalog: Optional[Dict[int, Tuple[str, object]]] = None,
    epoch: Optional[datetime] = None,
    window_s: float = 259200.0,
    primary_norad: Optional[int] = None,
    primary_sat=None,
) -> Dict:
    """
    Run every safety check on a proposed maneuver and return a single verdict.

    `primary_sat` is the primary's own Satrec. When it is supplied together with
    a catalog and an epoch, the post-burn re-screen runs against a genuine
    propagated orbit; without it the re-screen is skipped and the report says so,
    rather than silently approving on the strength of three checks out of four.
    """
    dv_kms = sign * dv_ms / 1000.0
    drift_s = burn_lead_h * 3600.0
    rp_post, vp_post = apply_along_track_dv(rp, vp, dv_kms, drift_s)

    checks: List[Dict] = [
        check_fuel(dv_ms, fuel_required_kg, fuel_available_kg),
        check_orbit_safety(rp_post, vp_post, mission_sma_km),
        check_threat_resolved(rp_post, vp_post, Cp, r2, v2, C2, hbr),
    ]

    rescreen_skipped = None
    if catalog and epoch is not None and primary_sat is not None:
        burn_offset_s = max(float(tca_s) - drift_s, 0.0)
        checks.append(
            check_no_new_conjunction(
                primary_sat, catalog, epoch, window_s, dv_kms, burn_offset_s,
                Cp_rtn_diag, hbr, primary_norad=primary_norad,
            )
        )
    else:
        rescreen_skipped = (
            "Post-burn catalog re-screen was not run: it needs the primary's "
            "element set, the catalog and an epoch. The verdict below covers "
            "fuel, orbit safety and the original threat only."
        )

    verdict = "APPROVED" if all(c["pass"] for c in checks) else "REJECTED"
    report = {
        "verdict": verdict,
        "failed_checks": [c["check"] for c in checks if not c["pass"]],
        "dv_ms": dv_ms,
        "direction": "prograde" if sign > 0 else "retrograde",
        "burn_lead_h": burn_lead_h,
        "fuel_required_kg": fuel_required_kg,
        "checks": checks,
    }
    if rescreen_skipped:
        report["incomplete"] = True
        report["incomplete_reason"] = rescreen_skipped
    return report
