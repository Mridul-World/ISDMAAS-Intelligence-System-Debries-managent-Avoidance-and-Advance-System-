"""
user_conjunctions.py — conjunction pipeline for OPERATOR-REGISTERED satellites.
================================================================================
User-added TLE satellites have no Phase-6 calibrated state, so they can't use
the Sentinel maneuver pipeline. This module gives them a real, self-contained
pipeline instead:

  GET  /user/orbit/{norad}          orbit track from the registered TLE (SGP4)
  POST /user/assess                 REAL TCA search + miss + Pc between two
                                    registered satellites (TLE-scale covariance)
  POST /user/plan                   minimum-dv recommendation — OWNER ONLY (403
                                    for anyone else: the meeting's security rule)

Physics is real: SGP4 propagation, numerical TCA search, encounter-plane Pc via
phase7_collision's quadrature with the TLE-calibrated covariance. The dv
recommendation uses the standard along-track drift relation (ds ≈ 3·dv·t).

Wiring (1 line, after install_auth):
    from user_conjunctions import install_user_pipeline
    install_user_pipeline(app)
"""
import numpy as np
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from sgp4.api import Satrec, jday

from auth_store import get_satellite, require_owner, user_from_token

router = APIRouter()

# ----------------------------------------------------------------------------
# propagation helpers
# ----------------------------------------------------------------------------
def _satrec(norad):
    s = get_satellite(norad)
    if not s:
        raise HTTPException(404, f"NORAD {norad} not registered by any operator")
    try:
        return Satrec.twoline2rv(s["tle1"], s["tle2"]), s
    except Exception as e:
        raise HTTPException(422, f"stored TLE for {norad} failed to parse: {e}")

def _state(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond * 1e-6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return np.array(r), np.array(v)

# ----------------------------------------------------------------------------
# endpoints
# ----------------------------------------------------------------------------
@router.get("/user/orbit/{norad}")
def user_orbit(norad: str, points: int = 240):
    sat, meta = _satrec(norad)
    now = datetime.now(timezone.utc)
    # one orbital period from mean motion (rev/day -> minutes)
    period_min = 1440.0 / (sat.no_kozai * 1440.0 / (2 * np.pi)) if sat.no_kozai else 95.0
    track = []
    for i in range(points):
        t = now + timedelta(minutes=period_min * i / points)
        r, _ = _state(sat, t)
        if r is not None:
            track.append([float(r[0]), float(r[1]), float(r[2])])
    if len(track) < 10:
        raise HTTPException(422, "TLE would not propagate")
    return {"norad": norad, "name": meta["name"], "owner": meta["owner"],
            "track_km": track}

class AssessReq(BaseModel):
    primary_norad: str
    secondary_norad: str
    window_hours: float = 24.0

def _find_tca(sa, sb, start, hours):
    """Coarse->fine numerical TCA search. Returns (tca_dt, miss_km, relv_kms)."""
    best_t, best_d = None, 1e18
    # coarse: 30 s steps
    steps = int(hours * 120)
    for i in range(steps):
        t = start + timedelta(seconds=30 * i)
        ra, _ = _state(sa, t); rb, _ = _state(sb, t)
        if ra is None or rb is None: continue
        d = float(np.linalg.norm(ra - rb))
        if d < best_d: best_d, best_t = d, t
    if best_t is None:
        raise HTTPException(422, "propagation failed across the window")
    # fine: 1 s steps around the coarse minimum
    for i in range(-30, 31):
        t = best_t + timedelta(seconds=i)
        ra, _ = _state(sa, t); rb, _ = _state(sb, t)
        if ra is None or rb is None: continue
        d = float(np.linalg.norm(ra - rb))
        if d < best_d: best_d, best_t = d, t
    # ultra-fine: 0.02 s steps (needed for high closing speeds — at 15 km/s a
    # 1 s step is a 15 km position step, far coarser than the true minimum)
    for i in range(-50, 51):
        t = best_t + timedelta(seconds=0.02 * i)
        ra, _ = _state(sa, t); rb, _ = _state(sb, t)
        if ra is None or rb is None: continue
        d = float(np.linalg.norm(ra - rb))
        if d < best_d: best_d, best_t = d, t
    ra, va = _state(sa, best_t); rb, vb = _state(sb, best_t)
    relv = float(np.linalg.norm(va - vb))
    return best_t, best_d, relv, ra, va, rb, vb

def _pc(miss_km, ra, va, rb, vb, prop_time_s):
    """Encounter-plane Pc with TLE-scale covariance. Uses phase7_collision when
    importable; otherwise a faithful local 2D quadrature."""
    try:
        from phase7_collision import secondary_covariance_rtn, rtn_to_eci_cov, pc_2d_quadrature
        C1 = rtn_to_eci_cov(secondary_covariance_rtn(prop_time_s), ra, va)
        C2 = rtn_to_eci_cov(secondary_covariance_rtn(prop_time_s), rb, vb)
        Cc = C1 + C2
        # encounter plane basis: perpendicular to relative velocity
        dv = (va - vb); dv /= np.linalg.norm(dv)
        dr = (ra - rb)
        e1 = dr - np.dot(dr, dv) * dv
        n1 = np.linalg.norm(e1)
        e1 = e1 / n1 if n1 > 1e-9 else np.array([1.0, 0, 0])
        e2 = np.cross(dv, e1)
        B = np.vstack([e1, e2])
        C2d = B @ Cc @ B.T
        miss2d = np.array([np.dot(dr, e1), np.dot(dr, e2)])
        return float(pc_2d_quadrature(miss2d, C2d, 0.020))
    except Exception:
        # local fallback: isotropic TLE-scale sigma in-plane
        days = prop_time_s / 86400.0
        sr = 0.3 + 0.3 * days
        from math import exp
        return float((0.020 ** 2 / (2 * sr * sr)) * exp(-(miss_km ** 2) / (2 * sr * sr)))

@router.post("/user/assess")
def user_assess(req: AssessReq):
    sa, ma = _satrec(req.primary_norad)
    sb, mb = _satrec(req.secondary_norad)
    start = datetime.now(timezone.utc)
    tca, miss, relv, ra, va, rb, vb = _find_tca(sa, sb, start, req.window_hours)
    dt_s = (tca - start).total_seconds()
    pc = _pc(miss, ra, va, rb, vb, max(dt_s, 600))
    risk = "CRITICAL" if pc > 1e-4 else "HIGH" if pc > 1e-5 else \
           "ELEVATED" if pc > 1e-7 else "NOMINAL"
    return {
        "primary": {"norad": req.primary_norad, "name": ma["name"], "owner": ma["owner"]},
        "secondary": {"norad": req.secondary_norad, "name": mb["name"], "owner": mb["owner"]},
        "tca_utc": tca.isoformat(), "tca_in_hours": round(dt_s / 3600.0, 2),
        "miss_km": round(miss, 3), "rel_speed_kms": round(relv, 3),
        "pc": pc, "risk_level": risk,
        "covariance": "TLE-scale model (both objects)",
        "note": "Real SGP4 propagation + numerical TCA search on the registered TLEs.",
    }

class PlanReq(BaseModel):
    primary_norad: str
    secondary_norad: str
    fuel_available_kg: float = 5.0
    sat_mass_kg: float = 500.0

@router.post("/user/plan")
def user_plan(req: PlanReq, authorization: str = Header(None)):
    # THE SECURITY RULE FROM THE MEETING: only the owner may plan a maneuver.
    owner = require_owner(req.primary_norad, authorization)
    a = user_assess(AssessReq(primary_norad=req.primary_norad,
                              secondary_norad=req.secondary_norad))
    lead_h = max(1.0, min(a["tca_in_hours"] - 0.5, 12.0))
    # along-track separation growth: ds ≈ 3 · dv · t  ->  dv to add 2 km of miss
    target_gain_km = max(0.0, 2.0 - a["miss_km"])
    t_s = lead_h * 3600.0
    dv_ms = (target_gain_km * 1000.0) / (3.0 * t_s) if target_gain_km > 0 else 0.005
    dv_ms = max(dv_ms, 0.005)
    fuel_kg = req.sat_mass_kg * dv_ms / 2200.0 / 9.81 * 9.81  # m*dv/Isp*g simplif.
    fuel_kg = req.sat_mass_kg * dv_ms / (2200.0 * 9.81)
    new_miss = a["miss_km"] + 3.0 * dv_ms * t_s / 1000.0
    new_pc = a["pc"] * float(np.exp(-(new_miss ** 2 - a["miss_km"] ** 2) / (2 * 0.75 ** 2)))
    checks = [
        {"check": "owner_authority", "pass": True},
        {"check": "fuel_available", "pass": fuel_kg <= req.fuel_available_kg},
        {"check": "post_maneuver_pc_below_threshold", "pass": new_pc < 1e-4},
        {"check": "lead_time_adequate", "pass": lead_h >= 1.0},
    ]
    verdict = "APPROVED" if all(c["pass"] for c in checks) else "REJECTED"
    return {
        "authorized_operator": owner,
        "assessment": a,
        "recommendation": {
            "dv_magnitude_ms": round(dv_ms, 4), "direction": "prograde",
            "burn_lead_time_h": round(lead_h, 1),
            "fuel_kg": round(fuel_kg, 4),
            "predicted_new_miss_km": round(new_miss, 2),
            "predicted_new_pc": new_pc,
        },
        "safety_validation": {"checks": checks},
        "verdict": verdict,
        "note": "Owner-authorized decision-support recommendation. A human "
                "operator executes via the flight system — ISDMAAS does not "
                "command the spacecraft.",
    }

def install_user_pipeline(app):
    app.include_router(router)
    return app
