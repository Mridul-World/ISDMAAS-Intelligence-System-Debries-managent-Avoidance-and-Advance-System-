"""
user_conjunctions.py — conjunction pipeline for OPERATOR-REGISTERED satellites.

Operator-added satellites have no precise-orbit solution, so they cannot use the
calibrated pipeline. This module gives them a real, self-contained one:

  GET  /user/orbit/{norad}     orbit track from the registered element set
  POST /user/assess            refined TCA, miss and Pc between two registered
                               satellites, with TLE-scale covariance
  POST /user/plan              minimum-dv recommendation — OWNER ONLY

What changed
------------
* The TCA search was a Python loop stepping 30 s at a time across the window,
  then 1 s, then 0.02 s — around 6 000 SGP4 calls for a single pair, and the
  screening endpoint ran it twenty times per request. It now uses the vectorized
  coarse grid plus Brent refinement, which is both faster and exact.

* The maneuver recommendation used a hand-rolled dv formula, an exponential
  guess at the post-burn Pc with a hard-coded 0.75 km sigma, and computed the
  propellant mass twice with the first result discarded. It now calls the same
  planner and the same safety checks as the calibrated pipeline, so an operator
  asset and a Sentinel get the same physics and the same audit trail.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
from fastapi import APIRouter, Header
from pydantic import BaseModel, Field
from sgp4.api import Satrec

from isdmaas_core import astrodynamics as astro
from isdmaas_core import screening as screening_service
from isdmaas_core.config import get_settings
from isdmaas_core.errors import ApiError
from isdmaas_core.logging_config import get_logger

from auth_store import get_satellite, require_owner
from phase8_maneuver import PC_THRESHOLD, plan_maneuver
from phase10_safety import orbital_elements, validate_maneuver
from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn

log = get_logger("isdmaas.user")
router = APIRouter(tags=["operator"])


# ----------------------------------------------------------------- helpers
def _satrec(norad: str) -> Tuple[Satrec, dict]:
    record = get_satellite(norad)
    if not record:
        raise ApiError(
            404, f"NORAD {norad} is not registered by any operator.", "not_found"
        )
    try:
        return Satrec.twoline2rv(record["tle1"], record["tle2"]), record
    except Exception as exc:
        raise ApiError(
            422, f"The stored element set for {norad} failed to parse: {exc}",
            code="unprocessable",
        ) from exc


def _state(sat: Satrec, when: datetime):
    """Kept for callers that import it; delegates to the shared propagator."""
    return astro.propagate(sat, when)


# --------------------------------------------------------------- endpoints
@router.get("/user/orbit/{norad}")
def user_orbit(norad: str, points: int = 240):
    points = max(8, min(int(points), 2000))
    sat, meta = _satrec(norad)
    now = datetime.now(timezone.utc)
    period_s = astro.orbital_period_s(sat)
    offsets = np.linspace(0.0, period_s, points, endpoint=False)
    positions, _, ok = astro.propagate_series(sat, now, offsets)
    track = [p.tolist() for p, good in zip(positions, ok, strict=True) if good]
    if len(track) < 10:
        raise ApiError(
            422,
            "The registered element set does not propagate. Check that it is "
            "current and correctly transcribed.",
            code="unprocessable",
        )
    return {
        "norad": norad,
        "name": meta["name"],
        "owner": meta["owner"],
        "track_km": track,
        "period_s": period_s,
    }


class AssessRequest(BaseModel):
    primary_norad: str = Field(min_length=1, max_length=12)
    secondary_norad: str = Field(min_length=1, max_length=12)
    window_hours: float = Field(default=24.0, gt=0.0, le=168.0)


def _assess(primary_norad: str, secondary_norad: str, window_hours: float) -> dict:
    settings = get_settings()
    sat_a, meta_a = _satrec(primary_norad)
    sat_b, meta_b = _satrec(secondary_norad)
    if primary_norad == secondary_norad:
        raise ApiError(
            400, "Primary and secondary must be different satellites.", "bad_request"
        )
    start = datetime.now(timezone.utc)
    conjunction = screening_service.assess_pair(
        primary_sat=sat_a,
        primary_name=meta_a["name"],
        secondary_sat=sat_b,
        secondary_name=meta_b["name"],
        secondary_norad=int(secondary_norad) if secondary_norad.isdigit() else 0,
        epoch=start,
        window_s=window_hours * 3600.0,
        hbr_km=settings.hbr_km,
        coarse_step_s=settings.screen_coarse_step_s,
    )
    if conjunction is None:
        raise ApiError(
            422, "Neither element set propagates across the window.", "unprocessable"
        )
    return {
        "primary": {
            "norad": primary_norad, "name": meta_a["name"], "owner": meta_a["owner"]
        },
        "secondary": {
            "norad": secondary_norad, "name": meta_b["name"], "owner": meta_b["owner"]
        },
        "tca_utc": conjunction.tca_utc,
        "tca_in_hours": round(conjunction.tca_hours, 3),
        "miss_km": round(conjunction.miss_km, 4),
        "rel_speed_kms": round(conjunction.relative_speed_kms, 4),
        "pc": conjunction.pc,
        "pc_chan_crosscheck": conjunction.pc_chan,
        "pc_methods_agree": conjunction.pc_methods_agree,
        "mahalanobis": conjunction.mahalanobis,
        "risk_level": conjunction.risk,
        "covariance": "TLE-scale model (both objects)",
        "note": (
            "Real SGP4 propagation with a Brent-refined time of closest approach "
            "on the registered element sets."
        ),
    }


@router.post("/user/assess")
def user_assess(request: AssessRequest):
    return _assess(
        request.primary_norad, request.secondary_norad, request.window_hours
    )


class PlanRequest(BaseModel):
    primary_norad: str = Field(min_length=1, max_length=12)
    secondary_norad: str = Field(min_length=1, max_length=12)
    fuel_available_kg: float = Field(default=5.0, ge=0.0, le=10000.0)
    sat_mass_kg: float = Field(default=500.0, gt=0.0, le=500000.0)
    window_hours: float = Field(default=24.0, gt=0.0, le=168.0)


@router.post("/user/plan")
def user_plan(request: PlanRequest, authorization: Optional[str] = Header(None)):
    """
    Owner-authorized avoidance plan for a registered satellite.

    Maneuver authority is restricted to the owning operator: planning a burn on
    another operator's asset is refused before any computation is done.
    """
    owner = require_owner(request.primary_norad, authorization)
    settings = get_settings()

    sat_a, meta_a = _satrec(request.primary_norad)
    sat_b, meta_b = _satrec(request.secondary_norad)
    start = datetime.now(timezone.utc)
    approach = astro.find_primary_close_approach(
        sat_a, sat_b, start, request.window_hours * 3600.0,
        coarse_step_s=settings.screen_coarse_step_s,
    )
    if approach is None:
        raise ApiError(
            422, "Neither element set propagates across the window.", "unprocessable"
        )

    prop_time_s = max(approach.tca_offset_s, 0.0)
    sigma = screening_service.TLE_PRIMARY_SIGMA_KM
    Cp = rtn_to_eci_cov(
        np.diag(np.asarray(sigma) ** 2), approach.r_primary, approach.v_primary
    )
    C2 = rtn_to_eci_cov(
        secondary_covariance_rtn(prop_time_s),
        approach.r_secondary, approach.v_secondary,
    )
    tca_s = max(approach.tca_offset_s, 600.0)

    assessment = _assess(
        request.primary_norad, request.secondary_norad, request.window_hours
    )
    plan = plan_maneuver(
        approach.r_primary, approach.v_primary, Cp,
        approach.r_secondary, approach.v_secondary, C2,
        hbr=settings.hbr_km, tca_s=tca_s, sat_mass_kg=request.sat_mass_kg,
    )

    response = {
        "authorized_operator": owner,
        "assessment": assessment,
        "pc_action_threshold": PC_THRESHOLD,
        "action_required": plan["action_required"],
        "note": (
            "Owner-authorized decision-support recommendation. A human operator "
            "executes via the flight system — ISDMAAS does not command the "
            "spacecraft."
        ),
    }

    recommendation = plan.get("recommendation")
    if not isinstance(recommendation, dict):
        response["verdict"] = "NO_ACTION"
        response["recommendation"] = recommendation
        return response

    mission_sma, _, _ = orbital_elements(approach.r_primary, approach.v_primary)
    sign = 1.0 if recommendation["direction"] == "prograde" else -1.0
    report = validate_maneuver(
        approach.r_primary, approach.v_primary, Cp,
        approach.r_secondary, approach.v_secondary, C2,
        settings.hbr_km, tca_s,
        recommendation["dv_magnitude_ms"], sign,
        recommendation["burn_lead_time_h"], recommendation["fuel_kg"],
        fuel_available_kg=request.fuel_available_kg,
        mission_sma_km=mission_sma,
        Cp_rtn_diag=sigma,
        primary_sat=sat_a,
    )
    # The owner check already passed, and it is part of the audit trail an
    # operator reads, so it is recorded alongside the physics checks.
    report["checks"].insert(
        0,
        {
            "check": "owner_authority",
            "pass": True,
            "reason": f"{owner} holds maneuver authority for NORAD "
                      f"{request.primary_norad}",
        },
    )
    response["recommendation"] = recommendation
    response["options"] = plan["options"]
    response["safety_validation"] = report
    response["verdict"] = report["verdict"]

    # Audit record. Captures the INPUTS as well as the outputs, because six
    # months from now "why did ISDMAAS recommend this burn?" has to be
    # answerable from the record rather than from someone's memory of what the
    # catalog looked like that day. A failure to write it must never deny an
    # operator their assessment, so it is best-effort and logged.
    try:
        import ops_db
        from isdmaas_core import __version__ as algorithm_version
        from isdmaas_core.tle import validate_tle

        assessment_id = ops_db.new_assessment_id()
        primary_report = validate_tle(meta_a["tle1"], meta_a["tle2"])
        secondary_report = validate_tle(meta_b["tle1"], meta_b["tle2"])
        ops_db.record_assessment({
            "assessment_id": assessment_id,
            "created_utc": start.isoformat(),
            "operator": owner,
            "primary_norad": request.primary_norad,
            "primary_name": meta_a["name"],
            "secondary_norad": request.secondary_norad,
            "secondary_name": meta_b["name"],
            "primary_epoch_utc": (primary_report.epoch_utc.isoformat()
                                  if primary_report.epoch_utc else None),
            "secondary_epoch_utc": (secondary_report.epoch_utc.isoformat()
                                    if secondary_report.epoch_utc else None),
            "catalog_fetched_utc": None,
            "catalog_age_hours": primary_report.age_days * 24.0
                                 if primary_report.age_days is not None else None,
            "algorithm_version": algorithm_version,
            # SGP4 alone. There is no ML model in the serving path; recording
            # "none" is the honest value and makes that visible in every record.
            "model_version": "none (SGP4 baseline)",
            "covariance_source": "tle_scale_model",
            "primary_sigma_rtn_km": list(sigma),
            "hbr_km": settings.hbr_km,
            "tca_utc": assessment["tca_utc"],
            "miss_km": assessment["miss_km"],
            "relative_speed_kms": assessment["rel_speed_kms"],
            "pc": assessment["pc"],
            "pc_method": "foster_2d_gauss_legendre_quadrature",
            "risk_level": assessment["risk_level"],
            "status": "ASSESSED" if assessment["pc"] is not None else "DATA_INVALID",
            "recommendation_json": recommendation,
            "safety_verdict": report["verdict"],
            "detail_json": {
                "options": plan["options"],
                "safety_checks": [
                    {"check": c["check"], "pass": c["pass"]}
                    for c in report["checks"]
                ],
            },
        })
        response["assessment_id"] = assessment_id
    except Exception:
        log.warning("assessment audit record could not be written", exc_info=True)

    return response


def install_user_pipeline(app):
    app.include_router(router)
    return app
