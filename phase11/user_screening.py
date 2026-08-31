"""
user_screening.py — full-catalog conjunction screening for operator satellites.

ISDMAAS's own SOCRATES-equivalent for a registered asset:

  GET /user/screen/{norad}?hours=24&gate_km=25

  1. CATALOG   the shared, cached public catalog service.
  2. FILTER    apogee/perigee rejection before any propagation.
  3. SCREEN    vectorized SGP4 across the window (SatrecArray, C speed).
  4. REFINE    Brent refinement of every hit to a millisecond-accurate TCA, then
               encounter-plane Pc with the TLE-scale covariance model.
  5. LOG       the run and its conjunctions go to the operational database.

Honest notes
  - Pc uses the documented TLE-scale covariance model. With real CDMs the
    /cdm/assess path uses the operator's own covariance instead.
  - Screening results are OPERATIONAL RECORDS, not training labels. Training
    stays truth-data-driven via training_pipeline.py.

Screening is now authenticated. It is the most expensive endpoint in the service
— a full-catalog screen against 16 000 objects — and it was previously reachable
without a token, which let any caller consume the whole machine. It also reads
an operator's registered assets, which is not public information.
"""
from __future__ import annotations

from datetime import datetime, timezone
from fastapi import APIRouter, Depends, Query
from sgp4.api import Satrec

from isdmaas_core import screening as screening_service
from isdmaas_core.catalog import get_catalog_service
from isdmaas_core.config import get_settings
from isdmaas_core.errors import ApiError
from isdmaas_core.logging_config import get_logger

import ops_db
from auth_store import get_satellite, require_user

log = get_logger("isdmaas.user_screen")
router = APIRouter(tags=["operator"])


@router.get("/user/screen/{norad}")
def screen(
    norad: str,
    hours: float = Query(default=24.0, gt=0.0, le=168.0),
    gate_km: float = Query(default=25.0, gt=0.0, le=500.0),
    max_results: int = Query(default=10, ge=1, le=100),
    refresh: bool = Query(default=False),
    username: str = Depends(require_user),
):
    settings = get_settings()
    record = get_satellite(norad)
    if not record:
        raise ApiError(
            404, f"NORAD {norad} is not registered by any operator.", "not_found"
        )
    if record["owner"] != username:
        raise ApiError(
            403,
            f"NORAD {norad} belongs to another operator. You can screen only "
            "your own assets.",
            code="forbidden",
        )
    try:
        user_sat = Satrec.twoline2rv(record["tle1"], record["tle2"])
    except Exception as exc:
        raise ApiError(
            422, f"The stored element set for {norad} failed to parse: {exc}",
            code="unprocessable",
        ) from exc

    hours = min(hours, settings.screen_max_hours)
    service = get_catalog_service(
        settings.catalog_tle_cache,
        settings.catalog_ttl_s,
        fallback_path=settings.catalog_tle_fallback,
    )
    snapshot = service.refresh(force=True) if refresh else service.snapshot()
    if snapshot is None or not snapshot.entries:
        raise ApiError(
            503,
            "The object catalog is unavailable: no local cache and the upstream "
            "fetch failed. Retry once network access is restored.",
            code="service_unavailable",
        )

    numeric_norad = int(norad) if str(norad).isdigit() else None
    result = screening_service.screen_primary(
        primary_sat=user_sat,
        primary_name=record["name"],
        catalog=snapshot.as_tuples(),
        epoch=datetime.now(timezone.utc),
        window_s=hours * 3600.0,
        gate_km=gate_km,
        hbr_km=settings.hbr_km,
        primary_norad=numeric_norad,
        coarse_step_s=settings.screen_coarse_step_s,
    )
    conjunctions = [
        {
            "secondary_norad": c.secondary_norad,
            "secondary_name": c.secondary_name,
            "tca_utc": c.tca_utc,
            "tca_in_hours": round(c.tca_hours, 3),
            "miss_km": round(c.miss_km, 4),
            "rel_speed_kms": round(c.relative_speed_kms, 4),
            "pc": c.pc,
            "pc_methods_agree": c.pc_methods_agree,
            "risk": c.risk,
        }
        for c in result.conjunctions[:max_results]
    ]

    # Operational record. A logging failure must never fail the screen.
    try:
        ops_db.log_screening(
            norad, record["name"], record["owner"],
            result.objects_considered, len(conjunctions), conjunctions,
        )
    except Exception:
        log.warning("screening run could not be logged", exc_info=True)

    return {
        "primary": {"norad": norad, "name": record["name"], "owner": record["owner"]},
        "window_hours": hours,
        "gate_km": gate_km,
        "catalog_fetched_utc": snapshot.fetched_utc.isoformat(),
        "catalog_age_hours": round(snapshot.age_s / 3600, 2),
        "objects_screened": result.objects_considered,
        "objects_after_geometric_filter": result.objects_after_geometric_filter,
        "coarse_candidates": result.coarse_candidates,
        "scan_seconds": result.scan_seconds,
        "conjunctions": conjunctions,
        "note": (
            "ISDMAAS's own full-catalog screen for this asset. Pc uses the "
            "documented TLE-scale covariance model; real CDM covariance is used "
            "when supplied via /cdm/assess."
        ),
    }


@router.get("/user/screen-history/{norad}")
def screen_history(
    norad: str,
    limit: int = Query(default=20, ge=1, le=100),
    username: str = Depends(require_user),
):
    """Past screening runs for one of the caller's own assets."""
    record = get_satellite(norad)
    if not record:
        raise ApiError(
            404, f"NORAD {norad} is not registered by any operator.", "not_found"
        )
    if record["owner"] != username:
        raise ApiError(403, "You can only read history for your own assets.", "forbidden")
    return {"norad": norad, "runs": ops_db.screening_history(norad, limit)}


def install_screening(app):
    app.include_router(router)
    return app
