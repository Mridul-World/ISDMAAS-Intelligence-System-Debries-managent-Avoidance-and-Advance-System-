"""
================================================================================
ISDMAAS API SERVICE  (FastAPI)
================================================================================
Unifies the pipeline — predict, assess risk, plan a maneuver, validate it —
behind REST endpoints.

  GET  /health                  liveness
  GET  /ready                   readiness (catalog loaded, store reachable)
  GET  /satellites              configured fleet
  GET  /resolve                 NORAD id / name / COSPAR id lookup
  GET  /orbit/{norad}           one orbit of ECI positions for the 3-D view
  GET  /screen/{norad}          full-catalog conjunction screen for one asset
  GET  /monitor                 watchlist screen with new-alert tracking
  POST /assess                  risk for a specific primary/secondary pair
  POST /cdm/assess              risk from a real CCSDS CDM (operator covariance)
  POST /maneuver-plan           planned + safety-validated avoidance burn
  POST /autonomous              the hands-off detect-plan-validate loop
  GET  /socrates                CelesTrak's published conjunction feed
  GET  /live/catalog            cached objects for browser-side propagation
  GET  /library, /library/{key} documented historical events

Operator accounts, satellite registration and the owner-only maneuver path are
mounted from auth_store, user_conjunctions and user_screening.

The /maneuver-plan endpoint is structured so a learned optimizer can drop in
behind it later: it calls a single `propose_maneuver()`. Swap that and every
safety guarantee still applies unchanged.

Run:  uvicorn phase11_api:app --port 8000
Docs: http://localhost:8000/docs
================================================================================
"""
from __future__ import annotations

from contextlib import asynccontextmanager
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from fastapi import Body, Depends, FastAPI, Header, Query, Request
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sgp4.api import Satrec

# The pipeline modules sit alongside this file; make that work regardless of the
# directory uvicorn was started from.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isdmaas_core import __version__ as ISDMAAS_VERSION
from isdmaas_core import astrodynamics as astro
from isdmaas_core import screening as screening_service
from isdmaas_core.catalog import get_catalog_service
from isdmaas_core.config import get_settings
from isdmaas_core.errors import ApiError, install_error_handlers
from isdmaas_core.logging_config import (
    configure_logging,
    get_logger,
    new_request_id,
    request_id_var,
)
from isdmaas_core.security import SlidingWindowLimiter, security_headers
from isdmaas_core.socrates import get_feed as get_socrates_feed
from isdmaas_core.store import get_store

from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn
from phase8_maneuver import PC_THRESHOLD, plan_maneuver
from phase10_safety import orbital_elements, validate_maneuver

settings = get_settings()
configure_logging(settings.log_level, settings.log_json)
log = get_logger("isdmaas.api")

MU = 398600.4418
SCREEN_WINDOW_S = 259200.0          # three days
SATELLITES = {
    39634: "SENTINEL-1A",
    40697: "SENTINEL-2A",
    42063: "SENTINEL-2B",
    41335: "SENTINEL-3A",
    43437: "SENTINEL-3B",
}

@asynccontextmanager
async def lifespan(_: FastAPI):
    on_startup()
    yield


app = FastAPI(
    lifespan=lifespan,
    title="ISDMAAS API",
    description=(
        "Intelligent Satellite Debris Management & Autonomous Avoidance — "
        "decision support for collision avoidance. This service recommends; "
        "a human operator decides and commands the spacecraft."
    ),
    version=ISDMAAS_VERSION,
    docs_url=None if settings.is_production else "/docs",
    redoc_url=None,
    openapi_url=None if settings.is_production else "/openapi.json",
)

install_error_handlers(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
    max_age=600,
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Correlation id, security headers, body-size cap and timing for every request."""
    incoming = request.headers.get("x-request-id", "")
    request_id = incoming[:64] if incoming.isascii() and len(incoming) <= 64 else ""
    token = request_id_var.set(request_id or new_request_id())
    started = time.monotonic()
    try:
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > settings.max_request_bytes:
            response: JSONResponse = JSONResponse(
                status_code=413,
                content={
                    "error": {
                        "code": "payload_too_large",
                        "message": (
                            f"Request body exceeds the "
                            f"{settings.max_request_bytes // 1024} KiB limit."
                        ),
                        "request_id": request_id_var.get(),
                    },
                    "detail": "Request body too large.",
                },
            )
        else:
            response = await call_next(request)
        for key, value in security_headers(settings.is_production).items():
            response.headers.setdefault(key, value)
        response.headers["X-Request-ID"] = request_id_var.get()
        elapsed_ms = int((time.monotonic() - started) * 1000)
        response.headers["X-Response-Time-ms"] = str(elapsed_ms)
        if elapsed_ms > 5000:
            log.info(
                "slow request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "ms": elapsed_ms,
                },
            )
        return response
    finally:
        request_id_var.reset(token)


# ----------------------------------------------------------------- services
catalog_service = get_catalog_service(
    settings.catalog_tle_cache,
    settings.catalog_ttl_s,
    fallback_path=settings.catalog_tle_fallback,
)


def require_catalog():
    """The current catalog snapshot, or a 503 that says why it is missing."""
    snapshot = catalog_service.snapshot()
    if snapshot is None or not snapshot.entries:
        raise ApiError(
            503,
            "The object catalog is unavailable: no local cache and the CelesTrak "
            "fetch failed. Retry once network access is restored.",
            code="service_unavailable",
        )
    return snapshot


# ------------------------------------------------------- compute guard
#
# Every screening endpoint runs a full-catalog propagation: 16 000 objects
# filtered, propagated and refined. Left anonymous and unlimited they are a
# denial-of-service amplifier - one HTTP request costs a CPU-second, and nothing
# stopped a caller issuing thousands.
#
# Two controls, deliberately separate:
#   * a per-client rate limit, always on, so no single caller can saturate the
#     service even when anonymous access is intended (the console's read-only
#     mode and the demo walkthrough both rely on it);
#   * an authentication requirement, on by default in production, so a public
#     deployment does not hand its CPU to anonymous callers at all.
_compute_limiter = SlidingWindowLimiter(
    settings.compute_rate_limit, settings.compute_rate_window_s
)


def compute_guard(request: Request, authorization: Optional[str] = Header(None)):
    """Rate-limit and (in production) authenticate an expensive endpoint."""
    if settings.require_auth_for_compute:
        from auth_store import user_from_token

        if not user_from_token(authorization):
            raise ApiError(
                401,
                "This endpoint runs a full-catalog screen and requires an "
                "operator token on this deployment.",
                code="unauthenticated",
                headers={"WWW-Authenticate": "Bearer"},
            )

    client = request.client.host if request.client else "unknown"
    decision = _compute_limiter.hit(client)
    if not decision.allowed:
        raise ApiError(
            429,
            "Too many screening requests. Each one propagates the whole "
            "catalog; please pace them.",
            code="rate_limited",
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    return True


def catalog_entry(snapshot, norad: int):
    entry = snapshot.get(norad)
    if entry is None:
        raise ApiError(404, f"NORAD {norad} is not in the catalog.", "not_found")
    return entry


# ------------------------------------------------------------------ primary
class PrimaryState:
    """A primary asset's state, covariance and provenance."""

    def __init__(self, norad, name, satrec, r, v, epoch, sigma_km, source):
        self.norad = norad
        self.name = name
        self.satrec = satrec
        self.r = r
        self.v = v
        self.epoch = epoch
        self.sigma_km = sigma_km
        self.covariance_source = source

    @property
    def covariance_eci(self) -> np.ndarray:
        return rtn_to_eci_cov(
            np.diag(np.asarray(self.sigma_km, dtype=float) ** 2), self.r, self.v
        )


# A precise-orbit determination describes the epoch it was computed for. Past
# this age it is a historical record, not a statement about where the satellite
# is now, and its uncertainty no longer applies.
CALIBRATED_STATE_MAX_AGE_DAYS = 7.0


def _read_calibrated_state(norad: int):
    """
    Load a precise-orbit state file, if it exists and is still current.

    Returns (name, sigma_km, record) or None. Two things this deliberately does
    NOT do, both of which the previous implementation did:

    * It does not use the file's position and velocity as the satellite's
      CURRENT state. Those are a snapshot at a determination epoch — the file
      shipped in this repository is dated 2026-06-16 — and treating a
      seventy-six-day-old state vector as "now" produces a position that is
      wrong by thousands of kilometres. Current state always comes from the
      current element set.

    * It does not substitute a hard-coded covariance. The file carries the
      per-axis uncertainty that was actually determined (`rtn_sigma_km`); for
      Sentinel-1A that is 3.13 km along-track, against the 0.05 km constant the
      service was applying. Understating the primary's uncertainty by a factor
      of sixty drives Pc down by orders of magnitude, which is precisely the
      calibration failure this project already fixed once for the secondary.
    """
    state_file = settings.state_file(norad)
    if not state_file.exists():
        return None
    import json

    try:
        record = json.loads(state_file.read_text(encoding="utf-8"))
        epoch = datetime.fromisoformat(record["epoch_utc"])
    except (OSError, KeyError, ValueError) as exc:
        log.warning("calibrated state for %s is unreadable: %s", norad, exc)
        return None

    age_days = (datetime.now(timezone.utc) - epoch).total_seconds() / 86400.0
    if age_days > CALIBRATED_STATE_MAX_AGE_DAYS:
        log.info(
            "ignoring the calibrated state for %s: it is %.1f days old "
            "(limit %.0f). Falling back to the element set with TLE-scale "
            "uncertainty.",
            norad, age_days, CALIBRATED_STATE_MAX_AGE_DAYS,
        )
        return None

    sigma = record.get("rtn_sigma_km")
    if not sigma or len(sigma) != 3 or not all(s > 0 for s in sigma):
        log.warning(
            "calibrated state for %s has no usable rtn_sigma_km; using the "
            "TLE-scale model instead of assuming a tighter one", norad)
        return None
    return record.get("name", SATELLITES.get(norad, str(norad))), tuple(sigma), record


def load_primary(norad: int, snapshot) -> PrimaryState:
    """
    Load a primary's current state and the uncertainty that goes with it.

    The state is always the current element set propagated to now. The
    uncertainty is the determined per-axis sigma from a recent precise-orbit
    solution when one exists, and the documented TLE-scale model otherwise.

    The TLE-scale fallback used to attach a 50 m radial / 1 km along-track
    covariance to a TLE-derived state. Those are precise-orbit numbers; applied
    to an element set they understate the uncertainty by an order of magnitude,
    which is what drove the engine to report Pc ~1e-19 for kilometre-class misses
    and made the console defer to SOCRATES as more authoritative. The fallback
    now carries honest TLE-scale uncertainty, so its Pc means something.
    """
    entry = snapshot.get(norad)
    calibrated = _read_calibrated_state(norad)

    if entry is None:
        # No element set. A recent precise-orbit state is the only thing left,
        # and only for the short window over which it is still meaningful.
        if calibrated is not None:
            name, sigma, record = calibrated
            return PrimaryState(
                norad=norad, name=name, satrec=None,
                r=np.asarray(record["position_teme_km"], dtype=float),
                v=np.asarray(record["velocity_teme_kms"], dtype=float),
                epoch=datetime.fromisoformat(record["epoch_utc"]),
                sigma_km=sigma, source="precise_orbit_determination",
            )
        raise ApiError(
            404,
            f"NORAD {norad} is not in the catalog and has no current "
            "precise-orbit state.",
            code="not_found",
        )

    epoch = datetime.now(timezone.utc)
    r, v = astro.propagate(entry.satrec, epoch)
    if r is None:
        raise ApiError(
            422,
            f"The element set for NORAD {norad} does not propagate to the current "
            "epoch. It is likely decayed or stale.",
            code="unprocessable",
        )
    sigma = screening_service.TLE_PRIMARY_SIGMA_KM
    source = "tle_scale_model"
    if calibrated is not None:
        # A current determination gives better uncertainty than the model, but
        # the state still comes from the element set: the determination is a
        # snapshot, and only the element set can be propagated to now.
        _, sigma, _ = calibrated
        source = "precise_orbit_sigma_on_element_set"
    return PrimaryState(
        norad=norad,
        name=entry.name,
        satrec=entry.satrec,
        r=r,
        v=v,
        epoch=epoch,
        sigma_km=sigma,
        source=source,
    )


def propose_maneuver(rp, vp, Cp, r2, v2, C2, tca_s, sat_mass_kg):
    """Maneuver proposer. Rule-based today; a learned policy plugs in here."""
    return plan_maneuver(
        rp, vp, Cp, r2, v2, C2,
        hbr=settings.hbr_km, tca_s=tca_s, sat_mass_kg=sat_mass_kg,
    )


# ------------------------------------------------------------------- models
class AssessRequest(BaseModel):
    primary_norad: int = Field(ge=1, le=999999)
    secondary_norad: int = Field(ge=1, le=999999)
    window_hours: float = Field(default=72.0, gt=0.0, le=168.0)


class ManeuverRequest(BaseModel):
    primary_norad: int = Field(ge=1, le=999999)
    secondary_norad: Optional[int] = Field(default=None, ge=1, le=999999)
    synthetic_miss_km: Optional[float] = Field(default=None, gt=0.0, le=1000.0)
    tca_hours: float = Field(default=8.0, gt=0.0, le=168.0)
    sat_mass_kg: float = Field(default=2300.0, gt=0.0, le=500000.0)
    fuel_available_kg: float = Field(default=5.0, ge=0.0, le=10000.0)


class AutoRequest(BaseModel):
    primary_norad: int = Field(ge=1, le=999999)
    mode: str = Field(default="auto", pattern="^(auto|inject|historical)$")
    inject_miss_km: float = Field(default=0.05, gt=0.0, le=1000.0)
    inject_tca_h: float = Field(default=8.0, gt=0.0, le=168.0)
    event_key: Optional[str] = Field(default=None, max_length=80)
    sat_mass_kg: float = Field(default=2300.0, gt=0.0, le=500000.0)
    fuel_available_kg: float = Field(default=5.0, ge=0.0, le=10000.0)
    risk_threshold_pc: float = Field(default=1e-7, ge=0.0, le=1.0)


# ---------------------------------------------------------------- lifecycle
def on_startup() -> None:
    log.info(
        "ISDMAAS starting",
        extra={
            "version": ISDMAAS_VERSION,
            "env": settings.env,
            "data_dir": str(settings.data_dir),
            "cors_origins": settings.cors_origins,
        },
    )
    if not settings.is_production:
        log.warning(
            "running in DEVELOPMENT mode: permissive CORS, demo accounts and "
            "interactive docs are enabled. Set ISDMAAS_ENV=production to deploy."
        )
    # Warm the catalog off the request path so the first screen is not the one
    # that pays for a 16 000-object download.
    threading.Thread(
        target=catalog_service.refresh, name="catalog-warmup", daemon=True
    ).start()


# ---------------------------------------------------------------- endpoints
@app.get("/health", tags=["ops"])
def health():
    return {
        "status": "ok",
        "service": "ISDMAAS",
        "version": ISDMAAS_VERSION,
        "environment": settings.env,
        "time_utc": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/ready", tags=["ops"])
def ready():
    """Readiness probe: the catalog is loaded and the registry is reachable."""
    snapshot = catalog_service.snapshot()
    catalog_ok = snapshot is not None and bool(snapshot.entries)
    try:
        get_store(settings.auth_db, settings.token_ttl_hours).user_count()
        store_ok = True
    except Exception:
        log.exception("registry unreachable")
        store_ok = False

    body = {
        "ready": catalog_ok and store_ok,
        "catalog": {
            "loaded": catalog_ok,
            "objects": len(snapshot.entries) if snapshot else 0,
            "fetched_utc": snapshot.fetched_utc.isoformat() if snapshot else None,
            "age_hours": round(snapshot.age_s / 3600, 2) if snapshot else None,
            "source": snapshot.source if snapshot else None,
        },
        "registry": {"reachable": store_ok},
    }
    return JSONResponse(status_code=200 if body["ready"] else 503, content=body)


@app.get("/satellites", tags=["catalog"])
def satellites():
    return {
        norad: {
            "name": name,
            "state_available": settings.state_file(norad).exists(),
        }
        for norad, name in SATELLITES.items()
    }


@app.get("/resolve", tags=["catalog"])
def resolve(q: str = Query(min_length=1, max_length=80)):
    """Resolve a NORAD id, satellite name or COSPAR id to a catalog object."""
    # Distinguishes "the catalog is down" (503) from "no such object" (404).
    # Without this, an unavailable catalog would report every query as not found.
    require_catalog()
    result = catalog_service.resolve(q)
    if "error" in result:
        raise ApiError(
            404, f"'{q}' was not found by NORAD id, name or COSPAR id.", "not_found"
        )
    return result


@app.get("/orbit/{norad}", tags=["catalog"])
def orbit_track(
    norad: int = PathParam(ge=1, le=999999),
    points: int = Query(default=180, ge=8, le=2000),
    period_fraction: float = Query(default=1.0, gt=0.0, le=10.0),
):
    """One orbit's worth of ECI positions (km) for the 3-D view."""
    snapshot = require_catalog()
    entry = snapshot.get(norad)

    if entry is not None:
        period = astro.orbital_period_s(entry.satrec)
        epoch = datetime.now(timezone.utc)
        offsets = np.linspace(0.0, period * period_fraction, points, endpoint=False)
        positions, _, ok = astro.propagate_series(entry.satrec, epoch, offsets)
        track = [p.tolist() for p, good in zip(positions, ok) if good]
        if len(track) < 8:
            raise ApiError(
                422,
                f"The element set for NORAD {norad} does not propagate; it is "
                "likely decayed.",
                code="unprocessable",
            )
        return {
            "norad": norad,
            "name": entry.name,
            "track_km": track,
            "period_s": period,
        }

    primary = load_primary(norad, snapshot)
    elements = astro.elements_from_state(primary.r, primary.v)
    track = [
        _two_body(primary.r, primary.v, i * elements.period_s * period_fraction / points).tolist()
        for i in range(points)
    ]
    return {
        "norad": norad,
        "name": primary.name,
        "track_km": track,
        "period_s": elements.period_s,
    }


def _two_body(r0, v0, dt, steps=50):
    """RK4 two-body propagator, used only when there is no element set to use."""
    r = np.asarray(r0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    if dt == 0.0:
        return r
    h = dt / steps

    def acceleration(position):
        return -MU * position / np.linalg.norm(position) ** 3

    for _ in range(steps):
        k1v, k1r = acceleration(r), v
        k2v, k2r = acceleration(r + 0.5 * h * k1r), v + 0.5 * h * k1v
        k3v, k3r = acceleration(r + 0.5 * h * k2r), v + 0.5 * h * k2v
        k4v, k4r = acceleration(r + h * k3r), v + h * k3v
        r = r + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r


# ---------------------------------------------------------------- screening
@app.get("/screen/{norad}", tags=["screening"])
def screen(
    norad: int,
    _guard: bool = Depends(compute_guard),
    gate_km: float = Query(default=25.0, gt=0.0, le=500.0),
    hours: float = Query(default=72.0, gt=0.0, le=168.0),
    max_results: int = Query(default=30, ge=1, le=200),
):
    """
    Full-catalog conjunction screen for one asset.

    Every reported miss distance comes from a Brent-refined time of closest
    approach, and each secondary's covariance is grown over the actual time to
    that approach.
    """
    hours = min(hours, settings.screen_max_hours)
    snapshot = require_catalog()
    primary = load_primary(norad, snapshot)
    if primary.satrec is None:
        raise ApiError(
            404,
            f"NORAD {norad} has a calibrated state but no element set, so it "
            "cannot be screened against the catalog.",
            code="not_found",
        )
    epoch = datetime.now(timezone.utc)
    result = screening_service.screen_primary(
        primary_sat=primary.satrec,
        primary_name=primary.name,
        catalog=snapshot.as_tuples(),
        epoch=epoch,
        window_s=hours * 3600.0,
        gate_km=gate_km,
        hbr_km=settings.hbr_km,
        primary_sigma_km=primary.sigma_km,
        covariance_source=primary.covariance_source,
        primary_norad=norad,
        coarse_step_s=settings.screen_coarse_step_s,
    )
    return {
        "primary": primary.name,
        "primary_norad": norad,
        "window_hours": hours,
        "gate_km": gate_km,
        "catalog_epoch_utc": snapshot.fetched_utc.isoformat(),
        "catalog_age_hours": round(snapshot.age_s / 3600, 2),
        "objects_screened": result.objects_considered,
        "objects_after_geometric_filter": result.objects_after_geometric_filter,
        "coarse_candidates": result.coarse_candidates,
        "scan_seconds": result.scan_seconds,
        "covariance_source": primary.covariance_source,
        "conjunctions": [c.as_dict() for c in result.conjunctions[:max_results]],
    }


class _MonitorState:
    """Conjunctions already reported, so 'new' means something across polls."""

    def __init__(self):
        self._seen: Dict[str, float] = {}
        self._lock = threading.Lock()

    def mark(self, key: str) -> bool:
        """Record a conjunction; returns True if it had not been seen before."""
        now = time.time()
        with self._lock:
            # Drop keys older than a week so a long-lived process does not grow
            # a set entry for every conjunction it has ever screened.
            if len(self._seen) > 20000:
                cutoff = now - 7 * 86400
                self._seen = {k: t for k, t in self._seen.items() if t > cutoff}
            is_new = key not in self._seen
            self._seen[key] = now
            return is_new

    def reset(self) -> int:
        with self._lock:
            count = len(self._seen)
            self._seen.clear()
            return count


monitor_state = _MonitorState()


@app.get("/monitor", tags=["screening"])
def monitor(
    _guard: bool = Depends(compute_guard),
    watchlist: str = Query(default="", max_length=400),
    gate_km: float = Query(default=25.0, gt=0.0, le=500.0),
    hours: float = Query(default=48.0, gt=0.0, le=168.0),
    alert_pc: float = Query(default=1e-6, ge=0.0, le=1.0),
    alert_miss_km: float = Query(default=5.0, gt=0.0, le=500.0),
):
    """
    Automatic conjunction detection across a watchlist.

    Screens each watched satellite against the full catalog, returns everything
    that breaches the alert thresholds, and flags which ones are NEW since the
    last call so the console can raise a fresh alert rather than re-alerting on
    a conjunction the operator has already seen.
    """
    hours = min(hours, settings.screen_max_hours)
    if watchlist.strip():
        try:
            ids = [int(x) for x in watchlist.split(",") if x.strip()]
        except ValueError as exc:
            raise ApiError(
                400, "watchlist must be comma-separated NORAD ids.", "bad_request"
            ) from exc
        if len(ids) > 20:
            raise ApiError(
                400,
                "A watchlist may hold at most 20 satellites per scan; each one is "
                "a full-catalog screen.",
                code="bad_request",
            )
    else:
        ids = list(SATELLITES.keys())

    snapshot = require_catalog()
    epoch = datetime.now(timezone.utc)
    started = time.monotonic()
    catalog_tuples = snapshot.as_tuples()

    alerts: List[Dict] = []
    all_conjunctions: List[Dict] = []
    errors: List[Dict] = []

    for norad in ids:
        try:
            primary = load_primary(norad, snapshot)
        except ApiError as exc:
            errors.append({"norad": norad, "error": exc.message})
            continue
        if primary.satrec is None:
            errors.append({"norad": norad, "error": "no element set in the catalog"})
            continue
        result = screening_service.screen_primary(
            primary_sat=primary.satrec,
            primary_name=primary.name,
            catalog=catalog_tuples,
            epoch=epoch,
            window_s=hours * 3600.0,
            gate_km=gate_km,
            hbr_km=settings.hbr_km,
            primary_sigma_km=primary.sigma_km,
            covariance_source=primary.covariance_source,
            primary_norad=norad,
            coarse_step_s=settings.screen_coarse_step_s,
        )
        for conjunction in result.conjunctions:
            # Only actionable assessments can raise an alert. A CO_LOCATED
            # pair is one physical object under two catalog entries — the ISS
            # and its own modules share a single element set, so their
            # separation is identically 0 km and the `miss <= alert_miss_km`
            # rule fired on every one of them. Alerting on that teaches
            # operators to ignore alerts.
            triggered = conjunction.is_actionable and (
                (conjunction.pc is not None and conjunction.pc >= alert_pc)
                or conjunction.miss_km <= alert_miss_km
            )
            if not triggered:
                continue
            record = conjunction.as_dict()
            key = (
                f"{norad}-{conjunction.secondary_norad}-{conjunction.tca_utc[:13]}"
            )
            record["is_new"] = monitor_state.mark(key)
            all_conjunctions.append(record)
            if record["is_new"]:
                alerts.append(record)

    def severity(level: str) -> int:
        return sum(
            1 for c in all_conjunctions if str(c.get("risk", "")).upper() == level
        )

    all_conjunctions.sort(key=lambda c: (-(c.get("pc") or 0.0), c["miss_km"]))
    alerts.sort(key=lambda c: (-(c.get("pc") or 0.0), c["miss_km"]))
    critical, high, elevated = severity("CRITICAL"), severity("HIGH"), severity("ELEVATED")

    return {
        "checked_utc": epoch.isoformat(),
        "scan_ms": int((time.monotonic() - started) * 1000),
        "window_hours": hours,
        "gate_km": gate_km,
        "catalog_age_hours": round(snapshot.age_s / 3600, 2),
        "satellites_monitored": len(ids),
        "satellites_ok": len(ids) - len(errors),
        "total_conjunctions": len(all_conjunctions),
        "new_alerts": len(alerts),
        "n_critical": critical,
        "n_high": high,
        "n_elevated": elevated,
        "n_warning": high + elevated,
        "alerts": alerts[:50],
        "conjunctions": all_conjunctions[:50],
        "errors": errors,
    }


@app.post("/monitor/reset", tags=["screening"])
def monitor_reset():
    """Clear the seen-conjunction memory so everything is treated as new again."""
    return {"reset": True, "cleared": monitor_state.reset()}


@app.post("/assess", tags=["screening"])
def assess(request: AssessRequest, _guard: bool = Depends(compute_guard)):
    """Risk assessment between a specific primary and secondary."""
    snapshot = require_catalog()
    primary = load_primary(request.primary_norad, snapshot)
    if primary.satrec is None:
        raise ApiError(
            404, f"NORAD {request.primary_norad} has no element set.", "not_found"
        )
    secondary = catalog_entry(snapshot, request.secondary_norad)

    conjunction = screening_service.assess_pair(
        primary_sat=primary.satrec,
        primary_name=primary.name,
        secondary_sat=secondary.satrec,
        secondary_name=secondary.name,
        secondary_norad=secondary.norad,
        epoch=datetime.now(timezone.utc),
        window_s=request.window_hours * 3600.0,
        hbr_km=settings.hbr_km,
        primary_sigma_km=primary.sigma_km,
        covariance_source=primary.covariance_source,
        primary_norad=request.primary_norad,
        coarse_step_s=settings.screen_coarse_step_s,
    )
    if conjunction is None:
        raise ApiError(
            422,
            "Neither object propagates across the requested window.",
            code="unprocessable",
        )
    payload = conjunction.as_dict()
    payload["primary"] = primary.name
    payload["secondary"] = secondary.name
    payload["risk_level"] = conjunction.risk
    payload["miss_distance_km"] = payload["miss_km"]
    return payload


# ------------------------------------------------------------ CDM ingestion
@app.post("/cdm/assess", tags=["screening"])
def cdm_assess(cdm_text: str = Body(..., embed=True, max_length=500_000)):
    """
    Conjunction assessment from a real CCSDS Conjunction Data Message.

    This is the operator bridge: a CDM carries the real covariance the operator's
    SSA provider computed, so the resulting Pc is operational-grade rather than
    modelled. Without a CDM there is nothing to assess — by design.
    """
    if not cdm_text or not cdm_text.strip():
        raise ApiError(
            400, "Empty CDM. Provide a CCSDS CDM in KVN or XML form.", "bad_request"
        )
    try:
        from cdm_ingest import assess_from_cdm
    except ImportError as exc:
        raise ApiError(503, f"CDM module unavailable: {exc}", "service_unavailable") from exc
    try:
        result = assess_from_cdm(cdm_text, hbr_km=settings.hbr_km)
    except Exception as exc:
        raise ApiError(422, f"Could not assess this CDM: {exc}", "unprocessable") from exc
    return {
        "source": "CDM (real operator covariance)",
        "primary": result.get("primary_name"),
        "secondary": result.get("secondary_name"),
        "tca": result.get("cdm_tca"),
        "miss_km": result.get("miss_distance_km"),
        "cdm_reported_miss_km": result.get("cdm_miss_km"),
        "pc": result.get("pc"),
        "pc_chan_crosscheck": result.get("pc_chan_crosscheck"),
        "risk_level": result.get("risk_level"),
        "relative_speed_kms": result.get("relative_speed_kms"),
        "mahalanobis": result.get("mahalanobis"),
        "note": (
            "Pc computed from the operator's real covariance in the CDM, not "
            "from the modelled TLE-scale covariance."
        ),
    }


# ------------------------------------------------------------------ planning
def _synthetic_threat(primary: PrimaryState, miss_km: float, tca_hours: float):
    """Build a crossing threat at a chosen miss distance, for what-if analysis."""
    rhat = primary.r / np.linalg.norm(primary.r)
    speed = float(np.linalg.norm(primary.v))
    vhat = primary.v / speed
    cross = np.cross(rhat, vhat)
    cross /= np.linalg.norm(cross)
    return primary.r + miss_km * rhat, cross * speed, tca_hours * 3600.0


@app.post("/maneuver-plan", tags=["planning"])
def maneuver_plan(request: ManeuverRequest,
                  _guard: bool = Depends(compute_guard)):
    """Plan and safety-validate an avoidance burn."""
    snapshot = require_catalog()
    primary = load_primary(request.primary_norad, snapshot)
    rp, vp = primary.r, primary.v

    if request.synthetic_miss_km is not None:
        r2, v2, tca_s = _synthetic_threat(
            primary, request.synthetic_miss_km, request.tca_hours
        )
        secondary_name = f"WHAT-IF {request.synthetic_miss_km:g} km"
        prop_time_s = tca_s
    elif request.secondary_norad is not None:
        if primary.satrec is None:
            raise ApiError(
                404,
                f"NORAD {request.primary_norad} has no element set, so a real "
                "conjunction cannot be located.",
                code="not_found",
            )
        secondary = catalog_entry(snapshot, request.secondary_norad)
        approach = astro.find_primary_close_approach(
            primary.satrec, secondary.satrec, primary.epoch, SCREEN_WINDOW_S,
            coarse_step_s=settings.screen_coarse_step_s,
        )
        if approach is None:
            raise ApiError(
                422,
                "No close approach between these objects in the next three days.",
                code="unprocessable",
            )
        rp, vp = approach.r_primary, approach.v_primary
        r2, v2 = approach.r_secondary, approach.v_secondary
        tca_s = max(approach.tca_offset_s, 600.0)
        prop_time_s = max(approach.tca_offset_s, 0.0)
        secondary_name = secondary.name
    else:
        raise ApiError(
            400,
            "Provide either secondary_norad (a real conjunction) or "
            "synthetic_miss_km (a what-if).",
            code="bad_request",
        )

    Cp = rtn_to_eci_cov(np.diag(np.asarray(primary.sigma_km) ** 2), rp, vp)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(prop_time_s), r2, v2)
    mission_sma, _, _ = orbital_elements(rp, vp)

    plan = propose_maneuver(rp, vp, Cp, r2, v2, C2, tca_s, request.sat_mass_kg)
    response = {
        "primary": primary.name,
        "primary_norad": request.primary_norad,
        "secondary": secondary_name,
        "covariance_source": primary.covariance_source,
        "pre_maneuver": plan["pre_maneuver"],
        "action_required": plan["action_required"],
        "pc_action_threshold": PC_THRESHOLD,
    }

    recommendation = plan.get("recommendation")
    if not isinstance(recommendation, dict):
        response["recommendation"] = recommendation or "no maneuver"
        response["verdict"] = "NO_ACTION"
        return response

    sign = 1.0 if recommendation["direction"] == "prograde" else -1.0
    report = validate_maneuver(
        rp, vp, Cp, r2, v2, C2, settings.hbr_km, tca_s,
        recommendation["dv_magnitude_ms"], sign,
        recommendation["burn_lead_time_h"], recommendation["fuel_kg"],
        fuel_available_kg=request.fuel_available_kg,
        mission_sma_km=mission_sma,
        Cp_rtn_diag=primary.sigma_km,
        catalog={e.norad: (e.name, e.satrec) for e in snapshot.entries.values()},
        epoch=primary.epoch,
        primary_norad=request.primary_norad,
        primary_sat=primary.satrec,
    )
    response["recommendation"] = recommendation
    response["options"] = plan["options"]
    response["safety_validation"] = report
    response["verdict"] = report["verdict"]
    return response


@app.post("/autonomous", tags=["planning"])
def autonomous(request: AutoRequest,
               _guard: bool = Depends(compute_guard)):
    """
    The hands-off loop on a real satellite:
      1. load the primary state
      2. screen the real catalog for the worst upcoming conjunction
      3. if its Pc exceeds the threshold that is the threat; otherwise (or in
         inject/historical mode) construct one, clearly labelled as such
      4. plan and safety-validate the maneuver with no human step
      5. return the full decision, labelled real or simulated
    """
    snapshot = require_catalog()
    primary = load_primary(request.primary_norad, snapshot)
    covariance_source = primary.covariance_source
    Cp = primary.covariance_eci
    rp, vp = primary.r, primary.v

    detected = None
    source = "none"
    historical_meta = None

    if request.mode == "auto" and primary.satrec is not None:
        result = screening_service.screen_primary(
            primary_sat=primary.satrec,
            primary_name=primary.name,
            catalog=snapshot.as_tuples(),
            epoch=primary.epoch,
            window_s=SCREEN_WINDOW_S,
            gate_km=25.0,
            hbr_km=settings.hbr_km,
            primary_sigma_km=primary.sigma_km,
            covariance_source=primary.covariance_source,
            primary_norad=request.primary_norad,
            coarse_step_s=settings.screen_coarse_step_s,
        )
        worst = next(
            (c for c in result.conjunctions
             if c.pc is not None and c.pc >= request.risk_threshold_pc),
            None,
        )
        if worst is not None:
            secondary = snapshot.get(worst.secondary_norad)
            approach = astro.find_primary_close_approach(
                primary.satrec, secondary.satrec, primary.epoch, SCREEN_WINDOW_S,
                coarse_step_s=settings.screen_coarse_step_s,
            )
            if approach is not None:
                detected = {
                    "norad": worst.secondary_norad,
                    "name": worst.secondary_name,
                    "miss": approach.miss_km,
                    "t": approach.tca_offset_s,
                    "rp": approach.r_primary,
                    "vp": approach.v_primary,
                    "r2": approach.r_secondary,
                    "v2": approach.v_secondary,
                    "C2": rtn_to_eci_cov(
                        secondary_covariance_rtn(max(approach.tca_offset_s, 0.0)),
                        approach.r_secondary, approach.v_secondary,
                    ),
                }
                Cp = rtn_to_eci_cov(
                    np.diag(np.asarray(primary.sigma_km) ** 2),
                    approach.r_primary, approach.v_primary,
                )
                source = "real_catalog"

    if detected is None and request.mode == "historical":
        detected, Cp, covariance_source, historical_meta = _historical_threat(
            request, primary
        )
        source = "historical_event"

    if detected is None:
        r2, v2, tca_s = _synthetic_threat(
            primary, request.inject_miss_km, request.inject_tca_h
        )
        detected = {
            "norad": None,
            "name": f"SIMULATED THREAT {request.inject_miss_km:g} km",
            "miss": request.inject_miss_km,
            "t": tca_s,
            "rp": rp, "vp": vp, "r2": r2, "v2": v2,
            "C2": rtn_to_eci_cov(secondary_covariance_rtn(tca_s), r2, v2),
        }
        source = "simulated_injection"

    tca_s = max(detected["t"], 600.0)
    mission_sma, _, _ = orbital_elements(detected["rp"], detected["vp"])
    plan = propose_maneuver(
        detected["rp"], detected["vp"], Cp, detected["r2"], detected["v2"],
        detected["C2"], tca_s, request.sat_mass_kg,
    )

    result = {
        "primary": primary.name,
        "detection_source": source,
        "covariance_source": covariance_source,
        "threat": {
            "norad": detected["norad"],
            "name": detected["name"],
            "miss_km": round(detected["miss"], 4),
            "tca_h": round(detected["t"] / 3600, 3),
        },
        "pre_maneuver": plan["pre_maneuver"],
        "action_required": plan["action_required"],
    }
    if historical_meta:
        result["historical"] = historical_meta

    recommendation = plan.get("recommendation")
    if not isinstance(recommendation, dict):
        result["verdict"] = "NO_ACTION"
        result["recommendation"] = (
            recommendation or "Pc below the action threshold — monitoring only."
        )
        return result

    sign = 1.0 if recommendation["direction"] == "prograde" else -1.0
    report = validate_maneuver(
        detected["rp"], detected["vp"], Cp, detected["r2"], detected["v2"],
        detected["C2"], settings.hbr_km, tca_s,
        recommendation["dv_magnitude_ms"], sign,
        recommendation["burn_lead_time_h"], recommendation["fuel_kg"],
        fuel_available_kg=request.fuel_available_kg,
        mission_sma_km=mission_sma,
        Cp_rtn_diag=primary.sigma_km,
        catalog={e.norad: (e.name, e.satrec) for e in snapshot.entries.values()},
        epoch=primary.epoch,
        primary_norad=request.primary_norad,
        primary_sat=primary.satrec,
    )
    result["recommendation"] = recommendation
    result["safety_validation"] = report
    result["verdict"] = report["verdict"]
    # The loop reaches a decision without a human step; the decision itself is
    # still a recommendation. ISDMAAS does not command the spacecraft.
    result["autonomous_action"] = (
        "RECOMMENDED_FOR_UPLINK" if report["verdict"] == "APPROVED"
        else "HELD_FOR_REVIEW"
    )
    return result


def _historical_threat(request: AutoRequest, primary: PrimaryState):
    """Seed a threat from a documented historical event's published geometry."""
    try:
        from historical_events import EVENTS
    except ImportError as exc:
        raise ApiError(
            503, f"Historical event library unavailable: {exc}", "service_unavailable"
        ) from exc
    event = EVENTS.get(request.event_key)
    if event is None:
        raise ApiError(
            404,
            f"Unknown event_key '{request.event_key}'. Available: "
            f"{', '.join(sorted(EVENTS))}",
            code="not_found",
        )
    scenario = event["scenario"]
    rhat = primary.r / np.linalg.norm(primary.r)
    speed = float(np.linalg.norm(primary.v))
    vhat = primary.v / speed
    cross = np.cross(rhat, vhat)
    cross /= np.linalg.norm(cross)

    r2 = primary.r + scenario["miss_km"] * rhat
    v2 = primary.v + cross * scenario.get("rel_velocity_kms", 10.0)

    # Tracking uncertainty at these encounters was comparable to the miss
    # distance — that is precisely why a sub-kilometre pass was a genuine
    # collision risk rather than a clean miss.
    scale = max(scenario["miss_km"], 0.05)
    tracking_rtn = np.diag([0.6 * scale, 1.5 * scale, 0.6 * scale]) ** 2
    Cp = rtn_to_eci_cov(tracking_rtn, primary.r, primary.v)
    detected = {
        "norad": event["secondary"]["norad"],
        "name": event["secondary"]["name"],
        "miss": scenario["miss_km"],
        "t": scenario.get("tca_lead_h", 8) * 3600.0,
        "rp": primary.r, "vp": primary.v, "r2": r2, "v2": v2,
        "C2": rtn_to_eci_cov(tracking_rtn, r2, v2),
    }
    meta = {
        "event_key": request.event_key,
        "title": event["title"],
        "date_utc": event["date_utc"],
        "outcome": event["outcome"],
        "primary_documented": event["primary"]["name"],
        "documented": event["documented"],
        "note": event["isdmaas_point"],
    }
    return detected, Cp, "documented_event_tracking", meta


# ------------------------------------------------------------- live tracking
@app.get("/live/catalog", tags=["catalog"])
def live_catalog(
    group: str = Query(default="debris", max_length=48, pattern=r"^[A-Za-z0-9_\-]+$"),
    limit: int = Query(default=400, ge=1, le=5000),
    shell_min: Optional[float] = Query(default=None, ge=0.0, le=100000.0),
    shell_max: Optional[float] = Query(default=None, ge=0.0, le=100000.0),
):
    """
    Cataloged objects for browser-side propagation with satellite.js.

    Reads the local cache built by `python debris_data.py sync`.
    """
    import json as json_module
    import sqlite3

    from sgp4 import omm as omm_module
    from sgp4.exporter import export_tle

    db_path = settings.object_cache_db
    if not db_path.exists():
        raise ApiError(
            503,
            "The object cache has not been built. Run: python debris_data.py sync",
            code="service_unavailable",
        )
    # Opened read-write, then restricted with PRAGMA query_only.
    #
    # `file:...?mode=ro` looks safer and is not usable here: the object cache is
    # a WAL database, and SQLite needs write access to the -wal and -shm
    # sidecars even to READ from one. A read-only URI connection fails with
    # "attempt to write a readonly database" the moment the WAL needs attaching,
    # which took this endpoint out entirely. query_only gives the same guarantee
    # - no statement on this connection can modify the database - without
    # blocking the housekeeping a WAL read requires.
    connection = sqlite3.connect(str(db_path), timeout=15.0)
    connection.execute("PRAGMA query_only=ON")
    connection.row_factory = sqlite3.Row
    try:
        query = "SELECT * FROM objects WHERE 1=1"
        args: List = []
        if group and group != "all":
            query += " AND grp=?"
            args.append(group)
        if shell_min is not None:
            query += " AND apogee_km >= ?"
            args.append(shell_min)
        if shell_max is not None:
            query += " AND perigee_km <= ?"
            args.append(shell_max)
        query += " LIMIT ?"
        args.append(limit)
        rows = connection.execute(query, args).fetchall()
    except sqlite3.Error as exc:
        raise ApiError(503, f"Object cache is unreadable: {exc}", "service_unavailable") from exc
    finally:
        connection.close()

    objects = []
    for row in rows:
        try:
            satrec = Satrec()
            omm_module.initialize(satrec, json_module.loads(row["omm_json"]))
            line1, line2 = export_tle(satrec)
        except Exception:
            continue
        objects.append(
            {
                "norad": row["norad"],
                "name": row["name"],
                "type": row["object_type"],
                "tle1": line1,
                "tle2": line2,
                "apogee_km": row["apogee_km"],
                "perigee_km": row["perigee_km"],
            }
        )
    return {"count": len(objects), "group": group, "objects": objects}


@app.get("/live/conjunctions/{norad}", tags=["screening"])
def live_conjunctions(
    norad: int,
    _guard: bool = Depends(compute_guard),
    range_km: float = Query(default=50.0, gt=0.0, le=500.0),
    hours: float = Query(default=24.0, gt=0.0, le=168.0),
):
    """Upcoming close approaches to a primary, from the cached object catalog."""
    snapshot = require_catalog()
    entry = catalog_entry(snapshot, norad)
    epoch = datetime.now(timezone.utc)
    result = screening_service.screen_primary(
        primary_sat=entry.satrec,
        primary_name=entry.name,
        catalog=snapshot.as_tuples(),
        epoch=epoch,
        window_s=hours * 3600.0,
        gate_km=range_km,
        hbr_km=settings.hbr_km,
        primary_norad=norad,
        coarse_step_s=settings.screen_coarse_step_s,
    )
    return {
        "primary": norad,
        "primary_name": entry.name,
        "range_km": range_km,
        "window_hours": hours,
        "scan_seconds": result.scan_seconds,
        "conjunctions": [c.as_dict() for c in result.conjunctions[:50]],
    }


# ---------------------------------------------------------------- SOCRATES
@app.get("/socrates", tags=["catalog"])
def socrates(
    norad: Optional[int] = Query(default=None, ge=1, le=999999),
    order: str = Query(default="MAXPROB", pattern="^(MAXPROB|MINRANGE)$"),
    maxrows: int = Query(default=25, ge=1, le=100),
):
    """
    CelesTrak's published SOCRATES conjunction feed — an independent cross-check
    on ISDMAAS's own screening. Cached for an hour to respect CelesTrak.
    """
    try:
        return get_socrates_feed().fetch(norad=norad, order=order, max_rows=maxrows)
    except RuntimeError as exc:
        raise ApiError(503, str(exc), code="service_unavailable") from exc


# ------------------------------------------------------------ event library
def _event_library():
    library_path = settings.data_dir.parent.parent / "phase12"
    if str(library_path) not in sys.path:
        sys.path.append(str(library_path))
    try:
        import event_library

        return event_library
    except ImportError as exc:
        raise ApiError(
            503, f"Event library unavailable: {exc}", "service_unavailable"
        ) from exc


@app.get("/library", tags=["library"])
def library():
    """The historical event library, grouped by category."""
    module = _event_library()
    return {
        "categories": module.by_category(),
        "n_events": len(module.EVENTS),
        "quantified": module.quantified_keys(),
    }


@app.get("/library/{key}", tags=["library"])
def library_event(key: str = PathParam(max_length=80, pattern=r"^[A-Za-z0-9_\-]+$")):
    """Full record for one documented event."""
    module = _event_library()
    event = module.get(key)
    if not event:
        raise ApiError(404, f"Event '{key}' not found.", "not_found")
    return event


@app.get("/historical-events", tags=["library"])
def historical_events():
    """Documented events available for replay in the console."""
    try:
        from historical_events import EVENTS
    except ImportError as exc:
        raise ApiError(
            503, f"Historical event library unavailable: {exc}", "service_unavailable"
        ) from exc
    return [
        {
            "event_key": key,
            "title": event["title"],
            "date_utc": event["date_utc"],
            "outcome": event["outcome"],
            "primary": event["primary"]["name"],
            "primary_norad": event["primary"]["norad"],
            "secondary": event["secondary"]["name"],
            "miss_km": event["scenario"]["miss_km"],
            "rel_velocity_kms": event["scenario"].get("rel_velocity_kms"),
        }
        for key, event in EVENTS.items()
    ]


# -------------------------------------------------------------- data syncing
@app.get("/sync-status", tags=["ops"])
def sync_status():
    """When each public data source was last refreshed."""
    snapshot = catalog_service.snapshot()
    return {
        "last_sync": {
            "catalog": snapshot.fetched_utc.isoformat() if snapshot else None,
        },
        "catalog_objects": len(snapshot.entries) if snapshot else 0,
        "catalog_age_hours": round(snapshot.age_s / 3600, 2) if snapshot else None,
        "catalog_source": snapshot.source if snapshot else None,
        "now_utc": datetime.now(timezone.utc).isoformat(),
    }


_sync_lock = threading.Lock()


@app.post("/sync-now", tags=["ops"])
def sync_now(
    _guard: bool = Depends(compute_guard),
    groups: str = Query(
        default="active,cosmos-2251-debris,cosmos-1408-debris,fengyun-1c-debris",
        max_length=400,
        pattern=r"^[A-Za-z0-9_,\-]+$",
    )
):
    """
    Refresh the public data caches.

    Serialized behind a lock: several console tabs opening at once used to fire
    several simultaneous full-catalog downloads at CelesTrak, which is exactly
    the behaviour their rate-limit guidance asks callers to avoid.
    """
    if not _sync_lock.acquire(blocking=False):
        return {
            "synced": {},
            "errors": {},
            "note": "A refresh is already running; this request was coalesced.",
        }
    try:
        result: Dict[str, Dict] = {"synced": {}, "errors": {}}
        snapshot = catalog_service.refresh(force=True)
        if snapshot is not None:
            result["synced"]["catalog"] = "ok"
            result["catalog_objects"] = len(snapshot.entries)
            result["catalog_synced_utc"] = snapshot.fetched_utc.isoformat()
            result["catalog_source"] = snapshot.source
        else:
            result["errors"]["catalog"] = "no cache and the upstream fetch failed"

        try:
            import debris_data

            group_list = [g.strip() for g in groups.split(",") if g.strip()]
            debris_data.sync(groups=group_list)
            for group in group_list:
                result["synced"][group] = "ok"
        except Exception as exc:
            log.warning("object cache sync failed", exc_info=True)
            result["errors"]["object_cache"] = str(exc)
        return result
    finally:
        _sync_lock.release()


# ------------------------------------------------------- mount the sub-apps
from auth_store import install_auth  # noqa: E402
from user_conjunctions import install_user_pipeline  # noqa: E402
from user_screening import install_screening  # noqa: E402

install_auth(app)
install_user_pipeline(app)
install_screening(app)
