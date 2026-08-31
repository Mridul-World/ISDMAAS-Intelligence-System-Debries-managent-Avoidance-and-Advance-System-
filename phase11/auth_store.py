"""
auth_store.py — operator accounts, sessions and satellite ownership.

Security model
--------------
Each satellite OPERATOR has an account. Each operator registers their own
satellite element sets. Only the owning operator has maneuver authority over
their satellites; planning a burn on someone else's asset is refused with 403.

What this is now (and was not before):

  * Passwords are PBKDF2-HMAC-SHA256 at 240 000 iterations, salted per user, and
    legacy single-round SHA-256 records are upgraded on the owner's next login.
  * Session tokens are 256-bit opaque values; only their SHA-256 digest is
    stored, and they expire.
  * Login and registration are rate limited per client address, and a failed
    login costs the same time as a successful one so the response does not
    reveal whether the username exists.
  * Credentials live in a gitignored SQLite database, not in a JSON file inside
    the repository.

Honest remaining limits, which belong in any pilot conversation: this is
single-node session storage with no multi-factor, no password reset flow and no
audit export. It is appropriate for a pilot behind TLS; it is not a replacement
for an identity provider.

Wiring (in phase11_api.py):
    from auth_store import install_auth
    install_auth(app)
"""
from __future__ import annotations

import secrets
import sqlite3
from datetime import datetime, timezone
from typing import Dict, Optional

from fastapi import APIRouter, Depends, Header, Request
from pydantic import BaseModel, Field

from isdmaas_core.config import get_settings
from isdmaas_core.errors import ApiError
from isdmaas_core.logging_config import get_logger
from isdmaas_core.security import (
    SlidingWindowLimiter,
    hash_password,
    normalize_username,
    new_session_token,
    parse_bearer,
    validate_password,
    verify_password,
)
from isdmaas_core.store import Store, get_store
from isdmaas_core.tle import validate_tle

log = get_logger("isdmaas.auth")
router = APIRouter(tags=["auth"])

# Credentials published in this repository's own demo material. They are seeded
# only in development, and the password policy refuses them for real accounts.
DEMO_OPERATORS = [
    ("operator_a", "alpha123", "90001", "ALPHASAT-DEMO"),
    ("operator_b", "bravo123", "90002", "BRAVOSAT-DEMO"),
]

_login_limiter: Optional[SlidingWindowLimiter] = None
_register_limiter: Optional[SlidingWindowLimiter] = None


def _store() -> Store:
    settings = get_settings()
    return get_store(settings.auth_db, settings.token_ttl_hours)


def _limiters():
    global _login_limiter, _register_limiter
    if _login_limiter is None:
        settings = get_settings()
        _login_limiter = SlidingWindowLimiter(
            settings.login_rate_limit, settings.login_rate_window_s
        )
        _register_limiter = SlidingWindowLimiter(
            max(3, settings.login_rate_limit // 2), settings.login_rate_window_s
        )
    return _login_limiter, _register_limiter


def client_key(request: Request) -> str:
    """
    Rate-limit key for a request: the peer address, and nothing else.

    It is tempting to read X-Forwarded-For here so the limiter sees the real
    client behind a proxy. Do not. That header is set by the caller, so keying on
    it hands an attacker a fresh, empty rate-limit bucket on every request simply
    by incrementing a number — the limiter would count to ten and never fire.
    Mixing it into a composite key does not help either: a new forwarded value
    still produces a new key.

    The correct place to resolve the real client is the ASGI server, which knows
    which peers are trusted proxies. Run uvicorn with `--proxy-headers
    --forwarded-allow-ips '<proxy addresses>'` and `request.client.host` becomes
    the real client address for requests that actually came through the proxy,
    and stays the socket address for everything else. DEPLOY.md carries that
    flag in every example command.
    """
    return request.client.host if request.client else "unknown"


# ---------------------------------------------------------------- demo TLEs
def _tle_checksum(line: str) -> str:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return str(total % 10)


def _finalize(line: str) -> str:
    line = line.ljust(68)[:68]
    return line + _tle_checksum(line)


def _demo_tles() -> Dict[str, tuple]:
    """
    Two fictitious satellites on crossing orbits.

    NORAD ids sit in the 90000+ analyst range so they can never collide with a
    real catalog id. The epoch is generated at seed time because differential
    nodal precession would separate the two planes within days of a fixed epoch,
    quietly turning the demo conjunction into a non-event.
    """
    now = datetime.now(timezone.utc)
    day_of_year = (
        now - datetime(now.year, 1, 1, tzinfo=timezone.utc)
    ).total_seconds() / 86400.0 + 1
    epoch = f"{now.year % 100:02d}{day_of_year:012.8f}"
    return {
        "90001": (
            "ALPHASAT-DEMO",
            _finalize(f"1 90001U 26900A   {epoch}  .00001000  00000-0  10000-3 0  999"),
            _finalize("2 90001  90.0000 120.0000 0001000  90.0000 250.0000 15.50000000    1"),
        ),
        "90002": (
            "BRAVOSAT-DEMO",
            _finalize(f"1 90002U 26900B   {epoch}  .00001000  00000-0  10000-3 0  999"),
            _finalize("2 90002  90.0000 300.0000 0001000  90.0000  90.2150 15.50000000    1"),
        ),
    }


def seed_demo_operators() -> None:
    """Create the demo accounts if the registry is empty. Development only."""
    settings = get_settings()
    if not settings.enable_demo_seed or settings.is_production:
        return
    store = _store()
    if store.user_count() > 0:
        return
    tles = _demo_tles()
    for username, password, norad, _name in DEMO_OPERATORS:
        try:
            store.create_user(
                username, hash_password(password, settings.pbkdf2_iterations)
            )
        except sqlite3.IntegrityError:
            continue
        name, line1, line2 = tles[norad]
        store.upsert_satellite(norad, username, name, line1, line2)
    log.warning(
        "seeded %d DEMO operator accounts with published passwords — "
        "development mode only; set ISDMAAS_ENV=production to disable",
        len(DEMO_OPERATORS),
    )


# ------------------------------------------------------- auth primitives
def user_from_token(authorization: Optional[str]) -> Optional[str]:
    """Resolve `Bearer <token>` to a username, or None."""
    token = parse_bearer(authorization)
    if not token:
        return None
    session = _store().resolve_session(token)
    return session.username if session else None


def get_satellite(norad) -> Optional[dict]:
    """Registered satellite record as a plain dict, or None."""
    satellite = _store().get_satellite(str(norad))
    if satellite is None:
        return None
    return {
        "owner": satellite.owner,
        "name": satellite.name,
        "tle1": satellite.tle1,
        "tle2": satellite.tle2,
        "updated_utc": satellite.updated_utc,
    }


def require_user(authorization: Optional[str] = Header(None)) -> str:
    """FastAPI dependency: the authenticated username, or 401."""
    username = user_from_token(authorization)
    if not username:
        raise ApiError(
            401,
            "Operator login required.",
            code="unauthenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return username


def require_owner(norad, authorization: Optional[str]) -> str:
    """403 unless the token's user owns the satellite. Returns the username."""
    username = user_from_token(authorization)
    if not username:
        raise ApiError(
            401,
            "Operator login required.",
            code="unauthenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    satellite = _store().get_satellite(str(norad))
    if satellite is None:
        raise ApiError(
            404, f"Satellite {norad} is not registered to any operator.", "not_found"
        )
    if satellite.owner != username:
        # The owner's identity is deliberately NOT disclosed: knowing which
        # operator holds an asset is itself commercially sensitive.
        log.warning(
            "authority denied", extra={"actor": username, "norad": str(norad)}
        )
        raise ApiError(
            403,
            f"You do not have maneuver authority over NORAD {norad}. "
            "Maneuver authority is restricted to the owning operator.",
            code="forbidden",
        )
    return username


# ------------------------------------------------------------------- models
class Credentials(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=1024)


class NewSatellite(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    tle1: str = Field(min_length=60, max_length=200)
    tle2: str = Field(min_length=60, max_length=200)


# ---------------------------------------------------------------- endpoints
@router.post("/auth/register", status_code=201)
def register(credentials: Credentials, request: Request):
    _, limiter = _limiters()
    decision = limiter.hit(client_key(request))
    if not decision.allowed:
        raise ApiError(
            429,
            "Too many registration attempts. Try again shortly.",
            code="rate_limited",
            headers={"Retry-After": str(decision.retry_after_s)},
        )
    try:
        username = normalize_username(credentials.username)
        validate_password(credentials.password)
    except ValueError as exc:
        raise ApiError(422, str(exc), code="unprocessable") from exc

    settings = get_settings()
    try:
        _store().create_user(
            username, hash_password(credentials.password, settings.pbkdf2_iterations)
        )
    except sqlite3.IntegrityError as exc:
        raise ApiError(409, "That username is already registered.", "conflict") from exc
    log.info("operator registered", extra={"actor": username})
    return {"ok": True, "username": username}


@router.post("/auth/login")
def login(credentials: Credentials, request: Request):
    limiter, _ = _limiters()
    key = client_key(request)
    decision = limiter.hit(key)
    if not decision.allowed:
        raise ApiError(
            429,
            "Too many failed sign-in attempts. Try again shortly.",
            code="rate_limited",
            headers={"Retry-After": str(decision.retry_after_s)},
        )

    settings = get_settings()
    store = _store()
    try:
        username = normalize_username(credentials.username)
    except ValueError:
        username = ""

    user = store.get_user(username) if username else None
    stored_hash = user["password_hash"] if user else ""
    valid, needs_rehash = verify_password(credentials.password, stored_hash)
    if not user or not valid:
        if not user:
            # Spend comparable work on an unknown username so response time does
            # not distinguish "no such account" from "wrong password".
            hash_password(secrets.token_urlsafe(16), settings.pbkdf2_iterations)
        log.info("failed sign-in", extra={"attempted_user": username or "(invalid)"})
        raise ApiError(401, "Invalid username or password.", code="unauthenticated")
    if user["disabled"]:
        raise ApiError(403, "This account is disabled.", code="forbidden")

    if needs_rehash:
        store.update_password_hash(
            username, hash_password(credentials.password, settings.pbkdf2_iterations)
        )
        log.info("password hash upgraded", extra={"actor": username})

    store.purge_expired_sessions()
    token = new_session_token()
    expires = store.create_session(username, token)
    store.mark_login(username)
    limiter.reset(key)
    log.info("operator signed in", extra={"actor": username})
    return {
        "token": token,
        "username": username,
        "expires_utc": expires.isoformat(),
        "token_type": "Bearer",
    }


@router.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    token = parse_bearer(authorization)
    if token:
        _store().delete_session(token)
    return {"ok": True}


@router.post("/auth/logout-all")
def logout_all(username: str = Depends(require_user)):
    """Invalidate every session for the caller — the response to a lost token."""
    revoked = _store().delete_sessions_for(username)
    log.info("all sessions revoked", extra={"actor": username, "revoked": revoked})
    return {"ok": True, "sessions_revoked": revoked}


@router.get("/auth/me")
def me(username: str = Depends(require_user)):
    satellites = {
        s.norad: {"name": s.name} for s in _store().satellites_for(username)
    }
    return {"username": username, "satellites": satellites}


@router.get("/my/satellites")
def my_satellites(username: str = Depends(require_user)):
    store = _store()
    mine = [
        {"norad": s.norad, "name": s.name, "updated_utc": s.updated_utc}
        for s in store.satellites_for(username)
    ]
    # Other operators' assets are listed so a cross-operator conjunction can be
    # assessed, but the owner is reported as an opaque label rather than an
    # account name — who flies what is commercially sensitive.
    others = [
        {"norad": s.norad, "name": s.name, "owner": "another operator"}
        for s in store.list_satellites()
        if s.owner != username
    ]
    return {"mine": mine, "others": others}


@router.post("/my/satellites", status_code=201)
def add_satellite(payload: NewSatellite, username: str = Depends(require_user)):
    line1, line2 = payload.tle1.strip(), payload.tle2.strip()

    # Full integrity validation, not just a parse. sgp4's parser accepts a
    # truncated line, a broken checksum, mismatched catalog numbers between the
    # two lines, and a corrupted inclination - and then returns a propagator that
    # produces a perfectly plausible altitude. An element set corrupted in
    # transit would otherwise be screened and reported with full confidence.
    report = validate_tle(line1, line2)
    if not report.valid:
        raise ApiError(
            422,
            "This element set failed validation: " + "; ".join(report.errors),
            code="unprocessable",
            extra={"tle_errors": report.errors},
        )
    norad = str(report.norad)

    store = _store()
    existing = store.get_satellite(norad)
    if existing and existing.owner != username:
        raise ApiError(
            409,
            f"NORAD {norad} is already registered to another operator.",
            code="conflict",
        )
    name = payload.name.strip() or norad
    store.upsert_satellite(norad, username, name, line1, line2)

    # Durable history for operations and future training. Optional by design:
    # a failure to log history must never block registering an asset.
    try:
        import ops_db

        ops_db.record_tle(norad, name, username, line1, line2)
    except Exception:
        log.warning("ops_db TLE history write failed", exc_info=True)
    log.info("satellite registered", extra={"actor": username, "norad": norad})
    return {"ok": True, "norad": norad, "name": name}


@router.delete("/my/satellites/{norad}")
def remove_satellite(norad: str, username: str = Depends(require_user)):
    if not _store().delete_satellite(norad, username):
        raise ApiError(
            404,
            f"NORAD {norad} is not registered to you.",
            code="not_found",
        )
    log.info("satellite removed", extra={"actor": username, "norad": norad})
    return {"ok": True, "norad": norad}


@router.get("/auth/demo-info")
def demo_info():
    """
    The demo scenario, for the console's help panel.

    Present only in development. In production this returns 404 rather than
    publishing working credentials for the deployment.
    """
    settings = get_settings()
    if settings.is_production or not settings.enable_demo_seed:
        raise ApiError(
            404, "Demo accounts are not enabled on this deployment.", "not_found"
        )
    return {
        "development_only": True,
        "operators": [
            {"username": u, "password": p, "satellite": f"{n} ({norad})"}
            for u, p, norad, n in DEMO_OPERATORS
        ],
        "scenario": (
            "ALPHASAT-DEMO and BRAVOSAT-DEMO are fictitious satellites on "
            "crossing orbits. Sign in as each operator to see that only the "
            "owner can approve a maneuver for their own satellite."
        ),
        "warning": (
            "These passwords are published in the project's source. They are "
            "seeded in development only and are rejected by the password policy "
            "for real accounts."
        ),
    }


def install_auth(app):
    app.include_router(router)
    settings = get_settings()
    store = get_store(settings.auth_db, settings.token_ttl_hours)
    store.import_legacy_json(settings.data_dir.parent / "auth_store.json")
    seed_demo_operators()
    store.purge_expired_sessions()
    return app
