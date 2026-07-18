"""
auth_store.py — multi-operator login & satellite ownership for the ISDMAAS demo.
=================================================================================
Implements the meeting-agreed demo flow (Jun 30, Partho Ghosh):
  - Each satellite OPERATOR logs in with their own account.
  - Each operator registers their OWN satellite TLEs.
  - Only the owning operator has authority over their satellites (maneuver
    approval on someone else's asset -> 403 SECURITY VIOLATION).
  - Two demo operators are pre-seeded with fictitious satellites on a collision
    course, so the two-owner approval story can be shown immediately.

DEMO-GRADE security (honest note): tokens are random hex held in a JSON file;
passwords are salted-SHA256. Fine for a live demo, NOT production auth (no
HTTPS enforcement, no expiry, no rate limiting). Say exactly that if asked.

Wiring (2 lines at the BOTTOM of phase11_api.py):
    from auth_store import install_auth
    install_auth(app)
"""
import os, json, hashlib, secrets, threading
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel

_STORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "auth_store.json")
_LOCK = threading.Lock()

# ----------------------------------------------------------------------------
# storage helpers
# ----------------------------------------------------------------------------
def _load():
    if not os.path.exists(_STORE):
        return {"users": {}, "tokens": {}, "satellites": {}}
    with open(_STORE, encoding="utf-8") as f:
        return json.load(f)

def _save(d):
    tmp = _STORE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=1)
    os.replace(tmp, _STORE)

def _hash_pw(pw, salt):
    return hashlib.sha256((salt + pw).encode()).hexdigest()

# ----------------------------------------------------------------------------
# fictitious demo TLEs — two satellites in the SAME orbital plane, opposite
# phasing drift, built so their paths cross (a plausible conjunction geometry).
# Checksums are computed programmatically so the TLEs are always valid.
# ----------------------------------------------------------------------------
def _tle_checksum(line):
    s = 0
    for ch in line[:68]:
        if ch.isdigit(): s += int(ch)
        elif ch == "-": s += 1
    return str(s % 10)

def _finalize(l):
    l = l.ljust(68)[:68]
    return l + _tle_checksum(l)

def _demo_tles():
    # NORADs in the 90000+ "analyst/fictitious" range so they never collide with
    # real catalog IDs. Geometry: SAME RAAN and node-crossing phase but planes
    # inclined 53° vs 73° -> the two orbits intersect at the node line, and both
    # satellites arrive there together each rev (same mean motion) with a
    # ~2.6 km/s crossing speed. A tiny mean-anomaly offset keeps the miss at a
    # few hundred metres instead of exactly zero. The epoch is generated AT SEED
    # TIME so the geometry is valid whenever the demo runs (differential nodal
    # precession would otherwise separate the planes within days).
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    doy = (now - datetime(now.year, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0 + 1
    epoch = f"{now.year % 100:02d}{doy:012.8f}"
    a1 = _finalize(f"1 90001U 26900A   {epoch}  .00001000  00000-0  10000-3 0  999")
    a2 = _finalize("2 90001  90.0000 120.0000 0001000  90.0000 250.0000 15.50000000    1")
    b1 = _finalize(f"1 90002U 26900B   {epoch}  .00001000  00000-0  10000-3 0  999")
    b2 = _finalize("2 90002  90.0000 300.0000 0001000  90.0000  90.2150 15.50000000    1")
    return {"90001": ("ALPHASAT-DEMO", a1, a2), "90002": ("BRAVOSAT-DEMO", b1, b2)}

def _seed_if_empty(d):
    if d["users"]:
        return d
    for uname, pw, norad in [("operator_a", "alpha123", "90001"),
                             ("operator_b", "bravo123", "90002")]:
        salt = secrets.token_hex(8)
        d["users"][uname] = {"salt": salt, "pw": _hash_pw(pw, salt)}
        name, l1, l2 = _demo_tles()[norad]
        d["satellites"][norad] = {"owner": uname, "name": name, "tle1": l1, "tle2": l2}
    return d

# ----------------------------------------------------------------------------
# auth primitives (used by user_conjunctions too)
# ----------------------------------------------------------------------------
def user_from_token(authorization):
    """Resolve 'Bearer <token>' -> username, or None."""
    if not authorization or not authorization.startswith("Bearer "):
        return None
    tok = authorization[7:].strip()
    with _LOCK:
        d = _load()
        return d["tokens"].get(tok)

def get_satellite(norad):
    with _LOCK:
        return _load()["satellites"].get(str(norad))

def require_owner(norad, authorization):
    """403 unless the token's user owns the satellite. Returns username."""
    user = user_from_token(authorization)
    if not user:
        raise HTTPException(401, "Not logged in. Operator login required.")
    sat = get_satellite(norad)
    if not sat:
        raise HTTPException(404, f"Satellite {norad} is not registered to any operator.")
    if sat["owner"] != user:
        raise HTTPException(403,
            f"SECURITY: {user} has no authority over NORAD {norad} "
            f"(owned by {sat['owner']}). Maneuver authority is restricted to the "
            f"owning operator.")
    return user

# ----------------------------------------------------------------------------
# API router
# ----------------------------------------------------------------------------
router = APIRouter()

class Creds(BaseModel):
    username: str
    password: str

class NewSat(BaseModel):
    name: str
    tle1: str
    tle2: str

@router.post("/auth/register")
def register(c: Creds):
    uname = c.username.strip().lower()
    if not uname or not c.password:
        raise HTTPException(400, "username and password required")
    with _LOCK:
        d = _seed_if_empty(_load())
        if uname in d["users"]:
            raise HTTPException(409, "username already exists")
        salt = secrets.token_hex(8)
        d["users"][uname] = {"salt": salt, "pw": _hash_pw(c.password, salt)}
        _save(d)
    return {"ok": True, "username": uname}

@router.post("/auth/login")
def login(c: Creds):
    uname = c.username.strip().lower()
    with _LOCK:
        d = _seed_if_empty(_load())
        u = d["users"].get(uname)
        if not u or _hash_pw(c.password, u["salt"]) != u["pw"]:
            raise HTTPException(401, "invalid username or password")
        tok = secrets.token_hex(16)
        d["tokens"][tok] = uname
        _save(d)
    return {"token": tok, "username": uname}

@router.post("/auth/logout")
def logout(authorization: str = Header(None)):
    if authorization and authorization.startswith("Bearer "):
        tok = authorization[7:].strip()
        with _LOCK:
            d = _load()
            d["tokens"].pop(tok, None)
            _save(d)
    return {"ok": True}

@router.get("/auth/me")
def me(authorization: str = Header(None)):
    user = user_from_token(authorization)
    if not user:
        raise HTTPException(401, "not logged in")
    with _LOCK:
        d = _load()
        mine = {n: {"name": s["name"]} for n, s in d["satellites"].items()
                if s["owner"] == user}
    return {"username": user, "satellites": mine}

@router.get("/my/satellites")
def my_sats(authorization: str = Header(None)):
    user = user_from_token(authorization)
    if not user:
        raise HTTPException(401, "not logged in")
    with _LOCK:
        d = _load()
    mine = [{"norad": n, "name": s["name"]} for n, s in d["satellites"].items()
            if s["owner"] == user]
    others = [{"norad": n, "name": s["name"], "owner": s["owner"]}
              for n, s in d["satellites"].items() if s["owner"] != user]
    return {"mine": mine, "others": others}

@router.post("/my/satellites")
def add_sat(s: NewSat, authorization: str = Header(None)):
    user = user_from_token(authorization)
    if not user:
        raise HTTPException(401, "not logged in — log in to register a satellite")
    l1, l2 = s.tle1.strip(), s.tle2.strip()
    if not (l1.startswith("1 ") and l2.startswith("2 ") and len(l1) >= 68 and len(l2) >= 68):
        raise HTTPException(422, "TLE lines look malformed (need standard 69-char lines 1/ and 2/)")
    # validate the TLE actually propagates
    try:
        from sgp4.api import Satrec
        sat = Satrec.twoline2rv(l1, l2)
        norad = str(sat.satnum)
    except Exception as e:
        raise HTTPException(422, f"TLE failed to parse: {e}")
    with _LOCK:
        d = _seed_if_empty(_load())
        existing = d["satellites"].get(norad)
        if existing and existing["owner"] != user:
            raise HTTPException(409, f"NORAD {norad} already registered to {existing['owner']}")
        d["satellites"][norad] = {"owner": user, "name": s.name.strip() or norad,
                                  "tle1": l1, "tle2": l2}
        _save(d)
    # durable history for operations + future training (ops_db is optional)
    try:
        import ops_db
        ops_db.record_tle(norad, s.name.strip() or norad, user, l1, l2)
    except Exception:
        pass
    return {"ok": True, "norad": norad, "name": s.name}

@router.get("/auth/demo-info")
def demo_info():
    """The demo scenario, for the console's help panel."""
    with _LOCK:
        _save(_seed_if_empty(_load()))
    return {
        "operators": [
            {"username": "operator_a", "password": "alpha123", "satellite": "ALPHASAT-DEMO (90001)"},
            {"username": "operator_b", "password": "bravo123", "satellite": "BRAVOSAT-DEMO (90002)"},
        ],
        "scenario": "ALPHASAT-DEMO and BRAVOSAT-DEMO are fictitious satellites on "
                    "crossing orbits. Log in as each operator to see that only the "
                    "owner can approve a maneuver for their own satellite.",
    }

def install_auth(app):
    app.include_router(router)
    with _LOCK:
        _save(_seed_if_empty(_load()))
    return app