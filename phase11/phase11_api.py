"""
================================================================================
PHASE 11 — ISDMAAS API SERVICE  (FastAPI)
================================================================================
Unifies the whole pipeline behind REST endpoints:

  GET  /health                      service status
  GET  /screen/{norad}              conjunctions for a satellite (catalog screen)
  POST /assess                      collision risk for primary vs a secondary
  POST /maneuver-plan               planned + SAFETY-VALIDATED avoidance burn
  GET  /satellites                  configured fleet

The /maneuver-plan endpoint is structured so a Phase-9 RL optimizer can drop in
behind it later: it calls a single `propose_maneuver()` function (currently the
rule-based Phase-8 planner). Swap that for RL and every safety guarantee
(Phase 10) still applies unchanged.

Run:  uvicorn phase11_api:app --reload --port 8000
Docs: http://localhost:8000/docs   (interactive Swagger UI, auto-generated)
================================================================================
"""
import os, json
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sgp4.api import Satrec, jday

from phase7_collision import (assess_conjunction, secondary_covariance_rtn,
                              rtn_to_eci_cov)
from phase8_maneuver import plan_maneuver
from phase10_safety import validate_maneuver, orbital_elements

# ----------------------------------------------------------------- config
PHASE55_DIR = os.environ.get("PHASE55_DIR", "../phase55")
CELESTRAK_ACTIVE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
HEADERS = {"User-Agent": "ISDMAAS/1.0 (research)"}
MU = 398600.4418
HBR_KM = 0.020
WINDOW_S = 259200
SATELLITES = {39634: "SENTINEL-1A", 40697: "SENTINEL-2A", 42063: "SENTINEL-2B",
              41335: "SENTINEL-3A", 43437: "SENTINEL-3B"}

app = FastAPI(title="ISDMAAS API",
              description="Intelligent Space Debris Mitigation & Autonomous Avoidance",
              version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_catalog = None      # lazy-loaded catalog cache


# ----------------------------------------------------------------- helpers
def load_primary(norad):
    """
    Return primary state. Priority:
      1. Phase-6 validated state file (real calibrated covariance) if it exists.
      2. Otherwise propagate the catalog TLE and attach a GENERIC covariance
         (so ANY catalog satellite can be used; flagged as 'generic').
    Returns (r, v, C_eci, epoch, name, cov_source).
    """
    fp = os.path.join(PHASE55_DIR, "reports", f"state_{norad}.json")
    if os.path.exists(fp):
        d = json.load(open(fp))
        return (np.array(d["position_teme_km"]), np.array(d["velocity_teme_kms"]),
                np.array(d["covariance_eci_km2"]), datetime.fromisoformat(d["epoch_utc"]),
                d["name"], "phase6_calibrated")
    # generic fallback via catalog TLE
    cat = get_catalog(required=False)
    entry = cat.get(norad)
    if entry is None:
        raise HTTPException(404, f"{norad}: no Phase-6 state and not in catalog.")
    name, sat = entry
    epoch = datetime.now(timezone.utc)
    r, v = sgp4_state(sat, epoch)
    if r is None:
        raise HTTPException(500, f"SGP4 failed for {norad}")
    # generic LEO covariance: modest along-track dominance (conservative, documented)
    C = rtn_to_eci_cov(np.diag([0.05**2, 1.0**2, 0.05**2]), r, v)
    return r, v, C, epoch, name, "generic_fallback"


def get_catalog(required=True):
    global _catalog
    if _catalog is not None:
        return _catalog
    cache = "data/catalog_active.tle"
    text = None
    if os.path.exists(cache) and (datetime.now().timestamp() - os.path.getmtime(cache)) < 86400:
        text = open(cache).read()
    else:
        try:
            r = requests.get(CELESTRAK_ACTIVE, headers=HEADERS, timeout=120)
            r.raise_for_status()
            text = r.text; os.makedirs("data", exist_ok=True); open(cache, "w").write(text)
        except Exception as e:
            if os.path.exists(cache):
                text = open(cache).read()              # stale cache better than nothing
            elif required:
                raise HTTPException(503, f"catalog unavailable: {e}")
            else:
                return {}
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    sats = {}
    for i in range(0, len(lines) - 2, 3):
        name, l1, l2 = lines[i], lines[i+1], lines[i+2]
        if not (l1.startswith("1 ") and l2.startswith("2 ")):
            continue
        try:
            sats[int(l2[2:7])] = (name.strip(), Satrec.twoline2rv(l1, l2))
        except Exception:
            continue
    _catalog = sats
    return sats


def sgp4_state(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return np.array(r), np.array(v)


def tca_search(psat, sat2, epoch, n=288):
    min_d, t_min = 1e9, 0.0
    for s in range(n):
        dt = s * (WINDOW_S / n); when = epoch + timedelta(seconds=dt)
        rp_t, _ = sgp4_state(psat, when); r2, _ = sgp4_state(sat2, when)
        if rp_t is None or r2 is None:
            continue
        d = np.linalg.norm(rp_t - r2)
        if d < min_d:
            min_d, t_min = d, dt
    return min_d, t_min


def propose_maneuver(rp, vp, Cp, r2, v2, C2, tca_s, sat_mass_kg):
    """Maneuver proposer. Rule-based (Phase 8) now; Phase-9 RL plugs in here."""
    return plan_maneuver(rp, vp, Cp, r2, v2, C2, hbr=HBR_KM,
                         tca_s=tca_s, sat_mass_kg=sat_mass_kg)


# ----------------------------------------------------------------- models
class AssessRequest(BaseModel):
    primary_norad: int
    secondary_norad: int


class ManeuverRequest(BaseModel):
    primary_norad: int
    secondary_norad: Optional[int] = None
    synthetic_miss_km: Optional[float] = None     # demo: inject a threat at this miss
    tca_hours: float = 8.0
    sat_mass_kg: float = 2300.0
    fuel_available_kg: float = 5.0


# ----------------------------------------------------------------- endpoints
@app.get("/health")
def health():
    return {"status": "ok", "service": "ISDMAAS", "version": "1.0",
            "time_utc": datetime.now(timezone.utc).isoformat()}


@app.get("/satellites")
def satellites():
    have_state = {}
    for n, name in SATELLITES.items():
        fp = os.path.join(PHASE55_DIR, "reports", f"state_{n}.json")
        have_state[n] = {"name": name, "state_available": os.path.exists(fp)}
    return have_state


@app.get("/screen/{norad}")
def screen(norad: int, coarse_km: float = 200.0, max_results: int = 30):
    rp, vp, Cp, epoch, pname, _cov = load_primary(norad)
    cat = get_catalog()
    psat = cat.get(norad, (None, None))[1]
    if psat is None:
        raise HTTPException(404, f"Primary {norad} not in catalog")
    rp_norm = np.linalg.norm(rp)
    hits, checked = [], 0
    for sec, (name2, sat2) in cat.items():
        if sec == norad:
            continue
        r2_now, _ = sgp4_state(sat2, epoch)
        if r2_now is None or abs(np.linalg.norm(r2_now) - rp_norm) > coarse_km:
            continue
        checked += 1
        min_d, t_min = tca_search(psat, sat2, epoch)
        if min_d > coarse_km:
            continue
        when = epoch + timedelta(seconds=t_min)
        rp_t, vp_t = sgp4_state(psat, when)
        C2 = rtn_to_eci_cov(secondary_covariance_rtn(WINDOW_S / 2), *sgp4_state(sat2, when))
        res = assess_conjunction(rp_t, vp_t, Cp, *sgp4_state(sat2, when), C2, HBR_KM, 600)
        hits.append({"norad": sec, "name": name2,
                     "miss_km": round(res["miss_distance_km"], 3),
                     "pc": res["pc"], "risk": res["risk_level"],
                     "tca_h": round(t_min / 3600, 2)})
    hits.sort(key=lambda x: -x["pc"])
    return {"primary": pname, "objects_screened": checked,
            "conjunctions": hits[:max_results]}


@app.post("/assess")
def assess(req: AssessRequest):
    rp, vp, Cp, epoch, pname, _cov = load_primary(req.primary_norad)
    cat = get_catalog()
    psat = cat.get(req.primary_norad, (None, None))[1]
    if req.secondary_norad not in cat:
        raise HTTPException(404, f"Secondary {req.secondary_norad} not in catalog")
    name2, sat2 = cat[req.secondary_norad]
    min_d, t_min = tca_search(psat, sat2, epoch)
    when = epoch + timedelta(seconds=t_min)
    rp_t, vp_t = sgp4_state(psat, when)
    r2, v2 = sgp4_state(sat2, when)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(WINDOW_S / 2), r2, v2)
    res = assess_conjunction(rp_t, vp_t, Cp, r2, v2, C2, HBR_KM, 600)
    return {"primary": pname, "secondary": name2,
            "tca_h": round(t_min / 3600, 2), **{k: res[k] for k in
            ("miss_distance_km", "pc", "mahalanobis", "risk_level",
             "relative_speed_kms", "pc_methods_agree")}}


@app.post("/maneuver-plan")
def maneuver_plan(req: ManeuverRequest):
    rp, vp, Cp, epoch, pname, _cov = load_primary(req.primary_norad)
    cat = get_catalog()

    # build the threat: real secondary OR synthetic injected miss
    if req.synthetic_miss_km is not None:
        rhat = rp / np.linalg.norm(rp); speed = np.linalg.norm(vp)
        vhat = vp / speed; cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
        r2 = rp + req.synthetic_miss_km * rhat
        v2 = cross * speed
        tca_s = req.tca_hours * 3600.0
        sec_name = f"SYNTHETIC_{req.synthetic_miss_km}km"
    elif req.secondary_norad is not None:
        psat = cat.get(req.primary_norad, (None, None))[1]
        if req.secondary_norad not in cat:
            raise HTTPException(404, "secondary not in catalog")
        sec_name, sat2 = cat[req.secondary_norad]
        min_d, t_min = tca_search(psat, sat2, epoch)
        when = epoch + timedelta(seconds=t_min)
        rp, vp = sgp4_state(psat, when)
        r2, v2 = sgp4_state(sat2, when)
        tca_s = max(t_min, 600.0)
    else:
        raise HTTPException(400, "provide secondary_norad or synthetic_miss_km")

    C2 = rtn_to_eci_cov(secondary_covariance_rtn(WINDOW_S / 2), r2, v2)
    mission_sma, _, _ = orbital_elements(rp, vp)

    plan = propose_maneuver(rp, vp, Cp, r2, v2, C2, tca_s, req.sat_mass_kg)
    out = {"primary": pname, "secondary": sec_name,
           "pre_maneuver": plan["pre_maneuver"],
           "action_required": plan["action_required"]}
    if not isinstance(plan.get("recommendation"), dict):
        out["recommendation"] = plan.get("recommendation", "no maneuver")
        out["verdict"] = "NO_ACTION"
        return out

    rec = plan["recommendation"]
    sign = 1.0 if rec["direction"] == "prograde" else -1.0
    report = validate_maneuver(
        rp, vp, Cp, r2, v2, C2, HBR_KM, tca_s,
        rec["dv_magnitude_ms"], sign, rec["burn_lead_time_h"], rec["fuel_kg"],
        fuel_available_kg=req.fuel_available_kg, mission_sma_km=mission_sma,
        Cp_rtn_diag=[0.02, 0.05, 0.02], catalog=cat, epoch=epoch,
        primary_norad=req.primary_norad)
    out["recommendation"] = rec
    out["options"] = plan["options"]
    out["safety_validation"] = report
    out["verdict"] = report["verdict"]
    return out


# ----------------------------------------------------------------- visualization
@app.get("/orbit/{norad}")
def orbit_track(norad: int, points: int = 180, period_fraction: float = 1.0):
    """
    Return one orbit's worth of ECI positions (km) for 3D plotting.
    points samples over `period_fraction` of the orbital period.
    """
    cat = get_catalog(required=False)
    sat = cat.get(norad, (None, None))[1]
    name = cat.get(norad, (str(norad),))[0]
    if sat is None:
        # fall back to the primary state file if not in catalog
        try:
            rp, vp, _, epoch0, name, _cov = load_primary(norad)
        except Exception:
            raise HTTPException(404, f"{norad} not available")
        # crude period from state
        a = -MU / (2 * (0.5*(vp@vp) - MU/np.linalg.norm(rp)))
        period = 2*np.pi*np.sqrt(a**3/MU)
        track = []
        for i in range(points):
            dt = i * period * period_fraction / points
            # propagate using two-body (no catalog TLE) — good enough for viz
            track.append(_twobody(rp, vp, dt).tolist())
        return {"norad": norad, "name": name, "track_km": track, "period_s": period}
    # use SGP4 from catalog TLE
    epoch = datetime.now(timezone.utc)
    # estimate period from mean motion in the TLE
    n_rev_per_day = sat.no_kozai * 1440 / (2*np.pi)   # approx
    period = 86400.0 / max(n_rev_per_day, 1e-6)
    track = []
    for i in range(points):
        dt = i * period * period_fraction / points
        when = epoch + timedelta(seconds=dt)
        r, _ = sgp4_state(sat, when)
        if r is not None:
            track.append(r.tolist())
    return {"norad": norad, "name": name, "track_km": track, "period_s": period}


def _twobody(r0, v0, dt, steps=50):
    """Tiny RK4 two-body propagator for visualization fallback."""
    r = np.array(r0, float); v = np.array(v0, float)
    h = dt / steps
    def acc(rr): return -MU * rr / np.linalg.norm(rr)**3
    for _ in range(steps):
        k1v = acc(r);           k1r = v
        k2v = acc(r+0.5*h*k1r); k2r = v+0.5*h*k1v
        k3v = acc(r+0.5*h*k2r); k3r = v+0.5*h*k2v
        k4v = acc(r+h*k3r);     k4r = v+h*k3v
        r = r + (h/6)*(k1r+2*k2r+2*k3r+k4r)
        v = v + (h/6)*(k1v+2*k2v+2*k3v+k4v)
    return r


# ----------------------------------------------------------------- autonomous loop
class AutoRequest(BaseModel):
    primary_norad: int
    mode: str = "auto"                 # "auto" = scan real catalog; "inject" = synthetic threat
    inject_miss_km: float = 0.05       # used when mode="inject" or no real threat found
    inject_tca_h: float = 8.0
    sat_mass_kg: float = 2300.0
    fuel_available_kg: float = 5.0
    risk_threshold_pc: float = 1e-7    # auto-act if any conjunction Pc exceeds this


@app.post("/autonomous")
def autonomous(req: AutoRequest):
    """
    The full hands-off loop on a REAL satellite:
      1. load real primary state
      2. screen the REAL catalog for the worst upcoming conjunction
      3. if its Pc exceeds threshold -> that's the threat (REAL detection)
         else (or mode=inject) -> inject a synthetic threat to demonstrate
      4. autonomously plan + safety-validate the maneuver (no human step)
      5. return the full decision + verdict, labelled real vs simulated
    """
    rp, vp, Cp, epoch, pname, cov_source = load_primary(req.primary_norad)
    cat = get_catalog()
    psat = cat.get(req.primary_norad, (None, None))[1]
    if psat is None:
        raise HTTPException(404, f"primary {req.primary_norad} not in catalog")

    detected = None
    source = "none"

    # ---- step 2-3: real-catalog scan for the worst conjunction ----
    if req.mode == "auto":
        rp_norm = np.linalg.norm(rp)
        worst = None
        for sec, (name2, sat2) in cat.items():
            if sec == req.primary_norad:
                continue
            r2_now, _ = sgp4_state(sat2, epoch)
            if r2_now is None or abs(np.linalg.norm(r2_now) - rp_norm) > 200.0:
                continue
            min_d, t_min = tca_search(psat, sat2, epoch)
            if min_d > 200.0:
                continue
            when = epoch + timedelta(seconds=t_min)
            rp_t, vp_t = sgp4_state(psat, when)
            r2, v2 = sgp4_state(sat2, when)
            C2 = rtn_to_eci_cov(secondary_covariance_rtn(WINDOW_S / 2), r2, v2)
            res = assess_conjunction(rp_t, vp_t, Cp, r2, v2, C2, HBR_KM, 600)
            if worst is None or res["pc"] > worst["pc"]:
                worst = {"norad": sec, "name": name2, "pc": res["pc"],
                         "miss": res["miss_distance_km"], "t": t_min,
                         "rp": rp_t, "vp": vp_t, "r2": r2, "v2": v2, "C2": C2}
        if worst and worst["pc"] >= req.risk_threshold_pc:
            detected = worst; source = "real_catalog"

    # ---- inject synthetic threat if no real one (or mode=inject) ----
    if detected is None:
        rhat = rp / np.linalg.norm(rp); speed = np.linalg.norm(vp)
        vhat = vp / speed; cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
        r2 = rp + req.inject_miss_km * rhat
        v2 = cross * speed
        C2 = rtn_to_eci_cov(secondary_covariance_rtn(WINDOW_S / 2), r2, v2)
        detected = {"norad": None, "name": f"SIM_THREAT_{req.inject_miss_km}km",
                    "pc": None, "miss": req.inject_miss_km,
                    "t": req.inject_tca_h * 3600.0,
                    "rp": rp, "vp": vp, "r2": r2, "v2": v2, "C2": C2}
        source = "simulated_injection"

    # ---- step 4: autonomous plan + validate ----
    d = detected
    tca_s = max(d["t"], 600.0)
    mission_sma, _, _ = orbital_elements(d["rp"], d["vp"])
    plan = propose_maneuver(d["rp"], d["vp"], Cp, d["r2"], d["v2"], d["C2"],
                            tca_s, req.sat_mass_kg)

    result = {"primary": pname, "detection_source": source,
              "covariance_source": cov_source,
              "threat": {"norad": d["norad"], "name": d["name"],
                         "miss_km": round(d["miss"], 3),
                         "tca_h": round(d["t"] / 3600, 2)},
              "pre_maneuver": plan["pre_maneuver"],
              "action_required": plan["action_required"]}

    if not isinstance(plan.get("recommendation"), dict):
        result["verdict"] = "NO_ACTION"
        result["recommendation"] = "Pc below action threshold — monitoring only."
        return result

    rec = plan["recommendation"]
    sign = 1.0 if rec["direction"] == "prograde" else -1.0
    report = validate_maneuver(
        d["rp"], d["vp"], Cp, d["r2"], d["v2"], d["C2"], HBR_KM, tca_s,
        rec["dv_magnitude_ms"], sign, rec["burn_lead_time_h"], rec["fuel_kg"],
        fuel_available_kg=req.fuel_available_kg, mission_sma_km=mission_sma,
        Cp_rtn_diag=[0.02, 0.05, 0.02], catalog=cat, epoch=epoch,
        primary_norad=req.primary_norad)
    result["recommendation"] = rec
    result["safety_validation"] = report
    result["verdict"] = report["verdict"]
    # autonomous execution flag: the loop acts without human approval if APPROVED
    result["autonomous_action"] = ("EXECUTED" if report["verdict"] == "APPROVED"
                                    else "HELD_FOR_REVIEW")
    return result


# ----------------------------------------------------------------- resolver
_name_index = None

def _build_index():
    """Index catalog by name (upper) and COSPAR id for flexible lookup."""
    global _name_index
    cat = get_catalog(required=False)
    idx = {"by_name": {}, "by_cospar": {}}
    for norad, (name, sat) in cat.items():
        idx["by_name"][name.upper()] = norad
        # COSPAR from TLE line 1 cols 10-17 (intldes): YYNNNAAA -> 20YY-NNNAAA
        try:
            intl = sat.intldesg.strip() if hasattr(sat, "intldesg") else ""
        except Exception:
            intl = ""
        if intl:
            yy = intl[:2]
            year = ("20" if int(yy) < 57 else "19") + yy
            idx["by_cospar"][f"{year}-{intl[2:]}".upper()] = norad
    _name_index = idx
    return idx


@app.get("/resolve")
def resolve(q: str):
    """
    Resolve a NORAD id, satellite name, or COSPAR id to a catalog object.
      /resolve?q=25544        /resolve?q=ISS        /resolve?q=1998-067A
    """
    cat = get_catalog(required=False)
    idx = _name_index or _build_index()
    qs = q.strip()
    # 1. pure NORAD id
    if qs.isdigit() and int(qs) in cat:
        n = int(qs)
        return {"norad": n, "name": cat[n][0], "matched_by": "norad"}
    # 2. COSPAR (contains a dash or matches index)
    up = qs.upper()
    if up in idx["by_cospar"]:
        n = idx["by_cospar"][up]; return {"norad": n, "name": cat[n][0], "matched_by": "cospar"}
    # 3. exact name
    if up in idx["by_name"]:
        n = idx["by_name"][up]; return {"norad": n, "name": cat[n][0], "matched_by": "name"}
    # 4. partial name match -> return candidates
    hits = [(nm, no) for nm, no in idx["by_name"].items() if up in nm][:8]
    if len(hits) == 1:
        return {"norad": hits[0][1], "name": cat[hits[0][1]][0], "matched_by": "name_partial"}
    if hits:
        return {"candidates": [{"norad": no, "name": cat[no][0]} for nm, no in hits],
                "matched_by": "ambiguous"}
    raise HTTPException(404, f"'{q}' not found by NORAD / name / COSPAR")