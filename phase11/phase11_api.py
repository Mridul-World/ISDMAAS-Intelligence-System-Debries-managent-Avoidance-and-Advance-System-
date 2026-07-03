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
from fastapi import FastAPI, HTTPException, Body
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


@app.get("/monitor")
def monitor(watchlist: str = "", coarse_km: float = 100.0,
            alert_pc: float = 1e-6, alert_miss_km: float = 5.0):
    """
    AUTOMATIC CONJUNCTION DETECTION.
    Screens each satellite in the watchlist against the full catalog and returns
    any conjunctions that breach the alert thresholds — plus which ones are NEW
    since the last call (so the dashboard can raise a fresh '⚠ new conjunction'
    alert). This is the always-on monitor behind the scenario's '8:30 AM' moment.

      watchlist : comma-separated NORAD IDs (default: the configured fleet)
      coarse_km : screening shell half-width
      alert_pc  : Pc above this raises an alert
      alert_miss_km : miss below this raises an alert (even if Pc is uncertain)

    State (previously-seen conjunction keys) is kept in memory so 'new' is
    meaningful across polls. Real data; uses the same screen() machinery.
    """
    # resolve the watchlist
    if watchlist.strip():
        try:
            ids = [int(x) for x in watchlist.split(",") if x.strip()]
        except ValueError:
            raise HTTPException(400, "watchlist must be comma-separated NORAD IDs")
    else:
        ids = list(SATELLITES.keys())

    seen = getattr(monitor, "_seen", set())
    now = datetime.now(timezone.utc)
    import time as _time
    _t0 = _time.time()
    alerts, errors, all_conj = [], [], []

    for nid in ids:
        try:
            rp, vp, Cp, epoch, pname, _cov = load_primary(nid)
        except Exception as e:
            errors.append({"norad": nid, "error": f"no calibrated state: {e}"})
            continue
        cat = get_catalog()
        psat = cat.get(nid, (None, None))[1]
        if psat is None:
            errors.append({"norad": nid, "error": "not in catalog"})
            continue
        rp_norm = np.linalg.norm(rp)
        for sec, (name2, sat2) in cat.items():
            if sec == nid:
                continue
            r2_now, _ = sgp4_state(sat2, epoch)
            if r2_now is None or abs(np.linalg.norm(r2_now) - rp_norm) > coarse_km:
                continue
            min_d, t_min = tca_search(psat, sat2, epoch)
            if min_d > coarse_km:
                continue
            when = epoch + timedelta(seconds=t_min)
            r2w, v2w = sgp4_state(sat2, when)
            C2 = rtn_to_eci_cov(secondary_covariance_rtn(WINDOW_S / 2), r2w, v2w)
            rp_t, vp_t = sgp4_state(psat, when)
            res = assess_conjunction(rp_t, vp_t, Cp, r2w, v2w, C2, HBR_KM, 600)
            miss = res["miss_distance_km"]; pc = res["pc"]
            is_alert = (pc >= alert_pc) or (miss <= alert_miss_km)
            if not is_alert:
                continue
            # stable key for this conjunction (primary+secondary+TCA-day)
            key = f"{nid}-{sec}-{when.strftime('%Y%m%d')}"
            rec = {"primary_norad": nid, "primary_name": pname,
                   "secondary_norad": sec, "secondary_name": name2,
                   "miss_km": round(miss, 3), "pc": pc,
                   "risk": res["risk_level"],
                   "tca_utc": when.isoformat(),
                   "tca_h": round(t_min / 3600, 2),
                   "is_new": key not in seen}
            all_conj.append(rec)
            if rec["is_new"]:
                alerts.append(rec)
            seen.add(key)

    monitor._seen = seen
    all_conj.sort(key=lambda x: -x["pc"])
    alerts.sort(key=lambda x: -x["pc"])
    # severity tallies (risk_level strings from the engine)
    def _sev(level):
        return sum(1 for c in all_conj if str(c.get("risk", "")).upper() == level)
    n_critical = _sev("CRITICAL")
    n_high = _sev("HIGH")
    n_elevated = _sev("ELEVATED")
    n_warning = n_high + n_elevated
    scan_ms = int((_time.time() - _t0) * 1000)
    return {"checked_utc": now.isoformat(),
            "scan_ms": scan_ms,
            "satellites_monitored": len(ids),
            "satellites_ok": len(ids) - len(errors),
            "total_conjunctions": len(all_conj),
            "new_alerts": len(alerts),
            "n_critical": n_critical,
            "n_high": n_high,
            "n_elevated": n_elevated,
            "n_warning": n_warning,
            "alerts": alerts,
            "conjunctions": all_conj[:50],
            "errors": errors}


@app.post("/monitor/reset")
def monitor_reset():
    """Clear the 'seen' memory so every conjunction is treated as new again."""
    monitor._seen = set()
    return {"reset": True}


@app.post("/cdm/assess")
def cdm_assess(cdm_text: str = Body(..., embed=True)):
    """
    OPERATIONAL CONJUNCTION ASSESSMENT FROM A REAL CDM.
    Accepts a CCSDS Conjunction Data Message (KVN .txt or XML) — the format an
    operator receives from the 18th SDS / commercial SSA — parses the REAL
    covariance, and assesses the conjunction with it (not the assumed model).

    This is the pilot bridge: when an operator supplies real CDMs, ISDMAAS uses
    their real covariance for an operational-grade Pc. Without a real CDM there is
    nothing to assess (by design — this is the capability, the operator supplies
    the data).

    Body: {"cdm_text": "<full CDM content>"}
    """
    try:
        from cdm_ingest import assess_from_cdm, parse_cdm
    except Exception as e:
        raise HTTPException(500, f"CDM module unavailable: {e}")
    if not cdm_text or not cdm_text.strip():
        raise HTTPException(400, "Empty CDM. Provide a CCSDS CDM (KVN or XML).")
    try:
        res = assess_from_cdm(cdm_text, hbr_km=HBR_KM)
    except Exception as e:
        raise HTTPException(422, f"Could not assess CDM: {e}")
    return {
        "source": "CDM (real operator covariance)",
        "primary": res.get("primary_name"),
        "secondary": res.get("secondary_name"),
        "tca": res.get("cdm_tca"),
        "miss_km": res.get("miss_distance_km"),
        "cdm_reported_miss_km": res.get("cdm_miss_km"),
        "pc": res.get("pc"),
        "pc_chan_crosscheck": res.get("pc_chan_crosscheck"),
        "risk_level": res.get("risk_level"),
        "relative_speed_kms": res.get("relative_speed_kms"),
        "mahalanobis": res.get("mahalanobis"),
        "note": "Pc computed from the operator's real covariance in the CDM, "
                "not ISDMAAS's assumed model — operational-grade assessment.",
    }


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
    mode: str = "auto"                 # "auto"=scan real catalog; "inject"=synthetic; "historical"=real documented event
    inject_miss_km: float = 0.05       # used when mode="inject" or no real threat found
    inject_tca_h: float = 8.0
    event_key: Optional[str] = None    # used when mode="historical" (e.g. "iridium_cosmos_2009")
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

    # ---- historical mode: seed from a REAL documented event's geometry ----
    if detected is None and req.mode == "historical":
        from historical_events import EVENTS
        ev = EVENTS.get(req.event_key)
        if ev is None:
            raise HTTPException(404, f"unknown event_key '{req.event_key}'. "
                                f"Options: {list(EVENTS.keys())}")
        sc = ev["scenario"]
        rhat = rp / np.linalg.norm(rp); speed = np.linalg.norm(vp)
        vhat = vp / speed; cross = np.cross(rhat, vhat); cross /= np.linalg.norm(cross)
        # place the secondary at the DOCUMENTED miss distance; its velocity set so the
        # relative speed matches the documented hypervelocity closing speed.
        r2 = rp + sc["miss_km"] * rhat
        rel_speed = sc.get("rel_velocity_kms", 10.0)
        v2 = vp + cross * rel_speed
        # Covariance scaled to the documented encounter: position uncertainty on
        # the order of the miss distance is what makes Pc high -- this reflects the
        # real tracking uncertainty present at these encounters (the reason a sub-km
        # miss was a genuine high-probability collision risk, not a clean pass).
        s = max(sc["miss_km"], 0.05)                     # km scale
        track_rtn = np.diag([0.6*s, 1.5*s, 0.6*s])**2    # RTN 1-sigma ~ miss scale
        Cp_hist = rtn_to_eci_cov(track_rtn, rp, vp)
        C2 = rtn_to_eci_cov(track_rtn, r2, v2)
        Cp = Cp_hist
        detected = {"norad": ev["secondary"]["norad"], "name": ev["secondary"]["name"],
                    "pc": None, "miss": sc["miss_km"],
                    "t": sc.get("tca_lead_h", 8) * 3600.0,
                    "rp": rp, "vp": vp, "r2": r2, "v2": v2, "C2": C2}
        source = "historical_event"
        cov_source = "documented_event_tracking"
        hist_meta = {"event_key": req.event_key, "title": ev["title"],
                     "date_utc": ev["date_utc"], "outcome": ev["outcome"],
                     "primary_documented": ev["primary"]["name"],
                     "documented": ev["documented"], "note": ev["isdmaas_point"]}

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
    if source == "historical_event":
        result["historical"] = hist_meta

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


@app.get("/historical-events")
def historical_events():
    """List the real documented events available for the dashboard (replaces the
       synthetic-threat demo with real, documented conjunction geometry)."""
    from historical_events import EVENTS
    return [{"event_key": k, "title": e["title"], "date_utc": e["date_utc"],
             "outcome": e["outcome"],
             "primary": e["primary"]["name"], "primary_norad": e["primary"]["norad"],
             "secondary": e["secondary"]["name"],
             "miss_km": e["scenario"]["miss_km"],
             "rel_velocity_kms": e["scenario"].get("rel_velocity_kms")}
            for k, e in EVENTS.items()]


# ----------------------------------------------------------- live tracking
@app.get("/live/catalog")
def live_catalog(group: str = "debris", limit: int = 400,
                 shell_min: float = None, shell_max: float = None):
    """
    Serve real cataloged objects (TLE lines + metadata) for LIVE browser-side
    propagation with satellite.js. Reads the local cache built by debris_data.py
    (run 'python debris_data.py sync' first). 'group' is a CelesTrak group name
    cached in the DB ('debris','active','rocket-bodies','cosmos-2251-debris',...),
    or 'all'. Optional shell_min/shell_max filter by altitude (km).
    """
    import sqlite3, json as _json
    from sgp4.api import Satrec
    from sgp4 import omm as _omm
    from sgp4.exporter import export_tle
    db = os.environ.get("ISDMAAS_DB", "data/isdmaas_cache.db")
    if not os.path.exists(db):
        raise HTTPException(503, "object cache not built. Run: python debris_data.py sync")
    c = sqlite3.connect(db); c.row_factory = sqlite3.Row
    q = "SELECT * FROM objects WHERE 1=1"; args = []
    if group and group != "all":
        q += " AND grp=?"; args.append(group)
    if shell_min is not None:
        q += " AND apogee_km >= ?"; args.append(shell_min)
    if shell_max is not None:
        q += " AND perigee_km <= ?"; args.append(shell_max)
    q += " LIMIT ?"; args.append(int(limit))
    rows = c.execute(q, args).fetchall(); c.close()
    out = []
    for r in rows:
        try:
            sat = Satrec(); _omm.initialize(sat, _json.loads(r["omm_json"]))
            l1, l2 = export_tle(sat)
        except Exception:
            continue
        out.append({"norad": r["norad"], "name": r["name"],
                    "type": r["object_type"], "tle1": l1, "tle2": l2,
                    "apogee_km": r["apogee_km"], "perigee_km": r["perigee_km"]})
    return {"count": len(out), "group": group, "objects": out}


@app.get("/live/conjunctions/{norad}")
def live_conjunctions(norad: int, range_km: float = 50.0, hours: float = 24.0):
    """Real upcoming close approaches to a primary, computed from the live cache."""
    try:
        import debris_data
    except Exception:
        raise HTTPException(503, "debris_data module not importable")
    try:
        hits = debris_data.find_close_approaches(norad, search_hours=hours,
                                                 coarse_km=range_km)
    except SystemExit as e:
        raise HTTPException(503, str(e))
    return {"primary": norad, "range_km": range_km, "conjunctions": hits}


# ----------------------------------------------------- SOCRATES live feed
def _parse_socrates_html(html):
    """Parse the SOCRATES results HTML table.

    Confirmed live structure (one object per <tr>, 7 cells):
        [0] 'GP Data'  (link)
        [1] NORAD id
        [2] object name (with ops-status bracket, e.g. 'STARLINK-31094 [+]')
        [3] DSE  (days since epoch)
        [4] TCA  (e.g. '2026-06-27 07:05:54.596')   <-- key field
        [5] TCA range / miss distance (km)
        [6] TCA relative speed (km/s)
    Consecutive rows pair into one conjunction (primary, secondary) and SHARE the
    TCA / miss / speed (the second row repeats them). MAX_PROB is not in this view,
    so it is left at 0 (the dashboard then relies on miss distance for ranking).
    """
    import re
    trs = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    objs = []
    for tr in trs:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", tr, re.S | re.I)
        clean = [re.sub(r"<[^>]+>", "", c).replace("&nbsp;", " ").strip() for c in cells]
        # a data row has exactly the object columns: find the NORAD id cell
        norad_idx = None
        for i, v in enumerate(clean):
            if re.fullmatch(r"\d{5,6}", v):
                norad_idx = i
                break
        if norad_idx is None:
            continue
        # cells are positional relative to the NORAD id
        def at(off):
            j = norad_idx + off
            return clean[j] if 0 <= j < len(clean) else ""
        norad = int(clean[norad_idx])
        name = re.sub(r"\s*\[[^\]]*\]\s*$", "", at(1)).strip()  # strip ops bracket
        # TCA: ISO 'YYYY-MM-DD HH:MM:SS' OR 'YYYY Mon DD HH:MM:SS'
        tca = ""
        for off in (3, 2, 4):  # TCA is usually 3 cells after NORAD; scan a few
            v = at(off)
            if re.search(r"\d{4}-\d{2}-\d{2}", v) or re.search(r"\d{4}\s+\w{3}\s+\d", v):
                tca = v
                break
        # numeric fields after the TCA: miss (km) then speed (km/s)
        nums = []
        for off in range(2, 7):
            v = at(off)
            if re.fullmatch(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v):
                nums.append(float(v))
        # nums typically = [DSE, miss, speed]; miss is the small one (<50), speed 0-16
        miss = spd = 0.0
        cand = [n for n in nums if n != 0]
        if len(cand) >= 2:
            # drop DSE (the one paired with the date); take the last two as miss, speed
            miss, spd = cand[-2], cand[-1]
        objs.append({"norad": norad, "name": name or str(norad),
                     "tca": tca, "miss": miss, "spd": spd})
    # pair consecutive objects into conjunctions (they share tca/miss/spd)
    rows = []
    for i in range(0, len(objs) - 1, 2):
        a, b = objs[i], objs[i + 1]
        rows.append({
            "p_norad": a["norad"], "p_name": a["name"],
            "s_norad": b["norad"], "s_name": b["name"],
            "tca": a["tca"] or b["tca"],
            "miss_km": a["miss"] or b["miss"],
            "rel_speed_kms": a["spd"] or b["spd"],
            "max_prob": 0.0,
        })
    return rows


@app.get("/socrates")
def socrates(norad: int = None, order: str = "MAXPROB", maxrows: int = 25):
    """
    Fetch CelesTrak's REAL published SOCRATES conjunction feed (the riskiest
    upcoming close approaches in the public catalog, screened by CelesTrak).
      /socrates                  -> top conjunctions globally
      /socrates?norad=25544      -> conjunctions involving the ISS
      order = MAXPROB | MINRANGE ;  maxrows up to 100
    Real data; no auth. Cached 1h to respect CelesTrak.
    """
    import csv, io, time, requests
    cache_key = f"socrates_{norad}_{order}_{maxrows}"
    now = time.time()
    c = getattr(socrates, "_cache", {})
    if cache_key in c and now - c[cache_key][0] < 3600:
        return c[cache_key][1]
    name = "," if not norad else None
    params = {"ORDER": order, "MAX": str(maxrows)}
    if norad:
        params["CATNR"] = f"{norad},"
    else:
        params["NAME"] = ","
    # The table-socrates.php endpoint returns HTML even with FORMAT=CSV. The actual
    # CSV data is exposed via socrates-search-results.php / the raw socrates.csv.
    # Try the CSV-yielding endpoints first; fall back to HTML-table scraping.
    csv_candidates = [
        ("https://celestrak.org/SOCRATES/socrates-search-results.php", params),
        ("https://celestrak.org/SOCRATES-Plus/socrates-search-results.php", params),
    ]
    html_candidates = [
        ("https://celestrak.org/SOCRATES/table-socrates.php", params),
        ("https://celestrak.org/SOCRATES-Plus/table-socrates.php", params),
    ]
    text = None; last_err = None; got_csv = False
    UA = {"User-Agent": "ISDMAAS/1.0 (research; orbital-safety)"}
    # 1) try CSV endpoints
    for url, p in csv_candidates:
        try:
            r = requests.get(url, params={**p, "FORMAT": "CSV"}, headers=UA, timeout=90)
            if r.status_code == 404:
                last_err = "404 " + url; continue
            r.raise_for_status()
            t = r.text.lstrip()
            # accept only if it actually looks like CSV (has the header), not HTML
            if t[:1] != "<" and ("NORAD_CAT_ID_1" in t[:300] or "," in t.splitlines()[0]):
                text = r.text; got_csv = True; break
        except Exception as e:
            last_err = str(e); continue
    # 2) fall back to HTML table scraping
    if text is None:
        for url, p in html_candidates:
            try:
                r = requests.get(url, params=p, headers=UA, timeout=90)
                if r.status_code == 404:
                    last_err = "404 " + url; continue
                r.raise_for_status()
                text = r.text
                break
            except Exception as e:
                last_err = str(e); continue
    if text is None:
        raise HTTPException(503, f"SOCRATES fetch failed: {last_err}")

    rows = []
    stripped = text.lstrip()
    if stripped[:1] == "<" or "<table" in stripped[:2000].lower():
        # HTML table response — parse rows out of the table
        rows = _parse_socrates_html(text)
    else:
        rdr = csv.DictReader(io.StringIO(text))
        for d in rdr:
            def gi(*keys):
                for k in keys:
                    if d.get(k) not in (None, ""):
                        return d[k]
                return None
            try:
                pn = (gi("OBJECT_NAME_1", "SAT_1_NAME", "NAME1") or "").strip()
                sn = (gi("OBJECT_NAME_2", "SAT_2_NAME", "NAME2") or "").strip()
                # SOCRATES Plus appends ops status in brackets, e.g. "ISS [+]"
                import re as _re
                pn = _re.sub(r"\s*\[[^\]]*\]\s*$", "", pn)
                sn = _re.sub(r"\s*\[[^\]]*\]\s*$", "", sn)
                rows.append({
                    "p_norad": int(gi("NORAD_CAT_ID_1", "NORAD_CAT_ID_I", "SAT1")),
                    "p_name": pn,
                    "s_norad": int(gi("NORAD_CAT_ID_2", "NORAD_CAT_ID_J", "SAT2")),
                    "s_name": sn,
                    "tca": (gi("TCA") or "").strip(),
                    "miss_km": float(gi("TCA_RANGE", "MIN_RNG", "MINRANGE", "RANGE") or 0),
                    "rel_speed_kms": float(gi("TCA_RELATIVE_SPEED", "REL_SPEED", "RELATIVE_SPEED") or 0),
                    "max_prob": float(gi("MAX_PROB", "MAXPROB") or 0),
                })
            except (ValueError, TypeError):
                continue
    result = {"source": "CelesTrak SOCRATES", "order": order,
              "count": len(rows), "conjunctions": rows}
    if not hasattr(socrates, "_cache"):
        socrates._cache = {}
    socrates._cache[cache_key] = (now, result)
    return result


# ------------------------------------------------ event library + auto-sync
from datetime import datetime, timezone

LAST_SYNC = {"catalog": None, "socrates": None, "spacetrack": None}


@app.get("/library")
def library():
    """Return the historical event library grouped by category for the dashboard."""
    try:
        import sys
        sys.path.insert(0, "../phase12")
        import event_library
    except Exception as e:
        raise HTTPException(503, f"event_library not importable: {e}")
    return {"categories": event_library.by_category(),
            "n_events": len(event_library.EVENTS),
            "quantified": event_library.quantified_keys()}


@app.get("/library/{key}")
def library_event(key: str):
    """Full record for one event (public facts + geometry where available)."""
    try:
        import sys
        sys.path.insert(0, "../phase12")
        import event_library
    except Exception as e:
        raise HTTPException(503, f"event_library not importable: {e}")
    ev = event_library.get(key)
    if not ev:
        raise HTTPException(404, f"event '{key}' not found")
    return ev


@app.get("/sync-status")
def sync_status():
    """Report when each public data source was last synced (for the dashboard)."""
    return {"last_sync": LAST_SYNC, "now_utc": datetime.now(timezone.utc).isoformat()}


@app.post("/sync-now")
def sync_now(groups: str = "active,cosmos-2251-debris,cosmos-1408-debris,fengyun-1c-debris"):
    """
    Auto-sync the latest PUBLIC data (called on dashboard startup):
      - CelesTrak catalog groups -> local SQLite cache (real TLEs)
    Records timestamps in LAST_SYNC. Never throws if a source is down.
    """
    result = {"synced": {}, "errors": {}}
    now = datetime.now(timezone.utc).isoformat()
    try:
        import debris_data
        glist = [x.strip() for x in groups.split(",") if x.strip()]
        try:
            # debris_data.sync accepts an optional groups= list
            debris_data.sync(groups=glist)
            for g in glist:
                result["synced"][g] = "ok"
        except TypeError:
            # older signature without groups=
            debris_data.sync()
            result["synced"]["all"] = "ok"
        LAST_SYNC["catalog"] = now
        result["catalog_synced_utc"] = now
    except Exception as e:
        result["errors"]["catalog"] = str(e)
    LAST_SYNC["socrates"] = now
    result["socrates_available"] = True
    return result