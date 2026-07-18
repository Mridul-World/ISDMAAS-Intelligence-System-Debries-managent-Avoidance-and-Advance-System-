"""
user_screening.py — full-catalog conjunction screening for operator satellites.
================================================================================
This is ISDMAAS's own SOCRATES-equivalent for a registered satellite:

  GET /user/screen/{norad}?hours=24&max_results=10

  1. CATALOG: fetches/caches the public CelesTrak GP catalog (active satellites
     + major debris groups) to catalog_cache.json (refresh with &refresh=true).
  2. COARSE SCREEN (vectorized): propagates the WHOLE catalog with SatrecArray
     (C++-speed, chunked) on a 60 s grid across the screening window, after an
     altitude-band prefilter, and finds each object's minimum distance to the
     operator's satellite.
  3. FINE TCA: candidates under the coarse gate get the full numerical TCA
     refinement (down to 0.02 s) + encounter-plane Pc with TLE-scale covariance.
  4. LOG: every screening run and its conjunctions are stored in the ops
     database (dataset accumulation for the pilot record).

HONEST NOTES:
  - Pc uses the TLE-scale covariance model (documented); with real CDMs the
    /cdm/assess path uses real covariance instead.
  - Screening results are OPERATIONAL RECORDS, not training labels. Training
    remains truth-data-driven via training_pipeline.py.
  - A full-catalog screen takes ~10–60 s depending on machine and catalog size;
    that time is local computation, not cloud latency.
"""
import os, json, time
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, HTTPException
from sgp4.api import Satrec, SatrecArray, jday

import ops_db
from auth_store import get_satellite
from user_conjunctions import _find_tca, _pc, _satrec, _state

router = APIRouter()

_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalog_cache.json")
_GROUPS = ["active", "cosmos-2251-debris", "iridium-33-debris",
           "fengyun-1c-debris", "cosmos-1408-debris"]
_GP_URL = "https://celestrak.org/NORAD/elements/gp.php?GROUP={g}&FORMAT=tle"


# ----------------------------------------------------------------------------
# catalog cache
# ----------------------------------------------------------------------------
def _fetch_catalog():
    objs = {}
    for g in _GROUPS:
        try:
            r = requests.get(_GP_URL.format(g=g),
                             headers={"User-Agent": "ISDMAAS/1.0"}, timeout=60)
            lines = [ln.strip() for ln in r.text.splitlines() if ln.strip()]
            for i in range(0, len(lines) - 2, 3):
                name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
                if not (l1.startswith("1 ") and l2.startswith("2 ")):
                    continue
                norad = l1[2:7].strip()
                objs[norad] = {"name": name, "tle1": l1, "tle2": l2, "group": g}
        except Exception as e:
            print(f"[screening] group {g} fetch failed: {e}")
    if objs:
        json.dump({"fetched_utc": datetime.now(timezone.utc).isoformat(),
                   "objects": objs}, open(_CACHE, "w"))
    return objs

def _catalog(refresh=False):
    if not refresh and os.path.exists(_CACHE):
        try:
            d = json.load(open(_CACHE))
            if d.get("objects"):
                return d["objects"], d.get("fetched_utc", "")
        except Exception:
            pass
    objs = _fetch_catalog()
    if not objs:
        raise HTTPException(503, "Catalog unavailable — no cache and CelesTrak "
                                 "fetch failed. Retry with &refresh=true when online.")
    return objs, datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------------
# coarse vectorized screen
# ----------------------------------------------------------------------------
def _alt_band(sat, when):
    """(perigee-ish, apogee-ish) altitude of the user satellite over one rev, km."""
    alts = []
    for m in range(0, 100, 5):
        r, _ = _state(sat, when + timedelta(minutes=m))
        if r is not None:
            alts.append(np.linalg.norm(r) - 6371.0)
    return (min(alts), max(alts)) if alts else (0, 2000)

def _coarse_screen(user_sat, cat_objs, start, hours, gate_km=150.0, band_pad_km=120.0):
    """Vectorized min-distance of every catalog object to the user track.
    Returns [(norad, min_km)] for objects under the gate."""
    n_steps = int(hours * 60)  # 60 s grid
    times = [start + timedelta(seconds=60 * i) for i in range(n_steps)]
    jds, frs = [], []
    for t in times:
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute,
                      t.second + t.microsecond * 1e-6)
        jds.append(jd); frs.append(fr)
    jds = np.array(jds); frs = np.array(frs)

    # user track
    ua = SatrecArray([user_sat])
    e, ur, _ = ua.sgp4(jds, frs)
    ur = ur[0]                                    # (n_steps, 3)
    ok_u = (e[0] == 0)
    if ok_u.sum() < 10:
        raise HTTPException(422, "operator TLE fails to propagate over the window")

    # altitude prefilter from the object's mean motion (cheap, no propagation):
    lo, hi = _alt_band(user_sat, start)
    lo -= band_pad_km; hi += band_pad_km
    cands, recs = [], []
    for norad, o in cat_objs.items():
        try:
            s = Satrec.twoline2rv(o["tle1"], o["tle2"])
        except Exception:
            continue
        # semi-major axis from mean motion (rad/min): a = (mu / n^2)^(1/3)
        n_rad_s = s.no_kozai / 60.0
        a_km = (398600.4418 / (n_rad_s ** 2)) ** (1.0 / 3.0)
        alt = a_km - 6371.0
        ecc = s.ecco
        apo, per = alt + ecc * a_km, alt - ecc * a_km
        if apo < lo or per > hi:
            continue
        cands.append(norad); recs.append(s)

    hits = []
    CH = 1000
    for i0 in range(0, len(recs), CH):
        chunk = SatrecArray(recs[i0:i0 + CH])
        e, rr, _ = chunk.sgp4(jds, frs)           # (chunk, n_steps, 3)
        d = np.linalg.norm(rr - ur[None, :, :], axis=2)
        d[e != 0] = 1e9
        d[:, ~ok_u] = 1e9
        mins = d.min(axis=1)
        for j, m in enumerate(mins):
            if m < gate_km:
                hits.append((cands[i0 + j], float(m)))
    hits.sort(key=lambda x: x[1])
    return hits, len(cands)


# ----------------------------------------------------------------------------
# endpoint
# ----------------------------------------------------------------------------
@router.get("/user/screen/{norad}")
def screen(norad: str, hours: float = 24.0, max_results: int = 10,
           refresh: bool = False):
    t_start = time.time()
    user_sat, meta = _satrec(norad)
    cat, fetched = _catalog(refresh=refresh)
    cat.pop(str(norad), None)  # don't screen against itself
    start = datetime.now(timezone.utc)

    hits, n_screened = _coarse_screen(user_sat, cat, start, hours)

    results = []
    for cn, coarse_km in hits[:max(20, max_results * 2)]:
        try:
            other = Satrec.twoline2rv(cat[cn]["tle1"], cat[cn]["tle2"])
            tca, miss, relv, ra, va, rb, vb = _find_tca(user_sat, other, start, hours)
            dt_s = (tca - start).total_seconds()
            pc = _pc(miss, ra, va, rb, vb, max(dt_s, 600))
            risk = ("CRITICAL" if pc > 1e-4 else "HIGH" if pc > 1e-5 else
                    "ELEVATED" if pc > 1e-7 else "NOMINAL")
            results.append({
                "secondary_norad": cn, "secondary_name": cat[cn]["name"],
                "group": cat[cn]["group"],
                "tca_utc": tca.isoformat(), "tca_in_hours": round(dt_s / 3600.0, 2),
                "miss_km": round(miss, 3), "rel_speed_kms": round(relv, 3),
                "pc": pc, "risk": risk,
            })
        except Exception:
            continue
    results.sort(key=lambda r: r["miss_km"])
    results = results[:max_results]

    # log the run + conjunctions to the ops dataset
    try:
        ops_db.log_screening(norad, meta["name"], meta["owner"],
                             n_screened, len(results), results)
    except Exception as e:
        print(f"[screening] log failed: {e}")

    return {
        "primary": {"norad": norad, "name": meta["name"], "owner": meta["owner"]},
        "window_hours": hours,
        "catalog_fetched_utc": fetched,
        "objects_screened": n_screened,
        "scan_seconds": round(time.time() - t_start, 1),
        "conjunctions": results,
        "note": "ISDMAAS's own full-catalog screen for this asset (SOCRATES-style). "
                "Pc uses the documented TLE-scale covariance model; real CDM "
                "covariance is used when supplied via /cdm/assess.",
    }

def install_screening(app):
    app.include_router(router)
    return app
