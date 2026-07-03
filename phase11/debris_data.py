"""
================================================================================
ISDMAAS — REAL DATA LAYER  (debris / rocket bodies / active satellites)
================================================================================
Replaces synthetic objects with REAL catalog data, cached locally in SQLite and
synced on a schedule so the dashboard is fast and respects CelesTrak rate limits.

PRIMARY SOURCE: CelesTrak GP catalog, FETCHED AS JSON (OMM).
  Why JSON, not TLE: CelesTrak runs out of 5-digit catalog numbers ~2026-07-12;
  after that, new objects get 6-digit IDs that DO NOT fit the TLE format. The
  JSON/OMM format already handles 6-digit IDs, so this code keeps working.

OPTIONAL SOURCES (need credentials/registration; CelesTrak alone is enough):
  - Space-Track  (SPACETRACK_USER / SPACETRACK_PASS env vars) — authoritative,
    supports historical queries.
  - ESA DISCOS   (DISCOS_TOKEN env var, register at discosweb.esoc.esa.int) —
    rich object metadata (size, mass), good for enrichment.

CelesTrak asks for < 1 request/second and no redundant downloads — this module
caches per group and only re-fetches when the cache is older than the interval.

SYNC INTERVALS (configurable):
  - element sets (debris/rocket-body/active groups): every 6 h
  - full SATCAT metadata: daily
  (space-weather sync lives in the phase55 pipeline already.)

WHAT THIS GIVES THE DEMO:
  real_satellite + real_debris + a real (computed) close approach — instead of a
  synthetic threat. Note honestly: live close approaches for a healthy satellite
  are usually tens of km (NOMINAL); genuinely CRITICAL live conjunctions are rare.
  For a dramatic-but-honest demo use the documented historical events (see
  historical_events.py) which carry real, documented sub-km geometry.

USAGE
    python debris_data.py sync                 # fetch + cache the standard groups
    python debris_data.py near 39634 50        # real objects within 50 km of Sentinel-1A
    python debris_data.py rocketbodies 700 800 # rocket bodies in the 700-800 km shell
    python debris_data.py stats                # cache status
================================================================================
"""
import os, sys, json, sqlite3, time, math
from datetime import datetime, timezone, timedelta

import requests
from sgp4.api import Satrec, jday
from sgp4 import omm

DB_PATH = os.environ.get("ISDMAAS_DB", "data/isdmaas_cache.db")
UA = {"User-Agent": "ISDMAAS/1.0 (research; orbital-safety)"}
CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"
CELESTRAK_SATCAT = "https://celestrak.org/satcat/records.php"

# CelesTrak GP groups we cache for the demo
GP_GROUPS = {
    "active":              "active",
    "rocket-bodies":       "rocket-bodies",          # all R/B
    "cosmos-1408-debris":  "cosmos-1408-debris",     # 2021 ASAT, ~480 km
    "fengyun-1c-debris":   "fengyun-1c-debris",      # 2007 ASAT, ~850 km
    "iridium-33-debris":   "iridium-33-debris",      # 2009 collision, ~780 km
    "cosmos-2251-debris":  "cosmos-2251-debris",     # 2009 collision, ~790 km
}

SYNC_INTERVAL_S = {"gp": 6 * 3600, "satcat": 24 * 3600}
EARTH_R = 6378.137
MU = 398600.4418


# ----------------------------------------------------------------- db
def _conn():
    os.makedirs(os.path.dirname(DB_PATH) or ".", exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("""CREATE TABLE IF NOT EXISTS objects(
        norad INTEGER PRIMARY KEY, name TEXT, object_id TEXT, grp TEXT,
        epoch TEXT, omm_json TEXT, mean_motion REAL, ecc REAL, inc REAL,
        apogee_km REAL, perigee_km REAL, object_type TEXT, fetched_at TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS sync_log(
        key TEXT PRIMARY KEY, last_synced TEXT)""")
    return c


def _get_synced(c, key):
    r = c.execute("SELECT last_synced FROM sync_log WHERE key=?", (key,)).fetchone()
    return datetime.fromisoformat(r[0]) if r else None


def _set_synced(c, key):
    c.execute("INSERT OR REPLACE INTO sync_log(key,last_synced) VALUES(?,?)",
              (key, datetime.now(timezone.utc).isoformat()))


def _is_stale(c, key, kind):
    last = _get_synced(c, key)
    if last is None:
        return True
    age = (datetime.now(timezone.utc) - last).total_seconds()
    return age > SYNC_INTERVAL_S[kind]


# --------------------------------------------------- elements helpers
def _orbit_alt(rec):
    """apogee/perigee altitude (km) from GP mean elements."""
    try:
        n = float(rec["MEAN_MOTION"])                 # rev/day
        ecc = float(rec["ECCENTRICITY"])
        a = (MU / (n * 2 * math.pi / 86400.0) ** 2) ** (1.0 / 3.0)
        return (a * (1 + ecc) - EARTH_R, a * (1 - ecc) - EARTH_R)
    except Exception:
        return (None, None)


def satrec_from_row(row):
    """Build an SGP4 propagator from a cached OMM JSON row."""
    rec = json.loads(row["omm_json"])
    sat = Satrec()
    omm.initialize(sat, rec)
    return sat


# ----------------------------------------------------------------- fetch
def fetch_group(group, fmt="JSON"):
    """Fetch one CelesTrak GP group as JSON (list of OMM dicts)."""
    params = {"GROUP": group, "FORMAT": fmt}
    r = requests.get(CELESTRAK_GP, params=params, headers=UA, timeout=120)
    r.raise_for_status()
    txt = r.text.strip()
    if not txt or txt.upper().startswith("NO GP DATA") or txt.startswith("<"):
        return []
    data = json.loads(txt)
    return data if isinstance(data, list) else [data]


def sync(force=False, groups=None):
    """Fetch+cache the standard groups if stale (respects intervals + rate limit)."""
    c = _conn()
    groups = groups or list(GP_GROUPS.values())
    total = 0
    for g in groups:
        key = f"gp:{g}"
        if not force and not _is_stale(c, key, "gp"):
            print(f"  {g}: fresh (skip)")
            continue
        try:
            recs = fetch_group(g)
        except Exception as e:
            print(f"  {g}: FETCH FAILED ({e})")
            continue
        now = datetime.now(timezone.utc).isoformat()
        for rec in recs:
            norad = rec.get("NORAD_CAT_ID")
            if norad is None:
                continue
            apo, per = _orbit_alt(rec)
            c.execute("""INSERT OR REPLACE INTO objects(norad,name,object_id,grp,
                epoch,omm_json,mean_motion,ecc,inc,apogee_km,perigee_km,
                object_type,fetched_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (int(norad), rec.get("OBJECT_NAME"), rec.get("OBJECT_ID"), g,
                 rec.get("EPOCH"), json.dumps(rec), rec.get("MEAN_MOTION"),
                 rec.get("ECCENTRICITY"), rec.get("INCLINATION"), apo, per,
                 _classify(rec.get("OBJECT_NAME", "")), now))
        _set_synced(c, key)
        c.commit()
        print(f"  {g}: cached {len(recs)} objects")
        total += len(recs)
        time.sleep(1.1)            # CelesTrak: < 1 req/sec
    print(f"sync done. {total} objects updated. db={DB_PATH}")
    c.close()
    return total


def _classify(name):
    n = (name or "").upper()
    if "DEB" in n: return "DEBRIS"
    if "R/B" in n or "ROCKET" in n: return "ROCKET BODY"
    return "PAYLOAD"


# ----------------------------------------------------------------- queries
def get_object(norad):
    c = _conn(); c.row_factory = sqlite3.Row
    row = c.execute("SELECT * FROM objects WHERE norad=?", (int(norad),)).fetchone()
    c.close()
    return row


def objects_in_shell(min_alt, max_alt, object_type=None):
    """All cached objects whose orbit overlaps [min_alt, max_alt] km."""
    c = _conn(); c.row_factory = sqlite3.Row
    q = ("SELECT * FROM objects WHERE perigee_km IS NOT NULL "
         "AND apogee_km >= ? AND perigee_km <= ?")
    args = [min_alt, max_alt]
    if object_type:
        q += " AND object_type=?"; args.append(object_type)
    rows = c.execute(q, args).fetchall()
    c.close()
    return rows


def _state(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    return (r, v) if e == 0 else (None, None)


def find_close_approaches(primary_norad, search_hours=24, step_s=30,
                          coarse_km=100, max_report=10):
    """
    REAL conjunction screening: propagate the primary and every cached object
    whose shell overlaps, find genuine closest approaches. Honest note: for a
    healthy satellite these are usually tens of km (NOMINAL).
    Returns list of dicts sorted by miss distance.
    """
    prow = get_object(primary_norad)
    if prow is None:
        raise SystemExit(f"{primary_norad} not in cache — run 'sync' first.")
    psat = satrec_from_row(prow)
    now = datetime.now(timezone.utc)
    r0, _ = _state(psat, now)
    if r0 is None:
        raise SystemExit("primary failed to propagate.")
    p_alt = math.sqrt(sum(x*x for x in r0)) - EARTH_R

    c = _conn(); c.row_factory = sqlite3.Row
    cand = c.execute("""SELECT * FROM objects WHERE norad!=? AND perigee_km IS NOT NULL
        AND apogee_km >= ? AND perigee_km <= ?""",
        (int(primary_norad), p_alt - coarse_km, p_alt + coarse_km)).fetchall()
    c.close()

    results = []
    times = [now + timedelta(seconds=i*step_s) for i in range(int(search_hours*3600/step_s))]
    for row in cand:
        try:
            sat = satrec_from_row(row)
        except Exception:
            continue
        min_d, t_min = 1e9, None
        for when in times:
            r1, _ = _state(psat, when); r2, _ = _state(sat, when)
            if r1 is None or r2 is None:
                continue
            d = math.dist(r1, r2)
            if d < min_d:
                min_d, t_min = d, when
        if t_min is not None and min_d < coarse_km:
            results.append({"norad": row["norad"], "name": row["name"],
                            "type": row["object_type"], "miss_km": round(min_d, 3),
                            "tca_utc": t_min.isoformat()})
    results.sort(key=lambda x: x["miss_km"])
    return results[:max_report]


# ----------------------------------------------------------------- cli
def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    if cmd == "sync":
        sync(force="--force" in sys.argv)
    elif cmd == "stats":
        c = _conn()
        tot = c.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        print(f"db={DB_PATH}  total objects={tot}")
        for g, in c.execute("SELECT DISTINCT grp FROM objects"):
            n = c.execute("SELECT COUNT(*) FROM objects WHERE grp=?", (g,)).fetchone()[0]
            last = _get_synced(c, f"gp:{g}")
            print(f"  {g:24s} {n:>6}   last sync: {last}")
        c.close()
    elif cmd == "near":
        norad = int(sys.argv[2]); rng = float(sys.argv[3]) if len(sys.argv) > 3 else 50
        hits = find_close_approaches(norad, coarse_km=rng)
        print(f"\nReal close approaches to {norad} within {rng} km "
              f"(next 24 h):")
        if not hits:
            print("  none in cache/shell — try a larger range or run 'sync'.")
        for h in hits:
            print(f"  {h['miss_km']:>8.2f} km  {h['name']} ({h['norad']}) "
                  f"[{h['type']}]  TCA {h['tca_utc']}")
    elif cmd == "rocketbodies":
        lo = float(sys.argv[2]) if len(sys.argv) > 2 else 700
        hi = float(sys.argv[3]) if len(sys.argv) > 3 else 800
        rows = objects_in_shell(lo, hi, "ROCKET BODY")
        print(f"\nRocket bodies in the {lo:.0f}-{hi:.0f} km shell ({len(rows)}):")
        for r in sorted(rows, key=lambda x: x["perigee_km"])[:40]:
            print(f"  {r['norad']:>7}  {r['name']:<28} "
                  f"peri {r['perigee_km']:.0f}  apo {r['apogee_km']:.0f} km")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
