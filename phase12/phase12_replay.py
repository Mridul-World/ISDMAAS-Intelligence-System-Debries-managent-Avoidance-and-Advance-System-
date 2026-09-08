"""
================================================================================
PHASE 12 — HISTORICAL REPLAY MODE  (credibility demonstration)
================================================================================
Replays real, documented orbital conjunctions using ONLY the information that
was available before the event: archived TLEs from before the event epoch.
ISDMAAS independently detects the close approach, computes Pc with calibrated
uncertainty, and plans an avoidance maneuver — then compares its output to the
documented historical outcome.

    python phase12_replay.py iridium_cosmos_2009
    python phase12_replay.py aeolus_starlink_2019
    python phase12_replay.py --list

TLE SOURCE
  Archived TLEs are fetched from Space-Track's historical API for an epoch
  shortly BEFORE the event (no future information). Set credentials:
      set SPACETRACK_USER=you@example.com
      set SPACETRACK_PASS=your_password
  If credentials are absent or offline, the replay falls back to any cached
  TLEs in tle_cache/<norad>_<date>.tle that you provide manually.

HONESTY
  - Only pre-event TLEs are used (the data operators had at the time).
  - Documented figures (Pc, miss, maneuver) are shown for comparison only;
    ISDMAAS recomputes everything itself from the TLEs.
  - For the collision case (Iridium-Cosmos), TLE-only data famously lacked the
    accuracy to flag it as top threat; ISDMAAS's contribution is adding an
    explicit calibrated Pc/risk band, which is reported honestly.
================================================================================
"""
import os, sys, json
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from sgp4.api import Satrec, jday
from phase7_collision import (assess_conjunction, secondary_covariance_rtn,
                              rtn_to_eci_cov)
from phase7_collision import pc_text
from phase8_maneuver import plan_maneuver
from historical_events import EVENTS

ST_BASE = "https://www.space-track.org"
HBR_KM = 0.020


def spacetrack_tle(norad, before_dt, session=None):
    """Fetch the latest TLE for `norad` with epoch <= before_dt from Space-Track."""
    user = os.environ.get("SPACETRACK_USER")
    pw = os.environ.get("SPACETRACK_PASS")
    # cache check first
    tag = before_dt.strftime("%Y%m%d")
    cache = f"tle_cache/{norad}_{tag}.tle"
    if os.path.exists(cache):
        lines = [l.strip() for l in open(cache) if l.strip()]
        if len(lines) >= 2:
            l1 = next(l for l in lines if l.startswith("1 "))
            l2 = next(l for l in lines if l.startswith("2 "))
            return l1, l2
    if not (user and pw):
        return None
    s = session or requests.Session()
    if session is None:
        r = s.post(ST_BASE + "/ajaxauth/login",
                   data={"identity": user, "password": pw}, timeout=60)
        r.raise_for_status()
    # query historical TLE: epoch window ending at before_dt
    end = before_dt.strftime("%Y-%m-%d")
    start = (before_dt - timedelta(days=5)).strftime("%Y-%m-%d")
    url = (f"{ST_BASE}/basicspacedata/query/class/gp_history/NORAD_CAT_ID/{norad}"
           f"/EPOCH/{start}--{end}/orderby/EPOCH desc/limit/1/format/tle")
    r = s.get(url, timeout=60)
    r.raise_for_status()
    lines = [l.strip() for l in r.text.splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    os.makedirs("tle_cache", exist_ok=True)
    open(cache, "w").write(r.text)
    l1 = next((l for l in lines if l.startswith("1 ")), None)
    l2 = next((l for l in lines if l.startswith("2 ")), None)
    return (l1, l2) if l1 and l2 else None


def state_at(sat, when):
    jd, fr = jday(when.year, when.month, when.day, when.hour, when.minute,
                  when.second + when.microsecond / 1e6)
    e, r, v = sat.sgp4(jd, fr)
    if e != 0:
        return None, None
    return np.array(r), np.array(v)


def find_tca(sat1, sat2, epoch, window_s, n=2880):
    """Sample relative distance over [epoch, epoch+window] to find closest approach."""
    min_d, t_min = 1e9, 0.0
    for i in range(n):
        dt = i * window_s / n
        when = epoch + timedelta(seconds=dt)
        r1, _ = state_at(sat1, when); r2, _ = state_at(sat2, when)
        if r1 is None or r2 is None:
            continue
        d = np.linalg.norm(r1 - r2)
        if d < min_d:
            min_d, t_min = d, dt
    return min_d, t_min


def replay(event_key):
    if event_key not in EVENTS:
        sys.exit(f"unknown event '{event_key}'. Use --list.")
    ev = EVENTS[event_key]
    event_dt = datetime.fromisoformat(ev["date_utc"]).replace(tzinfo=timezone.utc)
    # screening epoch: 1 day before the event (operators had ~days of lead time)
    screen_epoch = event_dt - timedelta(days=1)

    print("=" * 70)
    print(f"HISTORICAL REPLAY  \u2014  {ev['title']}")
    print(f"event epoch (UTC) : {ev['date_utc']}")
    print(f"screening from    : {screen_epoch.isoformat()}  (pre-event data only)")
    print(f"documented outcome: {ev['outcome']}")
    print("-" * 70)

    p, sname = ev["primary"], ev["secondary"]["name"]
    sess = requests.Session()
    user = os.environ.get("SPACETRACK_USER"); pw = os.environ.get("SPACETRACK_PASS")
    if user and pw:
        try:
            sess.post(ST_BASE + "/ajaxauth/login",
                      data={"identity": user, "password": pw}, timeout=60).raise_for_status()
        except Exception as e:
            print(f"(Space-Track login failed: {e}; will try cache)")

    t1 = spacetrack_tle(p["norad"], screen_epoch, sess)
    t2 = spacetrack_tle(ev["secondary"]["norad"], screen_epoch, sess)
    if not t1 or not t2:
        print("Could not obtain pre-event TLEs for both objects.")
        print("Provide them in tle_cache/<norad>_<YYYYMMDD>.tle or set "
              "SPACETRACK_USER / SPACETRACK_PASS, then retry.")
        print("\nDocumented figures for reference:")
        for k, v in ev["documented"].items():
            print(f"  {k}: {v}")
        return

    sat1 = Satrec.twoline2rv(*t1); sat2 = Satrec.twoline2rv(*t2)

    # search a wide window centred on the documented event time
    window = 48 * 3600
    search_start = event_dt - timedelta(hours=24)
    min_d, t_min = find_tca(sat1, sat2, search_start, window, n=5760)
    tca_when = search_start + timedelta(seconds=t_min)
    r1, v1 = state_at(sat1, tca_when); r2, v2 = state_at(sat2, tca_when)
    vrel = np.linalg.norm(v1 - v2)

    # covariances: primary gets a calibrated-style RTN cov; secondary modeled
    C1 = rtn_to_eci_cov(np.diag([0.05, 0.30, 0.08])**2, r1, v1)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(window / 2), r2, v2)
    res = assess_conjunction(r1, v1, C1, r2, v2, C2, HBR_KM, window)

    # honest TLE-accuracy assessment: how far is computed miss from documented?
    doc_miss_km = None
    if "socrates_predicted_miss_m" in ev["documented"]:
        doc_miss_km = ev["documented"]["socrates_predicted_miss_m"] / 1000.0
    tle_gap = (res["miss_distance_km"] - doc_miss_km) if doc_miss_km else None

    print("ISDMAAS DETECTION (from pre-event public TLEs):")
    print(f"  closest approach : {res['miss_distance_km']:.3f} km "
          f"at T{(t_min - 24*3600)/3600:+.1f} h rel. to event")
    print(f"  relative speed   : {vrel:.2f} km/s  (documented "
          f"{ev['documented'].get('rel_velocity_kms','?')} km/s)")
    print(f"  collision Pc     : {pc_text(res['pc'])}")
    print(f"  risk level       : {res['risk_level']}")
    if tle_gap is not None:
        print(f"  NOTE: public-TLE miss ({res['miss_distance_km']:.0f} km) vs "
              f"documented {doc_miss_km*1000:.0f} m \u2014 the TLE error is the "
              f"historical reason this was not flagged.")
    print("-" * 70)

    # maneuver recommendation (what ISDMAAS would have advised)
    plan = plan_maneuver(r1, v1, C1, r2, v2, C2, hbr=HBR_KM,
                         tca_s=max(t_min, 3600), sat_mass_kg=p["mass_kg"])
    if isinstance(plan.get("recommendation"), dict):
        r = plan["recommendation"]
        print("ISDMAAS RECOMMENDATION (avoidance maneuver):")
        print(f"  burn \u0394v        : {r['dv_magnitude_ms']:.4f} m/s {r['direction']}")
        print(f"  execute        : {r['burn_lead_time_h']} h before TCA")
        print(f"  fuel           : {r['fuel_kg']:.4f} kg")
        print(f"  -> new miss    : {r['predicted_new_miss_km']:.2f} km")
        print(f"  -> new Pc      : {r['predicted_new_pc']:.2e}")
    else:
        print(f"ISDMAAS: {plan.get('recommendation', 'no maneuver needed')}")
    print("-" * 70)

    print("DOCUMENTED HISTORICAL RECORD (for comparison):")
    for k, v in ev["documented"].items():
        print(f"  {k}: {v}")
    print("-" * 70)
    print("INTERPRETATION:")
    print(f"  {ev['isdmaas_point']}")
    if tle_gap is not None and abs(tle_gap) > 1.0:
        print("\n  FINDING (Phase 12 — data-sufficiency study, NOT collision prevention):")
        print("  Replaying with the only data available at the time (public TLEs)")
        print("  reproduces a documented limitation: TLE along-track timing error")
        print("  (seconds -> tens-to-hundreds of km) swamps sub-km conjunctions.")
        print(f"  Here the relative velocity matches the record but the computed miss")
        print(f"  ({res['miss_distance_km']:.0f} km) is far from the documented "
              f"{doc_miss_km*1000:.0f} m -- which is")
        print("  exactly why this event was not flagged from public TLEs. This")
        print("  QUANTIFIES why public-TLE screening is insufficient and empirically")
        print("  motivates ISDMAAS's precise-POD-ephemeris design (Phase 5.5) +")
        print("  calibrated uncertainty (Phase 6). We do NOT claim ISDMAAS would")
        print("  have prevented this collision from public TLEs -- the era's data")
        print("  makes that impossible; the lesson is the data requirement.")
    print("=" * 70)

    os.makedirs("reports", exist_ok=True)
    out = {"event": event_key, "title": ev["title"], "outcome": ev["outcome"],
           "isdmaas": {k: res[k] for k in
                       ("miss_distance_km", "pc", "risk_level", "mahalanobis",
                        "relative_speed_kms")},
           "tca_utc": tca_when.isoformat(),
           "recommendation": plan.get("recommendation"),
           "documented": ev["documented"]}
    json.dump(out, open(f"reports/replay_{event_key}.json", "w"), indent=2, default=str)
    print(f"saved: reports/replay_{event_key}.json")


def main():
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        print("Available historical events:")
        for k, ev in EVENTS.items():
            print(f"  {k:28s} {ev['title']}  ({ev['outcome']})")
        return
    replay(sys.argv[1])


if __name__ == "__main__":
    main()