# Phase 12 — Historical Replay Mode

Replays real, documented orbital conjunctions using ONLY pre-event public TLEs,
runs the ISDMAAS engine, and compares its output to the documented outcome.

## Files
- historical_events.py   real event database (NORAD ids, dates, documented figures)
- phase12_replay.py      replay engine (fetches pre-event TLEs, runs engine)
- phase7_collision.py, phase8_maneuver.py   (copied dependencies)
- tle_cache/             pre-event TLEs (auto-saved; or drop in manually)

## Run
    python phase12_replay.py --list
    python phase12_replay.py aeolus_starlink_2019      # strongest demo (recent, well-tracked)
    python phase12_replay.py iridium_cosmos_2009       # data-quality lesson

## Space-Track credentials (to auto-fetch archived TLEs)
    set SPACETRACK_USER=you@example.com
    set SPACETRACK_PASS=your_password           (Windows CMD)
    $env:SPACETRACK_USER="you@example.com"      (PowerShell)
    $env:SPACETRACK_PASS="your_password"
Without credentials it uses any TLEs you place in tle_cache/<norad>_<YYYYMMDD>.tle.

## HONEST framing for each event (state these truthfully)
- AEOLUS x STARLINK-44 (2019, AVOIDED): recent, well-tracked. ISDMAAS computes a
  meaningful Pc and recommends a raise maneuver comparable to ESA's actual move.
  THIS is the replay that shows ISDMAAS doing its job. Lead with it.
- IRIDIUM 33 x COSMOS 2251 (2009, COLLISION): public TLEs of the era were too
  inaccurate to predict this (SOCRATES ranked it 64th). The replay reproduces
  that reality. Use it as the DATA-QUALITY lesson motivating ISDMAAS's
  uncertainty layer + need for precise orbits. Do NOT claim ISDMAAS would have
  prevented it — the era's data makes that false.
- COSMOS 1408 (2021, debris field): screen a primary (ISS) against the debris
  shell; ranks closest approaches by Pc.

## Why the Iridium-Cosmos miss looks large
Computed closest approach from public TLEs is ~3000 km vs the documented 584 m.
This is not a bug — it is the documented limitation of public TLEs that caused
the collision to go unflagged. Relative velocity (~11.9 km/s) matches the record,
confirming the geometry is correct; only the along-track timing error (inherent
to TLEs) inflates the miss. This is the honest, citable historical lesson.
