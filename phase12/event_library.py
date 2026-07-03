"""
================================================================================
HISTORICAL EVENT LIBRARY  —  ISDMAAS  (expanded)
================================================================================
A structured, expandable database of documented space-safety events.

HONESTY STRUCTURE (critical — do not fabricate):
Each event carries `data_quality`:

  "quantified"   : real public geometry exists (real objects, documented facts,
                   trackable debris). Supports replay (conjunction or debris cloud).
  "partial"      : SOME numbers public (e.g. Aeolus Pc + altitude change), others
                   (miss/dv/covariance) are NOT — shown as "not public".
  "record_only"  : documented event/programme, but per-event conjunction geometry
                   is NOT public (operator-restricted CDM data). Listed with public
                   facts and aggregate statistics only. NO invented numbers.

`replay_mode`:
  "conjunction"  : two trackable objects -> conjunction assessment + 3D.
  "debris_cloud" : parent destroyed -> visualize the real debris fragments it made.
  "partial_facts": show the published numbers only (no live replay).
  (record_only events have no replay.)

Every number in a quantified/partial entry has a cited public source. record_only
entries carry NO geometry. To expand: add to EVENTS; if you lack a public source
for the geometry, use "record_only" and never guess miss/Pc/dv.
================================================================================
"""

EVENTS = [

    # ============================== COLLISIONS ==============================
    {
        "key": "iridium_cosmos_2009",
        "category": "Collisions",
        "replay_mode": "debris_cloud",
        "debris_group": "cosmos-2251-debris",
        "title": "Iridium 33 x Cosmos 2251 (2009)",
        "date_utc": "2009-02-10T16:56:00Z",
        "primary": {"name": "IRIDIUM 33", "norad": 24946},
        "secondary": {"name": "COSMOS 2251", "norad": 22675},
        "data_quality": "quantified",
        "outcome": "COLLISION (first accidental collision of two intact satellites)",
        "geometry": {"miss_km": 0.0, "rel_velocity_kms": 11.7},
        "public_facts": "First accidental hypervelocity collision between two intact "
                        "satellites, at ~790 km. Iridium 33 was active; Cosmos 2251 "
                        "was defunct. Created 2,000+ tracked fragments. Public TLE "
                        "screening (SOCRATES) had ranked the conjunction far down its "
                        "list - it could not resolve the sub-km approach.",
        "replay_note": "Public pre-event TLEs reconstruct only a ~km-scale miss, not "
                       "the actual collision - itself the data-sufficiency lesson.",
        "sources": ["NASA Orbital Debris Quarterly News (2009)", "CelesTrak"],
    },
    {
        "key": "cerise_1996",
        "category": "Collisions",
        "replay_mode": "partial_facts",
        "debris_group": None,
        "title": "Cerise x Ariane debris (1996)",
        "date_utc": "1996-07-24T00:00:00Z",
        "primary": {"name": "CERISE", "norad": 23606},
        "secondary": {"name": "ARIANE H-10 fragment (SPOT-1 launch)", "norad": None},
        "data_quality": "quantified",
        "outcome": "DAMAGED (first verified collision between two cataloged objects)",
        "geometry": {"rel_velocity_kms": 14.0,
                     "documented_miss_note": "Head-on impact at >14 km/s severed the "
                                             "boom; satellite survived and recovered."},
        "public_facts": "First verified collision between two cataloged objects. A "
                        "~10 cm fragment from a 1986 Ariane H-10 upper stage (SPOT-1 "
                        "launch) struck the French Cerise reconnaissance microsat at "
                        "~670 km, severing its ~4 m gravity-gradient boom. Cerise was "
                        "recovered via novel magnetic attitude-control software.",
        "replay_note": "Primary (Cerise, 23606) may be in the catalog; the impacting "
                       "fragment is a specific debris piece. Conjunction context only.",
        "sources": ["NASA ODQN Vol.1 Iss.2 (1996)", "SSTL press release (1996)",
                    "Acta Astronautica (Sweeting et al., 2004)"],
    },

    # ============================ ASAT / BREAKUPS ============================
    {
        "key": "fengyun1c_2007",
        "category": "ASAT / Breakups",
        "replay_mode": "debris_cloud",
        "debris_group": "fengyun-1c-debris",
        "title": "Fengyun-1C ASAT test (2007)",
        "date_utc": "2007-01-11T22:26:00Z",
        "primary": {"name": "FENGYUN 1C", "norad": 25730},
        "secondary": {"name": "SC-19 ASAT interceptor", "norad": None},
        "data_quality": "quantified",
        "outcome": "DELIBERATE DESTRUCTION (Chinese ASAT test)",
        "geometry": {"rel_velocity_kms": 8.0},
        "public_facts": "Chinese direct-ascent ASAT destroyed the defunct Fengyun-1C "
                        "weather satellite at ~865 km. Largest debris-generating event "
                        "in history (~3,000+ tracked fragments) - high enough that the "
                        "cloud persists for decades and remains a major conjunction "
                        "source.",
        "replay_note": "Debris cloud trackable today (group 'fengyun-1c-debris').",
        "sources": ["NASA Orbital Debris Program Office", "CelesTrak fengyun-1c-debris"],
    },
    {
        "key": "cosmos1408_2021",
        "category": "ASAT / Breakups",
        "replay_mode": "debris_cloud",
        "debris_group": "cosmos-1408-debris",
        "title": "Cosmos 1408 ASAT test (2021)",
        "date_utc": "2021-11-15T02:47:00Z",
        "primary": {"name": "COSMOS 1408", "norad": 13552},
        "secondary": {"name": "Nudol ASAT interceptor", "norad": None},
        "data_quality": "quantified",
        "outcome": "DELIBERATE DESTRUCTION (Russian ASAT test)",
        "geometry": {"rel_velocity_kms": 7.5},
        "public_facts": "Russian direct-ascent ASAT destroyed defunct Cosmos 1408 at "
                        "~480 km, creating 1,500+ tracked fragments in a shell "
                        "overlapping the ISS. ISS crew sheltered in their return craft "
                        "as a precaution.",
        "replay_note": "Debris cloud trackable today (group 'cosmos-1408-debris').",
        "sources": ["U.S. Space Command", "NASA", "CelesTrak cosmos-1408-debris"],
    },
    {
        "key": "mission_shakti_2019",
        "category": "ASAT / Breakups",
        "replay_mode": "partial_facts",
        "debris_group": None,
        "title": "Mission Shakti - Microsat-R (India, 2019)",
        "date_utc": "2019-03-27T05:40:00Z",
        "primary": {"name": "MICROSAT-R", "norad": 43947},
        "secondary": {"name": "PDV-MkII ASAT interceptor", "norad": None},
        "data_quality": "quantified",
        "outcome": "DELIBERATE DESTRUCTION (Indian ASAT test - Mission Shakti)",
        "geometry": {"altitude_km": 283,
                     "documented_miss_note": "Direct-ascent intercept at ~283 km - "
                                             "deliberately low to minimize debris "
                                             "persistence."},
        "public_facts": "India's first ASAT test (DRDO PDV-MkII) destroyed its own "
                        "Microsat-R at ~283 km, making India the 4th nation with a "
                        "demonstrated ASAT capability. The low altitude was chosen so "
                        "most debris would decay quickly; ~400 fragments were tracked, "
                        "the majority re-entering within weeks to months.",
        "replay_note": "Low-altitude intercept; most debris has decayed. Listed with "
                       "documented altitude. Relevant to India's space-safety posture.",
        "sources": ["DRDO / Government of India (2019)", "NASA",
                    "U.S. 18th Space Defense Squadron catalog"],
    },
    {
        "key": "usa193_2008",
        "category": "ASAT / Breakups",
        "replay_mode": "partial_facts",
        "debris_group": None,
        "title": "USA-193 intercept - Operation Burnt Frost (2008)",
        "date_utc": "2008-02-21T03:26:00Z",
        "primary": {"name": "USA-193 (NROL-21)", "norad": 29651},
        "secondary": {"name": "SM-3 interceptor (USS Lake Erie)", "norad": None},
        "data_quality": "quantified",
        "outcome": "DELIBERATE INTERCEPTION (US - stated hydrazine-safety rationale; "
                   "widely viewed as an ASAT demonstration)",
        "geometry": {"altitude_km": 247, "rel_velocity_kms": 10.0,
                     "documented_miss_note": "SM-3 kinetic kill at ~247 km; closing "
                                             "speed >10 km/s."},
        "public_facts": "A modified US Navy SM-3 fired from USS Lake Erie intercepted "
                        "the defunct, tumbling NRO satellite USA-193 at ~247 km. The "
                        "stated purpose was to destroy its ~450 kg hydrazine tank "
                        "before uncontrolled re-entry; many analysts viewed it as an "
                        "ASAT demonstration following China's 2007 test. Produced 174 "
                        "cataloged fragments; ~95% re-entered within months, the last "
                        "by Oct 2009 - the low altitude limited lasting debris.",
        "replay_note": "Clearly labeled as an interception. Documented altitude / "
                       "closing speed shown; debris has decayed.",
        "sources": ["US Missile Defense Agency / DoD (2008)",
                    "NASA Orbital Debris Program Office"],
    },
    {
        "key": "briz_m_breakups",
        "category": "ASAT / Breakups",
        "replay_mode": "partial_facts",
        "debris_group": None,
        "title": "Briz-M / Proton upper-stage breakups (2007, 2012, ...)",
        "date_utc": None,
        "primary": {"name": "BRIZ-M upper stages (multiple)", "norad": None},
        "secondary": {"name": "fuel/structural fragmentation", "norad": None},
        "data_quality": "record_only",
        "outcome": "DOCUMENTED (multiple accidental upper-stage explosions)",
        "public_facts": "Several Russian Briz-M (Proton) upper stages have exploded in "
                        "orbit after failed burns left residual propellant - notably a "
                        "2007 event and a Feb 2012 event that each created hundreds of "
                        "tracked fragments in eccentric orbits crossing many altitudes. "
                        "Upper-stage explosions are, collectively, one of the largest "
                        "historical debris sources. Individual fragmentation geometries "
                        "are cataloged by NORAD but vary per event.",
        "replay_note": "Documented breakup family; specific fragment sets vary. "
                       "Listed as a record (no single primary to replay).",
        "sources": ["NASA Orbital Debris Quarterly News (multiple)",
                    "ESA Space Debris Office"],
    },

    # =============================== MANEUVERS ==============================
    {
        "key": "aeolus_starlink_2019",
        "category": "ESA Maneuvers",
        "replay_mode": "partial_facts",
        "debris_group": None,
        "title": "Aeolus x Starlink-44 CAM (2019)",
        "date_utc": "2019-09-02T11:02:00Z",
        "primary": {"name": "AEOLUS", "norad": 43600},
        "secondary": {"name": "STARLINK-44", "norad": 44235},
        "data_quality": "partial",
        "outcome": "AVOIDED (ESA raised Aeolus altitude ~350 m)",
        "geometry": {"final_pc": 1.0e-3, "esa_threshold_pc": 1.0e-4,
                     "socrates_pc": 1.0e-6, "altitude_raise_m": 350,
                     "miss_km": None, "dv_ms": None, "rel_velocity_kms": None},
        "public_facts": "First Aeolus collision-avoidance maneuver. ESA's Space Debris "
                        "Office computed Pc rising to ~1e-3 (10x ESA's 1e-4 threshold) "
                        "from US military CDM data, and raised Aeolus' altitude by "
                        "~350 m about half an orbit before TCA. Public SOCRATES "
                        "screening had predicted < 1e-6 - illustrating the gap between "
                        "public TLE screening and operator CDM data.",
        "replay_note": "Uses the PUBLISHED Pc (1e-3) and ESA threshold. Miss distance, "
                       "dv, and covariance were never released (CDM-restricted) and are "
                       "marked 'not public'. The SOCRATES-vs-CDM gap is the key lesson.",
        "sources": ["ESA 'ESA spacecraft dodges large constellation' (2019)",
                    "SpaceNews", "Space.com"],
    },
    {
        "key": "esa_sentinel_cams",
        "category": "ESA Maneuvers",
        "replay_mode": None,
        "debris_group": None,
        "title": "ESA Sentinel / fleet CAM history (record)",
        "date_utc": None,
        "primary": {"name": "ESA fleet (Sentinels, Cryosat-2, Swarm, Aeolus)", "norad": None},
        "secondary": {"name": "various debris", "norad": None},
        "data_quality": "record_only",
        "outcome": "DOCUMENTED (ESA performs ~tens of CAMs per year fleet-wide)",
        "public_facts": "ESA's Space Debris Office reports performing on the order of "
                        "tens of collision-avoidance maneuvers per year across its "
                        "fleet, rising over time with the growth of LEO traffic and "
                        "megaconstellations. ESA publishes aggregate statistics in its "
                        "annual Space Environment Report and occasional case studies "
                        "(e.g. Aeolus), but per-maneuver conjunction geometry "
                        "(miss/Pc/covariance) is generally not released.",
        "replay_note": "Aggregate documented record; per-event geometry not public.",
        "sources": ["ESA Space Environment Report (annual)", "ESA Space Debris Office"],
    },
    {
        "key": "iss_pdam_history",
        "category": "ISS Maneuvers",
        "replay_mode": None,
        "debris_group": None,
        "title": "ISS Debris Avoidance Maneuver history (record)",
        "date_utc": None,
        "primary": {"name": "ISS (ZARYA)", "norad": 25544},
        "secondary": {"name": "various debris", "norad": None},
        "data_quality": "record_only",
        "outcome": "DOCUMENTED (ISS has performed 30+ debris-avoidance maneuvers)",
        "public_facts": "The ISS performs Pre-Determined Debris Avoidance Maneuvers "
                        "(PDAMs) when a conjunction enters its alert box. NASA has "
                        "documented 30+ such maneuvers since 1999, including avoidances "
                        "of Cosmos 1408 and Fengyun-1C debris, plus several occasions "
                        "where crews sheltered in their return craft. NASA publishes "
                        "that maneuvers occurred and approximate dates, but per-event "
                        "conjunction geometry comes from CDMs and is not released "
                        "per-event.",
        "replay_note": "Documented record; per-event geometry not public.",
        "sources": ["NASA ISS Trajectory Operations", "NASA Orbital Debris Program"],
    },
    {
        "key": "isro_cam_history",
        "category": "ISRO Maneuvers",
        "replay_mode": None,
        "debris_group": None,
        "title": "ISRO collision-avoidance history (record)",
        "date_utc": None,
        "primary": {"name": "ISRO fleet (Cartosat, RISAT, etc.)", "norad": None},
        "secondary": {"name": "various debris", "norad": None},
        "data_quality": "record_only",
        "outcome": "DOCUMENTED (ISRO reports a growing number of CAMs per year)",
        "public_facts": "ISRO, through its IS4OM centre, reports performing a growing "
                        "number of collision-avoidance maneuvers across its fleet "
                        "(dozens per year in recent annual figures), driven by rising "
                        "LEO traffic. ISRO publishes aggregate counts in its space "
                        "situational-awareness reports, but per-maneuver conjunction "
                        "geometry is not publicly released per event.",
        "replay_note": "Aggregate documented record; per-event geometry not public. "
                       "(Directly relevant to ISDMAAS's target operators.)",
        "sources": ["ISRO IS4OM reports", "ISRO annual reports"],
    },
    {
        "key": "starlink_cam_summary",
        "category": "Commercial Operators",
        "replay_mode": None,
        "debris_group": None,
        "title": "Starlink autonomous CAM summary (aggregate)",
        "date_utc": None,
        "primary": {"name": "Starlink fleet", "norad": None},
        "secondary": {"name": "various objects", "norad": None},
        "data_quality": "record_only",
        "outcome": "DOCUMENTED (tens of thousands of autonomous maneuvers per period)",
        "public_facts": "SpaceX reports its Starlink fleet performs large numbers of "
                        "autonomous collision-avoidance maneuvers - tens of thousands "
                        "per 6-month reporting period in recent FCC filings, with a "
                        "surge after the 2021 Cosmos-1408 debris event. SpaceX "
                        "publishes aggregate counts; individual conjunction geometry is "
                        "not released. Independent analyses estimate a large fraction "
                        "of Starlink movements are conjunction-driven.",
        "replay_note": "Aggregate statistics only - no fabricated per-event geometry.",
        "sources": ["SpaceX FCC semi-annual reports", "peer-reviewed analyses (arXiv)"],
    },
]


def by_category():
    out = {}
    for e in EVENTS:
        out.setdefault(e["category"], []).append({
            "key": e["key"], "title": e["title"],
            "data_quality": e["data_quality"], "outcome": e["outcome"],
            "primary_norad": e["primary"].get("norad"),
        })
    return out


def get(key):
    for e in EVENTS:
        if e["key"] == key:
            return e
    return None


def quantified_keys():
    return [e["key"] for e in EVENTS if e["data_quality"] in ("quantified", "partial")]


if __name__ == "__main__":
    cats = by_category()
    print("HISTORICAL EVENT LIBRARY")
    print("=" * 56)
    for cat, items in cats.items():
        print(f"\n{cat}:")
        for it in items:
            q = {"quantified": "[full replay]", "partial": "[partial data]",
                 "record_only": "[record only]"}[it["data_quality"]]
            print(f"  - {it['title']:<48} {q}")
    nq = len(quantified_keys())
    print(f"\n{len(EVENTS)} events across {len(cats)} categories; "
          f"{nq} with usable geometry, {len(EVENTS)-nq} record-only.")