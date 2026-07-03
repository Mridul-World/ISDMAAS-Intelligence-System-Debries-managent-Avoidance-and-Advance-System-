"""
PHASE 12 — historical event database.

Real, documented orbital conjunction / collision events with their public
metadata. The replay uses the archived TLEs for each object from BEFORE the
event epoch (the only information that was available at the time) and runs the
ISDMAAS engine to show what it would have detected, assessed, and recommended.

Sources are public reporting / agency statements (CelesTrak AMOS 2009, ESA,
SpaceNews, NASA). Values are the documented figures, cited for comparison —
ISDMAAS recomputes its own numbers from the TLEs independently.
"""

EVENTS = {
    "iridium_cosmos_2009": {
        "title": "Iridium 33 \u00d7 Cosmos 2251 collision",
        "date_utc": "2009-02-10T16:56:00",
        "primary": {"norad": 24946, "name": "IRIDIUM 33", "cospar": "1997-051C",
                    "mass_kg": 560, "status": "operational"},
        "secondary": {"norad": 22675, "name": "COSMOS 2251", "cospar": "1993-036A",
                      "mass_kg": 900, "status": "defunct"},
        "outcome": "COLLISION",
        # documented conjunction geometry (the REAL recorded values) used to seed
        # an honest dashboard scenario. NOT re-derived from public TLEs (which
        # cannot reconstruct a sub-km miss — see Phase 12).
        "scenario": {"miss_km": 0.584, "rel_velocity_kms": 11.7, "tca_lead_h": 8,
                     "altitude_km": 789},
        "documented": {
            "altitude_km": 789, "rel_velocity_kms": 11.7,
            "socrates_predicted_miss_m": 584,
            "socrates_rank": "64 of ~250 (not flagged as top threat)",
            "note": "First accidental hypervelocity collision of two intact "
                    "satellites. Public TLEs lacked covariance; the event was "
                    "NOT ranked among the most critical that day \u2014 a known "
                    "limitation of TLE-only screening without uncertainty."},
        "isdmaas_point": "ISDMAAS adds calibrated uncertainty to TLE-only data, "
                         "so it reports an explicit Pc and risk band rather than "
                         "an unranked miss distance.",
    },
    "aeolus_starlink_2019": {
        "title": "ESA Aeolus \u00d7 Starlink-44 near-miss (avoided)",
        "date_utc": "2019-09-02T11:02:00",
        "primary": {"norad": 43600, "name": "AEOLUS", "cospar": "2018-066A",
                    "mass_kg": 1360, "status": "operational"},
        "secondary": {"norad": 44235, "name": "STARLINK-44", "cospar": "2019-029K",
                      "mass_kg": 260, "status": "operational"},
        "outcome": "AVOIDED (ESA maneuvered)",
        "scenario": {"miss_km": 0.06, "rel_velocity_kms": 14.4, "tca_lead_h": 12,
                     "altitude_km": 320, "pc": 1e-3},
        "documented": {
            "collision_probability": 1e-3,
            "esa_threshold": 1e-4,
            "maneuver": "orbit raise, ~half an orbit before TCA, "
                        "thruster burns 10:14/10:17/10:18 UTC",
            "lead_time": "decision made the day before; detected ~5 days ahead",
            "note": "First ESA collision-avoidance maneuver against an active "
                    "constellation satellite. ESA raised Aeolus; coordination "
                    "with SpaceX failed (paging bug)."},
        "isdmaas_point": "ISDMAAS would flag Pc above the action threshold and "
                         "recommend a raise maneuver with a comparable lead time "
                         "\u2014 reproducing the decision ESA made manually.",
    },
    "cosmos1408_test_2021": {
        "title": "Cosmos 1408 ASAT debris field (2021)",
        "date_utc": "2021-11-15T02:47:00",
        "primary": {"norad": 25544, "name": "ISS (ZARYA)", "cospar": "1998-067A",
                    "mass_kg": 420000, "status": "crewed"},
        "secondary": {"norad": 13552, "name": "COSMOS 1408", "cospar": "1982-092A",
                      "mass_kg": 2200, "status": "destroyed (ASAT test)"},
        "outcome": "DEBRIS FIELD (ISS sheltered)",
        "scenario": {"miss_km": 0.5, "rel_velocity_kms": 7.0, "tca_lead_h": 6,
                     "altitude_km": 480},
        "documented": {
            "altitude_km": 480,
            "note": "Russian ASAT test destroyed Cosmos 1408, creating ~1500 "
                    "tracked debris pieces near the ISS altitude; the ISS crew "
                    "sheltered in capsules. Illustrates screening a primary "
                    "against a debris-generating event."},
        "isdmaas_point": "ISDMAAS screens the ISS against the resulting debris "
                         "shell and ranks the closest approaches by Pc.",
    },
}
