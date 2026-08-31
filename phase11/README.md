# ISDMAAS — service and mission console

The deployable part of ISDMAAS: a REST service that screens a satellite against
the public catalog, computes collision probability, plans an avoidance burn and
validates it — plus a browser console that drives all of it.

**This system recommends. A human operator decides and commands the spacecraft.**

---

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate       # Windows
pip install -r requirements-dev.txt

# terminal 1 — the service
uvicorn phase11_api:app --port 8000

# terminal 2 — the console
python -m http.server 8080
# open http://localhost:8080/dashboard.html
```

The top bar should show **Link active** and a green data indicator. Development
mode seeds two demo operators (`operator_a` / `alpha123`, `operator_b` /
`bravo123`) that own two fictitious satellites on crossing orbits — enough to
walk through detection, planning and the owner-only authorization rule without
any real data.

For anything beyond a laptop, read [DEPLOY.md](DEPLOY.md) first. The service
**refuses to start** in production mode without a secret key and an explicit CORS
origin list.

---

## Layout

```
phase11/
├── phase11_api.py            FastAPI application — endpoints and wiring
├── auth_store.py             operator accounts, sessions, satellite ownership
├── user_conjunctions.py      assess / plan for operator-registered satellites
├── user_screening.py         full-catalog screen for an operator's own asset
├── phase7_collision.py       collision probability engine (Foster + Chan)
├── phase8_maneuver.py        minimum-delta-v avoidance planner
├── phase10_safety.py         the safety gate every maneuver passes through
├── cdm_ingest.py             CCSDS CDM parser — real operator covariance
├── ops_db.py                 element-set history and screening records
├── debris_data.py            object cache builder (CelesTrak GP/OMM)
├── training_pipeline.py      truth-data scan and training-sample generation
│
├── isdmaas_core/             production support layer
│   ├── config.py             typed, environment-driven settings
│   ├── security.py           password hashing, tokens, rate limiting
│   ├── errors.py             one error envelope for the whole API
│   ├── logging_config.py     structured logs with request correlation ids
│   ├── store.py              SQLite registry for operators and assets
│   ├── astrodynamics.py      propagation, screening, exact TCA refinement
│   ├── screening.py          the single conjunction-screening implementation
│   ├── catalog.py            shared, cached public object catalog
│   └── socrates.py           CelesTrak SOCRATES feed client
│
├── dashboard.html            mission console markup
├── console.css               console styling
├── console.js                console logic
├── vendor/                   three.js and satellite.js, vendored — see its README
└── tests/                    pytest suite
```

---

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | — | liveness |
| GET | `/ready` | — | readiness: catalog loaded, registry reachable |
| GET | `/satellites` | — | the configured fleet |
| GET | `/resolve?q=` | — | NORAD id / name / COSPAR id lookup |
| GET | `/orbit/{norad}` | — | one orbit of ECI positions for the 3-D view |
| GET | `/screen/{norad}` | — | full-catalog conjunction screen |
| GET | `/monitor` | — | watchlist screen with new-alert tracking |
| POST | `/assess` | — | risk for a specific primary/secondary pair |
| POST | `/cdm/assess` | — | risk from a real CCSDS CDM |
| POST | `/maneuver-plan` | — | planned and safety-validated avoidance burn |
| POST | `/autonomous` | — | the hands-off detect → plan → validate loop |
| GET | `/socrates` | — | CelesTrak's published conjunction feed |
| GET | `/live/catalog` | — | cached objects for browser-side propagation |
| GET | `/library`, `/library/{key}` | — | documented historical events |
| POST | `/auth/register`, `/auth/login` | — | operator accounts (rate limited) |
| POST | `/auth/logout`, `/auth/logout-all` | token | end sessions |
| GET | `/auth/me`, `/my/satellites` | token | the caller's account and assets |
| POST | `/my/satellites` | token | register an element set |
| DELETE | `/my/satellites/{norad}` | token | deregister an asset |
| GET | `/user/screen/{norad}` | **owner** | full-catalog screen for your own asset |
| POST | `/user/assess` | — | conjunction between two registered assets |
| POST | `/user/plan` | **owner** | avoidance plan — owner only |

Interactive docs at `/docs` in development. They are disabled in production.

---

## How a conjunction is actually computed

This is worth reading before trusting a number the console shows.

1. **Geometric filter.** Objects whose perigee sits above the primary's apogee
   (plus the screening gate) can never come close, whatever the phasing. This
   rejects most of a 16 000-object catalog with no propagation at all.

2. **Coarse screen.** Survivors are propagated with SGP4 across the window on a
   30-second grid, vectorized through `SatrecArray`. Near a conjunction the
   squared separation is locally a parabola in time, so a grid sample that is
   lower than both its neighbours brackets a true minimum within one step.

3. **Refinement.** Every bracket is minimized with Brent's method down to a
   millisecond. This is the step that makes the miss distance mean something: a
   coarse grid alone is off by up to `relative_speed × step / 2`, which at
   14 km/s is a 200 km error.

4. **Probability.** Miss vector and combined covariance are projected onto the
   plane perpendicular to the relative velocity, and the 2-D Gaussian is
   integrated over a disk of the combined hard-body radius by Gauss-Legendre
   quadrature. Chan's series runs alongside as an independent cross-check, and
   the response reports whether the two agree.

5. **Covariance.** The secondary's uncertainty is grown over the **actual** time
   from its element-set epoch to the refined TCA. A primary with a calibrated
   precise-orbit solution uses that solution's uncertainty; anything else uses
   the documented TLE-scale model. Which one was used is reported in every
   response as `covariance_source`.

A conjunction whose relative speed falls below 10 m/s is reported as
`UNDETERMINED` rather than given a number: the short-encounter assumption that
Foster and Chan both rest on does not hold there.

---

## How a maneuver is planned

A tangential burn changes the orbital period, and that period change accumulates
into an along-track displacement. The planner uses the Clohessy-Wiltshire
impulse response:

```
radial(t)      = (2 Δv / n) (1 − cos n t)
along-track(t) = (4 Δv / n) sin(n t) − 3 Δv t
```

It bisects on Δv magnitude in both directions, at each of several lead times, for
the smallest burn that brings Pc under `PC_SAFE`, and reports the whole trade
table — because "burn earlier and spend less propellant" is the decision the
operator is actually making.

Every candidate then passes four safety checks:

1. **Fuel budget** — with a 30% margin.
2. **Orbit safety** — post-burn perigee and semi-major axis stay inside limits,
   evaluated from the actual post-burn velocity.
3. **Original threat resolved** — recomputed independently of the planner's claim.
4. **No new conjunction** — the post-burn trajectory is re-screened against the
   whole catalog. Solving one conjunction by creating another is the failure this
   check exists to catch, and it runs at the same accuracy as the original screen.

Any failure rejects the maneuver.

---

## Testing

```bash
pytest tests -q
```

115 tests covering the propagation and TCA layer, the probability engine, the
planner and safety gate, credential handling, and the API's routing,
authorization and validation. The physics tests compare against independent
ground truth — a brute-force 10 ms TCA scan, a dense numerical integration of
the probability integral, and an RK4-propagated two-body orbit — rather than
against the implementation's own previous output.

---

## Operating notes

- **Catalog freshness matters.** `/ready` and `/sync-status` report the
  catalog's age, and the console warns when it exceeds 24 hours. Screening
  against stale element sets produces confident, wrong answers.
- **Screening is CPU-bound.** A full-catalog screen takes roughly a second per
  satellite. `/monitor` caps a watchlist at 20 satellites per scan for that
  reason.
- **Respect CelesTrak.** The catalog is cached for six hours and concurrent
  refreshes are coalesced. Do not poll `/sync-now` more often than hourly.
