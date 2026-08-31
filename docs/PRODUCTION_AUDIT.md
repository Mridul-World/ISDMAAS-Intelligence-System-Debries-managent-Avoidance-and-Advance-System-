# ISDMAAS Production Audit

**Date:** 2026-08-31 · **Branch:** `production-hardening`
**Scope:** whole repository. Code was read, not modified, during the audit
itself; findings are tracked as numbered issues and fixed in later phases.

---

## 1. What this system is

A decision-support platform for satellite collision avoidance. It screens a
spacecraft against the public object catalog, computes collision probability,
proposes an avoidance burn, and validates that burn against the catalog before
presenting it. It recommends; a human operator decides and commands.

**It is not** an autonomous control system, and nothing should describe it as one.

---

## 2. Architecture

```
                       +------------------------------+
  CelesTrak GP ------->|  catalog.py                  |  6 h TTL, atomic cache,
  (element sets)       |  CatalogService              |  bundled offline fallback
                       +--------------+---------------+
                                      | (norad, name, Satrec)
  operator TLE --> store.py ----------+
  (registered)      SQLite            |
                                      v
                       +------------------------------+
                       | astrodynamics.py             |  apogee/perigee filter
                       |  propagate / screen / refine |  -> vectorized coarse grid
                       +--------------+---------------+  -> Brent refinement
                                      | CloseApproach
                                      v
                       +------------------------------+
  CDM (operator) ----->| phase7_collision.py          |  encounter-plane
  cdm_ingest.py        |  Pc: Foster quadrature       |  projection, Chan
                       |      + Chan cross-check      |  cross-check
                       +--------------+---------------+
                                      v
                       +------------------------------+
                       | phase8_maneuver.py           |  CW impulse response,
                       |  minimum-dv search           |  bisection per lead time
                       +--------------+---------------+
                                      v
                       +------------------------------+
                       | phase10_safety.py            |  fuel / orbit /
                       |  4-check safety gate         |  threat-resolved /
                       +--------------+---------------+  no-new-conjunction
                                      v
                    phase11_api.py (FastAPI) --> dashboard.html + console.js
```

**Prediction layer (`phase55/`)** trains a residual-correction Transformer over
SGP4. It is **not currently wired into the serving path** — see section 11.

### Module inventory (`phase11/`, 8 103 lines)

| Module | Lines | Responsibility |
|---|---:|---|
| `phase11_api.py` | 1352 | FastAPI app, endpoints, request/response models |
| `auth_store.py` | 475 | accounts, sessions, satellite ownership, demo seed |
| `isdmaas_core/astrodynamics.py` | 640 | propagation, screening, TCA refinement |
| `phase7_collision.py` | 446 | Pc engine, covariance, encounter plane |
| `phase10_safety.py` | 427 | post-burn re-screen and safety gate |
| `phase8_maneuver.py` | 360 | CW impulse model, minimum-dv search |
| `isdmaas_core/catalog.py` | 355 | catalog fetch, cache, resolve |
| `isdmaas_core/store.py` | 347 | SQLite registry (users/sessions/satellites) |
| `cdm_ingest.py` | 320 | CCSDS CDM parser (KVN + XML) |
| `debris_data.py` | 303 | object cache builder (OMM/JSON) |
| `isdmaas_core/socrates.py` | 285 | CelesTrak SOCRATES feed client |
| `isdmaas_core/security.py` | 263 | PBKDF2, tokens, rate limiting, headers |
| `user_conjunctions.py` | 260 | operator-asset assess/plan |
| `isdmaas_core/config.py` | 236 | typed settings, production guards |
| `isdmaas_core/screening.py` | 227 | the single screening implementation |
| `training_pipeline.py` | 216 | truth scan to labelled samples |
| `user_screening.py` | 164 | operator full-catalog screen |
| `ops_db.py` | 160 | element-set history, screening records |
| `isdmaas_core/errors.py` | 129 | uniform error envelope |
| `fix_*.py`, `add_*.py` | 713 | **archived**, disabled one-shot patch scripts |

---

## 3. Data flow and storage

| Store | Path | Contents | Durability need |
|---|---|---|---|
| Operator registry | `$DATA_DIR/isdmaas_auth.db` | users, password hashes, session digests, ownership | **irreplaceable** |
| Operations DB | `$DATA_DIR/isdmaas_ops.db` | element-set history, screening runs | **irreplaceable** (future training input) |
| Object cache | `$DATA_DIR/isdmaas_cache.db` | OMM records for the 3-D view | disposable |
| Catalog cache | `$DATA_DIR/catalog_cache.tle` | current element sets | disposable |
| Offline fallback | `phase11/data/catalog_active.tle` | committed element sets | read-only |
| Precise-orbit states | `phase55/reports/state_*.json` | determined state and sigma | reference |

All SQLite, WAL mode, thread-local connections, single node.

---

## 4. Authentication and authorization

- PBKDF2-HMAC-SHA256, 240 000 iterations, per-user salt; legacy single-round
  SHA-256 records verified and upgraded on next sign-in.
- Session tokens: 256-bit opaque, **only a SHA-256 digest persisted**, absolute
  expiry (default 12 h).
- Rate limiting: per-process sliding window keyed on `request.client.host`.
- Authorization model: **ownership only**. No roles, no admin, no read-only
  accounts, no organizational grouping.

---

## 5. API surface

23 routes. Authentication requirements as built:

| Auth | Endpoints |
|---|---|
| none | `/health` `/ready` `/satellites` `/resolve` `/orbit/{n}` `/screen/{n}` `/monitor` `/monitor/reset` `/assess` `/cdm/assess` `/maneuver-plan` `/autonomous` `/socrates` `/live/catalog` `/live/conjunctions/{n}` `/library` `/library/{k}` `/historical-events` `/sync-status` `/sync-now` `/user/assess` |
| token | `/auth/me` `/my/satellites` (GET/POST/DELETE) `/auth/logout*` |
| **owner** | `/user/screen/{n}` `/user/plan` |

---

## 6. Scientific and numerical posture

**Frames.** The engine is TEME end to end (SGP4's native output). There is no
ECI/ECEF conversion anywhere in the serving path, so there is no opportunity for
a frame-mixing error in the physics. Docstrings say "ECI/TEME" loosely; TEME is
what is meant.

**Units.** Kilometres and km/s throughout the engine; metres per second only at
the delta-v interface (`dv_ms`), converted at a single boundary. Degrees appear
only in reported orbital elements.

**Time.** UTC datetimes converted to Julian day/fraction via `sgp4.api.jday`.
UT1-UTC is not applied (<= 0.9 s). At 15 km/s that is a <= 13 km along-track
effect on *absolute* position, but it is common-mode between two objects on the
same time base, so relative geometry — the thing conjunction assessment depends
on — is unaffected.

**Already independently validated on this branch:** TCA refinement agrees with a
brute-force 10 ms scan to the metre and millisecond; the Pc quadrature agrees
with dense numerical integration to 6 significant figures; the CW impulse model
agrees with RK4 two-body propagation to 0.02-0.2%.

---

## 7. Tests as found

119 tests in 5 files (`test_api` 28, `test_security` 24, `test_maneuver` 13,
`test_risk_engine` 12, `test_astrodynamics` 8). No `tests/scientific/`, no
`tests/e2e/`, no performance suite, no ML regression test, **no CI**.

---

## 8. Deployment as found

`Dockerfile` (multi-stage, non-root, healthcheck) and `DEPLOY.md` exist.
No CI/CD. No migration framework. No backup automation. No rollback procedure.

---

# 9. Prioritized issue list

## P0 — safety, scientific or security failure

| ID | Issue | Evidence |
|---|---|---|
| **P0-1** | **Live credentials committed to git.** `dd.env` (Space-Track user+pass, DISCOS token, CDSE user+pass) and `phase2/data/build_ssa_dataset.py` (same values hardcoded in source). Present in **all 9 commits**. | `git log --all -- dd.env`; grep of tracked sources |
| **P0-2** | **Shipped ML checkpoint does not reproduce its validation report** and is anti-correlated with the target on 2 of 3 axes (r = -0.47 radial, -0.47 cross-track). Report claims +7.62% RMSE / +36.6% median; re-scoring gives +2.95% / -2.57%. | `docs/technical/09-model-artifact-audit.md` |
| **P0-3** | **No ML regression gate.** Nothing prevents a mismatched checkpoint / normalization / feature-schema pairing from shipping again — the mechanism behind P0-2. | absence |
| **P0-4** | **No end-to-end post-burn validation.** The safety gate passes unit tests, but no test walks detection to plan to burn to propagate to re-screen to confirmed improvement on real data. | absence of `tests/e2e/` |

## P1 — production-breaking defect

| ID | Issue | Evidence |
|---|---|---|
| **P1-1** | **Console renders two incompatible coordinate conventions.** Live debris uses `eciToScene` = `(x, z, -y)`; orbit tracks use raw `(x, y, z)`. The Earth mesh is Y-up, so orbit tracks render rotated 90 degrees relative to the Earth and to every live object. A real conjunction does not appear as one. | `console.js:561` vs `:489, :498` |
| **P1-2** | **Collision engine duplicated 6x and stale.** `phase7/ phase8/ phase9/ phase10/ phase12/ phase13/` each carry a byte-identical `phase7_collision.py` with the one-sided TCA clamp and the Chan overflow. Their callers do `pre["pc"] > threshold`, which raises `TypeError` under the corrected engine's `None` return. | md5 identical; `grep 't_lin = max(0.0'` |
| **P1-3** | **Every compute-heavy endpoint is unauthenticated** (`/screen`, `/assess`, `/maneuver-plan`, `/autonomous`, `/monitor`). Each is a full-catalog screen. Trivially abusable as a denial-of-service amplifier. | route table, section 5 |
| **P1-4** | **No failure-state vocabulary.** `pc: None` exists for sub-short-encounter geometry, but there is no `DATA_INVALID` / `ASSESSMENT_UNAVAILABLE` / `MODEL_UNAVAILABLE` state, and stale or invalid inputs are not distinguished from genuine low risk. | `phase7_collision.assess_conjunction` |
| **P1-5** | **No assessment audit record.** Nothing persists why a recommendation was made — no assessment ID, element-set epochs, algorithm version or covariance snapshot. "Why did ISDMAAS recommend this?" is unanswerable after the fact. | absence |

## P2 — reliability / performance

| ID | Issue |
|---|---|
| **P2-1** | No CI. Nothing runs the 119 tests automatically. |
| **P2-2** | Rate limiting is per-process; a multi-worker deployment multiplies the effective limit by worker count. |
| **P2-3** | Sessions are node-local; a rolling restart signs every operator out. |
| **P2-4** | No DB migration framework; schema changes need manual intervention. |
| **P2-5** | No performance baseline at any catalog size. |
| **P2-6** | No password reset or account recovery; recovery is a manual DB edit. |
| **P2-7** | `/sync-now` is unauthenticated and triggers outbound fetches. |

## P3 — quality / UX

| ID | Issue |
|---|---|
| **P3-1** | Docstrings say "ECI" where TEME is meant. |
| **P3-2** | No generated API documentation artifact (only live `/docs` in development). |
| **P3-3** | Console does not surface model version, algorithm version or assessment status. |
| **P3-4** | UI does not distinguish CALCULATED / SIMULATED / RECOMMENDED / EXECUTED. |

---

## 10. Risk register

**Scientific.** The physics engine is in good shape and independently validated
on its core paths. The residual risk is P0-2: the ML layer's claimed accuracy is
not reproducible, and the training dataset needed to re-derive it is absent.

**Numerical.** No cancellation or overflow paths remain in the Pc engine after
this branch's fixes; both were present before. Degenerate covariance is floored
rather than rejected — acceptable for round-off, but it means a *genuinely*
invalid covariance is silently accepted (P1-4).

**Reproducibility.** No checkpoint hash is recorded alongside any validation
number. That gap is what allowed P0-2.

**Security.** P0-1 is unbounded until the credentials are rotated. History
rewriting alone does not help, because the repository has been pushed.

---

## 11. Explicit note on the ML layer

`phase55` trains a model. `phase11` **does not load it.** No serving path imports
a checkpoint; the API's prediction layer is SGP4 alone.

This is currently the *safe* default, and it is why P0-2 is not also a
serving-path defect. But the README and the architecture diagram both imply the
model is in the loop, which is misleading. Either wire it behind a validated gate
with a baseline fallback, or state plainly that v1.0 serves SGP4.
