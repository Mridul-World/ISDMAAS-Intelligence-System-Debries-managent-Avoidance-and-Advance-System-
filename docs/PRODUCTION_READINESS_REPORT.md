# ISDMAAS Production Readiness Report

**Date:** 2026-08-31 · **Branch:** `production-hardening`
**Test run:** clean environment, caches cleared — **385 passed, 3 skipped, 0 failed**

---

# VERDICT: NOT PRODUCTION-READY

**One P0 remains open, and it cannot be closed by code.**

The scientific, safety and software engineering are in good shape and are backed
by evidence. What blocks a v1.0 release is credential exposure that requires
action at the credential providers, plus an ML layer whose claimed accuracy does
not reproduce.

The system is **suitable now for a single-tenant pilot behind TLS with a small
number of known operators**, once the credentials are rotated.

---

## Gate summary

| Area | Verdict | Evidence |
|---|---|---|
| Scientific correctness | **PASS** | `docs/TCA_VALIDATION.md`, `tests/scientific/` (173 tests) |
| Collision probability validation | **PASS** | `docs/COLLISION_PROBABILITY_VALIDATION.md` |
| Maneuver validation | **PASS** | `docs/MANEUVER_VALIDATION.md` |
| E2E workflow | **PASS** | `tests/e2e/test_mission_workflow.py` |
| API | **PASS** | 120 unit/API tests; uniform error envelope; request ids |
| Authentication / authorization | **PASS** | `docs/SECURITY_AUDIT.md` §2–3, IDOR all refused |
| Reproducibility | **PASS** | assessment audit trail, `/assessments/{id}` |
| Performance | **PASS** | `docs/PERFORMANCE_BENCHMARK.md` |
| Deployment | **PASS** | `phase11/DEPLOY.md`, container guard verified in CI |
| **Security** | **FAIL** | **P0-1: credentials in all 9 commits, not yet rotated** |
| **ML validation** | **FAIL** | **P0-2: checkpoint does not reproduce; not in production (safe)** |

---

# FAILURES

## FAIL 1 — Credential exposure (P0, blocking)

**Problem.** Live Space-Track, ESA DISCOS and Copernicus credentials were
committed in `dd.env` and hardcoded in `phase2/data/build_ssa_dataset.py`. They
are present in **all 9 commits** of a repository that has been pushed to GitHub.

**Severity.** P0. Highest of any finding — it is the only one with unbounded
blast radius and the only one outside this codebase's control.

**Impact.** Third-party account compromise under a named individual's identity.
The Space-Track username is a personal email address, so any password reuse
extends the exposure beyond this project. Possible account suspension for
terms-of-use violation.

**Evidence.**
```
git log --all --oneline -- dd.env            -> present from the first commit
git cat-file -e <every commit>:phase2/data/build_ssa_dataset.py  -> present in all 9
```

**Required fix (not code — provider action):**
1. Rotate all four credentials at Space-Track, DISCOS and CDSE. Do this first.
2. Check for password reuse; the Space-Track password was committed in plaintext.
3. Review provider access logs for the exposure window.
4. Purge history with `git filter-repo`, or abandon the repository for a fresh one.
5. Rotate **again** after the purge — pre-purge objects persist in forks and caches.

**Already done in code:** files untracked and gitignored, credentials moved to
environment variables, `.env.example` added, and a CI job that fails the build if
a credential file or credential-shaped literal is ever tracked again (verified to
fire on the real historical leak and stay silent on placeholders).

## FAIL 2 — ML layer not validated (P0 for the ML claim; NOT blocking the service)

**Problem.** The shipped checkpoint does not reproduce its committed validation
report, and its predictions are anti-correlated with the target on two of three
axes.

| Metric | SGP4 | Deployed checkpoint | Report claimed |
|---|---:|---:|---:|
| RMSE (km) | 1.9071 | 1.8508 | 1.7618 |
| Median (km) | 0.6090 | **0.6247** | 0.3859 |
| Improvement | — | **+2.95% / −2.57%** | +7.62% / +36.6% |

Per-axis correlation with the target: **radial −0.470**, along-track +0.278,
**cross-track −0.470**.

**Severity.** P0 as a *claim*; **not a serving-path defect**.

**Impact on the running system: none.** No module under `phase11/` imports torch
or loads a checkpoint. The prediction layer is SGP4, the validated baseline, and
every assessment record states `model_version: "none (SGP4 baseline)"`. This is
the safe fallback the requirement asks for, and it is the default.

**Impact on external claims: material.** The README and IEEE paper cite accuracy
figures that the artifacts do not reproduce.

**Required fix.** Six steps in `docs/ML_VALIDATION_REPORT.md` §8. The blocking
one is that the training set (`phase55/data/phase55_dataset.csv`, `X_train.pt`)
is absent from the repository and rebuilding it needs the ESA POD archive and
CDSE credentials — which must be rotated first, so FAIL 1 gates FAIL 2.

---

# PASSES, with evidence

## Scientific correctness — PASS

Validated against **independent reference implementations**, not against the
code's own previous output.

| Property | Result | Threshold | Margin |
|---|---:|---:|---:|
| TCA vs independent search (15 real pairs) | 0.544 ms | 2 ms | 3.7× |
| Miss distance vs independent search | 0.0052 m | 1 m | 190× |
| Pc vs dense numerical integration (19 cases) | 8.5e-4 rel | 2e-3 | 2.4× |
| CW maneuver model vs RK4 propagation (15 cases) | 0.75% | 2% | 2.7× |

Units, frames, epochs and time conversions are pinned to externally checkable
constants: JD 2460311.0, the vis-viva identity, ISS inclination 51.6416°,
covariance-rotation invariants (trace, determinant, eigenvalues).

## Collision probability — PASS

Three independent methods agree: Gauss-Legendre quadrature (production), a
3000×3000 log-stabilized dense integration, and 4×10⁶-sample Monte Carlo. Swept
across covariance magnitude, anisotropy to 500:1, miss distance from zero to
12σ, HBR from 1 m to 100 m, and HBR/σ from 1e-4 to 20.

Invalid covariance now **fails safely** (`pc: null`, `risk_level:
DATA_INVALID`). Previously a negative eigenvalue returned **Pc = 0.85** and a
zero matrix returned **Pc = 0.0** — a false alarm and a missed conjunction
respectively.

## Maneuver validation — PASS

The full chain is exercised: burn → post-burn state → propagation → re-screen →
new TCA → new Pc → safety gate. Every option independently reaches `PC_SAFE`;
the trade table is monotonic in lead time; a burn that worsens the geometry is
detected; an unsolvable conjunction returns escalation rather than a delta-v that
does not work.

## E2E workflow — PASS

All 18 steps, in order, on real data, with each step feeding the next — including
the assertion that the recommended burn increases miss distance **and** reduces
Pc.

## Security — PASS on everything except FAIL 1

IDOR fully refused across four principals. PBKDF2 240k iterations; digest-only
token storage; expiry; rate limiting; allowlist input validation; no information
disclosure across five error paths; five production configuration guards that
refuse to start.

## Performance — PASS

| Objects | Time | Note |
|---:|---:|---|
| 15 894 (real) | 0.239 s | full public catalog |
| 100 000 (synthesised) | 1.664 s | 233 MiB peak |

Linear at ~60 000 objects/s. The Brent refinement that makes the miss distance
correct costs 1.5% of screening time.

## Reproducibility — PASS

Every recommendation writes an audit record capturing inputs as well as outputs:
element-set epochs for both objects, catalog age, algorithm version, model
version, covariance source, per-axis sigma, hard-body radius, Pc method.
Retrievable at `/assessments/{id}`, scoped to the owning operator.

---

# Defects found and fixed in this pass

| # | Severity | Defect |
|---|---|---|
| 1 | P0 | Invalid covariance produced confident wrong Pc (0.85 for a false alarm, 0.0 for a missed conjunction) |
| 2 | P0 | Credentials committed in all 9 commits |
| 3 | P0 | No E2E test of the detection→plan→verify chain |
| 4 | P1 | Element-set validation accepted truncated lines, broken checksums, mismatched catalog numbers and a 48°-wrong inclination |
| 5 | P1 | Console rendered orbits **5 963 km** from the objects on them (two axis conventions) |
| 6 | P1 | `/live/catalog` returned 503 on every call (read-only open of a WAL database) |
| 7 | P1 | `/monitor` alerted on the ISS's own modules as 0 km conjunctions |
| 8 | P1 | Compute endpoints unauthenticated and unlimited (DoS amplifier) |
| 9 | P1 | No assessment audit record |
| 10 | P2 | Planner recommended operationally infeasible 2.5-minute lead times as "best" |
| 11 | P2 | Quadrature returned 6.2e-31 instead of 1.0 when HBR ≫ σ |
| 12 | P2 | Rate limiter keyed partly on a client-controlled header |
| 13 | P2 | Two fragile loop-variable closures; two `zip()` calls that could truncate silently |

Defects found in the **validation code itself** and corrected (recorded because
they affect how much the evidence is worth): the TCA reference implementation had
the same aliasing bug it was testing for; the relative-position threshold was
mis-derived as second-order when it is first-order; a maneuver test's burn
overshot its own premise; the E2E audit check compared two independent screens
taken seconds apart.

---

# Remaining risks

| Risk | Severity | Status |
|---|---|---|
| Credentials in git history, not rotated | **P0** | **OPEN — provider action required** |
| ML accuracy claims unreproducible | **P0** (claim only) | OPEN; serving path is safe |
| Collision engine duplicated 6× and stale in `phase7/`–`phase13/` | P1 | OPEN — deliberately not propagated; those callers do `pre["pc"] > threshold` and would crash on the corrected engine's `None`. Steps recorded in README §7. |
| No role model (ownership-only authorization) | P2 | Accepted; bounds deployment to single-tenant |
| Rate limiting per process | P2 | Accepted; run one worker or add a shared limiter |
| Sessions node-local | P2 | Accepted; rolling restart signs operators out |
| No password reset | P2 | Accepted; manual DB operation |
| No concurrency/soak testing | P2 | Not measured — stated in the performance report |
| Only tangential burns searched | P2 | Documented limitation |
| UT1−UTC not applied | P3 | ≤0.9 s, common-mode, no effect on relative geometry |

---

# Release decision

**Do not tag v1.0.**

Two P0 items are open. The first requires action at three credential providers
and a history rewrite. The second requires data that is not in the repository.

**What v1.0 could honestly claim once FAIL 1 is closed:**

> A conjunction screening and avoidance-planning service whose astrodynamics are
> validated against independent reference implementations to sub-millisecond TCA
> and sub-centimetre miss distance, whose collision probability agrees with dense
> numerical integration to 0.1%, and whose maneuver planner is validated against
> numerical orbit propagation to 0.75%. Prediction is SGP4. The ML layer is
> present in the repository but is **not** validated and is **not** in the
> serving path.

That is a defensible claim. "ML-enhanced orbit prediction" is not, until the ML
validation report says PASS.

---

# Commands

**Complete production test suite** (must be green before any release):

```bash
cd phase11 && python -m pytest tests -v
```

Individually:

```bash
cd phase11
python -m pytest tests/scientific -v      # 173 — safety-critical
python -m pytest tests/security -v        #  93 — safety-critical
python -m pytest tests/e2e -v             #   2 — safety-critical
python -m pytest tests/test_*.py -v       # 120 — unit and API
python -m ruff check .                    # lint
python tests/benchmark_screening.py       # performance baseline
cd ../phase55 && python train_phase55_rtn_v2.py --self-test   # ML pipeline guard
```

**Start the production system:**

```bash
export ISDMAAS_ENV=production
export ISDMAAS_SECRET_KEY="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
export ISDMAAS_CORS_ORIGINS="https://console.example.org"
export ISDMAAS_DATA_DIR=/var/lib/isdmaas

cd phase11
uvicorn phase11_api:app --host 127.0.0.1 --port 8000 \
        --workers 4 --proxy-headers --forwarded-allow-ips '<proxy-ip>'
```

Or containerised:

```bash
docker build -t isdmaas:latest phase11
docker run -d --name isdmaas -p 127.0.0.1:8000:8000 -v isdmaas-data:/data \
  -e ISDMAAS_SECRET_KEY="$ISDMAAS_SECRET_KEY" \
  -e ISDMAAS_CORS_ORIGINS="https://console.example.org" \
  isdmaas:latest
```

Serve the console from the **same origin** (see `phase11/DEPLOY.md` for the nginx
configuration). The service refuses to start in production without a secret key
and an explicit CORS origin list.
