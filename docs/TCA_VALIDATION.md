# TCA Validation

**Date:** 2026-08-31 · **Suite:** `phase11/tests/scientific/test_tca_validation.py`
**Verdict: PASS** — 9 tests, 15 real conjunction pairs compared against an
independent reference implementation.

---

## 1. What is being validated

The production time-of-closest-approach pipeline:

```
apogee/perigee filter  ->  vectorized coarse grid (30 s)  ->  bracket local
minima  ->  Brent refinement (1 ms absolute tolerance)
```

against an **independent reference** that shares no TCA logic with it:

```
dense uniform scan (5 s)  ->  bracket EVERY local minimum  ->  ternary section
(1 ms tolerance)  ->  take the best
```

Both call the same SGP4 propagator. That is deliberate: SGP4 is the shared
ground truth, and what is under test is the *search over it*. The refinement
algorithms differ (Brent vs ternary section) so that a shared bug in a single
minimizer cannot produce false agreement.

---

## 2. Acceptance thresholds and their derivation

Thresholds are derived from the numerical method. None was chosen to make a test
pass; where a threshold was initially wrong it was re-derived from the physics
and the reason is recorded in section 5.

| Quantity | Threshold | Derivation |
|---|---|---|
| TCA time | **2 ms** | Brent terminates on a 1 ms absolute abscissa tolerance; the reference refines to 1 ms too. Two independent searches each converged to +/-1 ms can differ by up to 2 ms. |
| Miss distance | **1 m** (or 1e-6 relative) | The miss distance is *stationary* at TCA: `d(t)^2 = miss^2 + v_rel^2 (t-tca)^2`, so a timing error enters at **second** order. At 15 km/s, 1 km miss and 1 ms error the perturbation is ~1e-4 m. 1 m is therefore a loose bound that still catches real disagreement. |
| Relative position vector | **\|v_rel\| x 2 ms** (1 m floor) | The position *vector* is **not** stationary — its derivative is the relative velocity — so timing error enters at **first** order. At 14 km/s that is 28 m. Holding the vector to the distance's 1 m bound would assert a precision the method does not have. |
| Relative velocity | **1e-6 relative** | Relative acceleration in LEO is ~1e-5 km/s^2, so 2 ms of timing changes velocity by ~2e-8 km/s. This bound is dominated by float64 round-off in SGP4, not by the search. |

The asymmetry between the miss *distance* (1 m) and the miss *vector* (28 m) is
physical, not a concession. It is the difference between a quantity that is
stationary at the minimum and one that is not.

---

## 3. Results

15 pairs, 3 primaries, 6-hour windows, epoch 2026-07-01T00:00:00Z, drawn from the
bundled 15 894-object catalog by the production coarse screen.

| Primary | Secondary | Name | Coarse km | Production km | Reference km | dTCA (ms) | dMiss (m) | v_rel km/s |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 25544 | 46277 | NEMO-HD | 21.875 | 21.2430 | 21.2430 | 0.037 | 0.0000 | 6.108 |
| 25544 | 48907 | MANDRAKE 2 ABLE | 28.813 | 23.6721 | 23.6721 | 0.207 | 0.0002 | 14.120 |
| 25544 | 61771 | SITRO-AIS 48 | 33.003 | 32.9062 | 32.9062 | 0.111 | 0.0000 | 14.745 |
| 25544 | 62636 | FLOCK 4G-20 | 41.480 | 26.0456 | 26.0456 | 0.427 | 0.0001 | 7.232 |
| 25544 | 62607 | STARLINK-32784 | 42.000 | 27.9855 | 27.9855 | 0.216 | 0.0001 | 9.344 |
| 39634 | 44874 | CHEOPS | 27.286 | **4.3856** | 4.3856 | 0.404 | 0.0052 | 14.860 |
| 39634 | 39423 | WNISAT-1 | 53.609 | 53.3130 | 53.3130 | 0.126 | 0.0000 | 14.039 |
| 39634 | 68585 | KUIPER-00254 | 70.110 | 69.7299 | 69.7299 | 0.079 | 0.0000 | 9.319 |
| 39634 | 37165 | YAOGAN-11 | 71.900 | 70.8148 | 70.8148 | 0.077 | 0.0000 | 14.012 |
| 39634 | 61018 | GEESAT-3 08 | 79.439 | 64.2043 | 64.2043 | 0.450 | 0.0003 | 14.422 |
| 43437 | 43689 | METOP-C | 31.212 | 30.9433 | 30.9433 | 0.167 | 0.0000 | 1.026 |
| 43437 | 42963 | IRIDIUM 139 | 40.429 | 35.5192 | 35.5192 | 0.436 | 0.0004 | 13.490 |
| 43437 | 43576 | IRIDIUM 156 | 106.857 | 80.0719 | 80.0719 | 0.544 | 0.0002 | 14.709 |
| 43437 | 63163 | QIANFAN-77 | 110.434 | 108.4849 | 108.4849 | 0.310 | 0.0000 | 5.928 |
| 43437 | 44078 | EMISAT | 111.588 | 111.5878 | 111.5878 | 0.044 | 0.0000 | 1.110 |

**Worst case across all 15 pairs**

| Metric | Observed | Threshold | Margin |
|---|---:|---:|---:|
| TCA time disagreement | 0.544 ms | 2 ms | 3.7x |
| Miss distance disagreement | 0.0052 m | 1 m | 190x |

---

## 4. Why the refinement stage is load-bearing

The "Coarse km" column is the 30-second grid's sampled minimum; "Production km"
is after Brent refinement. The gap is the error the original implementation
shipped, because it reported the coarse value as the miss distance:

| Pair | Coarse | Refined | Error the grid alone would have reported |
|---|---:|---:|---:|
| 39634 / 44874 (CHEOPS) | 27.286 km | 4.386 km | **22.9 km** |
| 43437 / 43576 (IRIDIUM 156) | 106.857 km | 80.072 km | 26.8 km |
| 39634 / 61018 (GEESAT-3 08) | 79.439 km | 64.204 km | 15.2 km |

The CHEOPS case is the operationally significant one: a 4.4 km conjunction
reported as 27.3 km. `test_refinement_materially_improves_on_the_grid` asserts
that this improvement is still happening, so a silent regression to grid
sampling fails the suite.

---

## 5. Two defects the validation itself surfaced

Both were found by the reference disagreeing with production. In each case the
investigation determined which side was wrong before anything was changed —
that ordering is the point of an independent reference.

### 5.1 The reference implementation had the aliasing bug (reference was wrong)

The first version of `reference_tca` refined only the **global minimum of the
coarse scan**. That is precisely the aliasing defect this project fixed in
production: at 14 km/s a 10 s step covers 140 km, so a sample landing near the
bottom of a shallow minimum can read lower than any sample near the bottom of the
true deepest one.

On ISS (25544) vs MANDRAKE 2 ABLE (48907) the reference reported **28.22 km at
t = 17 940 s**, while production reported **23.67 km at t = 20 723 s**. An
independent 1-second brute-force scan settled it: **23.694 km at t = 20 723 s**.
Production was right.

Fixed by refining *every* bracketed local minimum in the reference, not just the
sampled argmin.

### 5.2 The relative-position threshold was mis-derived (threshold was wrong)

The relative position vector was initially held to the same 1 m bound as the
miss distance. It failed at ~2.8 m disagreement. That is not a code defect: at
14 km/s closing speed and 0.2 ms of timing difference, 2.8 m is exactly what
first-order sensitivity predicts. The threshold now derives from `|v_rel| x
TCA tolerance` per pair, and the asymmetry is documented in the suite.

---

## 6. Degenerate pairs are excluded, and handled separately

Pairs with a separation below 0.5 km at every sample are excluded from the TCA
comparison, because **there is no time of closest approach for them**. The
separation function is flat, every instant is a minimum, and two correct
minimizers legitimately return times thousands of seconds apart.

This is not hypothetical: the public catalog lists the ISS modules (Unity,
Zvezda, Destiny, Poisk) as separate objects sharing one element set with ISS, so
their separation is identically 0.000 km with zero relative velocity.

Excluding them from *this* suite is correct; ignoring them in production was not.
They are now classified `CO_LOCATED` and made non-actionable — see
`tests/scientific/test_co_location.py` and section 3 of
`docs/PRODUCTION_AUDIT.md` (issue P1 false alerts).

---

## 7. Analytic edge cases

Beyond the catalog comparison, four cases with closed-form answers:

| Case | Assertion |
|---|---|
| Straight-line relative motion | TCA equals `-(dr.dv)/(dv.dv)` to 1e-6 s |
| Closest approach already past | TCA resolves **negative**, not clamped to 0 (the original defect) |
| Parallel motion, zero relative velocity | No division by zero; miss equals the constant separation |
| Multiple passes in 24 h | At least 2 distinct minima found, separated by > 60 s |

---

## 8. Reproducing

```bash
cd phase11
python -m pytest tests/scientific/test_tca_validation.py -v
```

Runtime ~30 s. Requires the bundled `phase11/data/catalog_active.tle`; the suite
skips rather than fails if it is absent.
