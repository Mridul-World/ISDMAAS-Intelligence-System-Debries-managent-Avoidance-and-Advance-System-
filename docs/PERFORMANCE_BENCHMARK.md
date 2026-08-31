# Performance Baseline

**Date:** 2026-08-31 · **Harness:** `phase11/tests/benchmark_screening.py`
**Purpose:** establish a baseline. Nothing here has been optimized in response to
these numbers; that would be premature.

## Environment

| | |
|---|---|
| Machine | Windows 11, Python 3.10.6, single process |
| Catalog | 15 894 real objects (bundled `catalog_active.tle`) |
| Primary | Sentinel-1A (39634) |
| Window | 24 h · Gate 25 km · Coarse step 30 s |
| Epoch | 2026-07-01T00:00:00Z |

Sizes above 15 894 are **synthesised** by perturbing real element sets in mean
anomaly and RAAN, so every object is a distinct orbit that SGP4 must actually
integrate. They are labelled as such below. No claim is made that 100 000 real
objects were screened.

---

## 1. Screening throughput

| Objects | Source | Seconds | Objects/s | Peak MiB | After geometric filter | Coarse candidates | Conjunctions |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 000 | real | 0.101 | 9 871 | 30.0 | 96 | 10 | 1 |
| 5 000 | real | 0.160 | 31 172 | 39.9 | 128 | 13 | 2 |
| 15 000 | real | 0.239 | 62 749 | 52.3 | 168 | 18 | 2 |
| 50 000 | synthesised | 0.866 | 57 731 | 193.0 | 621 | 71 | 8 |
| 100 000 | synthesised | 1.664 | 60 082 | 233.1 | 1 154 | 126 | 23 |

**Scaling is linear** from 15 000 objects upward at ~60 000 objects/second. Below
that the fixed cost of propagating the primary across the window dominates, which
is why the 1 000-object case shows a lower rate — it is not doing less work per
object, it is amortising the fixed cost over fewer of them.

**Memory** grows linearly with the number of objects surviving the geometric
filter, at roughly 2 MiB per 1 000 surviving objects. The 233 MiB peak at 100 000
objects is the chunked coarse-screen buffer (750 objects x 2 881 epochs x 3
coordinates, float64).

---

## 2. Component breakdown

Real catalog, one primary, 24 h window:

| Stage | Time | Work |
|---|---:|---|
| Geometric filter | 26.3 ms | 15 894 → 171 objects |
| Primary propagation | 1.1 ms | 2 881 epochs |
| Coarse screen | 153.7 ms | 171 objects × 2 881 epochs |
| TCA refinement | 2.7 ms | 19 candidates → 2 refined |
| **Total** | **~184 ms** | |

Two things are worth reading off this:

- **The apogee/perigee filter is doing the heavy lifting.** It removes 98.9% of
  the catalog for 26 ms, which is what makes the coarse screen affordable. Its
  cost is a single closed-form calculation per object, no propagation.
- **The Brent refinement is essentially free** — 2.7 ms, 1.5% of the total —
  while being the stage that makes the reported miss distance correct at all. On
  this primary the coarse grid samples 27.286 km where the true miss is
  4.386 km. There is no accuracy/performance trade to make here: the accurate
  answer costs 1.5%.

---

## 3. API latency

Measured through the ASGI stack with `TestClient` (no network), catalog warm.

| Endpoint | n | p50 | p95 | max |
|---|---:|---:|---:|---:|
| `GET /health` | 50 | 2.7 ms | 3.6 ms | 3.7 ms |
| `GET /ready` | 30 | 3.3 ms | 4.1 ms | 4.3 ms |
| `GET /satellites` | 30 | 3.0 ms | 3.3 ms | 3.4 ms |
| `GET /resolve?q=ISS` | 20 | 4.0 ms | 4.3 ms | 5.0 ms |
| `GET /orbit/25544` (240 pts) | 20 | 4.7 ms | 5.2 ms | 5.3 ms |
| `POST /assess` | 6 | 26.9 ms | 27.1 ms | 27.8 ms |
| `GET /screen/39634` (24 h, 25 km) | 6 | 194.3 ms | 196.4 ms | 208.9 ms |
| `GET /user/screen/90001` | 5 | 513.3 ms | 520.9 ms | 521.1 ms |
| `POST /maneuver-plan` (what-if) | 5 | 1 134.6 ms | 1 152.7 ms | 1 192.2 ms |

### Reading these

- **Probes and lookups are sub-5 ms.** `/health` and `/ready` are safe to poll at
  any sensible interval.
- **`/maneuver-plan` at ~1.1 s is the slowest path, and legitimately so.** It runs
  a bisection over delta-v at each of several lead times, then a **full-catalog
  post-burn re-screen**. That re-screen is the check that stops the system
  recommending a burn into a different object; a second of CPU is the correct
  price.
- **`/user/screen` is slower than `/screen`** (513 ms vs 194 ms) because the demo
  asset's orbit passes through a far denser altitude shell, so many more objects
  survive the geometric filter. The difference is the catalog, not the code path.

---

## 4. Operational implications

| Question | Answer from these numbers |
|---|---|
| Can a fleet be monitored continuously? | Yes. A 20-satellite watchlist is ~4 s of CPU; the `/monitor` cap of 20 per scan is sized for this. |
| What limits concurrency? | CPU. Screening is NumPy and SGP4, both of which release the GIL, so uvicorn workers scale close to linearly. Size workers to cores. |
| Is the rate limit sensible? | 30 requests/60 s/client against a ~200 ms screen is ~10% of one core per client. Reasonable for a pilot; revisit for many concurrent operators. |
| Would a 100 000-object catalog work? | Yes — 1.7 s and 233 MiB per primary. The public catalog is ~16 000 objects today; a future 100 000-object catalog needs no architectural change. |
| Where would optimization go first? | The coarse screen (84% of screening time). A larger coarse step with a correspondingly wider gate would trade memory for time. **Not done** — there is no performance problem to solve yet. |

---

## 5. What is not measured

Stated so the gaps are not mistaken for results:

- **No multi-user concurrency test.** Every number is single-request.
- **No sustained-load or soak test.** Nothing establishes behaviour over hours,
  or whether memory is stable across thousands of screens.
- **No network latency.** `TestClient` bypasses the socket; real deployments add
  TLS handshake and transit.
- **No cold-start measurement.** The catalog fetch on first use (a ~4 MB download
  from CelesTrak) is excluded.
- **ML inference is not benchmarked** because no model is in the serving path —
  see `docs/ML_VALIDATION_REPORT.md`.

---

## 6. Reproducing

```bash
cd phase11
python tests/benchmark_screening.py --sizes 1000 5000 15000 50000 100000
```

Runtime ~30 s. Sizes are configurable; `--json-out results.json` writes the raw
numbers.
