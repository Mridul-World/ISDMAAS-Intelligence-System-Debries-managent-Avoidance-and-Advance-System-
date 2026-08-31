# Collision Probability Validation

**Date:** 2026-08-31 · **Suite:** `phase11/tests/scientific/test_collision_probability.py`
**Verdict: PASS** — 72 tests, 3 skipped (Monte Carlo below sampling resolution).

---

## 1. What is validated

The reported Pc is Foster's 2-D encounter-plane integral, evaluated by composite
Gauss-Legendre quadrature. It is checked against **two independent methods**:

| Method | Independence |
|---|---|
| Dense Cartesian midpoint integration, 3000x3000, log-stabilized | Different coordinate system (Cartesian vs polar), different rule (midpoint vs Gauss-Legendre), different stabilization. Shares no code. |
| Monte Carlo, 4x10^6 samples | Fully independent of both — no quadrature at all. |

Chan's series runs alongside as the service's own cross-check and is reported,
not asserted equal (see section 5).

---

## 2. Acceptance thresholds and their derivation

| Comparison | Threshold | Derivation |
|---|---|---|
| Quadrature vs dense integration | **0.2% relative** | The 3000x3000 reference grid has ~1e-4 relative discretization error of its own, so this bound is dominated by the REFERENCE's accuracy, not the production method's. It still detects the ~0.5% bias the previous rectangle-rule implementation carried. |
| Quadrature vs Monte Carlo | **4 standard errors**, computed per case | Relative standard error is `sqrt((1-p)/(Np))`. A fixed tolerance would be meaningless at high p and perpetually flaky at low p, so the bound is derived from the sample count and hit count of each case. |
| Chan cross-check | **reported, not asserted** | Chan circularizes the encounter ellipse and is a genuine approximation. See section 5. |

---

## 3. Parameter sweep results

19 cases spanning covariance magnitude, anisotropy, miss distance, hard-body
radius and HBR/sigma ratio.

| Case | Quadrature | Dense reference | Rel. error | Chan |
|---|---:|---:|---:|---:|
| tight isotropic covariance | 1.12484e-02 | 1.12482e-02 | 2.2e-05 | 1.12484e-02 |
| moderate isotropic covariance | 1.08312e-04 | 1.08309e-04 | 2.2e-05 | 1.08312e-04 |
| loose isotropic covariance | 1.08269e-06 | 1.08266e-06 | 2.2e-05 | 1.08269e-06 |
| very loose isotropic covariance | 1.08268e-08 | 1.08266e-08 | 2.2e-05 | 1.08268e-08 |
| 4:1 anisotropy | 1.38661e-04 | 1.38658e-04 | 2.2e-05 | 1.38544e-04 |
| 4.5:1 anisotropy, zero miss | 1.10966e-03 | 1.10963e-03 | 2.2e-05 | 1.11049e-03 |
| 30:1 anisotropy | 2.03104e-25 | 2.03097e-25 | 3.1e-05 | 1.30695e-25 |
| **25:1 anisotropy, HBR above sigma_min** | 6.56031e-38 | 6.55477e-38 | 8.5e-04 | **1.04590e-48** |
| 500:1 anisotropy | 0 | 0 | — | 0 |
| zero miss distance | 7.99680e-04 | 7.99663e-04 | 2.2e-05 | 7.99680e-04 |
| miss well inside 1 sigma | 7.95693e-04 | 7.95676e-04 | 2.2e-05 | 7.95693e-04 |
| 5 sigma miss | 2.99505e-09 | 2.99499e-09 | 2.2e-05 | 2.99505e-09 |
| 10 sigma miss | 1.57343e-25 | 1.57340e-25 | 2.2e-05 | 1.57343e-25 |
| far tail (12 sigma, anisotropic) | 3.61248e-36 | 3.61240e-36 | 2.2e-05 | 3.59529e-36 |
| 1 m HBR (small debris) | 2.70671e-07 | 2.70665e-07 | 2.2e-05 | 2.70671e-07 |
| 10 m HBR | 2.70698e-05 | 2.70692e-05 | 2.2e-05 | 2.70698e-05 |
| 100 m HBR (large structure) | 2.73359e-03 | 2.73353e-03 | 2.2e-05 | 2.73359e-03 |
| HBR/sigma = 1e-4 | 5.00000e-09 | 4.99989e-09 | 2.2e-05 | 5.00000e-09 |
| HBR/sigma = 20, disk swallows distribution | 1.00000e+00 | 1.00000e+00 | 7.8e-16 | 1.00000e+00 |

**Worst relative error: 8.5e-04**, against a 2.0e-03 threshold — a 2.4x margin,
and that worst case is the extreme-anisotropy one where the dense reference is
itself least accurate.

Additional closed-form checks with no integration at all: the small-HBR limit
`Pc -> pi HBR^2 x N(miss)` matches to 1e-6 relative; monotonicity in miss
distance and in HBR; rotational invariance to 1e-9; bounds in [0, 1] everywhere.

---

## 4. A numerical defect this validation found and fixed

**Single-panel Gauss-Legendre failed when the covariance was much tighter than
the hard body.**

With sigma = 1 mm and HBR = 20 m the correct answer is **Pc = 1.0** — the disk
contains the entire distribution, so collision is certain. The implementation
returned **6.2e-31**.

The cause is structural, not a tuning issue: Gauss-Legendre nodes cluster toward
the *ends* of their panel. Across a single `[0, HBR]` span the smallest radial
node sat about 14 sigma from the origin, so the quadrature never sampled the
region carrying all the probability mass.

Fixed by integrating over **geometric panels** laid out from the covariance
scale outwards (`_radial_panels`), guaranteeing several nodes across the mass
whatever the HBR/sigma ratio. Node count per panel dropped from 48 to 16, so the
composite rule is no more expensive than the original.

This regime is reachable in operations: a large structure (ISS-class, HBR ~50 m)
against a well-determined object with metre-class covariance.

---

## 5. Chan's method: where it agrees and where it does not

Chan agrees with the quadrature to ~1e-5 relative across most of the sweep. It
fails in one identifiable regime — **high anisotropy with HBR above the smaller
sigma**:

| Case | Quadrature (correct) | Chan | Discrepancy |
|---|---:|---:|---|
| sigma 0.02 x 0.5 km, HBR 50 m | 6.56e-38 | 1.05e-48 | **10 orders of magnitude** |
| sigma 0.1 x 3.0 km, HBR 20 m | 2.03e-25 | 1.31e-25 | 36% |

This is documented behaviour of the method, which circularizes the encounter
ellipse to an equivalent area. It is **why quadrature is what the service
reports**, and why every response carries `pc_methods_agree`: the disagreement
is surfaced to the operator rather than hidden or averaged away.

---

## 6. Invalid covariance now fails safely

This was a **P0**. Measured behaviour before the guard, for a 1 km miss whose
true Pc is 1.08e-04:

| Input | Pc returned (quadrature) | Pc returned (Chan) |
|---|---:|---:|
| one negative eigenvalue | **8.5e-01** | **1.0** |
| both eigenvalues negative | 0.0 | 0.0 |
| all-zero matrix | 0.0 | 0.0 |
| singular (rank 1) | 0.0 | 0.0 |
| NaN entry | **NaN** (not valid JSON) | raises |
| asymmetric | silently accepted | silently accepted |

Both directions are unsafe, in opposite ways. A spurious Pc of 0.85 is a false
alarm that would trigger an unnecessary maneuver and burn propellant. A spurious
Pc of 0.0 is a **missed conjunction reported to the operator as safe**.

`validate_covariance` now rejects any matrix that is not finite, symmetric and
positive semi-definite, with a tolerance scaled to the matrix norm so that
genuine floating-point round-off (a -1e-18 eigenvalue) is still accepted.

`assess_conjunction` validates **each input covariance separately, before
combining them**. Validating only the sum would let a corrupt input through
whenever its partner was large enough to mask it — verified by test.

On invalid input the response carries:

```json
{
  "pc": null,
  "risk_level": "DATA_INVALID",
  "covariance_valid": false,
  "miss_distance_km": 1.0,
  "note": "Collision probability was not computed: ... is not positive semi-definite ..."
}
```

The geometry is still reported. The probability is `null`, never a number.

---

## 7. Numerical robustness

| Property | Result |
|---|---|
| No overflow/underflow at 1, 5, 10, 20, 30, 45, 60, 100, 1000 sigma | finite, in [0,1] for both methods |
| Chan cancellation floor (previously bottomed out at 2.2e-16) | now tracks quadrature to 5% at 1e-25 |
| Ill-conditioned but valid covariance (1e6:1 axis ratio) | computed, finite |
| Round-off negative eigenvalue (-1e-18) | accepted, not rejected |
| Zero hard-body radius | returns exactly 0 |

---

## 8. Reproducing

```bash
cd phase11
python -m pytest tests/scientific/test_collision_probability.py -v
```

Runtime ~18 s. Three Monte Carlo cases skip automatically when the probability
is too small for 4x10^6 samples to resolve — that is reported, not silent.
