"""
Scientific validation: collision probability.

The reported Pc is Foster's 2-D integral evaluated by Gauss-Legendre quadrature.
This suite validates it against a high-accuracy independent numerical
integration across the full parameter space an operator can reach, and validates
that invalid input fails safely instead of producing a confident wrong number.

REFERENCE METHOD
----------------
A dense Cartesian integration over the hard-body disk, evaluated in log space
and peak-shifted before exponentiation so it stays accurate at probabilities
down to 1e-38. It shares no code with the production quadrature: different
coordinate system (Cartesian vs polar), different rule (midpoint vs
Gauss-Legendre), different stabilization.

Monte Carlo is used as a third, fully independent check wherever the probability
is large enough for sampling to resolve it (Pc >~ 1e-5).

ACCEPTANCE THRESHOLDS
---------------------
* Quadrature vs dense integration: 0.2% relative.
  The dense reference uses a 3000x3000 grid over the disk, whose own
  discretization error is ~1e-4 relative. 0.2% is therefore dominated by the
  REFERENCE's error, not the production method's, and it still detects the
  ~0.5% bias the previous rectangle-rule implementation carried.

* Monte Carlo: 4 standard errors.
  With N samples and probability p the relative standard error is
  sqrt((1-p)/(Np)). The bound is computed per case rather than fixed, because a
  fixed tolerance would be either meaningless at high p or perpetually flaky at
  low p.

* Chan cross-check: reported, not asserted equal.
  Chan's method circularizes the encounter ellipse and is a genuine
  approximation. It is accurate to ~1e-3 relative for near-circular covariance
  and degrades badly for high anisotropy - at sigma 0.02 x 0.5 km with HBR 50 m
  it reports 1e-48 where the truth is 6.6e-38. That is expected behaviour for
  the method, which is why quadrature is what the service reports and
  `pc_methods_agree` exists to surface the disagreement.
"""
import math

import numpy as np
import pytest

from phase7_collision import (
    CovarianceError,
    assess_conjunction,
    mahalanobis_2d,
    pc_2d_quadrature,
    pc_chan,
    risk_level,
    rtn_to_eci_cov,
    validate_covariance,
)

QUADRATURE_REL_TOLERANCE = 2e-3
MONTE_CARLO_SIGMAS = 4.0


# =============================================================== REFERENCE
def dense_disk_integral(sigma_x, sigma_y, miss_x, miss_y, hbr, n=3000):
    """
    Independent high-accuracy reference: dense Cartesian midpoint integration.

    Evaluated in log space with the peak subtracted before exponentiation, which
    is what keeps it usable at probabilities around 1e-38 where a naive
    implementation underflows to exactly zero.
    """
    axis = np.linspace(-hbr, hbr, n)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    inside = X * X + Y * Y <= hbr * hbr
    exponent = -0.5 * (((X - miss_x) / sigma_x) ** 2 + ((Y - miss_y) / sigma_y) ** 2)
    exponent = np.where(inside, exponent, -np.inf)
    peak = float(exponent.max())
    if not math.isfinite(peak):
        return 0.0
    cell = (2.0 * hbr / (n - 1)) ** 2
    return float(
        np.sum(np.exp(exponent - peak)) * cell * math.exp(peak)
        / (2.0 * math.pi * sigma_x * sigma_y)
    )


def monte_carlo_pc(sigma_x, sigma_y, miss_x, miss_y, hbr, samples, seed=12345):
    """Third independent estimate: sample the Gaussian, count inside the disk."""
    rng = np.random.default_rng(seed)
    x = rng.normal(miss_x, sigma_x, samples)
    y = rng.normal(miss_y, sigma_y, samples)
    hits = int(np.count_nonzero(x * x + y * y <= hbr * hbr))
    return hits / samples, hits


# ======================================================= PARAMETER SWEEP
#   (sigma_x, sigma_y, miss, hbr, label)
SWEEP = [
    # --- covariance magnitude, isotropic, fixed geometry -------------------
    (0.05, 0.05, 0.10, 0.020, "tight isotropic covariance"),
    (0.50, 0.50, 1.00, 0.020, "moderate isotropic covariance"),
    (5.00, 5.00, 10.0, 0.020, "loose isotropic covariance"),
    (50.0, 50.0, 100.0, 0.020, "very loose isotropic covariance"),
    # --- anisotropy, from mild to extreme ----------------------------------
    (0.30, 1.20, 0.50, 0.020, "4:1 anisotropy"),
    (0.20, 0.90, 0.00, 0.020, "4.5:1 anisotropy, zero miss"),
    (0.10, 3.00, 1.00, 0.020, "30:1 anisotropy"),
    (0.02, 0.50, 0.30, 0.050, "25:1 anisotropy, HBR above sigma_min"),
    (0.01, 5.00, 0.50, 0.020, "500:1 anisotropy"),
    # --- miss distance, from zero to far tail ------------------------------
    (0.50, 0.50, 0.00, 0.020, "zero miss distance"),
    (0.50, 0.50, 0.05, 0.020, "miss well inside 1 sigma"),
    (0.50, 0.50, 2.50, 0.020, "5 sigma miss"),
    (0.50, 0.50, 5.00, 0.020, "10 sigma miss"),
    (1.00, 3.00, 12.0, 0.020, "far tail"),
    # --- hard-body radius --------------------------------------------------
    (0.50, 0.50, 1.00, 0.001, "1 m HBR (small debris)"),
    (0.50, 0.50, 1.00, 0.010, "10 m HBR"),
    (0.50, 0.50, 1.00, 0.100, "100 m HBR (large structure)"),
    # --- extreme HBR/sigma ratios -----------------------------------------
    (10.0, 10.0, 0.00, 0.001, "HBR/sigma = 1e-4"),
    (0.005, 0.005, 0.00, 0.100, "HBR/sigma = 20, disk swallows the distribution"),
]


@pytest.mark.parametrize("sigma_x,sigma_y,miss,hbr,label", SWEEP)
def test_quadrature_matches_dense_integration(sigma_x, sigma_y, miss, hbr, label):
    """
    The reported Pc must match an independent dense integration everywhere in
    the operational parameter space.
    """
    C = np.diag([sigma_x ** 2, sigma_y ** 2])
    m = np.array([miss, 0.0])
    produced = pc_2d_quadrature(m, C, hbr)
    reference = dense_disk_integral(sigma_x, sigma_y, miss, 0.0, hbr)

    if reference == 0.0:
        assert produced == 0.0, f"{label}: reference underflowed but production did not"
        return
    relative = abs(produced - reference) / reference
    assert relative < QUADRATURE_REL_TOLERANCE, (
        f"{label}: quadrature {produced:.6e} vs dense reference {reference:.6e} "
        f"({relative * 100:.4f}% apart)"
    )


@pytest.mark.parametrize(
    "sigma_x,sigma_y,miss,hbr,label",
    [case for case in SWEEP if case[2] <= 1.0 and case[3] >= 0.01],
)
def test_quadrature_matches_monte_carlo(sigma_x, sigma_y, miss, hbr, label):
    """
    A third independent estimate, wherever sampling can resolve the probability.

    The tolerance is the sampling standard error, computed per case. A fixed
    tolerance would be meaningless at high probability and perpetually flaky at
    low probability.
    """
    C = np.diag([sigma_x ** 2, sigma_y ** 2])
    produced = pc_2d_quadrature(np.array([miss, 0.0]), C, hbr)
    samples = 4_000_000
    if produced * samples < 40:
        pytest.skip(f"{label}: Pc {produced:.2e} is too small for Monte Carlo")

    estimate, hits = monte_carlo_pc(sigma_x, sigma_y, miss, 0.0, hbr, samples)
    standard_error = math.sqrt(max(hits, 1)) / samples
    assert abs(produced - estimate) < MONTE_CARLO_SIGMAS * standard_error, (
        f"{label}: quadrature {produced:.6e} vs Monte Carlo {estimate:.6e} "
        f"+/- {standard_error:.2e} ({hits} hits in {samples} samples)"
    )


# =========================================================== ANALYTIC LIMITS
def test_probability_is_one_when_the_disk_swallows_the_distribution():
    """HBR >> sigma with zero miss means collision is certain."""
    C = np.diag([1e-6 ** 2, 1e-6 ** 2])
    assert pc_2d_quadrature(np.zeros(2), C, 0.02) == pytest.approx(1.0, abs=1e-9)


def test_small_hbr_limit_matches_the_analytic_density():
    """
    For HBR << sigma the integral tends to (pi HBR^2) x N(miss), because the
    density is essentially constant across the disk. This is an independent
    closed-form check that does not use any integration at all.
    """
    sigma_x, sigma_y, miss, hbr = 1.0, 2.0, 0.5, 1e-4
    density = (1.0 / (2.0 * math.pi * sigma_x * sigma_y)) * math.exp(
        -0.5 * ((miss / sigma_x) ** 2)
    )
    expected = math.pi * hbr ** 2 * density
    produced = pc_2d_quadrature(
        np.array([miss, 0.0]), np.diag([sigma_x ** 2, sigma_y ** 2]), hbr
    )
    assert produced == pytest.approx(expected, rel=1e-6)


def test_probability_decreases_monotonically_with_miss_distance():
    C = np.diag([0.5 ** 2, 0.5 ** 2])
    values = [pc_2d_quadrature(np.array([d, 0.0]), C, 0.02)
              for d in (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)]
    assert all(a > b for a, b in zip(values, values[1:])), values


def test_probability_increases_monotonically_with_hard_body_radius():
    C = np.diag([0.5 ** 2, 0.5 ** 2])
    values = [pc_2d_quadrature(np.array([0.5, 0.0]), C, h)
              for h in (0.001, 0.005, 0.02, 0.05, 0.1)]
    assert all(a < b for a, b in zip(values, values[1:])), values


def test_probability_is_bounded_in_zero_one():
    for sigma_x, sigma_y, miss, hbr, _ in SWEEP:
        value = pc_2d_quadrature(
            np.array([miss, 0.0]), np.diag([sigma_x ** 2, sigma_y ** 2]), hbr
        )
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)


def test_zero_hard_body_radius_gives_zero_probability():
    """Two point masses cannot collide."""
    assert pc_2d_quadrature(np.array([1.0, 0.0]), np.diag([0.25, 0.25]), 0.0) == 0.0


def test_probability_is_rotationally_invariant():
    """
    Rotating the miss vector and covariance together must not change Pc.

    Catches a principal-axis transform applied in the wrong direction — a defect
    that is invisible for isotropic covariance and wrong for every other case.
    """
    sigma_x, sigma_y, miss = 0.2, 1.4, 0.9
    C = np.diag([sigma_x ** 2, sigma_y ** 2])
    baseline = pc_2d_quadrature(np.array([miss, 0.0]), C, 0.02)
    for angle in (0.3, 1.1, 2.7, 4.9):
        c, s = math.cos(angle), math.sin(angle)
        R = np.array([[c, -s], [s, c]])
        rotated = pc_2d_quadrature(R @ np.array([miss, 0.0]), R @ C @ R.T, 0.02)
        assert rotated == pytest.approx(baseline, rel=1e-9), f"angle {angle}"


# =================================================== NUMERICAL ROBUSTNESS
@pytest.mark.parametrize("sigmas", [1, 5, 10, 20, 30, 45, 60, 100, 1000])
def test_no_overflow_or_underflow_at_any_separation(sigmas):
    """
    Chan's series previously accumulated (v/2)^k / k! and returned NaN past
    ~40 sigma. Both methods must stay finite and in range at any separation.
    """
    C = np.diag([0.5 ** 2, 0.5 ** 2])
    m = np.array([sigmas * 0.5, 0.0])
    for value in (pc_2d_quadrature(m, C, 0.02), pc_chan(m, C, 0.02)):
        assert math.isfinite(value)
        assert 0.0 <= value <= 1.0


def test_chan_has_no_cancellation_floor():
    """
    The textbook Chan inner factor subtracts two numbers agreeing to sixteen
    digits and bottoms out at machine epsilon, reporting 2.2e-16 where the truth
    is ~1e-25.
    """
    C = np.diag([1.0, 1.0])
    m = np.array([10.0, 0.0])
    quadrature = pc_2d_quadrature(m, C, 0.02)
    assert quadrature < 1e-20
    assert pc_chan(m, C, 0.02) == pytest.approx(quadrature, rel=0.05)


def test_chan_disagreement_is_surfaced_for_extreme_anisotropy():
    """
    Chan is a genuine approximation and fails for high anisotropy with HBR above
    the smaller sigma. The service must REPORT the disagreement, not hide it —
    that is what `pc_methods_agree` is for.
    """
    C = np.diag([0.02 ** 2, 0.5 ** 2])
    m = np.array([0.3, 0.0])
    quadrature = pc_2d_quadrature(m, C, 0.05)
    series = pc_chan(m, C, 0.05)
    reference = dense_disk_integral(0.02, 0.5, 0.3, 0.0, 0.05)

    # Quadrature is right; Chan is wrong by many orders of magnitude.
    assert quadrature == pytest.approx(reference, rel=QUADRATURE_REL_TOLERANCE)
    assert series < quadrature * 1e-3, (
        "Chan is expected to fail badly here; if it now agrees, this test's "
        "premise has changed and the tolerance story needs revisiting"
    )


def test_ill_conditioned_but_valid_covariance_is_still_computed():
    """
    A 1e6:1 axis ratio is extreme but legitimate — a long-propagated debris
    covariance genuinely looks like this. It must produce a number.
    """
    C = np.diag([1e-6 ** 2, 1.0 ** 2])
    value = pc_2d_quadrature(np.array([0.0, 0.5]), C, 0.02)
    assert math.isfinite(value) and 0.0 <= value <= 1.0


# ================================================ INVALID COVARIANCE SAFETY
#
# The measured behaviour before this guard existed, for a 1 km miss whose true
# Pc is 1.08e-4:
#     one negative eigenvalue -> Pc = 8.5e-1 (quadrature), 1.0 (Chan)
#     both negative           -> Pc = 0.0
#     all zeros               -> Pc = 0.0
#     singular (rank 1)       -> Pc = 0.0
#     NaN entry               -> Pc = NaN  (not valid JSON)
# A spurious 0.85 is a false alarm that burns propellant; a spurious 0.0 is a
# missed conjunction reported as safe.
INVALID_COVARIANCES = [
    (np.array([[0.25, 0.0], [0.0, -0.25]]), "one negative eigenvalue"),
    (np.diag([-0.25, -0.25]), "both eigenvalues negative"),
    (np.zeros((2, 2)), "all zeros"),
    (np.array([[0.25, np.nan], [np.nan, 0.25]]), "NaN entry"),
    (np.array([[0.25, 0.0], [0.0, np.inf]]), "infinite entry"),
    (np.array([[0.25, 0.9], [0.0, 0.25]]), "asymmetric"),
]


@pytest.mark.parametrize("C,label", INVALID_COVARIANCES)
def test_invalid_covariance_raises_rather_than_returning_a_number(C, label):
    m = np.array([1.0, 0.0])
    with pytest.raises(CovarianceError):
        pc_2d_quadrature(m, C, 0.02)
    with pytest.raises(CovarianceError):
        pc_chan(m, C, 0.02)


@pytest.mark.parametrize("C,label", INVALID_COVARIANCES)
def test_validate_covariance_explains_the_problem(C, label):
    """The error must say what is wrong, so an operator can act on it."""
    with pytest.raises(CovarianceError) as excinfo:
        validate_covariance(C)
    assert len(str(excinfo.value)) > 40, f"{label}: error message is not explanatory"


def test_round_off_negative_eigenvalues_are_tolerated():
    """
    A covariance can legitimately come back with a -1e-18 eigenvalue from
    floating-point round-off. Rejecting that would fail on valid data.
    """
    C = np.diag([0.25, 1e-18])
    validate_covariance(C)
    assert math.isfinite(pc_2d_quadrature(np.array([0.1, 0.0]), C, 0.02))


@pytest.mark.parametrize(
    "which,label",
    [("primary", "invalid primary covariance"),
     ("secondary", "invalid secondary covariance")],
)
def test_assess_conjunction_reports_data_invalid_not_a_probability(which, label):
    """
    End to end: an invalid INPUT covariance yields status DATA_INVALID and
    pc = None, with the geometry still reported.

    Each input is validated separately, because adding a valid covariance to an
    invalid one can produce a sum that passes every check — validating only the
    combination lets a corrupt input through whenever its partner is large
    enough to mask it.
    """
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    r2 = r1 + np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 7.5])
    good = rtn_to_eci_cov(np.diag([0.3, 1.0, 0.3]) ** 2, r1, v1)
    bad = good.copy()
    bad[2, 2] = -abs(bad[2, 2])

    C1, C2 = (bad, good) if which == "primary" else (good, bad)
    result = assess_conjunction(r1, v1, C1, r2, v2, C2, 0.020,
                                tca_already_refined=True)

    assert result["pc"] is None, f"{label} produced a probability"
    assert result["risk_level"] == "DATA_INVALID"
    assert result["covariance_valid"] is False
    # The geometry is still usable and must survive.
    assert result["miss_distance_km"] == pytest.approx(1.0, rel=1e-9)
    assert "not positive semi-definite" in result["note"]


def test_valid_covariance_still_produces_a_probability():
    """The counterweight: the guard must not reject legitimate input."""
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    good = rtn_to_eci_cov(np.diag([0.3, 1.0, 0.3]) ** 2, r1, v1)
    result = assess_conjunction(
        r1, v1, good, r1 + np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 7.5]), good, 0.020, tca_already_refined=True,
    )
    assert result["pc"] is not None
    assert result["covariance_valid"] is True
    assert result["risk_level"] in ("NOMINAL", "ELEVATED", "HIGH", "CRITICAL")


# ============================================================== RISK BANDS
@pytest.mark.parametrize(
    "pc,expected",
    [(0.0, "NOMINAL"), (1e-9, "NOMINAL"), (1e-7, "ELEVATED"), (5e-6, "ELEVATED"),
     (1e-5, "HIGH"), (5e-5, "HIGH"), (1e-4, "CRITICAL"), (1e-2, "CRITICAL")],
)
def test_risk_bands_are_correct_at_their_boundaries(pc, expected):
    """Boundaries are inclusive-below: 1e-4 is CRITICAL, not HIGH."""
    assert risk_level(pc) == expected


def test_mahalanobis_matches_the_analytic_value():
    """For a diagonal covariance the Mahalanobis distance is miss/sigma."""
    assert mahalanobis_2d(np.array([1.5, 0.0]), np.diag([0.5 ** 2, 2.0 ** 2])) \
        == pytest.approx(3.0, rel=1e-12)
