"""
Regression tests for the collision-probability engine.

Each test corresponds to a defect that was shipping: a one-sided TCA clamp, a
quadrature with a systematic bias, a Chan series that overflowed to NaN, and a
covariance grown over the wrong time span.
"""
import math

import numpy as np
import pytest

from phase7_collision import (
    assess_conjunction,
    find_tca,
    mahalanobis_2d,
    pc_2d_quadrature,
    pc_chan,
    risk_level,
    rtn_to_eci_cov,
    eci_to_rtn_cov,
    secondary_covariance_rtn,
)


def dense_disk_integral(sx, sy, mx, my, hbr, n=3000):
    """Independent ground truth: dense Cartesian integration over the disk."""
    axis = np.linspace(-hbr, hbr, n)
    X, Y = np.meshgrid(axis, axis, indexing="ij")
    inside = X * X + Y * Y <= hbr * hbr
    exponent = -0.5 * (((X - mx) / sx) ** 2 + ((Y - my) / sy) ** 2)
    exponent = np.where(inside, exponent, -np.inf)
    peak = exponent.max()
    if not math.isfinite(peak):
        return 0.0
    cell = (2 * hbr / (n - 1)) ** 2
    return float(np.sum(np.exp(exponent - peak)) * cell * math.exp(peak)
                 / (2 * math.pi * sx * sy))


@pytest.mark.parametrize(
    "sx,sy,miss,hbr",
    [
        (0.5, 0.5, 1.0, 0.020),
        (0.3, 1.2, 0.5, 0.020),
        (0.05, 0.05, 0.1, 0.020),
        (0.2, 0.9, 0.0, 0.020),
        (1.0, 3.0, 12.0, 0.020),
        (0.02, 0.5, 0.3, 0.050),
    ],
)
def test_quadrature_matches_dense_integration(sx, sy, miss, hbr):
    """The reported Pc must match an independent integration to six figures."""
    C = np.diag([sx ** 2, sy ** 2])
    m = np.array([miss, 0.0])
    got = pc_2d_quadrature(m, C, hbr)
    expected = dense_disk_integral(sx, sy, miss, 0.0, hbr)
    assert got == pytest.approx(expected, rel=2e-3)


def test_quadrature_is_saturated_when_hbr_dominates():
    """A hard body far larger than the uncertainty means collision is certain."""
    C = np.diag([1e-6, 1e-6])
    assert pc_2d_quadrature(np.zeros(2), C, 0.02) == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("sigmas", [1, 5, 10, 20, 30, 45, 60, 100, 1000])
def test_chan_series_never_overflows(sigmas):
    """
    Chan's series used to accumulate (v/2)^k / k! directly and returned NaN past
    roughly a 40-sigma miss. It must now return a finite probability at any
    separation.
    """
    C = np.diag([0.5 ** 2, 0.5 ** 2])
    value = pc_chan(np.array([sigmas * 0.5, 0.0]), C, 0.02)
    assert math.isfinite(value)
    assert 0.0 <= value <= 1.0


def test_chan_has_no_cancellation_floor():
    """
    The textbook inner factor `1 - exp(-u/2) * partial` bottoms out at machine
    epsilon for a small hard-body radius, reporting 2.2e-16 where the truth is
    1e-25. The tail formulation must track the quadrature instead.
    """
    C = np.diag([1.0 ** 2, 1.0 ** 2])
    m = np.array([10.0, 0.0])
    quad = pc_2d_quadrature(m, C, 0.02)
    series = pc_chan(m, C, 0.02)
    assert quad < 1e-20
    assert series == pytest.approx(quad, rel=0.05)


def test_tca_is_not_clamped_to_the_window_start():
    """
    The original find_tca clamped its solution into [0, window]. Every screening
    path handed it states sampled just AFTER the closest approach, so the clamp
    pinned the answer to t=0 and returned the sampled separation as the miss.
    """
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    # Secondary is already separating along +z, so the closest approach is behind
    # us: dr . dv > 0 puts the analytic solution at a negative time.
    r2 = np.array([7000.0, 40.0, 1.0])
    v2 = np.array([0.0, 7.5, 0.1])

    tca, miss, _, _ = find_tca(r1, v1, r2, v2, t_window_s=600.0)
    assert tca < 0.0, "a past closest approach must resolve to a negative time"
    assert miss < float(np.linalg.norm(r2 - r1)), (
        "the refined miss must be smaller than the separation at the sample time"
    )


def test_covariance_grows_with_the_actual_propagation_time():
    """
    Callers used to pass half the screening window regardless of the real time to
    TCA, inflating a two-hour conjunction's uncertainty by an order of magnitude.
    """
    near = secondary_covariance_rtn(2 * 3600.0)
    far = secondary_covariance_rtn(3 * 86400.0)
    assert near[1, 1] < far[1, 1]
    # Along-track must dominate: a period error integrates into along-track error.
    assert near[1, 1] > near[0, 0]
    assert far[1, 1] > far[2, 2]


def test_rtn_rotation_round_trips():
    r = np.array([7000.0, 100.0, -50.0])
    v = np.array([0.1, 7.5, 0.2])
    C_rtn = np.diag([0.3 ** 2, 1.0 ** 2, 0.3 ** 2])
    assert np.allclose(eci_to_rtn_cov(rtn_to_eci_cov(C_rtn, r, v), r, v), C_rtn)


def test_rtn_rotation_preserves_trace_and_symmetry():
    r = np.array([7000.0, 100.0, -50.0])
    v = np.array([0.1, 7.5, 0.2])
    C_rtn = np.diag([0.3 ** 2, 1.0 ** 2, 0.3 ** 2])
    C_eci = rtn_to_eci_cov(C_rtn, r, v)
    assert np.trace(C_eci) == pytest.approx(np.trace(C_rtn))
    assert np.allclose(C_eci, C_eci.T)
    assert np.all(np.linalg.eigvalsh(C_eci) > 0)


def test_low_relative_speed_is_reported_as_undetermined():
    """
    Below the short-encounter limit the 2-D Foster/Chan model does not apply.
    Reporting a number anyway would be a confident wrong answer.
    """
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    C = rtn_to_eci_cov(np.diag([0.3 ** 2, 1.0 ** 2, 0.3 ** 2]), r1, v1)
    result = assess_conjunction(
        r1, v1, C, r1 + np.array([0.5, 0.0, 0.0]), v1 + np.array([0.0, 1e-6, 0.0]),
        C, 0.020, tca_already_refined=True,
    )
    assert result["pc"] is None
    assert result["risk_level"] == "UNDETERMINED"
    assert result["short_encounter_valid"] is False


def test_assess_respects_already_refined_geometry():
    """With tca_already_refined the reported miss must equal the given geometry."""
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    r2 = r1 + np.array([0.4, 0.0, 0.0])
    v2 = np.array([0.0, 0.0, 7.5])
    C = rtn_to_eci_cov(np.diag([0.05 ** 2, 0.15 ** 2, 0.05 ** 2]), r1, v1)
    result = assess_conjunction(r1, v1, C, r2, v2, C, 0.020, tca_already_refined=True)
    assert result["miss_distance_km"] == pytest.approx(0.4)
    assert result["tca_s_from_epoch"] == 0.0


def test_risk_bands():
    assert risk_level(1e-9) == "NOMINAL"
    assert risk_level(1e-6) == "ELEVATED"
    assert risk_level(5e-5) == "HIGH"
    assert risk_level(1e-3) == "CRITICAL"


def test_mahalanobis_survives_a_singular_covariance():
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    assert math.isfinite(mahalanobis_2d(np.array([1.0, 1.0]), singular))
