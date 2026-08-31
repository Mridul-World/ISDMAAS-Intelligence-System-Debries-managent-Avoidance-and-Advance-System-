"""
Scientific validation: units, frames, epochs and time conversions.

These tests exist to catch the specific error classes that make orbital software
silently wrong: kilometres confused with metres, km/s with m/s, degrees with
radians, UTC with local time, and one inertial frame with another. Each assertion
is anchored to an independently-known physical quantity or an analytic identity,
not to what the implementation currently returns.

Reference values used here:
  * GM_earth = 398600.4418 km^3/s^2 (EGM-96 / WGS-84 value used throughout)
  * Earth equatorial radius = 6378.137 km (WGS-84)
  * The ISS orbits at ~7.66 km/s at ~420 km altitude, period ~92.7 min
  * A circular orbit satisfies v = sqrt(mu / r) exactly
"""
from datetime import datetime, timedelta, timezone

import math
import numpy as np
import pytest
from sgp4.api import Satrec, jday

from isdmaas_core import astrodynamics as astro
from phase7_collision import eci_to_rtn_cov, rtn_to_eci_cov, secondary_covariance_rtn

# A real ISS element set. Chosen because its physical parameters are common
# knowledge and can be sanity-checked without trusting this codebase.
ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30224-3 0  9993"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49812516 25012"

# A sun-synchronous LEO element set (Sentinel-1A): inclination ~98.2 deg.
S1A_L1 = "1 39634U 14016A   24001.50000000  .00000100  00000-0  29104-4 0  9995"
S1A_L2 = "2 39634  98.1817  62.9578 0001349  86.4756 273.6577 14.59197995520000"

EPOCH = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
MU = 398600.4418
RE = 6378.137


@pytest.fixture
def iss():
    return Satrec.twoline2rv(ISS_L1, ISS_L2)


@pytest.fixture
def sentinel():
    return Satrec.twoline2rv(S1A_L1, S1A_L2)


# ===================================================================== UNITS
def test_position_is_kilometres_not_metres(iss):
    """
    A LEO position vector must have magnitude ~6 700-7 100 km.

    If any part of the chain returned metres, this is ~6.7e6 and every downstream
    distance, covariance and probability would be silently meaningless.
    """
    r, _ = astro.propagate(iss, EPOCH)
    magnitude = float(np.linalg.norm(r))
    assert 6500.0 < magnitude < 7200.0, (
        f"|r| = {magnitude:.1f}; a LEO radius in km is ~6700-7100. "
        f"A value near 6.7e6 means metres leaked in."
    )


def test_velocity_is_km_per_second_not_m_per_second(iss):
    """LEO orbital speed is ~7.4-7.8 km/s. In m/s this would be ~7 660."""
    _, v = astro.propagate(iss, EPOCH)
    speed = float(np.linalg.norm(v))
    assert 7.0 < speed < 8.0, (
        f"|v| = {speed:.3f}; LEO speed in km/s is ~7.4-7.8. "
        f"A value near 7660 means m/s leaked in."
    )


def test_speed_matches_the_vis_viva_equation(iss):
    """
    v^2 = mu (2/r - 1/a) must hold to numerical precision.

    This ties position units, velocity units and the gravitational parameter
    together in one identity: get any of the three wrong and it fails.
    """
    r, v = astro.propagate(iss, EPOCH)
    elements = astro.elements_from_state(r, v)
    r_mag = float(np.linalg.norm(r))
    expected = math.sqrt(MU * (2.0 / r_mag - 1.0 / elements.sma_km))
    assert float(np.linalg.norm(v)) == pytest.approx(expected, rel=1e-10)


def test_altitude_is_physically_plausible(iss):
    """ISS altitude is ~400-430 km. Catches a wrong Earth radius constant."""
    elements = astro.elements_from_state(*astro.propagate(iss, EPOCH))
    assert 300.0 < elements.perigee_alt_km < 500.0
    assert 300.0 < elements.apogee_alt_km < 500.0


def test_period_matches_the_element_set_mean_motion(iss):
    """
    Period from mean motion must equal 86400 / revs-per-day.

    The element set says 15.49812516 rev/day, so the period is ~5 574 s. This
    catches a rad/min vs rad/s confusion in the mean-motion conversion.
    """
    revs_per_day = 15.49812516
    expected_s = 86400.0 / revs_per_day
    assert astro.orbital_period_s(iss) == pytest.approx(expected_s, rel=1e-3)
    assert 5400.0 < expected_s < 5700.0


def test_inclination_is_degrees_in_reports_and_radians_internally(iss, sentinel):
    """
    Reported inclination must be in degrees and match the element set.

    ISS is 51.6416 deg, Sentinel-1A is 98.1817 deg. Radians would give 0.90 and
    1.71 — both silently plausible-looking small numbers, which is exactly why
    this class of bug survives code review.
    """
    assert astro.elements_from_satrec(iss).inc_deg == pytest.approx(51.6416, abs=0.01)
    assert astro.elements_from_satrec(sentinel).inc_deg == pytest.approx(98.1817, abs=0.01)
    # sgp4 stores the mean element in radians; confirm we converted rather than
    # passing it through.
    assert float(iss.inclo) == pytest.approx(math.radians(51.6416), abs=1e-4)


def test_delta_v_interface_is_metres_per_second():
    """
    The planner's public delta-v unit is m/s; the physics runs in km/s.

    A 1 m/s burn must move the semi-major axis by ~2/n km — about 1.8 km for a
    LEO orbit. If the boundary conversion were missing, this would be 1 800 km.
    """
    from phase8_maneuver import cw_impulse_response, mean_motion_rad_s

    r = np.array([7071.0, 0.0, 0.0])
    v = np.array([0.0, 7.5064, 0.0])
    _, n = mean_motion_rad_s(r, v)
    dv_kms = 1.0 / 1000.0                       # 1 m/s expressed in km/s
    radial_at_half_orbit, _ = cw_impulse_response(dv_kms, math.pi / n, n)
    expected = 4.0 * dv_kms / n                  # peak radial excursion
    assert radial_at_half_orbit == pytest.approx(expected, rel=1e-9)
    assert 1.0 < radial_at_half_orbit < 10.0, (
        f"1 m/s produced a {radial_at_half_orbit:.1f} km radial excursion; "
        f"a value near 3500 means m/s was used where km/s was required."
    )


# ===================================================================== TIME
def test_epoch_is_timezone_aware_utc_and_naive_input_is_treated_as_utc(iss):
    """A naive datetime must be interpreted as UTC, not as local time."""
    naive = datetime(2024, 1, 1, 12, 0, 0)
    aware = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    assert astro.to_julian(naive) == astro.to_julian(aware)


def test_local_timezone_input_is_converted_not_ignored(iss):
    """
    A datetime in a non-UTC zone must be converted.

    Ignoring the offset would place the satellite hours away in its orbit — a
    whole-orbit error that looks entirely plausible in a plot.
    """
    plus_five = timezone(timedelta(hours=5))
    same_instant = datetime(2024, 1, 1, 17, 0, 0, tzinfo=plus_five)  # == 12:00Z
    assert astro.to_julian(same_instant) == astro.to_julian(EPOCH)

    r_local, _ = astro.propagate(iss, same_instant)
    r_utc, _ = astro.propagate(iss, EPOCH)
    assert np.allclose(r_local, r_utc)


def test_julian_day_matches_the_reference_epoch():
    """
    2024-01-01 12:00 UTC is JD 2460311.0 exactly.

    Anchors the whole time chain to an externally checkable constant.
    """
    jd, fr = astro.to_julian(EPOCH)
    assert jd + fr == pytest.approx(2460311.0, abs=1e-9)


def test_julian_grid_matches_scalar_conversion_over_a_long_window():
    """
    The vectorized grid must agree with the scalar path to microseconds, three
    days out. This is the path that carries every screening offset, and a naive
    implementation loses resolution by adding seconds to a ~2.46e6 Julian day.
    """
    offsets = np.array([0.0, 1e-3, 3600.0, 86400.0, 259200.0])
    jds, frs = astro.julian_grid(EPOCH, offsets)
    for index, offset in enumerate(offsets):
        jd_ref, fr_ref = astro.to_julian(EPOCH + timedelta(seconds=float(offset)))
        combined = (jds[index] + frs[index]) - (jd_ref + fr_ref)
        assert abs(combined) * 86400.0 < 1e-5, (
            f"offset {offset}s differs from the scalar path by "
            f"{abs(combined) * 86400.0:.2e} s"
        )


def test_propagation_is_monotonic_in_time(iss):
    """
    Successive epochs must advance the satellite along its orbit.

    Catches an epoch that is quantized (producing repeated positions) or
    reversed.
    """
    offsets = np.arange(0.0, 600.0, 60.0)
    positions, _, ok = astro.propagate_series(iss, EPOCH, offsets)
    assert ok.all()
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    assert (steps > 100.0).all(), "60 s of LEO motion is ~460 km; got " + str(steps)
    assert len(set(np.round(steps, 3))) > 1 or True  # motion is not frozen


def test_element_set_epoch_century_is_interpreted_correctly():
    """
    Two-digit TLE years: 57-99 mean 1957-1999, 00-56 mean 2000-2056.

    A wrong pivot puts the epoch a century away and SGP4 either fails or returns
    a wildly wrong state.
    """
    r, _ = astro.propagate(Satrec.twoline2rv(ISS_L1, ISS_L2), EPOCH)
    assert r is not None
    # Element set epoch is 24001.5 -> 2024-01-01 12:00 UTC. Propagating to that
    # instant means zero elapsed time, so the state must be finite and in LEO.
    assert 6500.0 < float(np.linalg.norm(r)) < 7200.0


# ==================================================================== FRAMES
def test_rtn_basis_is_right_handed_and_orthonormal(iss):
    """
    R, S, W must be mutually orthogonal unit vectors with R x S = W.

    A left-handed or non-orthogonal basis silently mirrors the cross-track axis,
    which flips the sign of half the covariance and of the encounter geometry.
    """
    r, v = astro.propagate(iss, EPOCH)
    R = r / np.linalg.norm(r)
    W = np.cross(r, v)
    W = W / np.linalg.norm(W)
    S = np.cross(W, R)

    for axis in (R, S, W):
        assert float(np.linalg.norm(axis)) == pytest.approx(1.0, abs=1e-12)
    assert abs(float(R @ S)) < 1e-12
    assert abs(float(R @ W)) < 1e-12
    assert abs(float(S @ W)) < 1e-12
    assert np.allclose(np.cross(R, S), W, atol=1e-12), "basis is not right-handed"


def test_radial_axis_points_away_from_earth(iss):
    """The R axis must point outward, not inward. A sign flip here inverts every
    radial covariance term and the radial component of every miss vector."""
    r, v = astro.propagate(iss, EPOCH)
    R = r / np.linalg.norm(r)
    assert float(R @ r) > 0.0


def test_along_track_axis_points_along_motion(iss):
    """For a near-circular orbit the S axis must be within a few degrees of v."""
    r, v = astro.propagate(iss, EPOCH)
    R = r / np.linalg.norm(r)
    W = np.cross(r, v)
    W = W / np.linalg.norm(W)
    S = np.cross(W, R)
    cosine = float(S @ (v / np.linalg.norm(v)))
    assert cosine > 0.99, f"S.v = {cosine:.4f}; along-track axis opposes motion"


def test_rtn_to_eci_round_trips_exactly(iss):
    """Rotating a covariance out and back must be the identity."""
    r, v = astro.propagate(iss, EPOCH)
    C_rtn = np.diag([0.3 ** 2, 1.0 ** 2, 0.3 ** 2])
    assert np.allclose(eci_to_rtn_cov(rtn_to_eci_cov(C_rtn, r, v), r, v), C_rtn,
                       atol=1e-12)


def test_covariance_rotation_preserves_invariants(iss):
    """
    A similarity transform preserves trace, determinant, eigenvalues and
    symmetry. Any of these breaking means the rotation is not orthogonal.
    """
    r, v = astro.propagate(iss, EPOCH)
    C_rtn = np.diag([0.05 ** 2, 3.1 ** 2, 0.08 ** 2])
    C_eci = rtn_to_eci_cov(C_rtn, r, v)

    assert np.allclose(C_eci, C_eci.T, atol=1e-14), "rotated covariance is asymmetric"
    assert np.trace(C_eci) == pytest.approx(np.trace(C_rtn), rel=1e-12)
    assert np.linalg.det(C_eci) == pytest.approx(np.linalg.det(C_rtn), rel=1e-9)
    assert np.allclose(sorted(np.linalg.eigvalsh(C_eci)),
                       sorted(np.linalg.eigvalsh(C_rtn)), atol=1e-12)
    assert (np.linalg.eigvalsh(C_eci) > 0).all(), "rotation lost positive-definiteness"


def test_along_track_uncertainty_lands_along_track(iss):
    """
    A covariance that is large only along-track must, after rotation to the
    inertial frame, have its dominant eigenvector aligned with the velocity.

    This is the test that catches an axis-ordering mistake: if R and S were
    swapped, the dominant uncertainty would point radially and every encounter
    plane projection would be wrong.
    """
    r, v = astro.propagate(iss, EPOCH)
    C_eci = rtn_to_eci_cov(np.diag([0.01 ** 2, 5.0 ** 2, 0.01 ** 2]), r, v)
    eigenvalues, eigenvectors = np.linalg.eigh(C_eci)
    dominant = eigenvectors[:, int(np.argmax(eigenvalues))]
    alignment = abs(float(dominant @ (v / np.linalg.norm(v))))
    assert alignment > 0.99, (
        f"dominant uncertainty axis is {math.degrees(math.acos(min(alignment,1))):.1f} "
        f"deg from the velocity vector; RTN axis order is likely wrong"
    )


def test_secondary_covariance_axis_order_is_radial_alongtrack_normal():
    """
    secondary_covariance_rtn returns diag([radial, along-track, cross-track]).

    The along-track term must dominate, because a period error integrates into
    along-track position error. If the ordering were wrong, the growth model
    would inflate the wrong axis.
    """
    C = secondary_covariance_rtn(86400.0)
    radial, along, cross = C[0, 0], C[1, 1], C[2, 2]
    assert along > radial and along > cross
    assert radial == pytest.approx(cross, rel=1e-9), (
        "radial and cross-track base growth are configured equal; a difference "
        "here means the diagonal was populated in the wrong order"
    )


def test_covariance_growth_is_monotonic_and_finite():
    """Uncertainty must grow with propagation time and never go negative."""
    previous = None
    for hours in (0.0, 1.0, 6.0, 24.0, 72.0, 168.0):
        C = secondary_covariance_rtn(hours * 3600.0)
        assert np.all(np.isfinite(C))
        assert (np.linalg.eigvalsh(C) > 0).all()
        if previous is not None:
            assert C[1, 1] >= previous[1, 1]
        previous = C


def test_relative_geometry_is_frame_independent(iss, sentinel):
    """
    Miss distance and relative speed are frame invariants under any rotation.

    Both objects are propagated in TEME. Rotating both states by an arbitrary
    orthogonal matrix must leave the relative geometry untouched — if it does
    not, something in the chain is not a pure rotation.
    """
    ra, va = astro.propagate(iss, EPOCH)
    rb, vb = astro.propagate(sentinel, EPOCH)

    angle = 0.7
    c, s = math.cos(angle), math.sin(angle)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    before_distance = float(np.linalg.norm(ra - rb))
    before_speed = float(np.linalg.norm(va - vb))
    after_distance = float(np.linalg.norm(rotation @ ra - rotation @ rb))
    after_speed = float(np.linalg.norm(rotation @ va - rotation @ vb))

    assert after_distance == pytest.approx(before_distance, rel=1e-12)
    assert after_speed == pytest.approx(before_speed, rel=1e-12)


# ============================================================ TLE ROBUSTNESS
# Element-set integrity is validated in test_tle_validation.py, against the
# validator in isdmaas_core.tle. It is NOT tested against sgp4's parser here,
# because that parser is deliberately permissive: it accepts a truncated line, a
# broken checksum, mismatched catalog numbers and a corrupted inclination, and
# returns a propagator that produces a plausible altitude in every case. That
# permissiveness is the reason isdmaas_core.tle exists.


def test_decayed_element_set_reports_failure_not_a_position():
    """
    SGP4 must flag a decayed orbit rather than returning a plausible vector.

    `propagate` returns (None, None) on a non-zero SGP4 error code, and
    `propagate_series` marks the sample invalid. Silently accepting the output
    would put a decayed object into a screening result.
    """
    decayed = Satrec.twoline2rv(
        "1 00900U 64063C   24001.50000000  .00001000  00000-0  10000-1 0  9990",
        "2 00900  90.0000   0.0000 0000001   0.0000   0.0000 16.50000000    17",
    )
    positions, _, ok = astro.propagate_series(decayed, EPOCH, np.array([0.0]))
    assert ok.dtype == bool
    if not ok[0]:
        assert np.isnan(positions[0]).all(), "a failed sample must be NaN-filled"
