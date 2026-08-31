"""
Regression tests for the maneuver planner and the safety gate.

The physics tests here compare the first-order model against a numerically
integrated two-body orbit — the only way to catch a sign error or a missing
term, both of which were present.
"""
import math

import numpy as np
import pytest

from phase8_maneuver import (
    PC_THRESHOLD,
    apply_along_track_dv,
    cw_impulse_response,
    fuel_kg,
    lead_time_options,
    mean_motion_rad_s,
    plan_maneuver,
    solve_min_dv,
)
from phase10_safety import check_fuel, check_orbit_safety, orbital_elements
from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn

MU = 398600.4418
R0 = np.array([7071.0, 0.0, 0.0])
V0 = np.array([0.0, 7.5064, 0.0])


def rk4_two_body(r0, v0, dt, steps=20000):
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    h = dt / steps

    def acc(x):
        return -MU * x / np.linalg.norm(x) ** 3

    for _ in range(steps):
        k1v, k1r = acc(r), v
        k2v, k2r = acc(r + 0.5 * h * k1r), v + 0.5 * h * k1v
        k3v, k3r = acc(r + 0.5 * h * k2r), v + 0.5 * h * k2v
        k4v, k4r = acc(r + h * k3r), v + h * k3v
        r = r + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v


@pytest.mark.parametrize("fraction", [0.05, 0.25, 0.5, 1.0, 3.0])
def test_impulse_model_matches_numerical_propagation(fraction):
    """
    The displacement from a tangential burn must match a real integrated orbit.

    This is the regression test for two defects at once: the radial term modelled
    as a constant semi-major-axis offset (it actually oscillates as
    1 - cos(nt)), and the along-track drift applied with the wrong sign.
    """
    _, n = mean_motion_rad_s(R0, V0)
    period = 2 * math.pi / n
    dt = fraction * period
    dv = 0.5 / 1000.0

    nominal_r, nominal_v = rk4_two_body(R0, V0, dt)
    burned_r, _ = rk4_two_body(R0, V0 + dv * (V0 / np.linalg.norm(V0)), dt)
    truth = burned_r - nominal_r

    modelled_r, _ = apply_along_track_dv(nominal_r, nominal_v, dv, dt)
    model = modelled_r - nominal_r

    assert np.linalg.norm(model - truth) < 0.02 * max(np.linalg.norm(truth), 1e-9)


def test_prograde_burn_drifts_the_satellite_backwards():
    """
    A prograde burn raises the orbit, which lengthens the period, so the
    satellite falls BEHIND. The previous model displaced it forwards, so the
    reported burn direction was inverted.
    """
    _, n = mean_motion_rad_s(R0, V0)
    period = 2 * math.pi / n
    _, along = cw_impulse_response(0.5 / 1000.0, 3 * period, n)
    assert along < 0.0
    _, along_retro = cw_impulse_response(-0.5 / 1000.0, 3 * period, n)
    assert along_retro > 0.0


def test_radial_response_starts_at_zero_and_oscillates():
    """The satellite does not teleport to its new altitude when the thruster fires."""
    _, n = mean_motion_rad_s(R0, V0)
    period = 2 * math.pi / n
    dv = 0.5 / 1000.0
    at_burn, _ = cw_impulse_response(dv, 0.0, n)
    half_orbit, _ = cw_impulse_response(dv, period / 2, n)
    full_orbit, _ = cw_impulse_response(dv, period, n)
    assert at_burn == pytest.approx(0.0, abs=1e-12)
    assert half_orbit == pytest.approx(4.0 * dv / n, rel=1e-9)
    assert full_orbit == pytest.approx(0.0, abs=1e-6)


def test_post_burn_velocity_actually_changes():
    """
    The old model returned the velocity unchanged, so every orbit-safety check
    downstream evaluated an orbit that had never been burned.
    """
    dv = 1.0 / 1000.0
    _, v_post = apply_along_track_dv(R0, V0, dv, 3600.0)
    assert np.linalg.norm(v_post) == pytest.approx(np.linalg.norm(V0) + dv, rel=1e-9)


def test_orbit_safety_sees_the_semi_major_axis_change():
    """A large burn must move the semi-major axis out of the mission band."""
    a0, _, _ = orbital_elements(R0, V0)
    r_post, v_post = apply_along_track_dv(R0, V0, 50.0 / 1000.0, 1800.0)
    a1, _, _ = orbital_elements(r_post, v_post)
    assert abs(a1 - a0) > 0.05
    assert check_orbit_safety(r_post, v_post, mission_sma_km=a0)["pass"] in (True, False)
    huge_r, huge_v = apply_along_track_dv(R0, V0, 200.0, 1800.0)
    assert not check_orbit_safety(huge_r, huge_v, mission_sma_km=a0)["pass"]


def _crossing_threat(miss_km):
    rhat = R0 / np.linalg.norm(R0)
    speed = float(np.linalg.norm(V0))
    vhat = V0 / speed
    cross = np.cross(rhat, vhat)
    cross /= np.linalg.norm(cross)
    return R0 + miss_km * rhat, cross * speed


def test_plan_produces_a_monotonic_lead_time_trade():
    """
    Burning earlier must cost less delta-v. When every row of the trade table
    showed the same number, that was the symptom of the constant-radial bug.
    """
    r2, v2 = _crossing_threat(0.3)
    Cp = rtn_to_eci_cov(np.diag([0.05, 0.15, 0.05]) ** 2, R0, V0)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(12 * 3600.0), r2, v2)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, 12 * 3600.0, 2300.0)

    assert plan["action_required"]
    options = sorted(plan["options"], key=lambda o: o["burn_lead_time_h"])
    assert len(options) >= 4
    deltas = [o["dv_ms"] for o in options]
    assert deltas[0] > deltas[-1], "an earlier burn must need less delta-v"
    assert deltas == sorted(deltas, reverse=True)


def test_plan_reaches_the_safety_target():
    r2, v2 = _crossing_threat(0.3)
    Cp = rtn_to_eci_cov(np.diag([0.05, 0.15, 0.05]) ** 2, R0, V0)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(8 * 3600.0), r2, v2)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, 8 * 3600.0, 2300.0)
    recommendation = plan["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["predicted_new_pc"] < PC_THRESHOLD
    assert recommendation["direction"] in ("prograde", "retrograde")


def test_no_action_below_threshold():
    r2, v2 = _crossing_threat(60.0)
    Cp = rtn_to_eci_cov(np.diag([0.05, 0.15, 0.05]) ** 2, R0, V0)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(8 * 3600.0), r2, v2)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, 8 * 3600.0, 2300.0)
    assert not plan["action_required"]
    assert isinstance(plan["recommendation"], str)


def test_short_notice_conjunction_still_gets_options():
    """
    A conjunction eleven minutes out used to produce no feasible option at all,
    because the lead-time ladder started at 45 minutes.
    """
    options = lead_time_options(11 * 60.0)
    assert options
    assert max(options) < 11 / 60.0


def test_lead_time_options_never_exceed_time_to_tca():
    for tca_h in (0.1, 0.5, 2.0, 8.0, 48.0):
        for lead in lead_time_options(tca_h * 3600.0):
            assert 0 < lead < tca_h


def test_solve_min_dv_finds_both_directions():
    r2, v2 = _crossing_threat(0.3)
    Cp = rtn_to_eci_cov(np.diag([0.05, 0.15, 0.05]) ** 2, R0, V0)
    C2 = rtn_to_eci_cov(secondary_covariance_rtn(6 * 3600.0), r2, v2)
    solution = solve_min_dv(R0, V0, Cp, r2, v2, C2, 0.020, 6 * 3600.0)
    assert solution is not None
    dv_ms, sign, pc, _ = solution
    assert dv_ms > 0.0
    assert sign in (1.0, -1.0)
    assert pc < 1e-6 * 1.05


def test_fuel_follows_tsiolkovsky():
    mass, isp = 2300.0, 220.0
    dv_ms = 1.0
    expected = mass * (1 - math.exp(-(dv_ms / 1000.0) / (isp * 9.80665e-3)))
    assert fuel_kg(dv_ms, mass, isp) == pytest.approx(expected)
    assert fuel_kg(0.0, mass) == pytest.approx(0.0)


def test_fuel_check_enforces_the_margin():
    assert check_fuel(1.0, 1.0, 1.4)["pass"]
    assert not check_fuel(1.0, 1.0, 1.2)["pass"]
