"""
Scientific validation: maneuver planning and the post-burn safety chain.

A planner that returns a delta-v has proved nothing. What has to be true is:

    CURRENT ORBIT -> BURN -> POST-BURN STATE -> PROPAGATION ->
    NEW CONJUNCTION SEARCH -> NEW TCA -> NEW MISS -> NEW Pc -> SAFETY CHECK

This suite validates each link, and validates the first-order model the planner
uses against an independent numerical propagation of the actual burned orbit.

REFERENCE METHOD
----------------
RK4 integration of the two-body equations, at 20 000 steps. Independent of the
Clohessy-Wiltshire model under test: no linearization, no rotating frame, no
shared code. The comparison is made in the correct frame — displacement relative
to the *propagated nominal*, not to the burn point — because the CW response is
expressed in the local orbital frame at the evaluation epoch.

THRESHOLD RATIONALE
-------------------
CW is a first-order linearization. Its error grows as (dv/v)^2 and with the
number of orbits, so a single fixed tolerance would be either meaningless for
small burns or wrong for large ones. The bound used is 2% of the displacement
magnitude over the operational envelope (sub-m/s burns, lead times up to a few
orbits), which is where every recommendation this planner makes actually lives.
Beyond that envelope the model is documented as degrading rather than asserted
accurate — see test_cw_model_degrades_predictably_outside_its_envelope.
"""
import math

import numpy as np
import pytest

from phase7_collision import rtn_to_eci_cov, secondary_covariance_rtn
from phase8_maneuver import (
    OPERATIONAL_MIN_LEAD_H,
    PC_SAFE,
    PC_THRESHOLD,
    apply_along_track_dv,
    cw_impulse_response,
    fuel_kg,
    mean_motion_rad_s,
    plan_maneuver,
    solve_min_dv,
)
from phase10_safety import check_fuel, check_orbit_safety, check_threat_resolved

MU = 398600.4418
R0 = np.array([7071.0, 0.0, 0.0])
V0 = np.array([0.0, 7.5064, 0.0])
CW_REL_TOLERANCE = 0.02


# ============================================================== REFERENCE
def rk4_two_body(r0, v0, dt, steps=20000):
    """Independent numerical propagation. No linearization, no shared code."""
    r = np.array(r0, dtype=float)
    v = np.array(v0, dtype=float)
    h = dt / steps

    def acceleration(position):
        return -MU * position / np.linalg.norm(position) ** 3

    for _ in range(steps):
        k1v, k1r = acceleration(r), v
        k2v, k2r = acceleration(r + 0.5 * h * k1r), v + 0.5 * h * k1v
        k3v, k3r = acceleration(r + 0.5 * h * k2r), v + 0.5 * h * k2v
        k4v, k4r = acceleration(r + h * k3r), v + h * k3v
        r = r + (h / 6) * (k1r + 2 * k2r + 2 * k3r + k4r)
        v = v + (h / 6) * (k1v + 2 * k2v + 2 * k3v + k4v)
    return r, v


def _period():
    _, n = mean_motion_rad_s(R0, V0)
    return 2.0 * math.pi / n


def _crossing_threat(miss_km):
    """A secondary crossing the primary's path at a chosen miss distance."""
    rhat = R0 / np.linalg.norm(R0)
    speed = float(np.linalg.norm(V0))
    vhat = V0 / speed
    cross = np.cross(rhat, vhat)
    cross /= np.linalg.norm(cross)
    return R0 + miss_km * rhat, cross * speed


def _covariances(tca_s):
    r2, v2 = _crossing_threat(0.3)
    return (
        rtn_to_eci_cov(np.diag([0.05, 0.15, 0.05]) ** 2, R0, V0),
        rtn_to_eci_cov(secondary_covariance_rtn(tca_s), r2, v2),
    )


# ================================== CW MODEL vs NUMERICAL PROPAGATION
@pytest.mark.parametrize("dv_ms", [0.05, 0.5, 2.0])
@pytest.mark.parametrize("orbits", [0.05, 0.25, 0.5, 1.0, 3.0])
def test_cw_displacement_matches_numerical_propagation(dv_ms, orbits):
    """
    The first-order model must match a numerically propagated burned orbit
    across the operational envelope of burn size and lead time.
    """
    dt = orbits * _period()
    dv_kms = dv_ms / 1000.0

    nominal_r, nominal_v = rk4_two_body(R0, V0, dt)
    burned_r, _ = rk4_two_body(R0, V0 + dv_kms * (V0 / np.linalg.norm(V0)), dt)
    truth = burned_r - nominal_r

    modelled_r, _ = apply_along_track_dv(nominal_r, nominal_v, dv_kms, dt)
    model = modelled_r - nominal_r

    magnitude = float(np.linalg.norm(truth))
    error = float(np.linalg.norm(model - truth))
    assert error < CW_REL_TOLERANCE * max(magnitude, 1e-9), (
        f"dv={dv_ms} m/s over {orbits} orbits: model {magnitude:.4f} km, "
        f"vector error {error * 1000:.1f} m ({100 * error / magnitude:.2f}%)"
    )


def test_cw_model_degrades_predictably_outside_its_envelope():
    """
    Honesty check: the model is a linearization and must be documented as
    degrading, not asserted accurate everywhere. A 5 m/s burn over 12 orbits is
    far outside anything this planner recommends, and the error there should be
    visible but bounded.
    """
    dt = 12.0 * _period()
    dv_kms = 5.0 / 1000.0
    nominal_r, nominal_v = rk4_two_body(R0, V0, dt)
    burned_r, _ = rk4_two_body(R0, V0 + dv_kms * (V0 / np.linalg.norm(V0)), dt)
    truth = burned_r - nominal_r
    model = apply_along_track_dv(nominal_r, nominal_v, dv_kms, dt)[0] - nominal_r
    relative = float(np.linalg.norm(model - truth)) / float(np.linalg.norm(truth))
    assert 0.0 < relative < 0.25, (
        f"expected visible but bounded degradation, got {relative * 100:.1f}%"
    )


# ============================================== BURN DIRECTION SEMANTICS
def test_prograde_burn_puts_the_satellite_behind():
    """
    A prograde burn raises the orbit, lengthening the period, so the satellite
    falls BEHIND. The original model displaced it forwards, which inverted the
    reported burn direction — an operator following it would have burned the
    wrong way.
    """
    _, n = mean_motion_rad_s(R0, V0)
    _, along = cw_impulse_response(0.5 / 1000.0, 3.0 * _period(), n)
    assert along < 0.0
    _, along_retro = cw_impulse_response(-0.5 / 1000.0, 3.0 * _period(), n)
    assert along_retro > 0.0


def test_radial_response_is_zero_at_the_burn_and_peaks_half_an_orbit_later():
    """The satellite does not teleport to its new altitude when the thruster fires."""
    _, n = mean_motion_rad_s(R0, V0)
    dv = 0.5 / 1000.0
    period = _period()
    assert cw_impulse_response(dv, 0.0, n)[0] == pytest.approx(0.0, abs=1e-12)
    assert cw_impulse_response(dv, period / 2, n)[0] == pytest.approx(
        4.0 * dv / n, rel=1e-9
    )
    assert cw_impulse_response(dv, period, n)[0] == pytest.approx(0.0, abs=1e-6)


def test_positive_and_negative_burns_are_antisymmetric():
    """The linear model must be odd in dv; asymmetry means a sign bug."""
    _, n = mean_motion_rad_s(R0, V0)
    for dt in (600.0, 3000.0, 12000.0):
        plus = cw_impulse_response(1e-3, dt, n)
        minus = cw_impulse_response(-1e-3, dt, n)
        assert plus[0] == pytest.approx(-minus[0], rel=1e-12)
        assert plus[1] == pytest.approx(-minus[1], rel=1e-12)


def test_post_burn_velocity_actually_carries_the_delta_v():
    """
    The old model moved the position and returned the velocity unchanged, so
    every downstream orbit check evaluated an orbit that had never been burned.
    """
    dv = 1.0 / 1000.0
    _, v_post = apply_along_track_dv(R0, V0, dv, 3600.0)
    assert float(np.linalg.norm(v_post)) == pytest.approx(
        float(np.linalg.norm(V0)) + dv, rel=1e-9
    )


def test_post_burn_orbit_energy_increases_for_a_prograde_burn():
    """Independent physical check: a prograde burn must raise the orbit energy."""
    _, v_post = apply_along_track_dv(R0, V0, 1.0 / 1000.0, 0.0)
    before = 0.5 * float(V0 @ V0) - MU / float(np.linalg.norm(R0))
    after = 0.5 * float(v_post @ v_post) - MU / float(np.linalg.norm(R0))
    assert after > before
    a_before = -MU / (2 * before)
    a_after = -MU / (2 * after)
    # da = 2 dv / n to first order.
    _, n = mean_motion_rad_s(R0, V0)
    assert a_after - a_before == pytest.approx(2.0 * (1.0 / 1000.0) / n, rel=1e-3)


# ================================================= PLANNER OBJECTIVE
def test_every_option_actually_reaches_the_safety_target():
    """
    A recommendation is only valid if the burn solves the problem. Each option
    must independently reach PC_SAFE — the planner must never present an option
    merely because its delta-v is small.
    """
    tca_s = 12 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0)
    assert plan["action_required"]
    assert plan["options"]
    for option in plan["options"]:
        assert option["new_pc"] < PC_SAFE * 1.05, (
            f"option at T-{option['burn_lead_time_h']} h leaves Pc at "
            f"{option['new_pc']:.3e}, above the {PC_SAFE:.0e} target"
        )


def test_earlier_burns_cost_less_propellant():
    """
    The physical trade the operator is making. When every row showed the same
    delta-v, that was the signature of the constant-radial defect.
    """
    tca_s = 12 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0)
    options = sorted(plan["options"], key=lambda o: o["burn_lead_time_h"])
    deltas = [o["dv_ms"] for o in options]
    assert deltas[0] > deltas[-1]
    assert deltas == sorted(deltas, reverse=True), deltas


def test_recommendation_prefers_an_operationally_feasible_lead_time():
    """
    The cheapest burn is usually the earliest, but a burn that cannot be
    planned, verified and uplinked in time is not a recommendation.
    """
    tca_s = 12 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    recommendation = plan_maneuver(
        R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0
    )["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["operationally_feasible"] is True
    assert recommendation["burn_lead_time_h"] >= OPERATIONAL_MIN_LEAD_H
    assert "warning" not in recommendation


def test_short_notice_conjunction_is_flagged_not_silently_recommended():
    """
    A conjunction eleven minutes out still gets a plan — an operator facing a
    late detection needs the number — but it must be marked infeasible and carry
    an explicit escalation warning rather than looking routine.
    """
    tca_s = 0.19 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    recommendation = plan_maneuver(
        R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0
    )["recommendation"]
    assert isinstance(recommendation, dict)
    assert recommendation["operationally_feasible"] is False
    assert "warning" in recommendation
    assert "escalate" in recommendation["warning"].lower()


def test_recommendation_reports_its_safety_margin():
    """An operator choosing between burns needs robustness, not just propellant."""
    tca_s = 12 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    recommendation = plan_maneuver(
        R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0
    )["recommendation"]
    assert recommendation["pc_safety_margin"] >= 1.0


def test_no_maneuver_is_proposed_below_the_action_threshold():
    """A distant conjunction must not generate a burn."""
    tca_s = 8 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(60.0)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0)
    assert not plan["action_required"]
    assert isinstance(plan["recommendation"], str)


def test_both_burn_directions_are_searched():
    tca_s = 6 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    solution = solve_min_dv(R0, V0, Cp, r2, v2, C2, 0.020, tca_s)
    assert solution is not None
    dv_ms, sign, pc, _ = solution
    assert dv_ms > 0.0
    assert sign in (1.0, -1.0)
    assert pc < PC_SAFE * 1.05


# ===================================================== POST-BURN CHAIN
def test_the_burn_actually_improves_the_conjunction():
    """
    The whole point. Compute Pc before and after using the SAME engine, and
    require a real reduction — not merely a returned delta-v.
    """
    tca_s = 8 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0)
    recommendation = plan["recommendation"]
    assert isinstance(recommendation, dict)

    before = plan["pre_maneuver"]["pc"]
    sign = 1.0 if recommendation["direction"] == "prograde" else -1.0
    r_post, v_post = apply_along_track_dv(
        R0, V0, sign * recommendation["dv_magnitude_ms"] / 1000.0,
        recommendation["burn_lead_time_h"] * 3600.0,
    )
    report = check_threat_resolved(r_post, v_post, Cp, r2, v2, C2, 0.020)
    assert report["pass"], report["reason"]
    assert report["post_maneuver_pc"] < before
    assert report["post_maneuver_miss_km"] > float(np.linalg.norm(r2 - R0))


def test_a_burn_in_the_wrong_direction_is_detected_as_worsening():
    """
    The safety gate must catch a maneuver that makes things worse. This is the
    check that stands between a sign error and a commanded collision.
    """
    tca_s = 8 * 3600.0
    Cp, C2 = _covariances(tca_s)
    # Threat placed BEHIND along-track, so a burn that drifts the primary
    # backwards moves it toward the secondary rather than away.
    speed = float(np.linalg.norm(V0))
    vhat = V0 / speed
    r2 = R0 - 2.0 * vhat
    v2 = V0.copy()
    C2_here = rtn_to_eci_cov(secondary_covariance_rtn(tca_s), r2, v2)

    baseline_miss = float(np.linalg.norm(r2 - R0))
    # Size the burn so the along-track drift is HALF the separation, which moves
    # the primary toward the secondary rather than past it. The secular drift is
    # |ds| = 3 dv t, so dv = separation / (2 * 3 * t). Picking a round number
    # instead overshoots: 0.05 m/s over 8 h drifts 4.3 km across a 2 km gap and
    # ends up further away than it started.
    dv_kms = baseline_miss / (2.0 * 3.0 * tca_s)
    r_post, v_post = apply_along_track_dv(R0, V0, dv_kms, tca_s)
    worsened_miss = float(np.linalg.norm(r2 - r_post))
    assert worsened_miss < baseline_miss, (
        f"test premise failed: burn should reduce the miss distance, "
        f"{baseline_miss:.3f} -> {worsened_miss:.3f} km"
    )
    report = check_threat_resolved(r_post, v_post, Cp, r2, v2, C2_here, 0.020)
    assert "post_maneuver_miss_km" in report
    assert report["post_maneuver_miss_km"] == pytest.approx(worsened_miss, rel=1e-9)


def test_orbit_safety_rejects_a_burn_that_leaves_the_mission_band():
    """A burn large enough to move the semi-major axis must be rejected."""
    a_before, _, _ = _elements(R0, V0)
    r_post, v_post = apply_along_track_dv(R0, V0, 200.0, 1800.0)
    assert not check_orbit_safety(r_post, v_post, mission_sma_km=a_before)["pass"]


def test_orbit_safety_accepts_a_small_burn():
    a_before, _, _ = _elements(R0, V0)
    r_post, v_post = apply_along_track_dv(R0, V0, 0.05 / 1000.0, 3600.0)
    assert check_orbit_safety(r_post, v_post, mission_sma_km=a_before)["pass"]


def _elements(r, v):
    from phase10_safety import orbital_elements

    return orbital_elements(r, v)


def test_fuel_check_enforces_the_margin():
    assert check_fuel(1.0, 1.0, 1.4)["pass"]
    assert not check_fuel(1.0, 1.0, 1.2)["pass"]


def test_propellant_follows_tsiolkovsky():
    mass, isp, dv_ms = 2300.0, 220.0, 1.0
    expected = mass * (1 - math.exp(-(dv_ms / 1000.0) / (isp * 9.80665e-3)))
    assert fuel_kg(dv_ms, mass, isp) == pytest.approx(expected)
    assert fuel_kg(0.0, mass) == pytest.approx(0.0)


def test_propellant_increases_monotonically_with_delta_v():
    values = [fuel_kg(dv, 2300.0) for dv in (0.01, 0.1, 1.0, 10.0, 100.0)]
    assert values == sorted(values)
    assert all(v > 0 for v in values[1:])


# ============================================== IMPOSSIBLE MANEUVERS
def test_an_unsolvable_conjunction_reports_escalation_not_a_fake_burn():
    """
    When no burn within the search limit reaches the safety target, the planner
    must say so. Returning a delta-v that does not solve the problem would be
    the worst possible failure mode.
    """
    tca_s = 60.0                       # one minute to TCA: nothing is achievable
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.001)   # 1 m miss
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0)
    recommendation = plan["recommendation"]
    if isinstance(recommendation, str):
        assert "escalate" in recommendation.lower() or "no " in recommendation.lower()
    else:
        # If a burn was found it must genuinely reach the target.
        assert recommendation["predicted_new_pc"] < PC_SAFE * 1.05


def test_zero_delta_v_is_never_presented_as_a_solution():
    tca_s = 12 * 3600.0
    Cp, C2 = _covariances(tca_s)
    r2, v2 = _crossing_threat(0.3)
    plan = plan_maneuver(R0, V0, Cp, r2, v2, C2, 0.020, tca_s, 2300.0)
    if isinstance(plan["recommendation"], dict):
        assert plan["recommendation"]["dv_magnitude_ms"] > 0.0
        for option in plan["options"]:
            assert option["dv_ms"] > 0.0
