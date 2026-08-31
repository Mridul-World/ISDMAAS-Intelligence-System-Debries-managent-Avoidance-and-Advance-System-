"""
Scientific validation: time of closest approach.

The production algorithm is a three-stage pipeline (apogee/perigee filter ->
vectorized coarse grid -> Brent refinement). This suite validates it against an
INDEPENDENT reference implementation that shares no code with it: a dense
uniform scan over the same window, refined by ternary section.

The two implementations agree only if both are right. The reference is
deliberately naive and slow — that is what makes it independent evidence rather
than a restatement of the same assumptions.

THRESHOLD RATIONALE
-------------------
Thresholds are derived from the numerical method, not tuned until tests pass.

* TCA time: 2 ms.
  Brent terminates on an absolute abscissa tolerance of 1 ms (astrodynamics.py,
  `refine_close_approach` passes tol_s=1e-3). The reference refines to 1 ms as
  well. Two independent searches each converged to +/-1 ms can differ by up to
  2 ms. Anything larger means one of them is not converging.

* Miss distance: 1 metre, or 1e-6 relative, whichever is larger.
  Near TCA the separation is d(t)^2 = miss^2 + v_rel^2 (t - tca)^2, so the
  distance is STATIONARY in time: a 1 ms timing error at 15 km/s and a 1 km miss
  perturbs the distance by v_rel^2 * dt^2 / (2 * miss) ~ 1e-4 m. The distance
  agreement is therefore far tighter than the time agreement, and 1 m is a loose
  bound that still catches a real disagreement.

* Relative velocity: 1e-6 km/s relative.
  Velocity varies slowly through the encounter (its derivative is the relative
  acceleration, ~1e-5 km/s^2 in LEO), so a 2 ms timing difference changes it by
  ~2e-8 km/s. This threshold is dominated by float64 round-off in the SGP4 call,
  not by the search.

* Relative POSITION VECTOR: |v_rel| * TCA tolerance, with a 1 m floor.
  This one is deliberately looser than the miss distance, and the asymmetry is
  physical rather than a concession. The miss DISTANCE is stationary at TCA —
  d(t)^2 = miss^2 + v_rel^2 (t-tca)^2, so a timing error enters at SECOND order.
  The relative position VECTOR is not stationary: its time derivative is the
  relative velocity itself, so a timing error enters at FIRST order. Two searches
  that agree on the time to 2 ms therefore agree on the vector only to
  |v_rel| * 2 ms, which at 14 km/s is 28 m. Holding the vector to the same 1 m
  bound as the distance would be asserting a precision the method does not have,
  and the test would fail for a correct implementation.

These are not "close enough for a demo" numbers. They assert that the production
search is limited by SGP4 itself rather than by the search.
"""
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
from sgp4.api import Satrec

from isdmaas_core import astrodynamics as astro

CATALOG = Path(__file__).resolve().parent.parent.parent / "data" / "catalog_active.tle"
EPOCH = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

# Acceptance thresholds — see the module docstring for the derivation.
TCA_TIME_TOLERANCE_S = 2e-3
MISS_ABS_TOLERANCE_KM = 1e-3          # 1 metre
MISS_REL_TOLERANCE = 1e-6
REL_VELOCITY_REL_TOLERANCE = 1e-6
REL_POSITION_FLOOR_KM = 1e-3          # 1 metre floor for slow encounters


def _load_catalog():
    if not CATALOG.exists():
        return {}
    lines = [line.rstrip() for line in CATALOG.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    objects = {}
    for i in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            continue
        try:
            objects[int(line2[2:7])] = (name.strip(), Satrec.twoline2rv(line1, line2))
        except (ValueError, IndexError):
            continue
    return objects


CATALOG_OBJECTS = _load_catalog()
requires_catalog = pytest.mark.skipif(
    not CATALOG_OBJECTS, reason="bundled catalog not present"
)


# ============================================================== REFERENCE
def reference_separation(sat_a, sat_b, epoch, offset_s):
    """Separation in km at one instant. Deliberately simple and slow."""
    when = epoch + timedelta(seconds=float(offset_s))
    ra, _ = astro.propagate(sat_a, when)
    rb, _ = astro.propagate(sat_b, when)
    if ra is None or rb is None:
        return float("inf")
    return float(np.linalg.norm(ra - rb))


def reference_tca(sat_a, sat_b, epoch, window_s, coarse_step_s=5.0, tol_s=1e-3):
    """
    Independent TCA search: dense uniform scan, ternary-refine EVERY local
    minimum, take the best.

    Shares no TCA logic with production. Ternary section is used rather than
    Brent so the refinement algorithm differs too — if both implementations had
    a bug in the same minimizer, agreement would prove nothing. (Both call the
    same SGP4 propagator, which is the point: SGP4 is the shared ground truth,
    the search over it is what is being compared.)

    WHY EVERY LOCAL MINIMUM, NOT JUST THE SAMPLED ARGMIN
    ----------------------------------------------------
    An earlier version of this reference refined only the global minimum of the
    coarse scan. That is the very aliasing bug this project fixed in production:
    at 14 km/s a 10 s step covers 140 km, so a sample landing near the bottom of
    a shallow minimum can read lower than any sample near the bottom of the true
    deepest one. On NORAD 25544 vs 48907 that made the reference report 28.22 km
    at t=17940 s while the true closest approach — confirmed by an independent
    1 s brute-force scan — is 23.69 km at t=20723 s, which is what production
    reported. The reference was wrong, not production.

    Refining every bracketed minimum removes the aliasing entirely: the answer no
    longer depends on which sample happened to land closest to a bottom.
    """
    steps = int(window_s / coarse_step_s) + 1
    offsets = np.linspace(0.0, window_s, steps)

    # Vectorized SGP4 for the scan only. The search logic below is independent.
    ra, _, ok_a = astro.propagate_series(sat_a, epoch, offsets)
    rb, _, ok_b = astro.propagate_series(sat_b, epoch, offsets)
    valid = ok_a & ok_b
    separations = np.full(steps, np.inf)
    separations[valid] = np.linalg.norm(ra[valid] - rb[valid], axis=1)
    if not np.isfinite(separations).any():
        return float("nan"), float("inf")

    interior = np.flatnonzero(
        (separations[1:-1] <= separations[:-2])
        & (separations[1:-1] <= separations[2:])
    ) + 1
    brackets = sorted(set(interior.tolist()) | {0, steps - 1})

    def ternary(low, high):
        while high - low > tol_s:
            third = (high - low) / 3.0
            m1, m2 = low + third, high - third
            if reference_separation(sat_a, sat_b, epoch, m1) <                reference_separation(sat_a, sat_b, epoch, m2):
                high = m2
            else:
                low = m1
        t = 0.5 * (low + high)
        return t, reference_separation(sat_a, sat_b, epoch, t)

    best_t, best_d = float("nan"), float("inf")
    for index in brackets:
        if not np.isfinite(separations[index]):
            continue
        t, d = ternary(offsets[max(index - 1, 0)], offsets[min(index + 1, steps - 1)])
        if d < best_d:
            best_t, best_d = t, d
    return best_t, best_d


def _pairs_under_gate(primary_norad, gate_km, window_s, limit,
                      min_separation_km=0.5):
    """
    Find real conjunction pairs to test, using the production coarse screen.

    Pairs closer than `min_separation_km` at every sample are EXCLUDED, and that
    exclusion is essential rather than convenient. The public catalog lists the
    ISS modules (Unity, Zvezda, Destiny, Poisk) as separate objects sharing one
    element set, so their separation is identically 0.000 km with zero relative
    velocity. For such a pair the separation function is flat, every instant is
    a minimum, and "the" time of closest approach does not exist — two correct
    minimizers legitimately return times thousands of seconds apart. Comparing
    TCA on a degenerate pair tests nothing.

    Co-location is handled in production by
    `isdmaas_core.screening.STATUS_CO_LOCATED`; see test_co_location.py.
    """
    primary = CATALOG_OBJECTS[primary_norad][1]
    catalog = [(n, name, sat) for n, (name, sat) in CATALOG_OBJECTS.items()
               if n != primary_norad]
    candidates, _, _ = astro.screen_catalog(
        primary, catalog, EPOCH, window_s, gate_km=gate_km, coarse_step_s=30.0
    )
    return [c for c in candidates
            if c.coarse_miss_km >= min_separation_km][:limit]


# ======================================================= AGREEMENT TESTS
@requires_catalog
@pytest.mark.parametrize("primary_norad", [25544, 39634])
def test_production_tca_matches_independent_reference(primary_norad):
    """
    For real conjunction pairs, the production search and the independent
    reference must agree within the derived thresholds.
    """
    if primary_norad not in CATALOG_OBJECTS:
        pytest.skip(f"NORAD {primary_norad} not in the bundled catalog")

    window_s = 6 * 3600.0
    candidates = _pairs_under_gate(primary_norad, gate_km=200.0,
                                   window_s=window_s, limit=6)
    if not candidates:
        pytest.skip("no candidate pairs in the bundled catalog for this primary")

    primary = CATALOG_OBJECTS[primary_norad][1]
    compared = 0
    for candidate in candidates:
        secondary = CATALOG_OBJECTS[candidate.norad][1]

        production = astro.find_primary_close_approach(
            primary, secondary, EPOCH, window_s, coarse_step_s=30.0
        )
        if production is None:
            continue
        ref_tca, ref_miss = reference_tca(primary, secondary, EPOCH, window_s)

        # Both searches must find the SAME minimum, not merely a minimum.
        assert abs(production.tca_offset_s - ref_tca) < TCA_TIME_TOLERANCE_S, (
            f"NORAD {candidate.norad}: production TCA {production.tca_offset_s:.6f}s "
            f"vs reference {ref_tca:.6f}s "
            f"(delta {abs(production.tca_offset_s - ref_tca) * 1000:.3f} ms)"
        )
        tolerance = max(MISS_ABS_TOLERANCE_KM, MISS_REL_TOLERANCE * ref_miss)
        assert abs(production.miss_km - ref_miss) < tolerance, (
            f"NORAD {candidate.norad}: production miss {production.miss_km:.9f} km "
            f"vs reference {ref_miss:.9f} km"
        )
        compared += 1

    assert compared >= 1, "no pairs were actually compared"


@requires_catalog
def test_relative_position_and_velocity_agree_at_tca():
    """
    Agreement on the scalar miss distance is not enough — the full relative
    state must match, or a downstream encounter-plane projection could still be
    built from the wrong geometry.
    """
    candidates = _pairs_under_gate(25544, gate_km=200.0, window_s=6 * 3600.0, limit=3)
    if not candidates:
        pytest.skip("no candidate pairs available")
    primary = CATALOG_OBJECTS[25544][1]

    for candidate in candidates:
        secondary = CATALOG_OBJECTS[candidate.norad][1]
        production = astro.find_primary_close_approach(
            primary, secondary, EPOCH, 6 * 3600.0, coarse_step_s=30.0
        )
        if production is None:
            continue
        ref_tca, _ = reference_tca(primary, secondary, EPOCH, 6 * 3600.0)
        when = EPOCH + timedelta(seconds=ref_tca)
        ra, va = astro.propagate(primary, when)
        rb, vb = astro.propagate(secondary, when)

        production_rel_pos = production.r_primary - production.r_secondary
        production_rel_vel = production.v_primary - production.v_secondary
        reference_rel_pos = ra - rb
        reference_rel_vel = va - vb

        # First-order in the timing agreement — see the module docstring.
        speed = float(np.linalg.norm(reference_rel_vel))
        position_tolerance = max(REL_POSITION_FLOOR_KM,
                                 speed * TCA_TIME_TOLERANCE_S)
        assert np.allclose(production_rel_pos, reference_rel_pos,
                           atol=position_tolerance), (
            f"relative position mismatch for NORAD {candidate.norad}: "
            f"production {production_rel_pos} vs reference {reference_rel_pos}, "
            f"tolerance {position_tolerance * 1000:.1f} m at "
            f"{speed:.3f} km/s closing speed"
        )
        assert np.allclose(production_rel_vel, reference_rel_vel,
                           atol=REL_VELOCITY_REL_TOLERANCE * max(speed, 1.0)), (
            f"relative velocity mismatch for NORAD {candidate.norad}"
        )


@requires_catalog
def test_refined_miss_is_never_worse_than_the_coarse_grid():
    """
    Refinement must only ever improve on the sampled minimum.

    A refined value LARGER than the coarse sample means the refinement walked
    out of its bracket — the single most likely way for this pipeline to regress.
    """
    window_s = 12 * 3600.0
    candidates = _pairs_under_gate(39634, gate_km=300.0, window_s=window_s, limit=10)
    if not candidates:
        pytest.skip("no candidate pairs available")
    primary = CATALOG_OBJECTS[39634][1]

    for candidate in candidates:
        secondary = CATALOG_OBJECTS[candidate.norad][1]
        approach = astro.refine_candidate(
            primary, secondary, EPOCH, candidate, gate_km=1e9
        )
        if approach is None:
            continue
        assert approach.miss_km <= candidate.coarse_miss_km + 1e-9, (
            f"NORAD {candidate.norad}: refined {approach.miss_km:.6f} km is worse "
            f"than the coarse sample {candidate.coarse_miss_km:.6f} km"
        )


@requires_catalog
def test_refinement_materially_improves_on_the_grid():
    """
    The refinement must actually be doing work.

    A coarse 30 s grid at LEO closing speeds is wrong by up to
    v_rel * step / 2 ~ 200 km. If refinement changed nothing, the pipeline would
    have silently regressed to grid sampling — the original defect.
    """
    window_s = 12 * 3600.0
    candidates = _pairs_under_gate(39634, gate_km=300.0, window_s=window_s, limit=15)
    if not candidates:
        pytest.skip("no candidate pairs available")
    primary = CATALOG_OBJECTS[39634][1]

    improvements = []
    for candidate in candidates:
        secondary = CATALOG_OBJECTS[candidate.norad][1]
        approach = astro.refine_candidate(primary, secondary, EPOCH, candidate,
                                          gate_km=1e9)
        if approach is not None:
            improvements.append(candidate.coarse_miss_km - approach.miss_km)

    assert improvements, "no candidates refined"
    assert max(improvements) > 1.0, (
        f"refinement changed the miss distance by at most "
        f"{max(improvements):.6f} km across {len(improvements)} pairs; the "
        f"refinement stage appears to be inert"
    )


# ================================================== ANALYTIC EDGE CASES
def test_tca_of_two_objects_on_a_known_linear_approach():
    """
    A constructed geometry with an analytically known answer.

    Two points in straight-line relative motion have TCA at
    t* = -(dr . dv) / (dv . dv) exactly. This is independent of SGP4 entirely
    and pins down the linear solver used near the refined minimum.
    """
    from phase7_collision import find_tca

    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    # Secondary approaches, passes at 2 km, and recedes.
    r2 = np.array([7000.0, -75.0, 2.0])
    v2 = np.array([0.0, 7.5 + 0.5, 0.0])

    dr, dv = r2 - r1, v2 - v1
    expected_t = -float(dr @ dv) / float(dv @ dv)
    expected_miss = float(np.linalg.norm(dr + dv * expected_t))

    tca, miss, _, _ = find_tca(r1, v1, r2, v2, t_window_s=600.0)
    assert tca == pytest.approx(expected_t, abs=1e-6)
    assert miss == pytest.approx(expected_miss, rel=1e-9)


def test_tca_in_the_past_is_reported_as_negative():
    """
    A closest approach that has already happened must resolve to a negative
    time, not be clamped to zero. Clamping was the original defect: it made the
    engine report the sampled separation as the miss distance.
    """
    from phase7_collision import find_tca

    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    r2 = np.array([7000.0, 40.0, 1.0])
    v2 = np.array([0.0, 7.5, 0.1])          # already separating

    tca, miss, _, _ = find_tca(r1, v1, r2, v2, t_window_s=600.0)
    assert tca < 0.0
    assert miss < float(np.linalg.norm(r2 - r1))


def test_parallel_motion_has_no_finite_closest_approach():
    """Zero relative velocity must not divide by zero."""
    from phase7_collision import find_tca

    r1 = np.array([7000.0, 0.0, 0.0])
    v = np.array([0.0, 7.5, 0.0])
    r2 = r1 + np.array([0.0, 0.0, 5.0])
    tca, miss, _, _ = find_tca(r1, v, r2, v, t_window_s=600.0)
    assert math.isfinite(tca) and math.isfinite(miss)
    assert miss == pytest.approx(5.0, rel=1e-12)


@requires_catalog
def test_local_minima_bracketing_finds_every_pass():
    """
    Over a long window a pair produces one local minimum per approach. The
    bracketing must find several, not collapse to the global minimum only —
    otherwise a nearer later pass would be missed entirely.
    """
    if 25544 not in CATALOG_OBJECTS or 39634 not in CATALOG_OBJECTS:
        pytest.skip("required objects not in the bundled catalog")
    approaches = astro.find_close_approaches(
        CATALOG_OBJECTS[25544][1], CATALOG_OBJECTS[39634][1],
        EPOCH, 24 * 3600.0, gate_km=5000.0, coarse_step_s=30.0, max_results=8,
    )
    assert len(approaches) >= 2, (
        f"only {len(approaches)} approach(es) found in 24 h; distinct passes are "
        f"being merged or discarded"
    )
    times = sorted(a.tca_offset_s for a in approaches)
    assert all(b - a > 60.0 for a, b in zip(times, times[1:], strict=False)), (
        "two reported approaches are within a minute of each other; they are "
        "the same pass counted twice"
    )
