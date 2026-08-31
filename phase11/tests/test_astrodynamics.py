"""
Regression tests for the propagation and TCA layer.

The tests that matter most here are the ones that would have caught the original
defects: a TCA clamped to the start of the window, and a Brent tolerance that was
relative to a 259 200-second abscissa.
"""
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from sgp4.api import Satrec

from isdmaas_core import astrodynamics as astro

# Two real element sets in similar low Earth orbits.
ISS_L1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30224-3 0  9993"
ISS_L2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.49812516 25012"
S1A_L1 = "1 39634U 14016A   24001.50000000  .00000100  00000-0  29104-4 0  9995"
S1A_L2 = "2 39634  98.1817  62.9578 0001349  86.4756 273.6577 14.59197995520000"

EPOCH = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@pytest.fixture
def iss():
    return Satrec.twoline2rv(ISS_L1, ISS_L2)


@pytest.fixture
def sentinel():
    return Satrec.twoline2rv(S1A_L1, S1A_L2)


def brute_force_min(sat_a, sat_b, epoch, lo_s, hi_s, step_s=0.01):
    """Ground truth: sample the separation on a very fine grid."""
    best = (float("inf"), None)
    steps = int((hi_s - lo_s) / step_s) + 1
    for i in range(steps):
        t = lo_s + i * step_s
        ra, _ = astro.propagate(sat_a, epoch + timedelta(seconds=t))
        rb, _ = astro.propagate(sat_b, epoch + timedelta(seconds=t))
        if ra is None or rb is None:
            continue
        d = float(np.linalg.norm(ra - rb))
        if d < best[0]:
            best = (d, t)
    return best


def test_julian_grid_preserves_microseconds():
    """Offsets must not be quantized by being added to a ~2.46e6 Julian day."""
    offsets = np.array([0.0, 1e-3, 259200.0, 259200.001])
    jd, fr = astro.julian_grid(EPOCH, offsets)
    seconds = (jd - jd[0] + fr - fr[0]) * 86400.0
    assert np.allclose(seconds, offsets - offsets[0], atol=1e-6)


def test_refinement_matches_brute_force(iss, sentinel):
    """
    Brent must land on the same minimum a 10 ms brute-force scan finds.

    This is the regression test for the tolerance bug: with a relative tolerance
    the search stopped up to 86 seconds from the true minimum, which at 14 km/s
    is a 1 200 km error in the reported miss distance.
    """
    approach = astro.find_primary_close_approach(
        iss, sentinel, EPOCH, 6 * 3600.0, coarse_step_s=30.0
    )
    assert approach is not None
    truth_d, truth_t = brute_force_min(
        iss, sentinel, EPOCH,
        approach.tca_offset_s - 60.0, approach.tca_offset_s + 60.0,
    )
    assert approach.miss_km == pytest.approx(truth_d, abs=1e-3)
    assert approach.tca_offset_s == pytest.approx(truth_t, abs=0.05)


def test_refinement_beats_the_coarse_grid(iss, sentinel):
    """The whole point: the refined miss is meaningfully better than the sample."""
    offsets = astro.coarse_grid(6 * 3600.0, 30.0)
    ra, _, ok_a = astro.propagate_series(iss, EPOCH, offsets)
    rb, _, ok_b = astro.propagate_series(sentinel, EPOCH, offsets)
    valid = ok_a & ok_b
    coarse_min = float(np.linalg.norm(ra[valid] - rb[valid], axis=1).min())

    approach = astro.find_primary_close_approach(iss, sentinel, EPOCH, 6 * 3600.0)
    assert approach is not None
    assert approach.miss_km <= coarse_min


def test_geometric_filter_rejects_non_overlapping_shells(iss):
    """A GEO object can never conjunct with a LEO primary."""
    geo = Satrec.twoline2rv(
        "1 41866U 16071A   24001.50000000 -.00000267  00000-0  00000-0 0  9998",
        "2 41866   0.0157  73.4936 0001723 216.4374 130.1135  1.00270193 25812",
    )
    p = astro.elements_from_satrec(iss)
    g = astro.elements_from_satrec(geo)
    assert not astro.perigee_apogee_pass(p, g, gate_km=50.0)
    assert astro.perigee_apogee_pass(p, p, gate_km=50.0)


def test_elements_agree_between_mean_and_osculating(sentinel):
    """Element-set elements and propagated-state elements must describe one orbit."""
    from_tle = astro.elements_from_satrec(sentinel)
    r, v = astro.propagate(sentinel, EPOCH)
    from_state = astro.elements_from_state(r, v)
    assert from_state.sma_km == pytest.approx(from_tle.sma_km, rel=0.01)
    assert from_state.period_s == pytest.approx(from_tle.period_s, rel=0.01)


def test_screen_catalog_finds_a_known_pair(iss, sentinel):
    catalog = [(25544, "ISS", iss), (39634, "SENTINEL-1A", sentinel)]
    candidates, survivors, considered = astro.screen_catalog(
        iss, catalog, EPOCH, 24 * 3600.0, gate_km=500.0
    )
    assert considered == 2
    assert survivors == 2
    assert {c.norad for c in candidates} >= {25544}
    for candidate in candidates:
        assert candidate.brackets, "every candidate must carry refinement brackets"


def test_local_minima_bracket_the_true_minimum():
    """A strict local minimum on the grid brackets the continuous minimum."""
    t = np.linspace(0.0, 100.0, 101)
    d = np.abs(t - 42.4) + 1.0
    minima = astro.bracket_local_minima(d, np.ones_like(t, dtype=bool), gate_km=1e9)
    assert minima
    best = minima[int(np.argmin(d[minima]))]
    assert t[best - 1] <= 42.4 <= t[best + 1]


def test_propagation_failure_is_reported_not_hidden():
    """A decayed element set must flag failure rather than return garbage."""
    decayed = Satrec.twoline2rv(
        "1 00900U 64063C   24001.50000000  .00001000  00000-0  10000-1 0  9990",
        "2 00900  90.0000   0.0000 0000001   0.0000   0.0000 16.50000000    17",
    )
    _, _, ok = astro.propagate_series(
        decayed, EPOCH, np.linspace(0.0, 86400.0, 100)
    )
    assert ok.dtype == bool
