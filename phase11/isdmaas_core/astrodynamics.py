"""
astrodynamics.py — accurate propagation, screening and time-of-closest-approach.

Why this module exists
----------------------
The original screening code sampled the relative distance on a coarse grid and
reported the sampled minimum as the miss distance. Two objects in low Earth
orbit close at up to ~15 km/s, so a 900-second grid step moves them 13 500 km
between samples: the reported "miss distance" was a grid artefact, not a
conjunction. Everything downstream — Pc, risk level, the maneuver — inherited
that error.

The fix is the standard two-stage approach:

  1. FILTER    Cheap geometric rejection before any propagation. An object whose
               perigee is above the primary's apogee (plus the screening gate)
               can never come close, so it never gets propagated.

  2. BRACKET   Propagate survivors on a coarse time grid with SatrecArray (the
               vectorized C path). Near a conjunction the relative motion is
               linear, so |dr(t)|^2 is locally a parabola; the grid sample that
               is lower than both neighbours therefore brackets a true local
               minimum within one grid step.

  3. REFINE    Minimize the real SGP4 distance inside each bracket with Brent's
               method. Because the function is locally quadratic, the parabolic
               step converges in a handful of evaluations, and the golden-section
               fallback guarantees it converges at all. Result: TCA to
               milliseconds and a miss distance limited by SGP4 itself rather
               than by the grid.

All times are timezone-aware UTC datetimes. All states are TEME km / km per s,
matching what the SGP4 propagator natively produces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sgp4.api import Satrec, SatrecArray, jday

MU_KM3_S2 = 398600.4418
EARTH_RADIUS_KM = 6378.137
J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

# The golden ratio conjugate, used by the fallback section search.
_GOLDEN = 0.5 * (3.0 - math.sqrt(5.0))


# --------------------------------------------------------------- time handling
def to_julian(when: datetime) -> Tuple[float, float]:
    """Split a UTC datetime into the (jd, fraction) pair SGP4 expects."""
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    else:
        when = when.astimezone(timezone.utc)
    return jday(
        when.year, when.month, when.day, when.hour, when.minute,
        when.second + when.microsecond * 1e-6,
    )


def julian_grid(epoch: datetime, offsets_s: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Julian day/fraction arrays for `epoch + offsets_s`.

    The offsets are added to the fractional part and then renormalized so the
    integer day carries the magnitude. Keeping the fraction inside [0, 1) is what
    preserves microsecond resolution three days out from the epoch — adding
    seconds straight onto a ~2.46e6 Julian day would quantize to ~0.1 s.
    """
    jd0, fr0 = to_julian(epoch)
    fr = fr0 + np.asarray(offsets_s, dtype=float) / 86400.0
    whole = np.floor(fr)
    return np.full(fr.shape, jd0, dtype=float) + whole, fr - whole


# ------------------------------------------------------------------ propagation
def propagate(sat: Satrec, when: datetime) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Single-epoch state. Returns (None, None) if SGP4 reports an error."""
    jd, fr = to_julian(when)
    code, r, v = sat.sgp4(jd, fr)
    if code != 0:
        return None, None
    return np.asarray(r, dtype=float), np.asarray(v, dtype=float)


def propagate_series(
    sat: Satrec, epoch: datetime, offsets_s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate one object across many epochs at C speed.

    Returns (positions (n,3), velocities (n,3), ok (n,) bool). Failed steps are
    NaN-filled and flagged in `ok` rather than dropped, so the caller keeps the
    correspondence with `offsets_s`.
    """
    jds, frs = julian_grid(epoch, offsets_s)
    codes, r, v = SatrecArray([sat]).sgp4(jds, frs)
    ok = codes[0] == 0
    pos, vel = r[0].astype(float), v[0].astype(float)
    pos[~ok] = np.nan
    vel[~ok] = np.nan
    return pos, vel, ok


def propagate_batch(
    sats: Sequence[Satrec], epoch: datetime, offsets_s: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Propagate many objects across many epochs.

    Returns (positions (m,n,3), velocities (m,n,3), ok (m,n) bool).
    """
    if not sats:
        n = len(offsets_s)
        empty = np.empty((0, n, 3))
        return empty, empty, np.empty((0, n), dtype=bool)
    jds, frs = julian_grid(epoch, offsets_s)
    codes, r, v = SatrecArray(list(sats)).sgp4(jds, frs)
    ok = codes == 0
    pos, vel = r.astype(float), v.astype(float)
    pos[~ok] = np.nan
    vel[~ok] = np.nan
    return pos, vel, ok


# ------------------------------------------------------------ orbital elements
@dataclass(frozen=True)
class OrbitElements:
    sma_km: float
    ecc: float
    inc_deg: float
    perigee_alt_km: float
    apogee_alt_km: float
    period_s: float


def elements_from_state(r: np.ndarray, v: np.ndarray) -> OrbitElements:
    """Classical elements from a position/velocity pair (two-body osculating)."""
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    rn = float(np.linalg.norm(r))
    vn = float(np.linalg.norm(v))
    if rn <= 0.0:
        raise ValueError("position vector has zero magnitude")
    energy = 0.5 * vn * vn - MU_KM3_S2 / rn
    if energy >= 0.0:
        raise ValueError("state is not on a bound orbit (specific energy >= 0)")
    a = -MU_KM3_S2 / (2.0 * energy)
    h = np.cross(r, v)
    e_vec = np.cross(v, h) / MU_KM3_S2 - r / rn
    e = float(np.linalg.norm(e_vec))
    hn = float(np.linalg.norm(h))
    inc = math.degrees(math.acos(max(-1.0, min(1.0, h[2] / hn)))) if hn > 0 else 0.0
    return OrbitElements(
        sma_km=a,
        ecc=e,
        inc_deg=inc,
        perigee_alt_km=a * (1.0 - e) - EARTH_RADIUS_KM,
        apogee_alt_km=a * (1.0 + e) - EARTH_RADIUS_KM,
        period_s=2.0 * math.pi * math.sqrt(a ** 3 / MU_KM3_S2),
    )


def elements_from_satrec(sat: Satrec) -> OrbitElements:
    """
    Elements straight from the mean elements in the TLE.

    This costs no propagation, which is what makes the apogee/perigee filter
    cheap enough to run against the whole catalog.
    """
    n_rad_s = float(sat.no_kozai) / 60.0  # sgp4 stores mean motion in rad/min
    if n_rad_s <= 0.0:
        raise ValueError("non-positive mean motion")
    a = (MU_KM3_S2 / (n_rad_s * n_rad_s)) ** (1.0 / 3.0)
    e = float(sat.ecco)
    return OrbitElements(
        sma_km=a,
        ecc=e,
        inc_deg=math.degrees(float(sat.inclo)),
        perigee_alt_km=a * (1.0 - e) - EARTH_RADIUS_KM,
        apogee_alt_km=a * (1.0 + e) - EARTH_RADIUS_KM,
        period_s=2.0 * math.pi / n_rad_s,
    )


def orbital_period_s(sat: Satrec, default: float = 5700.0) -> float:
    try:
        return elements_from_satrec(sat).period_s
    except (ValueError, ZeroDivisionError):
        return default


# ------------------------------------------------------------- geometric filter
def perigee_apogee_pass(
    primary: OrbitElements, secondary: OrbitElements, gate_km: float
) -> bool:
    """
    Hoots' apogee/perigee filter: can these two orbits ever come within gate_km?

    Radial shells that do not overlap (after padding by the gate) can never
    produce a conjunction, whatever the phasing. This rejects the large majority
    of the catalog without a single propagation.
    """
    p_lo = primary.perigee_alt_km - gate_km
    p_hi = primary.apogee_alt_km + gate_km
    return not (secondary.perigee_alt_km > p_hi or secondary.apogee_alt_km < p_lo)


# ---------------------------------------------------------------- TCA refinement
def _brent_minimize(
    f, lo: float, hi: float, tol: float = 1e-3, max_iter: int = 80
) -> Tuple[float, float]:
    """
    Brent's method for the minimum of a scalar function on [lo, hi].

    `tol` is an ABSOLUTE tolerance on the abscissa, not the relative one the
    textbook formulation uses. That matters here: the abscissa is a time offset
    that runs to 259 200 s, so a relative tolerance of 1e-3 would stop the search
    260 seconds from the true minimum — which at 14 km/s closing speed is a
    3 600 km error in the reported miss distance.

    Near a conjunction the squared separation is very nearly a parabola in time,
    so the parabolic-interpolation step lands on the minimum almost immediately;
    the golden-section fallback is what makes it safe when that assumption breaks
    (a propagation failure inside the bracket, a shallow flyby, or a bracket that
    spans two minima). Returns (t_min, f_min).
    """
    a, b = (lo, hi) if lo <= hi else (hi, lo)
    x = w = v = a + _GOLDEN * (b - a)
    fx = fw = fv = f(x)
    d = e = 0.0
    tol1 = max(tol, 1e-9)
    tol2 = 2.0 * tol1

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        if abs(x - mid) <= tol2 - 0.5 * (b - a):
            break

        use_golden = True
        if abs(e) > tol1:
            # Fit a parabola through (v, fv), (w, fw), (x, fx).
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            prev_e, e = e, d
            if abs(p) < abs(0.5 * q * prev_e) and p > q * (a - x) and p < q * (b - x):
                d = p / q
                u = x + d
                if u - a < tol2 or b - u < tol2:
                    d = tol1 if x < mid else -tol1
                use_golden = False
        if use_golden:
            e = (b - x) if x < mid else (a - x)
            d = _GOLDEN * e

        u = x + (d if abs(d) >= tol1 else (tol1 if d > 0 else -tol1))
        fu = f(u)

        if fu <= fx:
            if u < x:
                b = x
            else:
                a = x
            v, w, x = w, x, u
            fv, fw, fx = fw, fx, fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if fu <= fw or w == x:
                v, w = w, u
                fv, fw = fw, fu
            elif fu <= fv or v == x or v == w:
                v, fv = u, fu
    return x, fx


@dataclass(frozen=True)
class CloseApproach:
    """One refined conjunction between two objects."""
    tca: datetime
    tca_offset_s: float
    miss_km: float
    r_primary: np.ndarray
    v_primary: np.ndarray
    r_secondary: np.ndarray
    v_secondary: np.ndarray

    @property
    def relative_speed_kms(self) -> float:
        return float(np.linalg.norm(self.v_primary - self.v_secondary))


StateFn = Callable[[float], Tuple[Optional[np.ndarray], Optional[np.ndarray]]]


def satrec_state_fn(sat: Satrec, epoch: datetime) -> StateFn:
    """Wrap a Satrec as a state function of seconds-since-epoch."""
    jd0, fr0 = to_julian(epoch)

    def state(offset_s: float):
        fr = fr0 + offset_s / 86400.0
        whole = math.floor(fr)
        code, r, v = sat.sgp4(jd0 + whole, fr - whole)
        if code != 0:
            return None, None
        return np.asarray(r, dtype=float), np.asarray(v, dtype=float)

    return state


def refine_between(
    state_a: StateFn,
    state_b: StateFn,
    epoch: datetime,
    bracket_lo_s: float,
    bracket_hi_s: float,
    tol_s: float = 1e-3,
) -> Optional[CloseApproach]:
    """
    Exact minimum separation between two arbitrary state functions in a bracket.

    Taking state functions rather than Satrecs is what lets the safety layer
    refine a POST-MANEUVER trajectory, which is not a catalog element set, with
    the same accuracy as a catalog-to-catalog conjunction.
    """
    def separation(offset_s: float) -> float:
        ra, _ = state_a(offset_s)
        rb, _ = state_b(offset_s)
        if ra is None or rb is None:
            return float("inf")
        return float(np.linalg.norm(ra - rb))

    t_min, d_min = _brent_minimize(separation, bracket_lo_s, bracket_hi_s, tol=tol_s)
    if not math.isfinite(d_min):
        return None
    ra, va = state_a(t_min)
    rb, vb = state_b(t_min)
    if ra is None or rb is None:
        return None
    return CloseApproach(
        tca=epoch + timedelta(seconds=t_min),
        tca_offset_s=t_min,
        miss_km=float(np.linalg.norm(ra - rb)),
        r_primary=ra,
        v_primary=va,
        r_secondary=rb,
        v_secondary=vb,
    )


def refine_close_approach(
    sat_a: Satrec,
    sat_b: Satrec,
    epoch: datetime,
    bracket_lo_s: float,
    bracket_hi_s: float,
    tol_s: float = 1e-3,
) -> Optional[CloseApproach]:
    """
    Find the exact minimum separation between two catalog objects in a bracket.

    `tol_s` of 1 ms bounds the position error at 15 km/s to ~15 m, which is far
    below SGP4's own accuracy — the refinement is no longer the limiting term.
    """
    return refine_between(
        satrec_state_fn(sat_a, epoch),
        satrec_state_fn(sat_b, epoch),
        epoch,
        bracket_lo_s,
        bracket_hi_s,
        tol_s,
    )


def bracket_local_minima(
    distances: np.ndarray, valid: np.ndarray, gate_km: float
) -> List[int]:
    """
    Indices of grid samples that are strict local minima under the gate.

    A sample lower than both neighbours brackets a true minimum of the smooth
    separation function within one grid step. Endpoints count when the sequence
    is descending into them, so a conjunction that peaks just outside the window
    is still caught.
    """
    n = distances.shape[0]
    if n == 0:
        return []
    d = np.where(valid, distances, np.inf)
    hits: List[int] = []
    for i in range(n):
        if not np.isfinite(d[i]) or d[i] > gate_km:
            continue
        left = d[i - 1] if i > 0 else np.inf
        right = d[i + 1] if i < n - 1 else np.inf
        if d[i] <= left and d[i] <= right:
            hits.append(i)
    return hits


def find_close_approaches(
    primary: Satrec,
    secondary: Satrec,
    epoch: datetime,
    window_s: float,
    gate_km: float = 50.0,
    coarse_step_s: float = 30.0,
    max_results: int = 4,
) -> List[CloseApproach]:
    """
    All refined close approaches between two objects inside a time window.

    The coarse step must resolve the *shape* of the separation curve, not the
    gate crossing: near a conjunction the curve is a parabola whose minimum the
    grid brackets within one step, so 30 s is ample for LEO while keeping the
    grid small. Each bracket is then refined to milliseconds.
    """
    n_steps = max(3, int(window_s / max(coarse_step_s, 1.0)) + 1)
    offsets = np.linspace(0.0, window_s, n_steps)
    ra, _, ok_a = propagate_series(primary, epoch, offsets)
    rb, _, ok_b = propagate_series(secondary, epoch, offsets)
    valid = ok_a & ok_b
    if not valid.any():
        return []
    separation = np.full(n_steps, np.inf)
    separation[valid] = np.linalg.norm(ra[valid] - rb[valid], axis=1)

    results: List[CloseApproach] = []
    for idx in bracket_local_minima(separation, valid, gate_km):
        lo = offsets[max(idx - 1, 0)]
        hi = offsets[min(idx + 1, n_steps - 1)]
        if hi <= lo:
            continue
        ca = refine_close_approach(primary, secondary, epoch, lo, hi)
        if ca is not None and ca.miss_km <= gate_km:
            results.append(ca)
    results.sort(key=lambda c: c.miss_km)
    return results[:max_results]


def find_primary_close_approach(
    primary: Satrec,
    secondary: Satrec,
    epoch: datetime,
    window_s: float,
    gate_km: float = 1e9,
    coarse_step_s: float = 30.0,
) -> Optional[CloseApproach]:
    """The single closest approach in the window, refined. None if propagation fails."""
    approaches = find_close_approaches(
        primary, secondary, epoch, window_s, gate_km, coarse_step_s, max_results=1
    )
    return approaches[0] if approaches else None


# -------------------------------------------------------------------- screening
@dataclass
class ScreenCandidate:
    """
    A catalog object that survived the coarse screen.

    `brackets` carries the (lo, hi) second-offsets around each coarse local
    minimum, so the refinement step never has to re-propagate the whole window:
    it goes straight to Brent inside the interval that already brackets the
    minimum. Several brackets are kept because the coarse gate is padded by the
    half-step travel distance, and the deepest sampled minimum is not always the
    one that refines to the smallest true miss.
    """
    norad: int
    name: str
    coarse_miss_km: float
    coarse_offset_s: float
    brackets: List[Tuple[float, float]] = field(default_factory=list)


def _row_local_minima(d: np.ndarray, gate_km: float, keep: int) -> List[int]:
    """Indices of the `keep` deepest strict local minima of a distance row."""
    n = d.shape[0]
    if n < 3:
        idx = int(np.argmin(d))
        return [idx] if math.isfinite(d[idx]) and d[idx] <= gate_km else []
    interior = np.flatnonzero(
        (d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] <= gate_km)
    ) + 1
    ends = [i for i in (0, n - 1) if math.isfinite(d[i]) and d[i] <= gate_km]
    hits = sorted(set(interior.tolist()) | set(ends), key=lambda i: d[i])
    return [i for i in hits if math.isfinite(d[i])][:keep]


def refine_candidate(
    primary: Satrec,
    secondary: Satrec,
    epoch: datetime,
    candidate: ScreenCandidate,
    gate_km: float,
) -> Optional[CloseApproach]:
    """
    Turn a coarse screening hit into an exact close approach.

    Each stored bracket is refined and the smallest result wins. Returns None
    when nothing in the brackets actually falls inside the reporting gate — that
    is the normal outcome for an object that only looked close because of the
    coarse grid's half-step padding.
    """
    brackets = candidate.brackets or [
        (candidate.coarse_offset_s - 60.0, candidate.coarse_offset_s + 60.0)
    ]
    best: Optional[CloseApproach] = None
    for lo, hi in brackets:
        if hi <= lo:
            continue
        approach = refine_close_approach(primary, secondary, epoch, lo, hi)
        if approach is None:
            continue
        if best is None or approach.miss_km < best.miss_km:
            best = approach
    if best is None or best.miss_km > gate_km:
        return None
    return best


def coarse_grid(window_s: float, coarse_step_s: float) -> np.ndarray:
    """Uniform sampling offsets covering [0, window_s]."""
    n_steps = max(3, int(window_s / max(coarse_step_s, 1.0)) + 1)
    return np.linspace(0.0, window_s, n_steps)


def screen_track(
    track_km: np.ndarray,
    track_ok: np.ndarray,
    catalog: Sequence[Tuple[int, str, Satrec]],
    epoch: datetime,
    offsets_s: np.ndarray,
    gate_km: float,
    chunk_size: int = 750,
) -> List[ScreenCandidate]:
    """
    Screen a precomputed primary track against catalog objects.

    Splitting this out from screen_catalog is what lets the safety layer
    re-screen a post-maneuver trajectory: that trajectory has no element set of
    its own, but it is still just a sequence of positions on the same time grid.

    The gate is padded by the distance a pair can cover in half a coarse step, so
    an approach whose true minimum falls between two samples survives to the
    refinement stage instead of being discarded here.
    """
    if not len(catalog) or track_ok.sum() < 3:
        return []
    step = float(offsets_s[1] - offsets_s[0]) if len(offsets_s) > 1 else 1.0
    coarse_gate = gate_km + 16.0 * step * 0.5
    last = len(offsets_s) - 1

    candidates: List[ScreenCandidate] = []
    for start in range(0, len(catalog), chunk_size):
        chunk = catalog[start:start + chunk_size]
        pos, _, ok = propagate_batch([s for _, _, s in chunk], epoch, offsets_s)
        dist = np.linalg.norm(pos - track_km[None, :, :], axis=2)
        dist = np.where(ok & track_ok[None, :], dist, np.inf)
        row_best = dist.min(axis=1)
        for j in np.flatnonzero(row_best <= coarse_gate):
            minima = _row_local_minima(dist[j], coarse_gate, keep=3)
            if not minima:
                continue
            norad, name, _ = chunk[j]
            candidates.append(
                ScreenCandidate(
                    norad=norad,
                    name=name,
                    coarse_miss_km=float(row_best[j]),
                    coarse_offset_s=float(offsets_s[minima[0]]),
                    brackets=[
                        (float(offsets_s[max(i - 1, 0)]),
                         float(offsets_s[min(i + 1, last)]))
                        for i in minima
                    ],
                )
            )
    candidates.sort(key=lambda c: c.coarse_miss_km)
    return candidates


def geometric_filter(
    primary: Satrec,
    catalog: Iterable[Tuple[int, str, Satrec]],
    gate_km: float,
) -> Tuple[List[Tuple[int, str, Satrec]], int]:
    """Apply the apogee/perigee filter. Returns (survivors, objects considered)."""
    entries = list(catalog)
    try:
        p_elem = elements_from_satrec(primary)
    except (ValueError, ZeroDivisionError):
        return [], len(entries)
    survivors = []
    for norad, name, sat in entries:
        try:
            s_elem = elements_from_satrec(sat)
        except (ValueError, ZeroDivisionError):
            continue
        if perigee_apogee_pass(p_elem, s_elem, gate_km):
            survivors.append((norad, name, sat))
    return survivors, len(entries)


def screen_catalog(
    primary: Satrec,
    catalog: Iterable[Tuple[int, str, Satrec]],
    epoch: datetime,
    window_s: float,
    gate_km: float = 50.0,
    coarse_step_s: float = 30.0,
    chunk_size: int = 750,
) -> Tuple[List[ScreenCandidate], int, int]:
    """
    Coarse full-catalog screen against one primary.

    Returns (candidates under the gate, objects that survived the geometric
    filter, objects considered).
    """
    survivors, considered = geometric_filter(primary, catalog, gate_km)
    if not survivors:
        return [], 0, considered
    offsets = coarse_grid(window_s, coarse_step_s)
    rp, _, ok_p = propagate_series(primary, epoch, offsets)
    candidates = screen_track(
        rp, ok_p, survivors, epoch, offsets, gate_km, chunk_size
    )
    return candidates, len(survivors), considered
