"""
Phase 5.5 shared utilities.
Vectorized orbital elements, RTN frame math (NumPy, mirrors phase5_TRAINER_rtn.py),
cubic-Hermite ephemeris interpolation, TLE epoch parsing.
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from config import MU


# ---------------------------------------------------------------- elements
def rv_to_elements_vec(r, v):
    """r,v: (N,3) km, km/s  ->  dict of (N,) arrays. Degrees for angles."""
    rn = np.linalg.norm(r, axis=1)
    vn = np.linalg.norm(v, axis=1)
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1)
    k = np.array([0.0, 0.0, 1.0])
    n = np.cross(np.broadcast_to(k, r.shape), h)
    nn = np.linalg.norm(n, axis=1)

    e_vec = np.cross(v, h) / MU - r / rn[:, None]
    e = np.linalg.norm(e_vec, axis=1)

    energy = 0.5 * vn**2 - MU / rn
    a = -MU / (2 * energy)
    inc = np.degrees(np.arccos(np.clip(h[:, 2] / hn, -1, 1)))

    raan = np.degrees(np.arccos(np.clip(n[:, 0] / np.maximum(nn, 1e-12), -1, 1)))
    raan = np.where(n[:, 1] < 0, 360 - raan, raan)
    raan = np.where(nn > 1e-8, raan, 0.0)

    argp = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", n, e_vec) / np.maximum(nn * e, 1e-12), -1, 1)))
    argp = np.where(e_vec[:, 2] < 0, 360 - argp, argp)
    argp = np.where((nn > 1e-8) & (e > 1e-8), argp, 0.0)

    ta = np.degrees(np.arccos(np.clip(
        np.einsum("ij,ij->i", e_vec, r) / np.maximum(e * rn, 1e-12), -1, 1)))
    rdotv = np.einsum("ij,ij->i", r, v)
    ta = np.where(rdotv < 0, 360 - ta, ta)
    ta = np.where(e > 1e-8, ta, 0.0)

    return {"semi_major_axis": a, "eccentricity": e, "inclination": inc,
            "raan": raan, "arg_perigee": argp, "true_anomaly": ta,
            "specific_energy": energy, "angular_momentum_mag": hn}


# ---------------------------------------------------------------- RTN
def rtn_basis_np(r, v):
    """(N,3),(N,3) -> R,S,W unit vectors. Same convention as trainer."""
    R = r / np.linalg.norm(r, axis=1, keepdims=True)
    W = np.cross(r, v); W = W / np.linalg.norm(W, axis=1, keepdims=True)
    S = np.cross(W, R)
    return R, S, W


def eci_to_rtn_np(vec, ref_r, ref_v):
    """Project ECI vectors (N,3) onto baseline RTN frame -> (N,3)."""
    R, S, W = rtn_basis_np(ref_r, ref_v)
    return np.stack([np.einsum("ij,ij->i", vec, R),
                     np.einsum("ij,ij->i", vec, S),
                     np.einsum("ij,ij->i", vec, W)], axis=1)


# ---------------------------------------------------------------- interpolation
def hermite_interp(t_query, t_grid, pos, vel):
    """
    Cubic Hermite interpolation of an ephemeris using position AND velocity.
    With 10 s POE sampling this is sub-millimeter — far below the 5 cm POD error.

    t_query: (M,) float seconds   t_grid: (N,) float seconds (sorted, ~uniform)
    pos, vel: (N,3) km, km/s      returns (M,3) pos_q, (M,3) vel_q
    """
    idx = np.searchsorted(t_grid, t_query, side="right") - 1
    idx = np.clip(idx, 0, len(t_grid) - 2)
    t0, t1 = t_grid[idx], t_grid[idx + 1]
    h = (t1 - t0)[:, None]
    s = ((t_query - t0) / (t1 - t0))[:, None]

    p0, p1 = pos[idx], pos[idx + 1]
    m0, m1 = vel[idx] * h, vel[idx + 1] * h            # scale slopes to unit interval

    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    p_q = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

    d00 = 6 * s**2 - 6 * s
    d10 = 3 * s**2 - 4 * s + 1
    d01 = -6 * s**2 + 6 * s
    d11 = 3 * s**2 - 2 * s
    v_q = (d00 * p0 + d10 * m0 + d01 * p1 + d11 * m1) / h
    return p_q, v_q


# ---------------------------------------------------------------- TLE helpers
def tle_epoch(t1: str) -> datetime:
    yy = int(t1[18:20]); yr = 2000 + yy if yy < 57 else 1900 + yy
    return datetime(yr, 1, 1, tzinfo=timezone.utc) + timedelta(days=float(t1[20:32]) - 1)


def load_tles(csv_path):
    """csv with columns epoch?,tle1,tle2 -> sorted DataFrame with epoch datetime."""
    df = pd.read_csv(csv_path)
    if "epoch" not in df.columns or df["epoch"].isna().any():
        df["epoch"] = df["tle1"].apply(tle_epoch)
    else:
        df["epoch"] = pd.to_datetime(df["epoch"], utc=True, format="mixed")
    return df.sort_values("epoch").reset_index(drop=True)


# ---------------------------------------------------------------- misc
def unix_seconds(ts_series):
    """tz-aware datetime Series -> float unix seconds (pandas-version-proof)."""
    import pandas as pd
    return (ts_series - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy()


def sgp4_at(sats, k_idx, t_unix):
    """Propagate sats[k_idx[i]] to t_unix[i]. Returns P(N,3), V(N,3), ok(N,)."""
    JD0 = 2440587.5
    N = len(t_unix)
    P = np.full((N, 3), np.nan); V = np.full((N, 3), np.nan)
    for i in range(N):
        jd = JD0 + t_unix[i] / 86400.0
        fr = jd % 1.0; jd = np.floor(jd)
        err, r, v = sats[int(k_idx[i])].sgp4(jd, fr)
        if err == 0:
            P[i], V[i] = r, v
    ok = np.isfinite(P[:, 0])
    return P, V, ok
