"""
================================================================================
PHASE 7.1 — COLLISION PROBABILITY ENGINE (deterministic foundation)
================================================================================
Given two space objects, each with a state (position+velocity, TEME km/km-s) and
a position covariance, compute:
    - TCA      time of closest approach   (refined to sub-second)
    - miss     miss distance at TCA       (km)
    - Pc       probability of collision   (Foster/Chan 2-D method)
    - Mahalanobis distance in the encounter plane
    - risk level   NOMINAL / ELEVATED / HIGH / CRITICAL

PRIMARY object  : AI-corrected state + Phase-6 calibrated covariance.
SECONDARY object: SGP4 state + analytic covariance growth model (debris has no
                  precise truth; this is standard operator practice).

The math (Foster/Chan 2-D Pc):
  1. At TCA, combine covariances:  C = C_primary + C_secondary  (independent).
  2. Project miss vector + C onto the plane perpendicular to relative velocity
     (the encounter plane). Motion along v_rel is ignored (objects flash past).
  3. Pc = integral of the 2-D Gaussian N(miss_2d, C_2d) over a disk of radius
     HBR (combined hard-body radius). Computed by direct polar quadrature and
     cross-checked with Chan's equivalent-area series.

This module is pure astrodynamics + statistics — no ML, runs in milliseconds.
================================================================================
"""
import numpy as np

# risk thresholds (industry-standard Pc bands)
RISK_BANDS = [(1e-7, "NOMINAL"), (1e-5, "ELEVATED"), (1e-4, "HIGH"), (np.inf, "CRITICAL")]


def risk_level(pc):
    for thresh, name in RISK_BANDS:
        if pc < thresh:
            return name
    return "CRITICAL"


# ---------------------------------------------------------------- TCA
def find_tca(r1, v1, r2, v2, t_window_s, n_coarse=4320, refine=True):
    """
    Linear-relative-motion TCA over [0, t_window_s].
    r,v in km, km/s (TEME). Returns (tca_s, miss_km, rel_pos, rel_vel).
    For short windows around a known approach, linear motion is accurate; for
    long screening windows the caller should pass SGP4-propagated states near
    the suspected TCA. Here we do an analytic linear TCA plus optional refine.
    """
    dr = r2 - r1
    dv = v2 - v1
    # analytic linear TCA: d/dt |dr + dv t|^2 = 0  ->  t = -(dr.dv)/(dv.dv)
    dvv = float(dv @ dv)
    t_lin = -float(dr @ dv) / dvv if dvv > 0 else 0.0
    t_lin = max(0.0, min(t_window_s, t_lin))
    if not refine:
        rel = dr + dv * t_lin
        return t_lin, float(np.linalg.norm(rel)), rel, dv
    # refine on a fine grid around t_lin (covers mild curvature if caller
    # supplied curved states; with linear states this just confirms t_lin)
    lo, hi = max(0.0, t_lin - 120), min(t_window_s, t_lin + 120)
    ts = np.linspace(lo, hi, 2401)
    d = np.linalg.norm(dr[None, :] + dv[None, :] * ts[:, None], axis=1)
    k = int(np.argmin(d))
    tca = float(ts[k])
    rel = dr + dv * tca
    return tca, float(np.linalg.norm(rel)), rel, dv


# ---------------------------------------------------------------- encounter plane
def encounter_plane_basis(v_rel):
    """Orthonormal 2-D basis spanning the plane perpendicular to v_rel."""
    vhat = v_rel / np.linalg.norm(v_rel)
    # pick any vector not parallel to vhat
    a = np.array([1.0, 0.0, 0.0]) if abs(vhat[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u1 = a - (a @ vhat) * vhat
    u1 /= np.linalg.norm(u1)
    u2 = np.cross(vhat, u1)
    return u1, u2          # 3-vectors spanning the encounter plane


def project_to_plane(rel_pos, C_combined, v_rel):
    """Project miss vector and 3x3 covariance onto the 2-D encounter plane."""
    u1, u2 = encounter_plane_basis(v_rel)
    P = np.stack([u1, u2], axis=0)          # (2,3)
    miss_2d = P @ rel_pos                    # (2,)
    C_2d = P @ C_combined @ P.T              # (2,2)
    return miss_2d, C_2d


# ---------------------------------------------------------------- Pc integral
def pc_2d_quadrature(miss_2d, C_2d, hbr, n_r=200, n_theta=180):
    """
    Pc = integral over disk radius HBR of N(miss_2d, C_2d).
    Direct polar quadrature in covariance principal-axis frame. Robust for any
    aspect ratio. Returns probability in [0,1].
    """
    # eigen-decompose covariance -> principal axes
    w, Vt = np.linalg.eigh(C_2d)
    w = np.clip(w, 1e-12, None)
    sx, sy = np.sqrt(w[0]), np.sqrt(w[1])
    m = Vt.T @ miss_2d                        # miss in principal frame
    # integrate Gaussian over disk centered at origin (object), Gaussian centered at m
    rr = np.linspace(0, hbr, n_r)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R, TH = np.meshgrid(rr, th, indexing="ij")
    X = R * np.cos(TH); Y = R * np.sin(TH)
    g = (1.0 / (2 * np.pi * sx * sy)) * np.exp(
        -0.5 * (((X - m[0]) ** 2) / sx ** 2 + ((Y - m[1]) ** 2) / sy ** 2))
    # area element r dr dtheta
    dr = rr[1] - rr[0] if n_r > 1 else hbr
    dth = th[1] - th[0] if n_theta > 1 else 2 * np.pi
    pc = float(np.sum(g * R) * dr * dth)
    return min(max(pc, 0.0), 1.0)


def pc_chan(miss_2d, C_2d, hbr, terms=20):
    """
    Chan's analytic series Pc (cross-check). Uses circular-equivalent reduction.
    Valid for the standard 2-D conjunction integral.
    """
    w, Vt = np.linalg.eigh(C_2d)
    w = np.clip(w, 1e-12, None)
    sx, sy = np.sqrt(w[0]), np.sqrt(w[1])
    m = Vt.T @ miss_2d
    # scale to circular: u = HBR^2 / (sx*sy);  v = (mx^2/sx^2 + my^2/sy^2)
    u = hbr ** 2 / (sx * sy)
    v = (m[0] ** 2) / (sx ** 2) + (m[1] ** 2) / (sy ** 2)
    # Chan series:  Pc = exp(-v/2) * sum_{k>=0} (v/2)^k/k! * (1 - exp(-u/2) sum_{j=0}^k (u/2)^j/j!)
    from math import exp, factorial
    pc = 0.0
    for k in range(terms):
        inner = sum((u / 2) ** j / factorial(j) for j in range(k + 1))
        pc += ((v / 2) ** k / factorial(k)) * (1 - exp(-u / 2) * inner)
    pc *= exp(-v / 2)
    return min(max(pc, 0.0), 1.0)


# ---------------------------------------------------------------- secondary covariance model
def secondary_covariance_rtn(prop_time_s, base_sigma_km=(0.05, 0.5, 0.05),
                             growth_along_km_per_day=2.0):
    """
    Analytic covariance growth for a secondary (debris) propagated by SGP4.
    RTN diagonal; along-track grows with propagation time (drag/period error).
    Conservative, documented modeling assumption (debris has no precise truth).
    """
    days = prop_time_s / 86400.0
    sr, st, sn = base_sigma_km
    st = st + growth_along_km_per_day * days
    sr = sr + 0.1 * days
    sn = sn + 0.1 * days
    return np.diag(np.array([sr, st, sn]) ** 2)


def rtn_to_eci_cov(C_rtn, r, v):
    """Rotate an RTN covariance into ECI/TEME using the object's state."""
    R = r / np.linalg.norm(r)
    W = np.cross(r, v); W /= np.linalg.norm(W)
    S = np.cross(W, R)
    Rt = np.stack([R, S, W], axis=1)        # columns = RTN axes in ECI
    return Rt @ C_rtn @ Rt.T


# ---------------------------------------------------------------- top-level
def assess_conjunction(r1, v1, C1_eci, r2, v2, C2_eci, hbr_km,
                       t_window_s=259200):
    """
    Full conjunction assessment between primary (1) and secondary (2).
    States in TEME km, km/s; covariances 3x3 ECI km^2; hbr_km combined radius.
    Returns a dict (CDM-style).
    """
    tca, miss, rel_pos, v_rel = find_tca(r1, v1, r2, v2, t_window_s)
    C_comb = C1_eci + C2_eci
    miss_2d, C_2d = project_to_plane(rel_pos, C_comb, v_rel)

    pc_q = pc_2d_quadrature(miss_2d, C_2d, hbr_km)
    pc_c = pc_chan(miss_2d, C_2d, hbr_km)
    pc = pc_q                                # quadrature is the reported value

    # Mahalanobis distance in encounter plane
    maha = float(np.sqrt(miss_2d @ np.linalg.solve(C_2d, miss_2d)))

    return {
        "tca_s_from_epoch": tca,
        "miss_distance_km": miss,
        "radial_miss_km": float(miss_2d[0]),
        "binormal_miss_km": float(miss_2d[1]),
        "pc": pc,
        "pc_chan_crosscheck": pc_c,
        "pc_methods_agree": bool(abs(pc_q - pc_c) < 0.1 * max(pc_q, 1e-30) + 1e-12),
        "mahalanobis": maha,
        "hbr_km": hbr_km,
        "relative_speed_kms": float(np.linalg.norm(v_rel)),
        "risk_level": risk_level(pc),
        "encounter_plane_cov_km2": C_2d.tolist(),
    }
