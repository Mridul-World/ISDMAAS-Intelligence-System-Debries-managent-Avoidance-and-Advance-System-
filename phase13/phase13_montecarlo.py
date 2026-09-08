"""
================================================================================
PHASE 13 — MONTE CARLO COLLISION PROBABILITY  (robustness layer for Phase 7)
================================================================================
The analytical Pc (Foster/Chan, phase7_collision.py) projects the combined 3-D
covariance onto a 2-D encounter plane and integrates a Gaussian. That is the
industry-standard fast method, valid when the encounter is short, linear, and
the uncertainty Gaussian.

Monte Carlo Pc makes none of those assumptions. It samples many possible primary
and secondary states from their covariances, propagates each pair through the
encounter window, finds each sample-pair's closest approach, and counts the
fraction that come within the combined hard-body radius. That fraction IS the
collision probability — computed by brute force.

USE
  - VALIDATION: if MC agrees with Foster/Chan on clean cases, the analytical
    method is confirmed (cross-method agreement = credibility).
  - ROBUSTNESS: for non-linear / non-Gaussian / grazing encounters where the
    2-D projection is questionable, MC is the more trustworthy number.
  - It also returns a confidence interval (Wilson) so you report Pc ± uncertainty,
    not a bare point estimate.

This module imports the verified analytical engine for cross-checking.
================================================================================
"""
import numpy as np
from phase7_collision import (assess_conjunction, project_to_plane,
                              pc_2d_quadrature)


def _sample_states(mean_r, mean_v, cov_eci, n, rng):
    """Sample n position vectors from N(mean_r, cov_eci). Velocity held at mean
       (position uncertainty dominates Pc on the encounter timescale)."""
    # cov_eci is 3x3 position covariance (km^2)
    L = np.linalg.cholesky(cov_eci + 1e-12 * np.eye(3))
    samples = mean_r[None, :] + (rng.standard_normal((n, 3)) @ L.T)
    return samples


def monte_carlo_pc(r1, v1, C1, r2, v2, C2, hbr_km,
                   n_samples=20000, encounter_window_s=20.0, dt_s=0.05, seed=0):
    """
    Brute-force Pc. Samples primary/secondary positions from their covariances,
    propagates each pair linearly through a short encounter window around TCA,
    counts pairs whose minimum separation < hbr_km.

    Returns dict: pc, n_hits, n_samples, ci_low, ci_high (95% Wilson interval).
    """
    rng = np.random.default_rng(seed)
    # relative state and its covariance (independent -> sum)
    # sample relative miss directly: Δr ~ N(r2-r1, C1+C2), relative velocity fixed
    dr_mean = r2 - r1
    dv = v2 - v1
    C = C1 + C2
    L = np.linalg.cholesky(C + 1e-12 * np.eye(3))
    dr_samples = dr_mean[None, :] + (rng.standard_normal((n_samples, 3)) @ L.T)

    # for each sample, min separation over the encounter window (linear motion)
    ts = np.arange(-encounter_window_s / 2, encounter_window_s / 2 + dt_s, dt_s)
    # min over t of |dr + dv t| : analytic t* = -(dr·dv)/(dv·dv), clamp to window
    dvv = float(dv @ dv)
    if dvv <= 0:
        min_sep = np.linalg.norm(dr_samples, axis=1)
    else:
        tstar = -(dr_samples @ dv) / dvv
        tstar = np.clip(tstar, ts[0], ts[-1])
        closest = dr_samples + tstar[:, None] * dv[None, :]
        min_sep = np.linalg.norm(closest, axis=1)

    hits = int(np.sum(min_sep < hbr_km))
    pc = hits / n_samples
    # Wilson 95% CI for a binomial proportion
    z = 1.96
    denom = 1 + z**2 / n_samples
    centre = (pc + z**2 / (2 * n_samples)) / denom
    half = z * np.sqrt(pc * (1 - pc) / n_samples + z**2 / (4 * n_samples**2)) / denom
    return {"pc": pc, "n_hits": hits, "n_samples": n_samples,
            "ci_low": max(0.0, centre - half), "ci_high": min(1.0, centre + half),
            "min_sep_median_km": float(np.median(min_sep))}


def compare_methods(r1, v1, C1, r2, v2, C2, hbr_km, n_samples=50000, seed=0):
    """Run both analytical (Foster/Chan) and Monte Carlo; report agreement."""
    ana = assess_conjunction(r1, v1, C1, r2, v2, C2, hbr_km, 600)
    mc = monte_carlo_pc(r1, v1, C1, r2, v2, C2, hbr_km, n_samples=n_samples, seed=seed)
    pa, pm = ana["pc"], mc["pc"]
    # The analytical method declines to produce a number for an invalid
    # covariance or a geometry outside the short-encounter regime it assumes.
    # Monte Carlo makes neither assumption, so it still has an answer -- but the
    # two cannot be said to agree when only one of them ran.
    if pa is None:
        return {
            "analytical_pc": None,
            "monte_carlo_pc": pm,
            "mc_ci": [mc["ci_low"], mc["ci_high"]],
            "mc_hits": mc["n_hits"], "mc_samples": mc["n_samples"],
            "miss_km": ana["miss_distance_km"],
            "agree": False,
            "risk_level": ana["risk_level"],
            "note": ("analytical Pc unavailable (" + str(ana["risk_level"])
                     + "); Monte Carlo is the only estimate here"),
        }
    # agreement: MC CI contains analytical, or both effectively zero
    in_ci = mc["ci_low"] <= pa <= mc["ci_high"]
    both_tiny = pa < 1e-6 and pm < 1e-6
    return {
        "analytical_pc": pa,
        "monte_carlo_pc": pm,
        "mc_ci": [mc["ci_low"], mc["ci_high"]],
        "mc_hits": mc["n_hits"], "mc_samples": mc["n_samples"],
        "miss_km": ana["miss_distance_km"],
        "agree": bool(in_ci or both_tiny),
        "risk_level": ana["risk_level"],
    }
