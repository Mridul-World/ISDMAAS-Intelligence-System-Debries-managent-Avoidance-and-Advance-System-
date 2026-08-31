"""
screening.py — the one conjunction-screening implementation.

/screen, /monitor, /assess and /autonomous each carried their own copy of the
same twelve lines: propagate, guess a TCA on a coarse grid, build a covariance
from half the screening window, assess. The copies had drifted, so the same
conjunction could be reported with different miss distances depending on which
endpoint asked. There is now one implementation, and the endpoints differ only
in how they present its output.

Every result here uses the refined TCA from `astrodynamics`, and the secondary's
covariance is grown over the ACTUAL time to that TCA rather than a fixed
fraction of the search window.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from sgp4.api import Satrec

from . import astrodynamics as astro
from .logging_config import get_logger

log = get_logger("isdmaas.screening")

# Position uncertainty (1 sigma, km, radial/along-track/cross-track) for an
# object whose only tracking source is a public element set. The service used to
# assume 50 m radial here — a precise-orbit figure applied to TLE-derived states,
# which put a 1 km miss at Pc ~1e-19 and made every real conjunction look safe.
TLE_PRIMARY_SIGMA_KM = (0.3, 1.0, 0.3)

# Fallback uncertainty for a primary with a precise-orbit solution that does not
# carry its own per-axis sigma. Prefer the determined `rtn_sigma_km` from the
# state file whenever it is present: a real Sentinel-1A determination reports
# 3.13 km along-track, sixty times this figure, and assuming the tighter value
# understates Pc by orders of magnitude.
CALIBRATED_PRIMARY_SIGMA_KM = (0.02, 0.05, 0.02)


@dataclass
class Conjunction:
    """One assessed close approach, ready for serialization."""
    primary_norad: Optional[int]
    primary_name: str
    secondary_norad: int
    secondary_name: str
    tca_utc: str
    tca_hours: float
    miss_km: float
    relative_speed_kms: float
    pc: Optional[float]
    pc_chan: Optional[float]
    pc_methods_agree: Optional[bool]
    mahalanobis: Optional[float]
    risk: str
    covariance_source: str

    def as_dict(self) -> Dict:
        return {
            "primary_norad": self.primary_norad,
            "primary_name": self.primary_name,
            "secondary_norad": self.secondary_norad,
            "secondary_name": self.secondary_name,
            "tca_utc": self.tca_utc,
            "tca_h": round(self.tca_hours, 3),
            "miss_km": round(self.miss_km, 4),
            "rel_speed_kms": round(self.relative_speed_kms, 4),
            "pc": self.pc,
            "pc_chan_crosscheck": self.pc_chan,
            "pc_methods_agree": self.pc_methods_agree,
            "mahalanobis": self.mahalanobis,
            "risk": self.risk,
            "covariance_source": self.covariance_source,
        }


def assess_approach(
    approach: astro.CloseApproach,
    primary_name: str,
    secondary_norad: int,
    secondary_name: str,
    hbr_km: float,
    primary_sigma_km: Sequence[float] = TLE_PRIMARY_SIGMA_KM,
    covariance_source: str = "tle_scale_model",
    primary_norad: Optional[int] = None,
) -> Conjunction:
    """Turn a refined close approach into a full risk assessment."""
    from phase7_collision import (
        assess_conjunction,
        rtn_to_eci_cov,
        secondary_covariance_rtn,
    )

    prop_time_s = max(approach.tca_offset_s, 0.0)
    Cp = rtn_to_eci_cov(
        np.diag(np.asarray(primary_sigma_km, dtype=float) ** 2),
        approach.r_primary,
        approach.v_primary,
    )
    C2 = rtn_to_eci_cov(
        secondary_covariance_rtn(prop_time_s),
        approach.r_secondary,
        approach.v_secondary,
    )
    result = assess_conjunction(
        approach.r_primary, approach.v_primary, Cp,
        approach.r_secondary, approach.v_secondary, C2,
        hbr_km, tca_already_refined=True,
    )
    return Conjunction(
        primary_norad=primary_norad,
        primary_name=primary_name,
        secondary_norad=secondary_norad,
        secondary_name=secondary_name,
        tca_utc=approach.tca.isoformat(),
        tca_hours=approach.tca_offset_s / 3600.0,
        miss_km=approach.miss_km,
        relative_speed_kms=approach.relative_speed_kms,
        pc=result["pc"],
        pc_chan=result["pc_chan_crosscheck"],
        pc_methods_agree=result["pc_methods_agree"],
        mahalanobis=result["mahalanobis"],
        risk=result["risk_level"],
        covariance_source=covariance_source,
    )


@dataclass
class ScreenResult:
    conjunctions: List[Conjunction]
    objects_considered: int
    objects_after_geometric_filter: int
    coarse_candidates: int
    scan_seconds: float


def screen_primary(
    primary_sat: Satrec,
    primary_name: str,
    catalog: Sequence[Tuple[int, str, Satrec]],
    epoch: datetime,
    window_s: float,
    gate_km: float,
    hbr_km: float,
    primary_sigma_km: Sequence[float] = TLE_PRIMARY_SIGMA_KM,
    covariance_source: str = "tle_scale_model",
    primary_norad: Optional[int] = None,
    coarse_step_s: float = 30.0,
    max_refinements: int = 400,
) -> ScreenResult:
    """
    Screen one primary against a catalog and assess every hit.

    `max_refinements` bounds the work an unauthenticated caller can trigger. The
    candidates are refined in order of coarse miss distance, so the bound removes
    the least interesting objects first and never silently drops the close ones.
    """
    import time

    started = time.monotonic()
    candidates, survivors, considered = astro.screen_catalog(
        primary_sat, catalog, epoch, window_s, gate_km, coarse_step_s
    )
    by_norad = {norad: sat for norad, _, sat in catalog}

    conjunctions: List[Conjunction] = []
    for candidate in candidates[:max_refinements]:
        if primary_norad is not None and candidate.norad == primary_norad:
            continue
        secondary = by_norad.get(candidate.norad)
        if secondary is None:
            continue
        approach = astro.refine_candidate(
            primary_sat, secondary, epoch, candidate, gate_km
        )
        if approach is None:
            continue
        try:
            conjunctions.append(
                assess_approach(
                    approach, primary_name, candidate.norad, candidate.name,
                    hbr_km, primary_sigma_km, covariance_source, primary_norad,
                )
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            log.debug("assessment failed for %s: %s", candidate.norad, exc)
            continue

    # Rank by risk: Pc first where it exists, then miss distance. A conjunction
    # whose Pc is undetermined sorts by geometry rather than vanishing.
    conjunctions.sort(key=lambda c: (-(c.pc or 0.0), c.miss_km))
    return ScreenResult(
        conjunctions=conjunctions,
        objects_considered=considered,
        objects_after_geometric_filter=survivors,
        coarse_candidates=len(candidates),
        scan_seconds=round(time.monotonic() - started, 2),
    )


def assess_pair(
    primary_sat: Satrec,
    primary_name: str,
    secondary_sat: Satrec,
    secondary_name: str,
    secondary_norad: int,
    epoch: datetime,
    window_s: float,
    hbr_km: float,
    primary_sigma_km: Sequence[float] = TLE_PRIMARY_SIGMA_KM,
    covariance_source: str = "tle_scale_model",
    primary_norad: Optional[int] = None,
    coarse_step_s: float = 30.0,
) -> Optional[Conjunction]:
    """Assess the closest approach between two specific objects in the window."""
    approach = astro.find_primary_close_approach(
        primary_sat, secondary_sat, epoch, window_s, coarse_step_s=coarse_step_s
    )
    if approach is None:
        return None
    return assess_approach(
        approach, primary_name, secondary_norad, secondary_name,
        hbr_km, primary_sigma_km, covariance_source, primary_norad,
    )
