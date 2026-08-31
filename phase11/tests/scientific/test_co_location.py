"""
Scientific validation: co-located objects must not become false alerts.

The public catalog lists several physically-attached objects as separate
entries sharing ONE element set. The ISS is the clearest case: Unity (25575),
Zvezda (26400), Destiny (26700) and Poisk (36086) all carry the ISS element set,
so their computed separation from ISS (25544) is identically 0.000 km with zero
relative velocity, at every instant, forever.

Two consequences, both of which were live defects:

1. `/monitor` alerts when `miss <= alert_miss_km` (default 5 km). Screening the
   ISS produced six 0.000 km "conjunctions" against its own modules and raised
   an alert for every one. An alert system that cries wolf on the primary's own
   structure trains operators to dismiss alerts.

2. There is no time of closest approach for such a pair. The separation function
   is flat, so every instant is a minimum, and two correct minimizers
   legitimately return times thousands of seconds apart. Any TCA reported for
   them is arbitrary.

The fix is to classify rather than to delete: the pair is reported with
`status = CO_LOCATED` and `actionable = False`. Silently dropping it would be
its own failure mode — two genuinely distinct spacecraft flying in close
formation land in the same bucket, and an operator needs to see that.
"""
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sgp4.api import Satrec

from isdmaas_core import astrodynamics as astro
from isdmaas_core import screening as screening_service
from isdmaas_core.screening import (
    STATUS_ASSESSED,
    STATUS_CO_LOCATED,
    STATUS_UNDETERMINED,
)

CATALOG = Path(__file__).resolve().parent.parent.parent / "data" / "catalog_active.tle"
EPOCH = datetime(2026, 7, 1, 0, 0, 0, tzinfo=timezone.utc)

# ISS and four modules that share its element set in the public catalog.
ISS = 25544
ISS_MODULES = [25575, 26400, 26700, 36086]


def _load():
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


CATALOG_OBJECTS = _load()
requires_catalog = pytest.mark.skipif(
    ISS not in CATALOG_OBJECTS, reason="bundled catalog not present"
)


def _assess(primary_norad, secondary_norad):
    primary = CATALOG_OBJECTS[primary_norad][1]
    secondary = CATALOG_OBJECTS[secondary_norad][1]
    approach = astro.find_primary_close_approach(
        primary, secondary, EPOCH, 6 * 3600.0, coarse_step_s=30.0
    )
    if approach is None:
        pytest.skip(f"{primary_norad} vs {secondary_norad} did not propagate")
    return screening_service.assess_approach(
        approach,
        CATALOG_OBJECTS[primary_norad][0],
        secondary_norad,
        CATALOG_OBJECTS[secondary_norad][0],
        hbr_km=0.020,
        primary_norad=primary_norad,
    )


@requires_catalog
@pytest.mark.parametrize("module_norad", ISS_MODULES)
def test_iss_modules_are_classified_co_located(module_norad):
    """Each ISS module must be CO_LOCATED, not a conjunction."""
    if module_norad not in CATALOG_OBJECTS:
        pytest.skip(f"NORAD {module_norad} not in the bundled catalog")
    conjunction = _assess(ISS, module_norad)
    assert conjunction.miss_km < 0.001
    assert conjunction.relative_speed_kms < 1e-6
    assert conjunction.status == STATUS_CO_LOCATED, (
        f"{conjunction.secondary_name} at {conjunction.miss_km} km and "
        f"{conjunction.relative_speed_kms} km/s was classified "
        f"{conjunction.status}"
    )


@requires_catalog
@pytest.mark.parametrize("module_norad", ISS_MODULES)
def test_co_located_pairs_are_not_actionable(module_norad):
    """
    The property `/monitor` gates on. This is the assertion that stops the
    false-alert regression from coming back.
    """
    if module_norad not in CATALOG_OBJECTS:
        pytest.skip(f"NORAD {module_norad} not in the bundled catalog")
    assert _assess(ISS, module_norad).is_actionable is False


@requires_catalog
def test_co_located_pairs_are_reported_not_hidden():
    """
    Classification, not deletion. The pair must still appear in the result with
    an explicit status so an operator can see it — a genuine close formation
    would land here too.
    """
    module = next((m for m in ISS_MODULES if m in CATALOG_OBJECTS), None)
    if module is None:
        pytest.skip("no ISS module in the bundled catalog")
    payload = _assess(ISS, module).as_dict()
    assert payload["status"] == STATUS_CO_LOCATED
    assert payload["actionable"] is False
    assert payload["secondary_norad"] == module
    assert "miss_km" in payload and "rel_speed_kms" in payload


@requires_catalog
def test_a_genuine_conjunction_remains_actionable():
    """
    The guard must not suppress real conjunctions.

    A co-location filter that also silences genuine close approaches would be
    far worse than the false alerts it removes, so this is the counterweight to
    every test above.
    """
    primary = CATALOG_OBJECTS[ISS][1]
    catalog = [(n, name, sat) for n, (name, sat) in CATALOG_OBJECTS.items() if n != ISS]
    result = screening_service.screen_primary(
        primary_sat=primary,
        primary_name=CATALOG_OBJECTS[ISS][0],
        catalog=catalog,
        epoch=EPOCH,
        window_s=6 * 3600.0,
        gate_km=200.0,
        hbr_km=0.020,
        primary_norad=ISS,
        coarse_step_s=30.0,
    )
    genuine = [c for c in result.conjunctions if c.miss_km > 1.0]
    assert genuine, "no genuine conjunctions found to check against"
    assert all(c.status == STATUS_ASSESSED for c in genuine), (
        "a real conjunction was classified as non-actionable: "
        + str([(c.secondary_name, c.miss_km, c.status) for c in genuine[:3]])
    )
    assert all(c.is_actionable for c in genuine)


@requires_catalog
def test_screening_the_iss_separates_modules_from_real_conjunctions():
    """End to end: the module noise is present but never actionable."""
    primary = CATALOG_OBJECTS[ISS][1]
    catalog = [(n, name, sat) for n, (name, sat) in CATALOG_OBJECTS.items() if n != ISS]
    result = screening_service.screen_primary(
        primary_sat=primary,
        primary_name=CATALOG_OBJECTS[ISS][0],
        catalog=catalog,
        epoch=EPOCH,
        window_s=6 * 3600.0,
        gate_km=200.0,
        hbr_km=0.020,
        primary_norad=ISS,
        coarse_step_s=30.0,
    )
    co_located = [c for c in result.conjunctions if c.status == STATUS_CO_LOCATED]
    actionable = [c for c in result.conjunctions if c.is_actionable]

    assert co_located, "the ISS modules should have been detected and classified"
    assert all(not c.is_actionable for c in co_located)
    assert all(c.miss_km > 1.0 for c in actionable), (
        "a zero-separation pair reached the actionable set"
    )


def test_status_vocabulary_is_distinct():
    """The three states must not collide — they gate different behaviour."""
    assert len({STATUS_ASSESSED, STATUS_CO_LOCATED, STATUS_UNDETERMINED}) == 3
