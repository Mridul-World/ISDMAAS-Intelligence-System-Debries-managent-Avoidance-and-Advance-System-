"""
Scientific validation: element-set integrity.

The regression this suite locks down: `sgp4.api.Satrec.twoline2rv` is a parser,
not a validator. Against the sgp4 version this repository pins, it accepts all of
the following and returns a propagator that yields a plausible LEO altitude:

    truncated line 1 (40 of 69 chars)     accepted, 406.5 km
    non-numeric catalog number            accepted, satnum taken from line 2
    inclination corrupted 51.6 -> 99.9    accepted, 406.1 km
    broken line-1 checksum                accepted
    catalog numbers disagreeing           accepted, line 2 wins

The corrupted-inclination case is the one that matters operationally: altitude
barely moves, so nothing downstream looks wrong, while the orbital plane is off
by 48 degrees. Screening against that produces a confident, wrong answer.

Two properties are tested, and both matter equally:
  1. Every corruption is REJECTED.
  2. Every one of the 15 894 real element sets in the bundled catalog is
     ACCEPTED. A validator with false positives is worse than none, because it
     turns a legitimate catalog refresh into an outage.
"""
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from isdmaas_core.tle import (
    TleValidationError,
    checksum,
    validate_tle,
)

CATALOG = Path(__file__).resolve().parent.parent.parent / "data" / "catalog_active.tle"


def _real_element_sets():
    """Every (name, line1, line2) triple in the bundled offline catalog."""
    if not CATALOG.exists():
        return []
    lines = [line.rstrip() for line in CATALOG.read_text(encoding="utf-8").splitlines()
             if line.strip()]
    return [(lines[i], lines[i + 1], lines[i + 2])
            for i in range(0, len(lines) - 2, 3)]


REAL = _real_element_sets()
requires_catalog = pytest.mark.skipif(not REAL, reason="bundled catalog not present")


@pytest.fixture(scope="module")
def iss():
    """A real, checksum-valid ISS element set from the bundled catalog."""
    for name, line1, line2 in REAL:
        if name.startswith("ISS (ZARYA)"):
            return line1, line2
    pytest.skip("ISS not in the bundled catalog")


# ================================================== no false positives
@requires_catalog
def test_every_real_element_set_validates():
    """
    The whole bundled catalog must pass.

    This is the anti-false-positive guard. If a future tightening of the
    validator rejects real CelesTrak data, this fails loudly rather than
    silently shrinking the screening catalog.
    """
    failures = []
    for name, line1, line2 in REAL:
        report = validate_tle(line1, line2)
        if not report.valid:
            failures.append((name, report.errors))
    assert not failures, (
        f"{len(failures)} of {len(REAL)} real element sets were rejected. "
        f"First few: {failures[:3]}"
    )


@requires_catalog
def test_checksums_of_real_element_sets_are_self_consistent():
    """The checksum function must reproduce the published digit on real data."""
    for name, line1, line2 in REAL[:500]:
        assert checksum(line1) == int(line1[68]), f"line 1 checksum: {name}"
        assert checksum(line2) == int(line2[68]), f"line 2 checksum: {name}"


@requires_catalog
def test_catalog_numbers_and_epochs_are_extracted_correctly():
    for _name, line1, line2 in REAL[:200]:
        report = validate_tle(line1, line2)
        assert report.norad == int(line1[2:7])
        assert report.norad == int(line2[2:7])
        assert report.epoch_utc is not None
        assert report.epoch_utc.tzinfo is timezone.utc
        assert 1957 <= report.epoch_utc.year <= 2060


# ===================================================== corruption rejection
def test_clean_element_set_is_accepted(iss):
    report = validate_tle(*iss)
    assert report.valid, report.errors


def test_truncated_line_is_rejected(iss):
    line1, line2 = iss
    report = validate_tle(line1[:40], line2)
    assert not report.valid
    assert any("characters" in e for e in report.errors)


def test_broken_checksum_is_rejected(iss):
    """The single most important check: it is what detects transit corruption."""
    line1, line2 = iss
    tampered = line1[:68] + ("0" if line1[68] != "0" else "1")
    report = validate_tle(tampered, line2)
    assert not report.valid
    assert any("checksum" in e for e in report.errors)


def test_corrupted_inclination_is_rejected(iss):
    """
    The dangerous case. sgp4 accepts this and reports a normal altitude while
    the orbital plane is wrong by 48 degrees.
    """
    line1, line2 = iss
    corrupted = line2[:8] + "99.9999" + line2[15:]
    report = validate_tle(line1, corrupted)
    assert not report.valid


def test_mismatched_catalog_numbers_are_rejected(iss):
    """sgp4 silently takes the line-2 value; the element set is not trustworthy."""
    line1, line2 = iss
    report = validate_tle(line1, "2 12345" + line2[7:])
    assert not report.valid


@pytest.mark.parametrize(
    "line1,line2",
    [
        ("", ""),
        ("garbage", "garbage"),
        (None, None),
        ("1 " + "x" * 67, "2 " + "x" * 67),
    ],
)
def test_structurally_invalid_input_is_rejected_without_raising(line1, line2):
    """Validation must return a report, never explode, on arbitrary input."""
    report = validate_tle(line1, line2)
    assert not report.valid
    assert report.errors


def test_line_order_swap_is_rejected(iss):
    line1, line2 = iss
    report = validate_tle(line2, line1)
    assert not report.valid


def test_out_of_range_fields_are_rejected(iss):
    """Physical ranges, checked independently of the checksum."""
    line1, line2 = iss
    # Inclination 200 degrees is impossible; rebuild the checksum so the failure
    # is attributable to the range check and not to the checksum.
    body = line2[:8] + "200.0000" + line2[16:68]
    report = validate_tle(line1, body + str(checksum(body)))
    assert not report.valid
    assert any("inclination" in e.lower() for e in report.errors)


# =========================================================== staleness
def test_stale_element_set_warns_but_does_not_fail(iss):
    """
    Age is reported, never fatal.

    A stale element set is a legitimate operational input as long as the
    staleness is surfaced. Rejecting it here would break screening against
    catalog objects that simply have not been refreshed.
    """
    line1, line2 = iss
    far_future = datetime.now(timezone.utc) + timedelta(days=400)
    report = validate_tle(line1, line2, now=far_future)
    assert report.valid
    assert report.age_days > 365
    assert any("old" in w for w in report.warnings)


def test_future_epoch_is_flagged(iss):
    line1, line2 = iss
    long_past = datetime(1990, 1, 1, tzinfo=timezone.utc)
    report = validate_tle(line1, line2, now=long_past)
    assert any("future" in w for w in report.warnings)


def test_raise_if_invalid_raises_with_all_reasons():
    report = validate_tle("bad", "worse")
    with pytest.raises(TleValidationError):
        report.raise_if_invalid()
    assert validate_tle(*_real_or_skip()).valid


def _real_or_skip():
    if not REAL:
        pytest.skip("bundled catalog not present")
    return REAL[0][1], REAL[0][2]
