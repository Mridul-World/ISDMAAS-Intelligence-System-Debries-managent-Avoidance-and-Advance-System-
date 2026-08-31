"""
tle.py — element-set integrity validation.

Why this module exists
----------------------
`sgp4.api.Satrec.twoline2rv` is a parser, not a validator. Measured against this
repository's own pinned sgp4 version, it accepts every one of these and returns a
propagator that produces a plausible-looking orbit:

    truncated line 1 (40 of 69 chars)  -> accepted, 406.5 km altitude
    non-numeric catalog number         -> accepted, satnum silently taken from line 2
    inclination corrupted 51.6 -> 99.9 -> accepted, 406.1 km altitude
    broken line-1 checksum             -> accepted
    catalog numbers disagree between
      line 1 and line 2                -> accepted, line 2 wins

The corrupted-inclination case is the dangerous one: the altitude barely moves,
so nothing downstream looks wrong, while the orbital plane is off by 48 degrees.
An element set like that entering a conjunction screen produces a confident,
completely wrong answer.

Operators paste element sets by hand, and TLEs travel through email and chat
where they get wrapped, truncated and re-typed. The checksum digit exists
precisely to catch that, and nothing in the pipeline was checking it.

What is validated
-----------------
1. Structure: two lines, 69 characters, correct line numbers.
2. Checksum: the standard mod-10 sum over each line (digits count as their
   value, a minus sign counts as 1, everything else as 0).
3. Cross-line agreement: the catalog number must match on both lines.
4. Physical ranges: inclination, eccentricity, angles and mean motion.
5. Propagation: the element set must actually produce a finite state.

Deliberately NOT enforced
-------------------------
Element-set age is reported, not rejected. A stale element set is a legitimate
operational input as long as the staleness is surfaced — that is a decision for
the caller, and rejecting it here would break screening against catalog objects
that simply have not been updated recently.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import List, Optional

TLE_LINE_LENGTH = 69

# Physical bounds. These are the ranges the TLE format itself defines, not
# mission-specific limits, so a legitimate catalog object can never fail them.
INCLINATION_RANGE = (0.0, 180.0)
ANGLE_RANGE = (0.0, 360.0)
ECCENTRICITY_RANGE = (0.0, 1.0)
MEAN_MOTION_RANGE = (0.0, 20.0)          # rev/day; 20 is below the Earth's surface


class TleValidationError(ValueError):
    """An element set failed integrity validation."""


@dataclass
class TleReport:
    """The outcome of validating one element set."""
    valid: bool
    norad: Optional[int] = None
    epoch_utc: Optional[datetime] = None
    age_days: Optional[float] = None
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.valid:
            raise TleValidationError("; ".join(self.errors))


def checksum(line: str) -> int:
    """
    Standard TLE mod-10 checksum over the first 68 characters.

    Digits contribute their value, a minus sign contributes 1, everything else
    contributes nothing. The 69th character is the expected result.
    """
    total = 0
    for character in line[:68]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return total % 10


def _decode_epoch(field_text: str) -> datetime:
    """
    Decode the YYDDD.DDDDDDDD epoch field.

    Two-digit years pivot at 57: 57-99 are 1957-1999, 00-56 are 2000-2056. That
    is the convention the format defines; getting it wrong moves the epoch by a
    century and SGP4 then returns a state that is nowhere near the truth.
    """
    year_part = int(field_text[:2])
    day_part = float(field_text[2:])
    year = 2000 + year_part if year_part < 57 else 1900 + year_part
    if not 1.0 <= day_part < 367.0:
        raise ValueError(f"day-of-year {day_part} is outside 1-366")
    return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=day_part - 1.0)


def _decode_eccentricity(field_text: str) -> float:
    """Line 2 columns 27-33: an implied leading decimal point."""
    digits = field_text.strip()
    if not digits.isdigit():
        raise ValueError(f"eccentricity field {field_text!r} is not numeric")
    return float("0." + digits)


def validate_tle(
    line1: str,
    line2: str,
    now: Optional[datetime] = None,
    stale_after_days: float = 14.0,
) -> TleReport:
    """
    Validate an element set and report every problem found.

    All checks run; the report lists everything wrong rather than stopping at the
    first failure, because an operator fixing a pasted element set wants to see
    all of it at once.
    """
    report = TleReport(valid=False)
    errors, warnings = report.errors, report.warnings

    line1 = (line1 or "").rstrip("\r\n")
    line2 = (line2 or "").rstrip("\r\n")

    # ---------------------------------------------------------- structure
    for index, line in ((1, line1), (2, line2)):
        if len(line) != TLE_LINE_LENGTH:
            errors.append(
                f"line {index} is {len(line)} characters; a TLE line is exactly "
                f"{TLE_LINE_LENGTH}"
            )
        if not line.startswith(f"{index} "):
            errors.append(f"line {index} must begin with '{index} '")

    if errors:
        return report  # nothing further can be trusted

    # ----------------------------------------------------------- checksum
    for index, line in ((1, line1), (2, line2)):
        expected = line[68]
        if not expected.isdigit():
            errors.append(f"line {index} checksum column is {expected!r}, not a digit")
            continue
        actual = checksum(line)
        if actual != int(expected):
            errors.append(
                f"line {index} checksum is {expected} but the line sums to "
                f"{actual} - the element set is corrupted or was altered"
            )

    # -------------------------------------------------- cross-line agreement
    try:
        catalog_1 = int(line1[2:7])
        catalog_2 = int(line2[2:7])
        if catalog_1 != catalog_2:
            errors.append(
                f"catalog number disagrees between lines: {catalog_1} on line 1, "
                f"{catalog_2} on line 2"
            )
        report.norad = catalog_1
    except ValueError:
        errors.append("catalog number is not numeric")

    # ------------------------------------------------------------- epoch
    try:
        epoch = _decode_epoch(line1[18:32])
        report.epoch_utc = epoch
        reference = now or datetime.now(timezone.utc)
        age = (reference - epoch).total_seconds() / 86400.0
        report.age_days = age
        if age < -1.0:
            warnings.append(
                f"epoch is {abs(age):.1f} days in the future; check the year field"
            )
        elif age > stale_after_days:
            warnings.append(
                f"element set is {age:.1f} days old; propagation error grows "
                f"quickly past about {stale_after_days:.0f} days"
            )
    except (ValueError, IndexError) as exc:
        errors.append(f"epoch field is unreadable: {exc}")

    # ---------------------------------------------------- physical ranges
    numeric_fields = [
        ("inclination", line2[8:16], INCLINATION_RANGE, "deg"),
        ("RAAN", line2[17:25], ANGLE_RANGE, "deg"),
        ("argument of perigee", line2[34:42], ANGLE_RANGE, "deg"),
        ("mean anomaly", line2[43:51], ANGLE_RANGE, "deg"),
        ("mean motion", line2[52:63], MEAN_MOTION_RANGE, "rev/day"),
    ]
    for name, text, (low, high), unit in numeric_fields:
        try:
            value = float(text)
        except ValueError:
            errors.append(f"{name} field {text.strip()!r} is not numeric")
            continue
        if not low <= value <= high:
            errors.append(
                f"{name} is {value} {unit}, outside the valid range "
                f"{low}-{high} {unit}"
            )

    try:
        eccentricity = _decode_eccentricity(line2[26:33])
        if not ECCENTRICITY_RANGE[0] <= eccentricity < ECCENTRICITY_RANGE[1]:
            errors.append(
                f"eccentricity is {eccentricity}, which is not a closed orbit"
            )
    except ValueError as exc:
        errors.append(str(exc))

    # -------------------------------------------------- propagation check
    # The element set must produce a finite state at its own epoch. This catches
    # combinations that are individually in range but jointly unphysical.
    if not errors:
        try:
            from sgp4.api import Satrec, jday

            satrec = Satrec.twoline2rv(line1, line2)
            epoch = report.epoch_utc or datetime.now(timezone.utc)
            jd, fr = jday(epoch.year, epoch.month, epoch.day,
                          epoch.hour, epoch.minute,
                          epoch.second + epoch.microsecond * 1e-6)
            code, position, velocity = satrec.sgp4(jd, fr)
            if code != 0:
                errors.append(
                    f"SGP4 rejects this element set at its own epoch (error {code})"
                )
            elif not all(math.isfinite(component) for component in position + velocity):
                errors.append("SGP4 produced a non-finite state")
            else:
                radius = math.sqrt(sum(component ** 2 for component in position))
                if radius < 6378.137:
                    errors.append(
                        f"the orbit passes below the Earth's surface "
                        f"(radius {radius:.1f} km at epoch)"
                    )
        except Exception as exc:  # noqa: BLE001 - any parser failure is a rejection
            errors.append(f"element set failed to parse: {exc}")

    report.valid = not errors
    return report
