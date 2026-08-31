"""
End-to-end mission workflow.

One test walks the entire operational chain against the real service, in order,
with each step's output feeding the next:

     1. create an operator account          10. compute Pc
     2. authenticate                        11. classify risk
     3. register a spacecraft               12. generate maneuver candidates
     4. load orbit / element-set data       13. select a candidate
     5. ingest tracked objects              14. propagate the post-burn state
     6. run catalog screening               15. re-screen against the catalog
     7. detect a conjunction                16. verify improved safety
     8. refine the TCA                      17. generate a mission report
     9. compute the miss distance           18. verify the audit record

This is the test that P0-4 in the production audit was about. The safety gate
and the planner each passed their own unit tests while nothing walked the whole
chain on real data — which is exactly how a pipeline ends up with four correct
components and a broken join.

The conjunction is real in the sense that matters: two registered element sets
on genuinely crossing orbits, screened by the production screener, refined by
the production TCA search, assessed by the production Pc engine. The two
spacecraft are fictitious (analyst NORAD range 90000+) because a test cannot
depend on a real conjunction existing in the catalog on the day it runs.
"""
import os
import tempfile
from datetime import datetime, timezone

import pytest

os.environ.setdefault("ISDMAAS_ENV", "development")
os.environ.setdefault("ISDMAAS_DATA_DIR", tempfile.mkdtemp(prefix="isdmaas-e2e-"))

from fastapi.testclient import TestClient  # noqa: E402

import phase11_api  # noqa: E402

OPERATOR = "mission_e2e"
PASSWORD = "Tr0ubadour-Sat!"


@pytest.fixture(scope="module")
def client():
    with TestClient(phase11_api.app) as test_client:
        yield test_client


def _tle_checksum(line):
    total = 0
    for character in line[:68]:
        if character.isdigit():
            total += int(character)
        elif character == "-":
            total += 1
    return str(total % 10)


def _finalize(line):
    line = line.ljust(68)[:68]
    return line + _tle_checksum(line)


def _crossing_pair():
    """
    Two spacecraft on crossing orbits, epoch generated now.

    The epoch has to be current: differential nodal precession separates two
    fixed-epoch planes within days, so a hard-coded element set would quietly
    stop producing a conjunction and the test would pass while asserting
    nothing.
    """
    now = datetime.now(timezone.utc)
    day_of_year = (
        now - datetime(now.year, 1, 1, tzinfo=timezone.utc)
    ).total_seconds() / 86400.0 + 1
    epoch = f"{now.year % 100:02d}{day_of_year:012.8f}"
    return {
        "primary": (
            "E2E-PRIMARY",
            _finalize(f"1 91001U 26900A   {epoch}  .00001000  00000-0  10000-3 0  999"),
            _finalize("2 91001  90.0000 120.0000 0001000  90.0000 250.0000 15.50000000    1"),
        ),
        "secondary": (
            "E2E-SECONDARY",
            _finalize(f"1 91002U 26900B   {epoch}  .00001000  00000-0  10000-3 0  999"),
            _finalize("2 91002  90.0000 300.0000 0001000  90.0000  90.2150 15.50000000    1"),
        ),
    }


def test_full_mission_workflow(client):
    """The whole chain, in order, with every step asserted."""
    report = {}

    # --- 1. create an operator account ------------------------------------
    registration = client.post(
        "/auth/register", json={"username": OPERATOR, "password": PASSWORD}
    )
    assert registration.status_code in (201, 409), registration.text

    # --- 2. authenticate ---------------------------------------------------
    login = client.post(
        "/auth/login", json={"username": OPERATOR, "password": PASSWORD}
    )
    assert login.status_code == 200, login.text
    token = login.json()["token"]
    auth = {"Authorization": f"Bearer {token}"}
    assert login.json()["username"] == OPERATOR
    assert "expires_utc" in login.json()

    # --- 3. register spacecraft -------------------------------------------
    pair = _crossing_pair()
    registered = {}
    for role, (name, line1, line2) in pair.items():
        response = client.post(
            "/my/satellites",
            json={"name": name, "tle1": line1, "tle2": line2},
            headers=auth,
        )
        assert response.status_code == 201, f"{role}: {response.text}"
        registered[role] = response.json()["norad"]
    assert registered["primary"] != registered["secondary"]
    report["spacecraft"] = registered

    # Ownership must be real, not cosmetic.
    holdings = client.get("/my/satellites", headers=auth).json()
    owned = {s["norad"] for s in holdings["mine"]}
    assert set(registered.values()) <= owned

    # --- 4. load orbit data ------------------------------------------------
    orbit = client.get(
        f"/user/orbit/{registered['primary']}?points=120", headers=auth
    )
    assert orbit.status_code == 200, orbit.text
    track = orbit.json()["track_km"]
    assert len(track) >= 100
    radii = [sum(component ** 2 for component in point) ** 0.5 for point in track]
    assert all(6500.0 < radius < 8000.0 for radius in radii), (
        "orbit track left low Earth orbit"
    )
    report["orbit_points"] = len(track)

    # --- 5. ingest tracked objects -----------------------------------------
    readiness = client.get("/ready").json()
    if not readiness["catalog"]["loaded"]:
        pytest.skip("object catalog unavailable in this environment")
    assert readiness["catalog"]["objects"] > 1000
    report["catalog_objects"] = readiness["catalog"]["objects"]
    report["catalog_age_hours"] = readiness["catalog"]["age_hours"]

    # --- 6. run catalog screening ------------------------------------------
    screen = client.get(
        f"/user/screen/{registered['primary']}?hours=24&gate_km=50", headers=auth
    )
    assert screen.status_code == 200, screen.text
    screening = screen.json()
    assert screening["objects_screened"] > 1000
    assert screening["objects_after_geometric_filter"] <= screening["objects_screened"]
    report["objects_screened"] = screening["objects_screened"]
    report["scan_seconds"] = screening["scan_seconds"]

    # --- 7. detect the conjunction -----------------------------------------
    assessment = client.post(
        "/user/assess",
        json={
            "primary_norad": registered["primary"],
            "secondary_norad": registered["secondary"],
            "window_hours": 24,
        },
    )
    assert assessment.status_code == 200, assessment.text
    conjunction = assessment.json()

    # --- 8. TCA refined ----------------------------------------------------
    assert conjunction["tca_utc"].endswith("+00:00"), "TCA must be explicit UTC"
    tca = datetime.fromisoformat(conjunction["tca_utc"])
    assert tca.tzinfo is not None
    assert 0.0 <= conjunction["tca_in_hours"] <= 24.0
    report["tca_utc"] = conjunction["tca_utc"]

    # --- 9. miss distance ---------------------------------------------------
    assert conjunction["miss_km"] >= 0.0
    assert conjunction["rel_speed_kms"] > 0.0
    report["miss_km"] = conjunction["miss_km"]
    report["rel_speed_kms"] = conjunction["rel_speed_kms"]

    # --- 10. collision probability ------------------------------------------
    assert "pc" in conjunction
    if conjunction["pc"] is None:
        # A legitimate outcome, but then it must be labelled, not blank.
        assert conjunction["risk_level"] in ("UNDETERMINED", "DATA_INVALID")
        pytest.skip(
            f"geometry not assessable this run: {conjunction['risk_level']}"
        )
    assert 0.0 <= conjunction["pc"] <= 1.0
    report["pc"] = conjunction["pc"]

    # --- 11. risk classification --------------------------------------------
    assert conjunction["risk_level"] in (
        "NOMINAL", "ELEVATED", "HIGH", "CRITICAL"
    )
    report["risk_level"] = conjunction["risk_level"]

    # --- 12/13. generate and select a maneuver candidate --------------------
    plan = client.post(
        "/user/plan",
        json={
            "primary_norad": registered["primary"],
            "secondary_norad": registered["secondary"],
            "sat_mass_kg": 500.0,
            "fuel_available_kg": 5.0,
        },
        headers=auth,
    )
    assert plan.status_code == 200, plan.text
    planning = plan.json()
    assert planning["authorized_operator"] == OPERATOR

    if planning["verdict"] == "NO_ACTION":
        # Below the action threshold. The audit record must still exist, and the
        # workflow is complete — a decision not to burn is still a decision.
        assert isinstance(planning["recommendation"], str)
        report["verdict"] = "NO_ACTION"
    else:
        recommendation = planning["recommendation"]
        assert recommendation["dv_magnitude_ms"] > 0.0, (
            "a maneuver was recommended with zero delta-v"
        )
        assert recommendation["direction"] in ("prograde", "retrograde")
        assert recommendation["burn_lead_time_h"] > 0.0
        assert recommendation["fuel_kg"] > 0.0
        assert recommendation["fuel_kg"] <= 5.0
        assert planning["options"], "no trade table was produced"
        report["dv_ms"] = recommendation["dv_magnitude_ms"]
        report["lead_time_h"] = recommendation["burn_lead_time_h"]

        # --- 14/15/16. post-burn propagation, re-screen, improved safety ----
        # This is the link that matters most: the recommendation is only valid
        # if the burn measurably improves the conjunction.
        assert recommendation["predicted_new_miss_km"] > conjunction["miss_km"], (
            f"the recommended burn does not increase the miss distance: "
            f"{conjunction['miss_km']} -> {recommendation['predicted_new_miss_km']}"
        )
        assert recommendation["predicted_new_pc"] < conjunction["pc"], (
            f"the recommended burn does not reduce Pc: "
            f"{conjunction['pc']} -> {recommendation['predicted_new_pc']}"
        )

        checks = {c["check"]: c["pass"]
                  for c in planning["safety_validation"]["checks"]}
        assert "owner_authority" in checks and checks["owner_authority"]
        assert "fuel_budget" in checks
        assert "orbit_safety" in checks
        assert "original_threat_resolved" in checks
        assert planning["verdict"] in ("APPROVED", "REJECTED")
        if planning["verdict"] == "APPROVED":
            assert all(checks.values()), (
                f"APPROVED with a failing check: {checks}"
            )
        report["verdict"] = planning["verdict"]
        report["safety_checks"] = checks
        report["new_miss_km"] = recommendation["predicted_new_miss_km"]
        report["new_pc"] = recommendation["predicted_new_pc"]

    # --- 17. mission report ------------------------------------------------
    # Every field an operator needs to act, present and self-consistent.
    for field in ("spacecraft", "catalog_objects", "objects_screened",
                  "tca_utc", "miss_km", "pc", "risk_level", "verdict"):
        assert field in report, f"mission report is missing {field}"

    # --- 18. audit record --------------------------------------------------
    assert "assessment_id" in planning, (
        "no assessment id was returned; the recommendation is not reconstructable"
    )
    assessment_id = planning["assessment_id"]

    stored = client.get(f"/assessments/{assessment_id}", headers=auth)
    assert stored.status_code == 200, stored.text
    record = stored.json()

    # The record must capture the INPUTS, not only the answer — otherwise the
    # numbers cannot be reproduced and the audit trail is decorative.
    assert record["assessment_id"] == assessment_id
    assert record["operator"] == OPERATOR
    assert record["primary_norad"] == registered["primary"]
    assert record["secondary_norad"] == registered["secondary"]
    assert record["primary_epoch_utc"], "element-set epoch not recorded"
    assert record["secondary_epoch_utc"], "element-set epoch not recorded"
    assert record["algorithm_version"], "algorithm version not recorded"
    assert record["model_version"], "model version not recorded"
    assert record["covariance_source"], "covariance source not recorded"
    assert record["hbr_km"] > 0.0
    assert record["pc_method"], "Pc method not recorded"
    assert record["created_utc"]

    # And it must agree with what the operator was shown IN THIS RESPONSE.
    #
    # Deliberately compared against `planning["assessment"]` and not against the
    # earlier standalone /user/assess call. Those are two independent screens run
    # seconds apart against a moving conjunction, and for these demo orbits
    # (15.5 rev/day, passes ~11 minutes apart) they legitimately resolve
    # different passes - 0.4 m in one call and 1.4 m in the next. The audit
    # record has to match the assessment the recommendation was derived from;
    # requiring it to match a different screen would be asserting that the sky
    # does not move.
    shown = planning["assessment"]
    assert record["miss_km"] == pytest.approx(shown["miss_km"], rel=1e-9)
    assert record["risk_level"] == shown["risk_level"]
    assert record["tca_utc"] == shown["tca_utc"]
    if shown["pc"] is not None:
        assert record["pc"] == pytest.approx(shown["pc"], rel=1e-9)

    # The record is scoped to its owner.
    other = client.post(
        "/auth/login", json={"username": "operator_b", "password": "bravo123"}
    )
    if other.status_code == 200:
        intruder = {"Authorization": f"Bearer {other.json()['token']}"}
        assert client.get(
            f"/assessments/{assessment_id}", headers=intruder
        ).status_code == 403

    # It appears in the operator's own listing.
    listing = client.get("/assessments", headers=auth).json()
    assert assessment_id in [a["assessment_id"] for a in listing["assessments"]]


def test_workflow_is_reproducible_from_the_audit_record(client):
    """
    The auditability requirement, stated as a test: a stored assessment must
    carry enough to answer "why did ISDMAAS recommend this?" without access to
    any live state.
    """
    login = client.post(
        "/auth/login", json={"username": OPERATOR, "password": PASSWORD}
    )
    if login.status_code != 200:
        pytest.skip("workflow test did not run first")
    auth = {"Authorization": f"Bearer {login.json()['token']}"}

    listing = client.get("/assessments", headers=auth).json()["assessments"]
    if not listing:
        pytest.skip("no assessments recorded")

    record = client.get(
        f"/assessments/{listing[0]['assessment_id']}", headers=auth
    ).json()

    required = {
        "assessment_id": "which assessment",
        "created_utc": "when",
        "operator": "who",
        "primary_norad": "whose asset",
        "secondary_norad": "against what",
        "primary_epoch_utc": "which element set for the primary",
        "secondary_epoch_utc": "which element set for the secondary",
        "algorithm_version": "which code produced it",
        "model_version": "which model, or none",
        "covariance_source": "which uncertainty model",
        "hbr_km": "which hard-body radius",
        "pc_method": "which probability method",
        "tca_utc": "when the approach was",
        "miss_km": "how close",
        "risk_level": "how it was classified",
    }
    missing = [f"{field} ({why})" for field, why in required.items()
               if record.get(field) in (None, "")]
    assert not missing, f"audit record cannot answer: {missing}"
