"""
End-to-end API tests: routing, authentication, authorization and validation.

Everything runs against a temporary data directory so a test never touches the
developer's real operator registry.
"""
import os
import tempfile

import pytest

os.environ.setdefault("ISDMAAS_ENV", "development")
os.environ.setdefault("ISDMAAS_DATA_DIR", tempfile.mkdtemp(prefix="isdmaas-test-"))

from fastapi.testclient import TestClient  # noqa: E402

import phase11_api  # noqa: E402

STRONG_PASSWORD = "Tr0ubadour-Sat!"


@pytest.fixture(scope="module")
def client():
    with TestClient(phase11_api.app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def operator(client):
    response = client.post(
        "/auth/login", json={"username": "operator_a", "password": "alpha123"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


# -------------------------------------------------------------------- ops
def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["service"] == "ISDMAAS"


def test_security_headers_on_every_response(client):
    response = client.get("/health")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["X-Request-ID"]


def test_request_id_echoes_back(client):
    response = client.get("/health", headers={"X-Request-ID": "trace-me-123"})
    assert response.headers["X-Request-ID"] == "trace-me-123"


def test_ready_reports_catalog_state(client):
    body = client.get("/ready").json()
    assert "catalog" in body and "registry" in body


# ------------------------------------------------------------------- auth
def test_login_rejects_a_wrong_password(client):
    response = client.post(
        "/auth/login", json={"username": "operator_a", "password": "definitely-wrong"}
    )
    assert response.status_code == 401


def test_login_error_does_not_reveal_whether_the_account_exists(client):
    known = client.post(
        "/auth/login", json={"username": "operator_a", "password": "definitely-wrong"}
    )
    unknown = client.post(
        "/auth/login", json={"username": "nobody_here", "password": "definitely-wrong"}
    )
    assert known.status_code == unknown.status_code == 401
    assert known.json()["detail"] == unknown.json()["detail"]


def test_registration_enforces_the_password_policy(client):
    response = client.post(
        "/auth/register", json={"username": "weakop", "password": "alpha123"}
    )
    assert response.status_code == 422
    assert "at least" in response.json()["detail"]


def test_registration_and_duplicate(client):
    payload = {"username": "pilot_ops", "password": STRONG_PASSWORD}
    first = client.post("/auth/register", json=payload)
    assert first.status_code in (201, 409)
    assert client.post("/auth/register", json=payload).status_code == 409


def test_me_requires_a_token(client):
    assert client.get("/auth/me").status_code == 401
    assert client.get(
        "/auth/me", headers={"Authorization": "Bearer not-a-real-token"}
    ).status_code == 401


def test_me_returns_the_caller(client, operator):
    body = client.get("/auth/me", headers=operator).json()
    assert body["username"] == "operator_a"


def test_logout_invalidates_the_token(client):
    login = client.post(
        "/auth/login", json={"username": "operator_b", "password": "bravo123"}
    ).json()
    headers = {"Authorization": f"Bearer {login['token']}"}
    assert client.get("/auth/me", headers=headers).status_code == 200
    client.post("/auth/logout", headers=headers)
    assert client.get("/auth/me", headers=headers).status_code == 401


# ---------------------------------------------------------- authorization
def test_screening_requires_authentication(client):
    assert client.get("/user/screen/90001").status_code == 401


def test_an_operator_cannot_screen_another_operators_asset(client, operator):
    response = client.get("/user/screen/90002", headers=operator)
    assert response.status_code == 403


def test_maneuver_authority_is_owner_only(client, operator):
    response = client.post(
        "/user/plan",
        json={"primary_norad": "90002", "secondary_norad": "90001"},
        headers=operator,
    )
    assert response.status_code == 403
    # The refusal must not disclose which operator holds the asset.
    assert "operator_b" not in response.text


def test_maneuver_authority_requires_a_token(client):
    response = client.post(
        "/user/plan", json={"primary_norad": "90001", "secondary_norad": "90002"}
    )
    assert response.status_code == 401


def test_owner_may_plan_their_own_asset(client, operator):
    response = client.post(
        "/user/plan",
        json={"primary_norad": "90001", "secondary_norad": "90002"},
        headers=operator,
    )
    assert response.status_code == 200
    body = response.json()
    assert body["authorized_operator"] == "operator_a"
    assert body["verdict"] in ("APPROVED", "REJECTED", "NO_ACTION")


def test_other_operators_are_listed_without_naming_them(client, operator):
    body = client.get("/my/satellites", headers=operator).json()
    assert body["mine"]
    for entry in body["others"]:
        assert entry["owner"] == "another operator"


# -------------------------------------------------------------- validation
@pytest.mark.parametrize(
    "payload",
    [
        {"primary_norad": -1, "secondary_norad": 25544},
        {"primary_norad": 39634},
        {"primary_norad": 39634, "secondary_norad": 25544, "window_hours": 0},
        {"primary_norad": 39634, "secondary_norad": 25544, "window_hours": 100000},
    ],
)
def test_assess_rejects_invalid_input(client, payload):
    assert client.post("/assess", json=payload).status_code == 422


def test_maneuver_plan_requires_a_threat_source(client):
    response = client.post("/maneuver-plan", json={"primary_norad": 39634})
    assert response.status_code == 400
    assert "synthetic_miss_km" in response.json()["detail"]


def test_unknown_satellite_is_a_clean_404(client):
    response = client.get("/orbit/999998")
    assert response.status_code in (404, 503)
    assert "error" in response.json()


def test_monitor_watchlist_is_bounded(client):
    watchlist = ",".join(str(30000 + i) for i in range(25))
    response = client.get(f"/monitor?watchlist={watchlist}")
    assert response.status_code == 400
    assert "at most 20" in response.json()["detail"]


def test_monitor_rejects_a_non_numeric_watchlist(client):
    assert client.get("/monitor?watchlist=abc,def").status_code == 400


def test_error_envelope_shape(client):
    body = client.post("/assess", json={"primary_norad": "nope"}).json()
    assert body["error"]["code"] == "unprocessable"
    assert body["error"]["request_id"]
    assert body["detail"]


def test_tle_registration_validates_the_element_set(client, operator):
    response = client.post(
        "/my/satellites",
        json={"name": "BAD", "tle1": "not a tle line at all ................",
              "tle2": "also not a tle line ..................."},
        headers=operator,
    )
    assert response.status_code == 422


# ------------------------------------------------------------------ physics
def test_screen_returns_refined_geometry(client):
    response = client.get("/screen/39634?gate_km=50&hours=12&max_results=5")
    if response.status_code == 503:
        pytest.skip("catalog unavailable in this environment")
    body = response.json()
    assert body["objects_screened"] > 1000
    assert body["objects_after_geometric_filter"] <= body["objects_screened"]
    for conjunction in body["conjunctions"]:
        assert conjunction["miss_km"] <= 50.0
        assert conjunction["tca_utc"].endswith("+00:00")
        assert conjunction["risk"] in (
            "NOMINAL", "ELEVATED", "HIGH", "CRITICAL", "UNDETERMINED"
        )


def test_maneuver_plan_runs_the_full_safety_gate(client):
    response = client.post(
        "/maneuver-plan",
        json={
            "primary_norad": 39634, "synthetic_miss_km": 0.3, "tca_hours": 12,
            "sat_mass_kg": 2300, "fuel_available_kg": 5,
        },
    )
    if response.status_code == 503:
        pytest.skip("catalog unavailable in this environment")
    body = response.json()
    assert body["verdict"] in ("APPROVED", "REJECTED", "NO_ACTION")
    if body["verdict"] != "NO_ACTION":
        names = {c["check"] for c in body["safety_validation"]["checks"]}
        assert names == {
            "fuel_budget", "orbit_safety", "original_threat_resolved",
            "no_new_conjunction",
        }


# ------------------------------------------------- calibrated state handling
def test_stale_precise_orbit_state_is_not_used_as_current(client):
    """
    A precise-orbit determination is a snapshot at its epoch, not a statement
    about where the satellite is now. The state file shipped for Sentinel-1A is
    dated 2026-06-16; using its position vector as the current state would place
    the satellite thousands of kilometres from where it is.

    The service must fall back to the current element set and say so.
    """
    response = client.get("/screen/39634?gate_km=25&hours=6&max_results=1")
    if response.status_code == 503:
        pytest.skip("catalog unavailable in this environment")
    source = response.json()["covariance_source"]
    assert source in (
        "tle_scale_model",                      # determination too old to trust
        "precise_orbit_sigma_on_element_set",   # determination current
    ), f"unexpected covariance source {source!r}"


def test_primary_uncertainty_is_never_tighter_than_the_determination(client):
    """
    Whatever covariance is used, its along-track sigma must not be tighter than
    what a real precise-orbit solution reports. Understating the primary's
    uncertainty deflates Pc and is the exact class of calibration bug this
    project already fixed once for the secondary object.
    """
    import phase11_api
    from isdmaas_core import screening as screening_service

    snapshot = phase11_api.catalog_service.snapshot()
    if snapshot is None:
        pytest.skip("catalog unavailable in this environment")
    primary = phase11_api.load_primary(39634, snapshot)
    assert primary.sigma_km[1] >= screening_service.CALIBRATED_PRIMARY_SIGMA_KM[1]
    # The state must be the propagated element set, i.e. an epoch of ~now.
    age_s = abs(
        (phase11_api.datetime.now(phase11_api.timezone.utc) - primary.epoch)
        .total_seconds()
    )
    assert age_s < 3600, "the primary state must be current, not a stale snapshot"
