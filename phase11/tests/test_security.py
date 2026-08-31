"""
Tests for credential handling, session lifetime and abuse control.
"""
import hashlib
import time
from datetime import datetime, timedelta, timezone

import pytest

from isdmaas_core.security import (
    SlidingWindowLimiter,
    hash_password,
    legacy_hash_record,
    normalize_username,
    new_session_token,
    parse_bearer,
    security_headers,
    token_digest,
    validate_password,
    verify_password,
)
from isdmaas_core.store import Store


# ------------------------------------------------------------------ hashing
def test_hash_is_salted_and_verifies():
    a = hash_password("Correct-Horse-9!", iterations=1000)
    b = hash_password("Correct-Horse-9!", iterations=1000)
    assert a != b, "two hashes of the same password must not be identical"
    assert verify_password("Correct-Horse-9!", a) == (True, True)
    assert verify_password("wrong", a)[0] is False


def test_current_iterations_do_not_request_a_rehash():
    stored = hash_password("Correct-Horse-9!", iterations=240_000)
    valid, needs_rehash = verify_password("Correct-Horse-9!", stored)
    assert valid and not needs_rehash


def test_legacy_hashes_verify_and_request_an_upgrade():
    """
    The old store used a single unsalted-round SHA-256. Existing accounts must
    keep working, and must be flagged for upgrade on the owner's next sign-in.
    """
    salt, password = "abc123", "alpha123"
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    stored = legacy_hash_record(salt, digest)
    assert verify_password(password, stored) == (True, True)
    assert verify_password("nope", stored)[0] is False


def test_malformed_hash_never_raises():
    for junk in ("", "garbage", "pbkdf2_sha256$notanint$x$y", "a$b$c$d$e"):
        assert verify_password("anything", junk) == (False, False)


# ----------------------------------------------------------------- policy
@pytest.mark.parametrize(
    "password", ["short1!", "alllowercase123", "alpha123", "password", "12345678"]
)
def test_weak_passwords_are_refused(password):
    with pytest.raises(ValueError):
        validate_password(password)


def test_published_demo_passwords_are_refused():
    """
    These appear in the project's own README and demo panel. Accepting them for
    a real account would turn published documentation into working credentials.
    """
    for published in ("alpha123", "bravo123"):
        with pytest.raises(ValueError):
            validate_password(published)


def test_a_strong_password_is_accepted():
    validate_password("Tr0ubadour-Sat!")


@pytest.mark.parametrize("bad", ["", "  ", "a" * 65, "has space", "semi;colon", "../x"])
def test_usernames_are_validated(bad):
    with pytest.raises(ValueError):
        normalize_username(bad)


def test_usernames_are_normalized():
    assert normalize_username("  Operator_A  ") == "operator_a"


# ------------------------------------------------------------------ tokens
def test_tokens_are_high_entropy_and_unique():
    tokens = {new_session_token() for _ in range(500)}
    assert len(tokens) == 500
    assert all(len(t) >= 40 for t in tokens)


def test_only_the_digest_is_ever_stored(tmp_path):
    """A stolen database must not yield usable sessions."""
    store = Store(tmp_path / "auth.db", token_ttl_hours=1)
    store.create_user("op", hash_password("Tr0ubadour-Sat!", iterations=1000))
    token = new_session_token()
    store.create_session("op", token)
    # WAL mode keeps recent writes in the -wal sidecar, so scan every file the
    # database is made of, not just the main one.
    raw = b"".join(
        path.read_bytes() for path in sorted(tmp_path.glob("auth.db*"))
    )
    assert token.encode() not in raw
    assert token_digest(token).encode() in raw


@pytest.mark.parametrize(
    "header,expected",
    [
        ("Bearer abc", "abc"),
        ("bearer abc", "abc"),
        ("Basic abc", None),
        ("", None),
        (None, None),
        ("Bearer   ", None),
    ],
)
def test_bearer_parsing(header, expected):
    assert parse_bearer(header) == expected


# --------------------------------------------------------------- sessions
def test_session_resolves_then_expires(tmp_path):
    store = Store(tmp_path / "auth.db", token_ttl_hours=1)
    store.create_user("op", hash_password("Tr0ubadour-Sat!", iterations=1000))
    token = new_session_token()
    store.create_session("op", token)
    assert store.resolve_session(token).username == "op"

    # Force expiry by rewriting the stored deadline.
    conn = store._connection()
    conn.execute(
        "UPDATE sessions SET expires_utc=? WHERE token_hash=?",
        ((datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
         token_digest(token)),
    )
    assert store.resolve_session(token) is None


def test_logout_all_revokes_every_session(tmp_path):
    store = Store(tmp_path / "auth.db", token_ttl_hours=1)
    store.create_user("op", hash_password("Tr0ubadour-Sat!", iterations=1000))
    tokens = [new_session_token() for _ in range(3)]
    for token in tokens:
        store.create_session("op", token)
    assert store.delete_sessions_for("op") == 3
    assert all(store.resolve_session(t) is None for t in tokens)


def test_ownership_is_not_transferable_by_re_registration(tmp_path):
    store = Store(tmp_path / "auth.db", token_ttl_hours=1)
    for name in ("alice", "mallory"):
        store.create_user(name, hash_password("Tr0ubadour-Sat!", iterations=1000))
    store.upsert_satellite("90001", "alice", "ALICE-1", "1 ...", "2 ...")
    store.upsert_satellite("90001", "mallory", "STOLEN", "1 ...", "2 ...")
    assert store.get_satellite("90001").owner == "alice"
    assert store.get_satellite("90001").name == "ALICE-1"


def test_legacy_json_import_discards_tokens(tmp_path):
    """
    The old auth_store.json was committed to the repository, so every session
    token in it is public. Users and satellites migrate; tokens must not.
    """
    import json

    legacy = tmp_path / "auth_store.json"
    legacy.write_text(json.dumps({
        "users": {"opa": {"salt": "s", "pw": hashlib.sha256(b"salpha123").hexdigest()}},
        "tokens": {"deadbeef": "opa", "cafebabe": "opa"},
        "satellites": {"90001": {"owner": "opa", "name": "A", "tle1": "1", "tle2": "2"}},
    }))
    store = Store(tmp_path / "auth.db", token_ttl_hours=1)
    result = store.import_legacy_json(legacy)
    assert result == {"users": 1, "satellites": 1, "tokens_discarded": 2}
    assert store.get_user("opa") is not None
    assert store.get_satellite("90001").owner == "opa"
    assert store.resolve_session("deadbeef") is None


# ------------------------------------------------------------ rate limiting
def test_limiter_blocks_after_the_limit():
    limiter = SlidingWindowLimiter(limit=3, window_s=60)
    for _ in range(3):
        assert limiter.hit("ip").allowed
    blocked = limiter.hit("ip")
    assert not blocked.allowed
    assert blocked.retry_after_s > 0


def test_limiter_is_per_key():
    limiter = SlidingWindowLimiter(limit=1, window_s=60)
    assert limiter.hit("a").allowed
    assert not limiter.hit("a").allowed
    assert limiter.hit("b").allowed


def test_limiter_window_slides():
    limiter = SlidingWindowLimiter(limit=1, window_s=1)
    assert limiter.hit("ip").allowed
    assert not limiter.hit("ip").allowed
    time.sleep(1.05)
    assert limiter.hit("ip").allowed


def test_limiter_reset_clears_a_key():
    limiter = SlidingWindowLimiter(limit=1, window_s=60)
    limiter.hit("ip")
    limiter.reset("ip")
    assert limiter.hit("ip").allowed


# ---------------------------------------------------------------- headers
def test_security_headers_present():
    headers = security_headers(is_production=True)
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"
    assert "Strict-Transport-Security" in headers
    assert "frame-ancestors 'none'" in headers["Content-Security-Policy"]


def test_hsts_only_in_production():
    assert "Strict-Transport-Security" not in security_headers(is_production=False)


# ------------------------------------------------- rate-limit key integrity
def test_rate_limit_key_ignores_client_supplied_forwarding_headers():
    """
    X-Forwarded-For is set by the caller. If the limiter keys on it, an attacker
    mints a fresh empty bucket per request by incrementing a number and the limit
    never fires. Resolving the real client behind a proxy is the ASGI server's
    job (`--forwarded-allow-ips`), not this function's.
    """
    from auth_store import client_key

    class _Client:
        host = "203.0.113.9"

    class _Request:
        client = _Client()

        def __init__(self, headers):
            self.headers = headers

    plain = _Request({})
    spoofed_a = _Request({"x-forwarded-for": "10.0.0.1"})
    spoofed_b = _Request({"x-forwarded-for": "10.0.0.2, 10.0.0.3"})

    assert client_key(plain) == client_key(spoofed_a) == client_key(spoofed_b)
    assert client_key(plain) == "203.0.113.9"


def test_a_spoofed_forwarding_header_cannot_extend_the_limit():
    """End to end: varying X-Forwarded-For must not buy extra attempts."""
    from auth_store import client_key
    from isdmaas_core.security import SlidingWindowLimiter

    class _Client:
        host = "203.0.113.9"

    class _Request:
        client = _Client()

        def __init__(self, index):
            self.headers = {"x-forwarded-for": f"10.0.0.{index}"}

    limiter = SlidingWindowLimiter(limit=3, window_s=60)
    allowed = sum(limiter.hit(client_key(_Request(i))).allowed for i in range(10))
    assert allowed == 3
