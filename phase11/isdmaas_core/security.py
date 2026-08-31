"""
security.py — credential handling, session tokens and abuse control.

Design decisions, and why:

* Passwords are stretched with PBKDF2-HMAC-SHA256 (stdlib, no build deps) at a
  high iteration count. The previous scheme was a single unsalted-round SHA-256,
  which a commodity GPU brute-forces at billions of guesses per second. Legacy
  hashes are still verifiable so existing accounts keep working, and they are
  transparently upgraded on the next successful login.

* Session tokens are opaque 256-bit random strings. Only a SHA-256 digest of the
  token is persisted, so a stolen database does not hand the attacker live
  sessions. Tokens carry an absolute expiry.

* Every comparison of a secret uses hmac.compare_digest to avoid leaking a
  prefix through timing.

* The rate limiter is a per-process sliding window. It is deliberately simple:
  it stops credential stuffing against a single instance. A multi-instance
  deployment should front this with a shared limiter (documented in DEPLOY.md).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple
from collections import deque, defaultdict

PBKDF2_PREFIX = "pbkdf2_sha256"
LEGACY_PREFIX = "sha256_salted"

MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 1024
MAX_USERNAME_LENGTH = 64

# Passwords that appear in the project's own docs/demos must never be accepted
# for a real account — otherwise the published demo credentials become a valid
# login on someone's deployment.
_BANNED_PASSWORDS = frozenset(
    {"alpha123", "bravo123", "password", "password1", "12345678", "changeme",
     "isdmaas", "operator", "letmein", "qwertyui", "admin123", "satellite"}
)


# --------------------------------------------------------------------- hashing
def hash_password(password: str, iterations: int = 240_000) -> str:
    """Return a self-describing PBKDF2 hash string."""
    if not isinstance(password, str) or not password:
        raise ValueError("password must be a non-empty string")
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return "$".join(
        [
            PBKDF2_PREFIX,
            str(iterations),
            base64.b64encode(salt).decode("ascii"),
            base64.b64encode(dk).decode("ascii"),
        ]
    )


def verify_password(password: str, stored: str) -> Tuple[bool, bool]:
    """
    Check a password against a stored hash.

    Returns (is_valid, needs_rehash). needs_rehash is True when the stored hash
    used the legacy scheme or a lower iteration count than we now require, so
    the caller can silently upgrade it after a successful login.
    """
    if not password or not stored:
        return False, False
    parts = stored.split("$")
    try:
        if parts[0] == PBKDF2_PREFIX and len(parts) == 4:
            iterations = int(parts[1])
            salt = base64.b64decode(parts[2])
            expected = base64.b64decode(parts[3])
            actual = hashlib.pbkdf2_hmac(
                "sha256", password.encode("utf-8"), salt, iterations
            )
            ok = hmac.compare_digest(actual, expected)
            return ok, ok and iterations < 240_000
        if parts[0] == LEGACY_PREFIX and len(parts) == 3:
            salt, expected_hex = parts[1], parts[2]
            actual = hashlib.sha256((salt + password).encode()).hexdigest()
            ok = hmac.compare_digest(actual, expected_hex)
            return ok, ok
    except (ValueError, IndexError, TypeError):
        return False, False
    return False, False


def legacy_hash_record(salt: str, pw_hex: str) -> str:
    """Wrap an old auth_store.json {salt, pw} pair in the new stored format."""
    return f"{LEGACY_PREFIX}${salt}${pw_hex}"


def validate_password(password: str) -> None:
    """Raise ValueError with an actionable message if the password is too weak."""
    if not isinstance(password, str):
        raise ValueError("password must be a string")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise ValueError(
            f"password must be at least {MIN_PASSWORD_LENGTH} characters"
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise ValueError(
            f"password must be at most {MAX_PASSWORD_LENGTH} characters"
        )
    if password.lower() in _BANNED_PASSWORDS:
        raise ValueError(
            "that password appears in the project's public demo material and "
            "cannot be used for a real account"
        )
    classes = sum(
        [
            any(c.islower() for c in password),
            any(c.isupper() for c in password),
            any(c.isdigit() for c in password),
            any(not c.isalnum() for c in password),
        ]
    )
    if classes < 3:
        raise ValueError(
            "password must mix at least three of: lowercase, uppercase, "
            "digits, symbols"
        )


def normalize_username(username: str) -> str:
    """Lowercase, trim, and reject anything that is not a safe identifier."""
    if not isinstance(username, str):
        raise ValueError("username must be a string")
    name = username.strip().lower()
    if not name:
        raise ValueError("username is required")
    if len(name) > MAX_USERNAME_LENGTH:
        raise ValueError(f"username must be at most {MAX_USERNAME_LENGTH} characters")
    if not all(c.isalnum() or c in "._-" for c in name):
        raise ValueError(
            "username may contain only letters, digits, dot, underscore, hyphen"
        )
    return name


# ---------------------------------------------------------------------- tokens
def new_session_token() -> str:
    """A fresh opaque bearer token (256 bits of entropy, URL-safe)."""
    return secrets.token_urlsafe(32)


def token_digest(token: str) -> str:
    """The value actually persisted. The raw token never touches storage."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def parse_bearer(authorization: Optional[str]) -> Optional[str]:
    """Extract the token from an `Authorization: Bearer <token>` header."""
    if not authorization:
        return None
    scheme, _, value = authorization.partition(" ")
    if scheme.lower() != "bearer":
        return None
    value = value.strip()
    return value or None


# ---------------------------------------------------------------- rate limiting
@dataclass
class RateLimitDecision:
    allowed: bool
    remaining: int
    retry_after_s: int


class SlidingWindowLimiter:
    """
    Thread-safe sliding-window counter.

    `hit(key)` records an attempt and reports whether the caller is still under
    the limit. Buckets that fall out of the window are dropped lazily, and the
    whole table is swept when it grows past `max_keys` so a stream of unique
    source addresses cannot grow memory without bound.
    """

    def __init__(self, limit: int, window_s: int, max_keys: int = 10_000):
        self.limit = max(1, int(limit))
        self.window_s = max(1, int(window_s))
        self.max_keys = max_keys
        self._hits: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _prune(self, bucket: Deque[float], now: float) -> None:
        cutoff = now - self.window_s
        while bucket and bucket[0] < cutoff:
            bucket.popleft()

    def _sweep(self, now: float) -> None:
        for key in [k for k, b in self._hits.items() if not b or b[-1] < now - self.window_s]:
            self._hits.pop(key, None)

    def check(self, key: str) -> RateLimitDecision:
        """Report the current state without recording an attempt."""
        now = time.monotonic()
        with self._lock:
            bucket = self._hits.get(key)
            if not bucket:
                return RateLimitDecision(True, self.limit, 0)
            self._prune(bucket, now)
            used = len(bucket)
            if used < self.limit:
                return RateLimitDecision(True, self.limit - used, 0)
            retry = int(self.window_s - (now - bucket[0])) + 1
            return RateLimitDecision(False, 0, max(retry, 1))

    def hit(self, key: str) -> RateLimitDecision:
        now = time.monotonic()
        with self._lock:
            if len(self._hits) > self.max_keys:
                self._sweep(now)
            bucket = self._hits[key]
            self._prune(bucket, now)
            if len(bucket) >= self.limit:
                retry = int(self.window_s - (now - bucket[0])) + 1
                return RateLimitDecision(False, 0, max(retry, 1))
            bucket.append(now)
            return RateLimitDecision(True, self.limit - len(bucket), 0)

    def reset(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


# ------------------------------------------------------------ response headers
def security_headers(is_production: bool) -> Dict[str, str]:
    """
    Headers applied to every API response.

    The API serves JSON only, so a restrictive CSP costs nothing here and blocks
    an attacker who finds a way to get HTML reflected out of an error path.
    HSTS is only meaningful over TLS, so it is limited to production.
    """
    headers = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Resource-Policy": "same-site",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'",
        "Cache-Control": "no-store",
    }
    if is_production:
        headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains"
        )
    return headers
