# ISDMAAS Security Audit

**Date:** 2026-08-31 · **Branch:** `production-hardening`
**Suites:** `phase11/tests/security/` (93 tests), `phase11/tests/test_security.py` (26 tests)

---

# 1. CREDENTIAL EXPOSURE — ROTATE THESE NOW

**Severity: P0. This is the only finding with unbounded blast radius, and code
changes do not close it.**

Live credentials were committed to this repository and are present in **every one
of its 9 commits**. The repository has been pushed to GitHub. Untracking the
files removes them from `HEAD`; it does **not** remove them from history, and it
does not undo any access already obtained.

## 1.1 What was exposed

| Credential | Location | Service | Impact if abused |
|---|---|---|---|
| Space-Track username + password | `dd.env`, `phase2/data/build_ssa_dataset.py` | space-track.org | Full catalog query access under your identity; account suspension for terms-of-use violation; the account is tied to a named individual |
| DISCOS API token | `dd.env`, `phase2/data/build_ssa_dataset.py` | ESA DISCOS | Metadata API access under your identity, rate-limit exhaustion |
| CDSE username + password | `dd.env` | dataspace.copernicus.eu | Copernicus data download under your identity |

The Space-Track credential is the serious one: the username is a personal email
address, so a password reused anywhere else is now also exposed.

## 1.2 Required actions, in order

1. **Rotate all four credentials at the provider.** Do this first and
   independently of anything else in this document. Assume they are public.
2. **Check for password reuse.** The Space-Track password was committed in
   plaintext; if it appears on any other account, change that too.
3. **Review provider access logs** for the exposure window (first commit to
   today) for activity you do not recognise.
4. **Purge history** with `git filter-repo --path dd.env --path
   phase2/data/build_ssa_dataset.py --invert-paths`, or accept the repository as
   compromised and start a fresh one. Force-pushing a rewritten history breaks
   every existing clone, so coordinate it.
5. **Rotate again after the purge**, because the pre-purge objects may persist in
   forks, caches and the GitHub API for some time.

## 1.3 What has been fixed in code

- `dd.env` untracked; `*.env` and `dd.env` gitignored.
- Hardcoded credentials removed from `phase2/data/build_ssa_dataset.py` and
  replaced with `os.environ.get(...)` plus an explicit warning when unset.
- `.env.example` added as the template, with no real values.
- `phase11/auth_store.json` (password hashes and **live session tokens**) was
  untracked in an earlier commit on this branch; the store now discards imported
  session tokens on migration precisely because they were public.

---

# 2. Authentication

| Control | Status | Evidence |
|---|---|---|
| Password hashing | PBKDF2-HMAC-SHA256, 240 000 iterations, 16-byte per-user salt | `test_security.py::test_hash_is_salted_and_verifies` |
| Legacy hash migration | Single-round SHA-256 records verified and upgraded on next sign-in | `test_legacy_hashes_verify_and_request_an_upgrade` |
| Password policy | >= 10 chars, 3 of 4 character classes, published demo passwords banned | `test_weak_passwords_are_refused`, `test_published_demo_passwords_are_refused` |
| Token entropy | 256-bit, `secrets.token_urlsafe(32)`; 500 tokens all distinct | `test_tokens_are_high_entropy_and_unique` |
| Token storage | **Only a SHA-256 digest is persisted** — a stolen database yields no live sessions | `test_only_the_digest_is_ever_stored` (scans the WAL sidecar too) |
| Token expiry | Absolute, default 12 h, enforced on read | `test_session_resolves_then_expires` |
| Logout | Immediate revocation, verified against an owned resource | `test_a_revoked_token_stops_working_immediately` |
| Logout-all | Revokes every session for the account | `test_logout_all_revokes_every_session` |
| Brute-force protection | Sliding-window limiter, 10 failed logins / 5 min / client | `test_limiter_blocks_after_the_limit` |
| Username enumeration | Identical status and message for unknown user and wrong password; comparable work is spent on both | `test_login_does_not_reveal_whether_an_account_exists` |
| Timing | `hmac.compare_digest` for every secret comparison | `security.py` |
| Forged tokens | 5 malformed `Authorization` shapes all rejected | `test_malformed_or_forged_tokens_are_rejected` |

**A defect found in this pass:** the rate-limit key originally included the
client-supplied `X-Forwarded-For` header. Keying on it hands an attacker a fresh,
empty bucket per request by incrementing a number — the limiter would count to
ten and never fire. It now keys on `request.client.host` only, and resolving the
real client behind a proxy is delegated to uvicorn's `--forwarded-allow-ips`,
which is the layer that knows which peers are trusted.

---

# 3. Authorization and IDOR

**Model: ownership only.** There are no roles, no admin, no read-only accounts
and no organizational grouping. That is a limitation, recorded in section 8 —
not a defect, but it bounds what this system can be used for.

Every check below is made **against the server**. A console that hides a button
is not an access control.

## Principals exercised

| Principal | Holdings |
|---|---|
| `operator_a` | asset 90001 |
| `operator_b` | asset 90002 |
| `operator_c` | nothing (freshly registered) |
| anonymous | no token |

## IDOR results — all PASS

| Attack | Result |
|---|---|
| Screen another operator's asset (`/user/screen/{norad}`) | **403** for both `operator_b` and `operator_c` |
| Plan a maneuver on another operator's asset (`/user/plan`) | **403** — the highest-consequence boundary in the system |
| Read another operator's screening history | **403** |
| Deregister another operator's asset | **403/404**, asset survives |
| Hijack an asset by re-registering its element set | **409**, ownership unchanged |
| Enumerate which operator owns what | Others are labelled `"another operator"`; no account name appears in any response |
| Use a 403 as an ownership oracle | The owner is not named in the refusal |
| Use operator A's token as operator B | Each token resolves only to its own account |

`/my/satellites` was also verified to list only the caller's own assets under
`mine`.

---

# 4. Input security

| Class | Result |
|---|---|
| XSS (`<script>`, `{{7*7}}`, `${jndi:...}`) | Rejected in usernames; stored inertly as data in asset names; the console builds DOM nodes and never parses API values as HTML |
| SQL injection (`'; DROP TABLE users; --`, `' OR '1'='1`) | Every query is parameterized; payloads rejected by the username allowlist or stored verbatim |
| Command injection (`` `id` ``, `$(rm -rf /)`, `; cat /etc/shadow`) | Rejected; no `subprocess`/`os.system` anywhere in the request path |
| Path traversal (`../../../../etc/passwd`, Windows form) | 403/404/422, never 500; no path is ever built from user input |
| Malformed JSON | 400/422 |
| Oversized request (3 MB) | 413, refused before parsing |
| Malicious element-set content | 422 from `validate_tle` |
| Unicode direction-override / BOM / NUL | Rejected by the username allowlist |

**Usernames use an allowlist** (`[a-z0-9._-]`), which is why the payload list
above is sufficient: no denylist has to keep pace with new injection syntax.

**A NUL byte in a URL path** is rejected by the HTTP client and by every
conforming proxy before it reaches the service. That is noted in the suite rather
than tested through the server, because the test would be measuring httpx.

---

# 5. Information disclosure

Verified across five error paths (unknown NORAD, unknown asset, bad type, missing
parameter, unknown library key) that **no** response contains: `Traceback`,
`File "`, `site-packages`, `sqlite3`, `SELECT `, a module filename, or a
filesystem path.

- All errors use one envelope with a stable machine-readable `code`.
- Unhandled exceptions are logged with a traceback server-side and returned as a
  generic `internal_error` carrying only a request id.
- Security headers are present on **error** responses too, not just successes.
- Every response carries `X-Request-ID`.

---

# 6. Transport and browser controls

| Header | Value | Applies |
|---|---|---|
| `X-Content-Type-Options` | `nosniff` | always |
| `X-Frame-Options` | `DENY` | always |
| `Referrer-Policy` | `no-referrer` | always |
| `Content-Security-Policy` | `default-src 'none'; frame-ancestors 'none'` | always (API) |
| `Cross-Origin-Opener-Policy` | `same-origin` | always |
| `Cache-Control` | `no-store` | always |
| `Strict-Transport-Security` | `max-age=31536000; includeSubDomains` | production only |

The console runs under `script-src 'self'` with no `unsafe-inline`: all script is
in `console.js`, all style in `console.css`, and three.js and satellite.js are
vendored locally rather than loaded from a CDN.

**CSRF is not applicable**: authentication is a `Authorization: Bearer` header,
never a cookie, so a cross-site form post cannot carry credentials. This is
stated explicitly because "no CSRF token" otherwise looks like an omission.

---

# 7. Denial of service — a defect found and fixed

`/screen`, `/monitor`, `/assess`, `/maneuver-plan`, `/autonomous`,
`/live/conjunctions` and `/sync-now` were **unauthenticated and unlimited**. Each
propagates the entire 16 000-object catalog: one HTTP request costs roughly a
CPU-second, and nothing stopped a caller issuing thousands. `/sync-now`
additionally triggers outbound fetches to CelesTrak, so it could be used to
direct traffic at a third party.

Two controls, deliberately separate:

- **Rate limit**, always on: 30 requests / 60 s / client, so no single caller can
  saturate the service even where anonymous access is intended.
- **Authentication requirement**, on by default in production
  (`ISDMAAS_REQUIRE_AUTH_FOR_COMPUTE`), so a public deployment does not hand its
  CPU to anonymous callers at all.

Development leaves them open so the console's read-only mode and the demo
walkthrough keep working. Verified in production mode: `/screen`, `/monitor` and
`/sync-now` return **401**, while `/health` stays open and `/docs` returns 404.

---

# 8. Production configuration guards

The service **refuses to start** in production without a safe configuration.
Each is a test.

| Misconfiguration | Result |
|---|---|
| No `ISDMAAS_SECRET_KEY` | `ConfigError`, refuses to start |
| Secret key shorter than 32 chars | `ConfigError` |
| No `ISDMAAS_CORS_ORIGINS` | `ConfigError` |
| Wildcard CORS origin | `ConfigError` |
| Unknown `ISDMAAS_ENV` value | `ConfigError` |

And a valid production configuration was verified to disable demo seeding,
require auth for compute endpoints, emit JSON logs, and turn off `/docs` and
`/openapi.json`.

---

# 9. Accepted limitations

These are real and they bound the deployment envelope. None is a defect; all
would need addressing before a multi-tenant public service.

| Limitation | Consequence | Mitigation today |
|---|---|---|
| **No role model** — authorization is ownership only | No admin, no read-only auditor, no organizational grouping | Single-tenant or small trusted-operator pilot only |
| **Rate limiting is per process** | N uvicorn workers multiply every limit by N | Run one worker, or front with a shared limiter (Redis) |
| **Sessions are node-local** | A rolling restart signs every operator out | Single node, or sticky sessions |
| **No MFA** | Password compromise is full account compromise | Strong password policy, rate limiting, TLS |
| **No password reset** | Account recovery is a manual database edit | Documented in DEPLOY.md |
| **No tamper-evident audit log** | Authorization decisions are logged but not signed | Structured logs with request ids |
| **Element-set history is not encrypted at rest** | Disk access reveals operator asset history | Filesystem permissions, encrypted volume |

---

# 10. Reproducing

```bash
cd phase11
python -m pytest tests/security tests/test_security.py -v
```

119 security tests, ~5 s. The production-configuration tests spawn subprocesses
with patched environments, because settings are a process-wide singleton read at
import time.
