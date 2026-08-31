# Deploying ISDMAAS

This document covers running the service somewhere other than a developer's
laptop. If you only want it running locally, skip to [Local development](#local-development).

---

## 1. What "production mode" changes

Setting `ISDMAAS_ENV=production` is not cosmetic. The service refuses to start
unless it is configured safely, and it turns off everything that exists only to
make a demo convenient:

| | development | production |
|---|---|---|
| `ISDMAAS_SECRET_KEY` | generated per start | **required**, ≥ 32 chars |
| `ISDMAAS_CORS_ORIGINS` | localhost defaults | **required**, wildcard refused |
| Demo operator accounts | seeded | not seeded |
| `GET /auth/demo-info` | returns credentials | 404 |
| Interactive docs (`/docs`, `/openapi.json`) | on | off |
| `Strict-Transport-Security` | off | on |
| Log format | human-readable | JSON, one object per line |

The service exits with an explanatory error rather than starting in a state that
looks fine and is not. That is deliberate: a misconfigured deployment that boots
successfully is worse than one that refuses to.

---

## 2. Required configuration

```bash
# Generate once, then store it in your secret manager. Rotating it signs every
# operator out; it does not invalidate their passwords.
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

```bash
export ISDMAAS_ENV=production
export ISDMAAS_SECRET_KEY='<the value you just generated>'
export ISDMAAS_CORS_ORIGINS='https://console.example.org'
export ISDMAAS_DATA_DIR=/var/lib/isdmaas       # must be writable and backed up
export ISDMAAS_PHASE55_DIR=/opt/isdmaas/phase55
```

Optional, with their defaults:

| Variable | Default | Notes |
|---|---|---|
| `ISDMAAS_TOKEN_TTL_HOURS` | 12 | session lifetime |
| `ISDMAAS_LOGIN_RATE` | 10 | failed sign-ins per window, per client |
| `ISDMAAS_LOGIN_WINDOW_S` | 300 | that window, in seconds |
| `ISDMAAS_SCREEN_MAX_HOURS` | 72 | ceiling on any screening window |
| `ISDMAAS_HBR_KM` | 0.020 | default combined hard-body radius |
| `ISDMAAS_LOG_LEVEL` | INFO | |
| `ISDMAAS_LOG_JSON` | on in production | |

---

## 3. Run it

### Container

```bash
docker build -t isdmaas:latest phase11
docker run -d --name isdmaas \
  -p 127.0.0.1:8000:8000 \
  -v isdmaas-data:/data \
  -e ISDMAAS_SECRET_KEY="$ISDMAAS_SECRET_KEY" \
  -e ISDMAAS_CORS_ORIGINS="https://console.example.org" \
  isdmaas:latest
```

The image runs as an unprivileged user and writes only to `/data`. Bind the
published port to loopback and put TLS in front of it — the service speaks plain
HTTP by design, because terminating TLS is the reverse proxy's job.

### Directly

```bash
cd phase11
uvicorn phase11_api:app --host 127.0.0.1 --port 8000 \
        --workers 4 --proxy-headers --forwarded-allow-ips '*'
```

`--proxy-headers` is what makes `X-Forwarded-For` reach the rate limiter. Without
it every request appears to come from the proxy and one client can exhaust the
login limit for everyone.

**On worker count.** Screening is CPU-bound NumPy and SGP4, and it releases the
GIL, so workers scale nearly linearly. Size them to available cores, not to
expected concurrency: a single full-catalog screen will use a core for a second
or two.

---

## 4. Serving the console

The console is static files (`dashboard.html`, `console.css`, `console.js`,
`vendor/`). Serve them from **the same origin as the API**. That removes CORS
from the picture, avoids mixed-content problems, and lets the console's
`connect-src 'self'` policy do its job with nothing else allowed.

An nginx front end that does both:

```nginx
server {
    listen 443 ssl http2;
    server_name console.example.org;

    ssl_certificate     /etc/ssl/certs/isdmaas.crt;
    ssl_certificate_key /etc/ssl/private/isdmaas.key;

    # The console is a static page; the API sets its own headers, and
    # `always` makes sure these still apply on error responses.
    add_header X-Frame-Options            "DENY"                always;
    add_header X-Content-Type-Options     "nosniff"             always;
    add_header Referrer-Policy            "no-referrer"         always;
    add_header Content-Security-Policy    "frame-ancestors 'none'" always;
    add_header Strict-Transport-Security  "max-age=31536000; includeSubDomains" always;

    root /opt/isdmaas/app;
    index dashboard.html;

    location / {
        try_files $uri $uri/ =404;
    }

    # Everything the API owns, proxied to the service on the same origin.
    location ~ ^/(health|ready|satellites|resolve|orbit|screen|monitor|assess|
                  cdm|maneuver-plan|autonomous|socrates|live|library|
                  historical-events|sync-status|sync-now|auth|my|user) {
        proxy_pass         http://127.0.0.1:8000;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # A full-catalog screen legitimately takes tens of seconds.
        proxy_read_timeout 300s;
    }
}
```

`frame-ancestors` must be delivered as an HTTP header — browsers ignore it in a
`<meta>` tag. That is why it is set here and not in `dashboard.html`.

---

## 5. Data the service needs

| Path | What it is | If it is missing |
|---|---|---|
| `$ISDMAAS_DATA_DIR/catalog_active.tle` | public element-set catalog | fetched from CelesTrak on first use |
| `$ISDMAAS_DATA_DIR/isdmaas_auth.db` | operators, sessions, registered assets | created empty |
| `$ISDMAAS_DATA_DIR/isdmaas_ops.db` | element-set history, screening records | created empty |
| `$ISDMAAS_DATA_DIR/isdmaas_cache.db` | object cache for the 3-D view | `/live/catalog` returns 503 |
| `$ISDMAAS_PHASE55_DIR/reports/state_*.json` | calibrated precise-orbit states | falls back to element sets with TLE-scale uncertainty |

Build the object cache once after deployment:

```bash
python debris_data.py sync
```

CelesTrak asks callers to stay under one request per second and not to
re-download unchanged data. The catalog service honours that: it caches for six
hours, coalesces concurrent refreshes behind a lock, and falls back to a stale
cache rather than retrying when the upstream is down. **Do not** add a cron job
that calls `/sync-now` more often than hourly.

---

## 6. Back these up

`$ISDMAAS_DATA_DIR/isdmaas_auth.db` and `isdmaas_ops.db` are the only files that
cannot be regenerated. They hold operator accounts and the element-set history
that future model training depends on. The catalog and object caches are
disposable.

---

## 7. Health and monitoring

| Endpoint | Use it for |
|---|---|
| `GET /health` | liveness — is the process up |
| `GET /ready` | readiness — catalog loaded and registry reachable; 503 otherwise |

Point a container or load-balancer liveness probe at `/health` and a readiness
probe at `/ready`. `/ready` also reports the catalog's age, which is the number
to alert on: a catalog more than 24 hours old means screening results reflect
stale element sets, and the console shows a warning to the operator when it does.

Every response carries `X-Request-ID`. Logs are JSON in production and include
that id, so one failed screen can be traced end to end from a single grep.

---

## 8. Security posture, stated plainly

What is in place:

- PBKDF2-HMAC-SHA256 password hashing (240 000 iterations), per-user salt,
  transparent upgrade of legacy hashes on next sign-in.
- Opaque session tokens with absolute expiry; only a SHA-256 digest is stored.
- Per-client rate limiting on sign-in and registration; constant-work responses
  for unknown usernames.
- Owner-only maneuver authority, enforced server-side before any computation.
- Strict CSP on the console, no inline script, libraries vendored locally.
- Security headers on every API response; request size cap; bounded and
  validated query parameters on every endpoint.

What is **not** in place, and should be discussed before any pilot:

- **No multi-factor authentication and no password reset flow.** Account
  recovery is a manual database operation today.
- **Rate limiting is per process.** Behind more than one worker or instance an
  attacker gets one bucket per instance. A shared limiter (Redis) is needed for
  a real deployment.
- **No audit log export.** Authorization decisions are logged, but there is no
  tamper-evident audit trail.
- **Sessions are stored on one node.** There is no session replication, so
  rolling restarts sign operators out.
- **The service has no authorization model beyond ownership.** There are no
  roles, no read-only accounts, and no organizational grouping.

None of these block a pilot behind TLS with a small number of known operators.
All of them block a multi-tenant public deployment.

---

## 9. Verifying a deployment

```bash
# Refuses to start without a secret key
ISDMAAS_ENV=production python -c "from isdmaas_core.config import load_settings; load_settings()"
# -> ConfigError: ISDMAAS_SECRET_KEY must be set when ISDMAAS_ENV=production

# Demo credentials are not served
curl -s https://console.example.org/auth/demo-info | jq .error.code   # "not_found"

# Interactive docs are off
curl -so /dev/null -w '%{http_code}\n' https://console.example.org/docs   # 404

# Security headers are present
curl -sI https://console.example.org/health | grep -Ei 'x-frame|nosniff|strict-transport'

# The full test suite
cd phase11 && pytest tests -q
```
