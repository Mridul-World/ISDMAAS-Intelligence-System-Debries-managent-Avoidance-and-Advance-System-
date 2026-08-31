"""
Security validation: the production configuration guards.

These run `load_settings()` in a subprocess with a patched environment, because
settings are a process-wide singleton and the API module reads them at import
time. Doing it in-process would either contaminate the rest of the suite or
require reload gymnastics that prove nothing.
"""
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

APP_DIR = Path(__file__).resolve().parent.parent.parent
LONG_KEY = "k" * 40


def _load_settings(env):
    """Run load_settings() in a clean subprocess and report the outcome."""
    script = textwrap.dedent(
        """
        import json, sys
        sys.path.insert(0, sys.argv[1])
        try:
            from isdmaas_core.config import load_settings
            s = load_settings()
            print(json.dumps({
                "ok": True,
                "env": s.env,
                "demo_seed": s.enable_demo_seed,
                "require_auth_for_compute": s.require_auth_for_compute,
                "log_json": s.log_json,
                "cors": s.cors_origins,
            }))
        except Exception as exc:
            print(json.dumps({"ok": False, "error": type(exc).__name__,
                              "message": str(exc)}))
        """
    )
    base = {
        "PATH": "",
        "SYSTEMROOT": "C:\\\\Windows",
        "PYTHONIOENCODING": "utf-8",
    }
    completed = subprocess.run(
        [sys.executable, "-c", script, str(APP_DIR)],
        capture_output=True, text=True, env={**base, **env}, timeout=120,
    )
    line = [l for l in completed.stdout.splitlines() if l.startswith("{")]
    assert line, f"no result from subprocess: {completed.stdout} {completed.stderr}"
    return json.loads(line[-1])


# ==================================================== refuses unsafe setups
def test_production_requires_a_secret_key():
    result = _load_settings({"ISDMAAS_ENV": "production"})
    assert not result["ok"]
    assert result["error"] == "ConfigError"
    assert "ISDMAAS_SECRET_KEY" in result["message"]


def test_production_rejects_a_short_secret_key():
    result = _load_settings(
        {"ISDMAAS_ENV": "production", "ISDMAAS_SECRET_KEY": "tooshort"}
    )
    assert not result["ok"]
    assert "at least 32" in result["message"]


def test_production_requires_explicit_cors_origins():
    result = _load_settings(
        {"ISDMAAS_ENV": "production", "ISDMAAS_SECRET_KEY": LONG_KEY}
    )
    assert not result["ok"]
    assert "ISDMAAS_CORS_ORIGINS" in result["message"]


def test_production_refuses_wildcard_cors():
    """
    A wildcard origin on a credentialed API lets any page on the internet make
    authenticated requests on an operator's behalf.
    """
    result = _load_settings({
        "ISDMAAS_ENV": "production",
        "ISDMAAS_SECRET_KEY": LONG_KEY,
        "ISDMAAS_CORS_ORIGINS": "*",
    })
    assert not result["ok"]
    assert "Wildcard" in result["message"]


def test_an_unknown_environment_name_is_refused():
    result = _load_settings({"ISDMAAS_ENV": "staging"})
    assert not result["ok"]
    assert "development" in result["message"]


# ================================================= correct production posture
def _valid_production():
    return {
        "ISDMAAS_ENV": "production",
        "ISDMAAS_SECRET_KEY": LONG_KEY,
        "ISDMAAS_CORS_ORIGINS": "https://console.example.org",
    }


def test_valid_production_configuration_loads():
    result = _load_settings(_valid_production())
    assert result["ok"], result
    assert result["env"] == "production"


def test_production_disables_demo_account_seeding():
    """Published demo passwords must never exist on a production deployment."""
    assert _load_settings(_valid_production())["demo_seed"] is False


def test_production_requires_auth_for_compute_endpoints():
    """
    Screening endpoints propagate the entire catalog. Anonymous access to them
    on a public deployment is a denial-of-service amplifier.
    """
    assert _load_settings(_valid_production())["require_auth_for_compute"] is True


def test_production_emits_structured_logs():
    assert _load_settings(_valid_production())["log_json"] is True


def test_development_keeps_compute_endpoints_open():
    """
    The console's read-only mode and the demo walkthrough rely on anonymous
    access; the rate limiter still applies. This is the deliberate difference
    between the two environments, so it is asserted rather than assumed.
    """
    result = _load_settings({"ISDMAAS_ENV": "development"})
    assert result["ok"]
    assert result["require_auth_for_compute"] is False
    assert result["demo_seed"] is True


def test_compute_auth_requirement_can_be_overridden_explicitly():
    """An operator running behind a private network may opt out knowingly."""
    env = {**_valid_production(), "ISDMAAS_REQUIRE_AUTH_FOR_COMPUTE": "0"}
    assert _load_settings(env)["require_auth_for_compute"] is False
