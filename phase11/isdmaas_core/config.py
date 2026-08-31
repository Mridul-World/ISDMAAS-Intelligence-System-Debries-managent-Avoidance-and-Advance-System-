"""
config.py — typed, environment-driven settings for ISDMAAS.

Every tunable in the service is declared here once. Nothing else in the codebase
reads os.environ directly, and nothing resolves a path relative to the current
working directory: paths are anchored to the package so the service behaves the
same whether it is started from phase11/, from the repo root, or from a
container WORKDIR.

Environment variables (all optional; defaults are safe for local development):

  ISDMAAS_ENV                 development | production      (default development)
  ISDMAAS_DATA_DIR            writable directory for caches/DBs
  ISDMAAS_PHASE55_DIR         location of phase55 (calibrated state files)
  ISDMAAS_CORS_ORIGINS        comma-separated allowed origins
  ISDMAAS_SECRET_KEY          required in production; used to sign session tokens
  ISDMAAS_TOKEN_TTL_HOURS     session lifetime                    (default 12)
  ISDMAAS_LOGIN_RATE          max failed logins per IP per window (default 10)
  ISDMAAS_LOGIN_WINDOW_S      rate-limit window in seconds        (default 300)
  ISDMAAS_ENABLE_DEMO_SEED    seed the two demo operators         (default 1 in
                              development, 0 in production)
  ISDMAAS_SCREEN_MAX_HOURS    upper bound on a screening window   (default 72)
  ISDMAAS_LOG_LEVEL           DEBUG | INFO | WARNING | ERROR      (default INFO)
  ISDMAAS_LOG_JSON            1 to emit JSON logs                 (default 1 in
                              production, 0 in development)
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# phase11/isdmaas_core/config.py  ->  phase11/
PACKAGE_DIR = Path(__file__).resolve().parent
APP_DIR = PACKAGE_DIR.parent
REPO_DIR = APP_DIR.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if not raw:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _resolve(raw: str, fallback: Path) -> Path:
    """Resolve a configured path; relative values are anchored to phase11/."""
    if not raw:
        return fallback
    p = Path(raw).expanduser()
    return p if p.is_absolute() else (APP_DIR / p).resolve()


class ConfigError(RuntimeError):
    """Raised when the process is configured in a way that is unsafe to run."""


@dataclass(frozen=True)
class Settings:
    # ---------------------------------------------------------------- runtime
    env: str = "development"
    log_level: str = "INFO"
    log_json: bool = False

    # ------------------------------------------------------------------ paths
    data_dir: Path = field(default_factory=lambda: APP_DIR / "data")
    phase55_dir: Path = field(default_factory=lambda: REPO_DIR / "phase55")

    # ----------------------------------------------------------------- server
    cors_origins: List[str] = field(default_factory=list)
    secret_key: str = ""
    request_timeout_s: float = 300.0
    max_request_bytes: int = 2 * 1024 * 1024

    # ------------------------------------------------------------------- auth
    token_ttl_hours: int = 12
    login_rate_limit: int = 10
    login_rate_window_s: int = 300
    enable_demo_seed: bool = True
    pbkdf2_iterations: int = 240_000

    # ------------------------------------------------------------ screening
    screen_max_hours: float = 72.0
    screen_coarse_step_s: float = 30.0
    catalog_ttl_s: int = 6 * 3600
    hbr_km: float = 0.020

    # ------------------------------------------------------------- derived
    @property
    def is_production(self) -> bool:
        return self.env == "production"

    @property
    def catalog_tle_cache(self) -> Path:
        """Writable catalog cache. Refreshed on a TTL; never version-controlled."""
        return self.data_dir / "catalog_cache.tle"

    @property
    def catalog_tle_fallback(self) -> Path:
        """
        Read-only offline catalog, committed to the repository.

        Kept separate from the live cache on purpose: when the service wrote its
        refreshed catalog straight over the committed file, every six hours
        produced a sixty-thousand-line diff, and the offline fallback quietly
        became whatever was last downloaded. This copy is only ever read.
        """
        return APP_DIR / "data" / "catalog_active.tle"

    @property
    def object_cache_db(self) -> Path:
        return self.data_dir / "isdmaas_cache.db"

    @property
    def ops_db(self) -> Path:
        return self.data_dir / "isdmaas_ops.db"

    @property
    def auth_db(self) -> Path:
        return self.data_dir / "isdmaas_auth.db"

    @property
    def catalog_json_cache(self) -> Path:
        return self.data_dir / "catalog_cache.json"

    def state_file(self, norad: int) -> Path:
        return self.phase55_dir / "reports" / f"state_{norad}.json"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_settings() -> Settings:
    """Build Settings from the environment, refusing unsafe production setups."""
    env = (_env("ISDMAAS_ENV", "development") or "development").lower()
    if env not in ("development", "production"):
        raise ConfigError(
            f"ISDMAAS_ENV must be 'development' or 'production', got {env!r}"
        )
    is_prod = env == "production"

    origins_raw = _env("ISDMAAS_CORS_ORIGINS")
    origins = [o.strip() for o in origins_raw.split(",") if o.strip()]

    secret = _env("ISDMAAS_SECRET_KEY")
    if is_prod:
        # A production deployment that silently invents a key on every restart
        # would log every operator out on each deploy and make token forgery
        # analysis impossible. Fail loudly instead.
        if not secret:
            raise ConfigError(
                "ISDMAAS_SECRET_KEY must be set when ISDMAAS_ENV=production. "
                "Generate one with: python -c \"import secrets;"
                "print(secrets.token_urlsafe(48))\""
            )
        if len(secret) < 32:
            raise ConfigError("ISDMAAS_SECRET_KEY must be at least 32 characters")
        if not origins:
            raise ConfigError(
                "ISDMAAS_CORS_ORIGINS must list the console's origin(s) when "
                "ISDMAAS_ENV=production (wildcard CORS is refused in production)"
            )
        if "*" in origins:
            raise ConfigError(
                "Wildcard CORS origin is refused in production. List the exact "
                "origins that serve the console."
            )
    else:
        secret = secret or secrets.token_urlsafe(48)
        origins = origins or [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:5500",
            "http://127.0.0.1:5500",
        ]

    settings = Settings(
        env=env,
        log_level=(_env("ISDMAAS_LOG_LEVEL", "INFO") or "INFO").upper(),
        log_json=_env_bool("ISDMAAS_LOG_JSON", is_prod),
        data_dir=_resolve(_env("ISDMAAS_DATA_DIR"), APP_DIR / "data"),
        phase55_dir=_resolve(_env("ISDMAAS_PHASE55_DIR"), REPO_DIR / "phase55"),
        cors_origins=origins,
        secret_key=secret,
        token_ttl_hours=_env_int("ISDMAAS_TOKEN_TTL_HOURS", 12),
        login_rate_limit=_env_int("ISDMAAS_LOGIN_RATE", 10),
        login_rate_window_s=_env_int("ISDMAAS_LOGIN_WINDOW_S", 300),
        enable_demo_seed=_env_bool("ISDMAAS_ENABLE_DEMO_SEED", not is_prod),
        screen_max_hours=_env_float("ISDMAAS_SCREEN_MAX_HOURS", 72.0),
        hbr_km=_env_float("ISDMAAS_HBR_KM", 0.020),
    )
    settings.ensure_dirs()
    return settings


_settings: Settings | None = None


def get_settings() -> Settings:
    """Process-wide settings singleton."""
    global _settings
    if _settings is None:
        _settings = load_settings()
    return _settings


def reset_settings_for_tests() -> None:
    """Drop the cached singleton so a test can re-read a patched environment."""
    global _settings
    _settings = None
