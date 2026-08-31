"""
isdmaas_core — production support layer for the ISDMAAS service.

Modules
-------
config          Typed, environment-driven settings (single source of truth).
logging_config  Structured JSON logging with per-request correlation ids.
errors          Uniform error envelope + FastAPI exception handlers.
security        Password hashing (PBKDF2), opaque session tokens, rate limiting.
astrodynamics   Accurate SGP4 propagation, vectorized screening, exact TCA.

Nothing in here imports the pipeline modules (phase7/8/10), so it can be reused
by the trainer, the CLI tools and the API without circular imports.
"""

__version__ = "2.0.0"
