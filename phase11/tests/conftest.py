"""
Test configuration.

The pipeline modules (phase7_collision, phase11_api, ...) live in phase11/, one
level above this directory, and are imported by module name rather than as a
package. Put that directory on the path once, here, so every test file can
import them the same way the service does.

The data directory is redirected to a temporary location before anything is
imported: importing phase11_api creates the operator registry as a side effect,
and a test run must never write into a developer's real one.
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_DATA_DIR = Path(os.environ.setdefault(
    "ISDMAAS_DATA_DIR", tempfile.mkdtemp(prefix="isdmaas-tests-")
))
os.environ.setdefault("ISDMAAS_ENV", "development")
os.environ.setdefault("ISDMAAS_LOG_LEVEL", "WARNING")

# Seed the temporary data directory with the committed offline catalog. Without
# it the suite depends on CelesTrak being reachable, and a rate-limited upstream
# turns every screening assertion into a spurious failure.
_FALLBACK = APP_DIR / "data" / "catalog_active.tle"
if _FALLBACK.exists():
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _seeded = _DATA_DIR / "catalog_cache.tle"
    shutil.copy(_FALLBACK, _seeded)
    # Mark it fresh so the catalog service uses it directly instead of spending
    # a network timeout discovering that CelesTrak is unreachable.
    os.utime(_seeded, None)
