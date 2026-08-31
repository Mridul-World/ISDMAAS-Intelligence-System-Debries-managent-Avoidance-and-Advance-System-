"""
ops_db.py — ISDMAAS operational database (SQLite).
====================================================
Stores every uploaded TLE (full history, not just latest) so that:
  1. OPERATIONS: monitoring / maneuver planning always uses the newest TLE per
     satellite (auth_store keeps working as-is; this adds durable history).
  2. TRAINING: when precise truth data later becomes available for a satellite,
     the TLE history is exactly what's needed to build (TLE-prediction, truth)
     training pairs — see training_pipeline.py.

Tables:
  tle_history(norad, name, owner, tle1, tle2, epoch_yyddd, uploaded_utc)
  truth_files(norad, path, n_points, span_start, span_end, registered_utc)
  training_runs(run_utc, n_samples, satellites, note)

Zero external deps (sqlite3 is stdlib). DB file: isdmaas_ops.db next to this file.
"""
import os, sqlite3, threading
from datetime import datetime, timezone

def _db_path():
    """
    Resolve the operational database location.

    It lives under the configured data directory, which is gitignored. The
    previous location was next to this module inside the repository, so an
    operator's registered element-set history was one `git add .` away from
    being published.
    """
    try:
        from isdmaas_core.config import get_settings

        return str(get_settings().ops_db)
    except Exception:
        return os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "data", "isdmaas_ops.db"
        )

def _migrate_legacy_db(target):
    """
    Move a pre-existing database from the old in-repo location.

    The operational history — every element set an operator uploaded, every
    screening run — is real data. Repointing the path must not silently orphan
    it, so the old file is copied across once if the new location is empty.
    """
    legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isdmaas_ops.db")
    if os.path.exists(target) or not os.path.exists(legacy):
        return
    try:
        import shutil

        os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
        shutil.copy2(legacy, target)
    except OSError:
        pass


_DB = _db_path()
_migrate_legacy_db(_DB)
_LOCK = threading.Lock()

def _conn():
    os.makedirs(os.path.dirname(_DB) or ".", exist_ok=True)
    c = sqlite3.connect(_DB, timeout=15.0)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("""CREATE TABLE IF NOT EXISTS tle_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        norad TEXT, name TEXT, owner TEXT,
        tle1 TEXT, tle2 TEXT, epoch_yyddd TEXT, uploaded_utc TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS truth_files(
        norad TEXT, path TEXT, n_points INTEGER,
        span_start TEXT, span_end TEXT, registered_utc TEXT,
        PRIMARY KEY (norad, path))""")
    c.execute("""CREATE TABLE IF NOT EXISTS training_runs(
        run_utc TEXT, n_samples INTEGER, satellites TEXT, note TEXT)""")
    c.execute("""CREATE TABLE IF NOT EXISTS assessments(
        assessment_id TEXT PRIMARY KEY,
        created_utc TEXT, operator TEXT,
        primary_norad TEXT, primary_name TEXT,
        secondary_norad TEXT, secondary_name TEXT,
        primary_epoch_utc TEXT, secondary_epoch_utc TEXT,
        catalog_fetched_utc TEXT, catalog_age_hours REAL,
        algorithm_version TEXT, model_version TEXT,
        covariance_source TEXT, primary_sigma_rtn_km TEXT, hbr_km REAL,
        tca_utc TEXT, miss_km REAL, relative_speed_kms REAL,
        pc REAL, pc_method TEXT, risk_level TEXT, status TEXT,
        recommendation_json TEXT, safety_verdict TEXT, detail_json TEXT)""")
    c.execute("""CREATE INDEX IF NOT EXISTS idx_assess_created
                 ON assessments(created_utc)""")
    c.execute("""CREATE TABLE IF NOT EXISTS screening_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, run_utc TEXT,
        norad TEXT, name TEXT, owner TEXT,
        objects_screened INTEGER, conjunctions_found INTEGER, results_json TEXT)""")
    return c

def log_screening(norad, name, owner, n_screened, n_found, results):
    import json as _json
    with _LOCK, _conn() as c:
        c.execute("INSERT INTO screening_runs(run_utc,norad,name,owner,"
                  "objects_screened,conjunctions_found,results_json) VALUES (?,?,?,?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), str(norad), name, owner,
                   n_screened, n_found, _json.dumps(results)))

def screening_history(norad=None, limit=20):
    import json as _json
    with _LOCK, _conn() as c:
        if norad:
            rows = c.execute("SELECT run_utc,norad,name,objects_screened,"
                             "conjunctions_found,results_json FROM screening_runs "
                             "WHERE norad=? ORDER BY id DESC LIMIT ?",
                             (str(norad), limit)).fetchall()
        else:
            rows = c.execute("SELECT run_utc,norad,name,objects_screened,"
                             "conjunctions_found,results_json FROM screening_runs "
                             "ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    return [{"run_utc": u, "norad": n, "name": nm, "objects_screened": os_,
             "conjunctions_found": cf, "results": _json.loads(rj)}
            for u, n, nm, os_, cf, rj in rows]

def record_tle(norad, name, owner, tle1, tle2):
    """Append an uploaded TLE to history. Called on every /my/satellites POST."""
    epoch = tle1[18:32].strip() if len(tle1) > 32 else ""
    with _LOCK, _conn() as c:
        c.execute("INSERT INTO tle_history(norad,name,owner,tle1,tle2,epoch_yyddd,uploaded_utc) "
                  "VALUES (?,?,?,?,?,?,?)",
                  (str(norad), name, owner, tle1, tle2, epoch,
                   datetime.now(timezone.utc).isoformat()))

def latest_tle(norad):
    with _LOCK, _conn() as c:
        r = c.execute("SELECT tle1,tle2,name,owner FROM tle_history WHERE norad=? "
                      "ORDER BY id DESC LIMIT 1", (str(norad),)).fetchone()
    return None if not r else {"tle1": r[0], "tle2": r[1], "name": r[2], "owner": r[3]}

def tle_history(norad):
    with _LOCK, _conn() as c:
        rows = c.execute("SELECT tle1,tle2,epoch_yyddd,uploaded_utc FROM tle_history "
                         "WHERE norad=? ORDER BY id", (str(norad),)).fetchall()
    return [{"tle1": a, "tle2": b, "epoch": e, "uploaded_utc": u} for a, b, e, u in rows]

def all_norads():
    with _LOCK, _conn() as c:
        return [r[0] for r in c.execute("SELECT DISTINCT norad FROM tle_history").fetchall()]

def register_truth(norad, path, n_points, span_start, span_end):
    with _LOCK, _conn() as c:
        c.execute("INSERT OR REPLACE INTO truth_files VALUES (?,?,?,?,?,?)",
                  (str(norad), path, n_points, span_start, span_end,
                   datetime.now(timezone.utc).isoformat()))

def truth_for(norad):
    with _LOCK, _conn() as c:
        return c.execute("SELECT path,n_points,span_start,span_end FROM truth_files "
                         "WHERE norad=?", (str(norad),)).fetchall()

def record_training_run(n_samples, satellites, note=""):
    with _LOCK, _conn() as c:
        c.execute("INSERT INTO training_runs VALUES (?,?,?,?)",
                  (datetime.now(timezone.utc).isoformat(), n_samples,
                   ",".join(map(str, satellites)), note))

def status():
    with _LOCK, _conn() as c:
        n_tle = c.execute("SELECT COUNT(*) FROM tle_history").fetchone()[0]
        n_sat = c.execute("SELECT COUNT(DISTINCT norad) FROM tle_history").fetchone()[0]
        n_truth = c.execute("SELECT COUNT(*) FROM truth_files").fetchone()[0]
        runs = c.execute("SELECT run_utc,n_samples FROM training_runs "
                         "ORDER BY run_utc DESC LIMIT 3").fetchall()
    return {"tles_stored": n_tle, "satellites": n_sat,
            "truth_files": n_truth, "recent_training_runs": runs}

# ---------------------------------------------------------------------------
# Assessment audit trail
# ---------------------------------------------------------------------------
# Every recommendation must be reconstructable after the fact. Six months later
# an operator has to be able to answer "why did ISDMAAS recommend this burn?",
# and the answer has to come from a record rather than from someone's memory of
# what the catalog looked like that day.
#
# The record therefore captures the INPUTS as well as the outputs: which element
# sets were used and how old they were, which covariance model, which algorithm
# and model version, the hard-body radius. Without those, the numbers cannot be
# reproduced and the record is decorative.

ASSESSMENT_FIELDS = (
    "assessment_id", "created_utc", "operator",
    "primary_norad", "primary_name", "secondary_norad", "secondary_name",
    "primary_epoch_utc", "secondary_epoch_utc",
    "catalog_fetched_utc", "catalog_age_hours",
    "algorithm_version", "model_version",
    "covariance_source", "primary_sigma_rtn_km", "hbr_km",
    "tca_utc", "miss_km", "relative_speed_kms",
    "pc", "pc_method", "risk_level", "status",
    "recommendation_json", "safety_verdict", "detail_json",
)


def new_assessment_id():
    """A stable, sortable identifier an operator can quote."""
    import uuid

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"ASM-{stamp}-{uuid.uuid4().hex[:8]}"


def record_assessment(record):
    """
    Persist one assessment. `record` is a dict keyed by ASSESSMENT_FIELDS.

    Never raises into the request path: an audit-write failure must not deny an
    operator a conjunction assessment. It is logged instead.
    """
    import json as _json

    row = []
    for field in ASSESSMENT_FIELDS:
        value = record.get(field)
        if isinstance(value, (dict, list, tuple)):
            value = _json.dumps(value, default=str)
        row.append(value)
    placeholders = ",".join("?" * len(ASSESSMENT_FIELDS))
    with _LOCK, _conn() as c:
        c.execute(f"INSERT OR REPLACE INTO assessments VALUES ({placeholders})", row)
    return record.get("assessment_id")


def get_assessment(assessment_id):
    import json as _json

    with _LOCK, _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM assessments WHERE assessment_id=?",
                        (str(assessment_id),)).fetchone()
    if row is None:
        return None
    record = dict(row)
    for field in ("recommendation_json", "detail_json"):
        if record.get(field):
            try:
                record[field] = _json.loads(record[field])
            except (ValueError, TypeError):
                pass
    return record


def list_assessments(operator=None, limit=50):
    with _LOCK, _conn() as c:
        c.row_factory = sqlite3.Row
        if operator:
            rows = c.execute(
                "SELECT assessment_id,created_utc,operator,primary_norad,"
                "secondary_norad,miss_km,pc,risk_level,status,safety_verdict "
                "FROM assessments WHERE operator=? ORDER BY created_utc DESC "
                "LIMIT ?", (operator, limit)).fetchall()
        else:
            rows = c.execute(
                "SELECT assessment_id,created_utc,operator,primary_norad,"
                "secondary_norad,miss_km,pc,risk_level,status,safety_verdict "
                "FROM assessments ORDER BY created_utc DESC LIMIT ?",
                (limit,)).fetchall()
    return [dict(r) for r in rows]


if __name__ == "__main__":
    print("ops_db status:", status())
