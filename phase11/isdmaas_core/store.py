"""
store.py — durable operator, session and satellite registry (SQLite).

Replaces the previous auth_store.json. That file held password hashes and LIVE
SESSION TOKENS in plaintext, and it was committed to the git repository — so
every clone of the repo carried working credentials for any deployment seeded
from it. Three things change here:

  * Storage moves to SQLite under the configured data directory, which is
    gitignored, so credentials cannot be committed by accident.
  * Only a SHA-256 digest of a session token is persisted. A copy of the
    database no longer yields usable sessions.
  * Sessions carry an absolute expiry and are swept on read.

Existing auth_store.json files are imported once on first start so nobody loses
their registered satellites. User records are imported with their legacy hash
format, which verify_password still understands and upgrades on next login.
Session tokens from the JSON file are deliberately NOT imported: they were
public, so every one of them is treated as compromised and the operator has to
log in again.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from .logging_config import get_logger
from .security import legacy_hash_record, token_digest

log = get_logger("isdmaas.store")

SCHEMA_VERSION = 1


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


@dataclass(frozen=True)
class Satellite:
    norad: str
    owner: str
    name: str
    tle1: str
    tle2: str
    updated_utc: str


@dataclass(frozen=True)
class Session:
    username: str
    expires_utc: datetime


class Store:
    """Thread-safe SQLite registry. One instance per process."""

    def __init__(self, db_path: Path, token_ttl_hours: int = 12):
        self.db_path = Path(db_path)
        self.token_ttl = timedelta(hours=max(1, int(token_ttl_hours)))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()

    # ------------------------------------------------------------- plumbing
    def _connection(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path), timeout=15.0, isolation_level=None
            )
            conn.row_factory = sqlite3.Row
            # WAL lets the screening endpoints read while a login writes.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        with self._write_lock:
            conn = self._connection()
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    def _init_schema(self) -> None:
        # executescript() issues its own COMMIT before running, so it cannot sit
        # inside an explicit transaction. DDL here is idempotent, and the lock
        # still serializes concurrent first-start races.
        with self._write_lock:
            conn = self._connection()
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS meta(
                    key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS users(
                    username      TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    created_utc   TEXT NOT NULL,
                    last_login_utc TEXT,
                    disabled      INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS sessions(
                    token_hash   TEXT PRIMARY KEY,
                    username     TEXT NOT NULL,
                    created_utc  TEXT NOT NULL,
                    expires_utc  TEXT NOT NULL,
                    last_seen_utc TEXT NOT NULL,
                    FOREIGN KEY(username) REFERENCES users(username)
                        ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(username);
                CREATE INDEX IF NOT EXISTS idx_sessions_exp ON sessions(expires_utc);
                CREATE TABLE IF NOT EXISTS satellites(
                    norad        TEXT PRIMARY KEY,
                    owner        TEXT NOT NULL,
                    name         TEXT NOT NULL,
                    tle1         TEXT NOT NULL,
                    tle2         TEXT NOT NULL,
                    updated_utc  TEXT NOT NULL,
                    FOREIGN KEY(owner) REFERENCES users(username)
                        ON DELETE CASCADE);
                CREATE INDEX IF NOT EXISTS idx_sat_owner ON satellites(owner);
                """
            )
            conn.execute(
                "INSERT OR REPLACE INTO meta(key,value) VALUES('schema_version',?)",
                (str(SCHEMA_VERSION),),
            )

    # ---------------------------------------------------------------- users
    def user_count(self) -> int:
        row = self._connection().execute("SELECT COUNT(*) AS n FROM users").fetchone()
        return int(row["n"])

    def get_user(self, username: str) -> Optional[sqlite3.Row]:
        return self._connection().execute(
            "SELECT * FROM users WHERE username=?", (username,)
        ).fetchone()

    def create_user(self, username: str, password_hash: str) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO users(username,password_hash,created_utc) VALUES(?,?,?)",
                (username, password_hash, _iso(_utc_now())),
            )

    def update_password_hash(self, username: str, password_hash: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE users SET password_hash=? WHERE username=?",
                (password_hash, username),
            )

    def mark_login(self, username: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE users SET last_login_utc=? WHERE username=?",
                (_iso(_utc_now()), username),
            )

    # ------------------------------------------------------------- sessions
    def create_session(self, username: str, token: str) -> datetime:
        now = _utc_now()
        expires = now + self.token_ttl
        with self._write() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions"
                "(token_hash,username,created_utc,expires_utc,last_seen_utc)"
                " VALUES(?,?,?,?,?)",
                (token_digest(token), username, _iso(now), _iso(expires), _iso(now)),
            )
        return expires

    def resolve_session(self, token: str) -> Optional[Session]:
        """Return the session for a token, or None if absent or expired."""
        digest = token_digest(token)
        row = self._connection().execute(
            "SELECT username, expires_utc FROM sessions WHERE token_hash=?", (digest,)
        ).fetchone()
        if row is None:
            return None
        expires = datetime.fromisoformat(row["expires_utc"])
        if expires <= _utc_now():
            self.delete_session(token)
            return None
        return Session(username=row["username"], expires_utc=expires)

    def touch_session(self, token: str) -> None:
        with self._write() as conn:
            conn.execute(
                "UPDATE sessions SET last_seen_utc=? WHERE token_hash=?",
                (_iso(_utc_now()), token_digest(token)),
            )

    def delete_session(self, token: str) -> None:
        with self._write() as conn:
            conn.execute(
                "DELETE FROM sessions WHERE token_hash=?", (token_digest(token),)
            )

    def delete_sessions_for(self, username: str) -> int:
        with self._write() as conn:
            cur = conn.execute("DELETE FROM sessions WHERE username=?", (username,))
            return cur.rowcount

    def purge_expired_sessions(self) -> int:
        with self._write() as conn:
            cur = conn.execute(
                "DELETE FROM sessions WHERE expires_utc <= ?", (_iso(_utc_now()),)
            )
            return cur.rowcount

    # ----------------------------------------------------------- satellites
    def get_satellite(self, norad: str) -> Optional[Satellite]:
        row = self._connection().execute(
            "SELECT * FROM satellites WHERE norad=?", (str(norad),)
        ).fetchone()
        return None if row is None else Satellite(**dict(row))

    def list_satellites(self) -> List[Satellite]:
        rows = self._connection().execute(
            "SELECT * FROM satellites ORDER BY name"
        ).fetchall()
        return [Satellite(**dict(r)) for r in rows]

    def satellites_for(self, owner: str) -> List[Satellite]:
        rows = self._connection().execute(
            "SELECT * FROM satellites WHERE owner=? ORDER BY name", (owner,)
        ).fetchall()
        return [Satellite(**dict(r)) for r in rows]

    def upsert_satellite(
        self, norad: str, owner: str, name: str, tle1: str, tle2: str
    ) -> None:
        with self._write() as conn:
            conn.execute(
                "INSERT INTO satellites(norad,owner,name,tle1,tle2,updated_utc) "
                "VALUES(?,?,?,?,?,?) "
                "ON CONFLICT(norad) DO UPDATE SET "
                "name=excluded.name, tle1=excluded.tle1, tle2=excluded.tle2, "
                "updated_utc=excluded.updated_utc "
                "WHERE satellites.owner=excluded.owner",
                (str(norad), owner, name, tle1, tle2, _iso(_utc_now())),
            )

    def delete_satellite(self, norad: str, owner: str) -> bool:
        with self._write() as conn:
            cur = conn.execute(
                "DELETE FROM satellites WHERE norad=? AND owner=?", (str(norad), owner)
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------ migration
    def import_legacy_json(self, path: Path) -> Dict[str, int]:
        """
        One-time import of an old auth_store.json.

        Users and satellites come across; tokens do not. Everything in that file
        was committed to the repository, so its live sessions are treated as
        compromised and the operator re-authenticates once.
        """
        result = {"users": 0, "satellites": 0, "tokens_discarded": 0}
        if not path.exists() or self.user_count() > 0:
            return result
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            log.warning("could not read legacy auth store %s: %s", path, exc)
            return result

        for username, record in (data.get("users") or {}).items():
            salt, pw = record.get("salt"), record.get("pw")
            if not salt or not pw:
                continue
            try:
                self.create_user(username, legacy_hash_record(salt, pw))
                result["users"] += 1
            except sqlite3.IntegrityError:
                continue

        for norad, record in (data.get("satellites") or {}).items():
            owner = record.get("owner")
            if not owner or self.get_user(owner) is None:
                continue
            try:
                self.upsert_satellite(
                    norad, owner, record.get("name") or norad,
                    record.get("tle1", ""), record.get("tle2", ""),
                )
                result["satellites"] += 1
            except sqlite3.IntegrityError:
                continue

        result["tokens_discarded"] = len(data.get("tokens") or {})
        if result["users"] or result["satellites"]:
            log.warning(
                "imported legacy auth store: %d users, %d satellites; %d session "
                "tokens discarded as compromised (the file was in version control)",
                result["users"], result["satellites"], result["tokens_discarded"],
            )
        return result

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            conn.close()
            self._local.conn = None


_store: Optional[Store] = None
_store_lock = threading.Lock()


def get_store(db_path: Optional[Path] = None, token_ttl_hours: int = 12) -> Store:
    """Process-wide Store singleton."""
    global _store
    with _store_lock:
        if _store is None:
            if db_path is None:
                raise RuntimeError("get_store() needs a db_path on first call")
            _store = Store(db_path, token_ttl_hours)
        return _store


def reset_store_for_tests() -> None:
    global _store
    with _store_lock:
        if _store is not None:
            _store.close()
        _store = None
