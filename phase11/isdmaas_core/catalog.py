"""
catalog.py — the public object catalog, fetched once and shared safely.

The previous implementation kept the catalog in a module-level global that was
populated without a lock, wrote its cache file with a bare `open(...).write()`
(so a crash mid-write left a truncated file that every later start would happily
parse into a partial catalog), and resolved that file relative to the process
working directory. It also never expired: a server left running for a week
screened against week-old element sets without saying so.

This service fixes all four:
  * a lock around the refresh, so concurrent requests do one fetch, not N;
  * atomic replace via a temporary file in the same directory;
  * paths anchored to the configured data directory;
  * an explicit age, surfaced to callers so the console can show how stale the
    data is instead of implying it is live.
"""
from __future__ import annotations

import os
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from sgp4.api import Satrec

from .logging_config import get_logger

log = get_logger("isdmaas.catalog")

CELESTRAK_GP = "https://celestrak.org/NORAD/elements/gp.php"
DEFAULT_GROUP = "active"
USER_AGENT = "ISDMAAS/2.0 (orbital safety research)"
FETCH_TIMEOUT_S = 120
J2000 = datetime(2000, 1, 1, 12, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class CatalogEntry:
    norad: int
    name: str
    satrec: Satrec


@dataclass(frozen=True)
class CatalogSnapshot:
    entries: Dict[int, CatalogEntry]
    fetched_utc: datetime
    source: str

    @property
    def age_s(self) -> float:
        return (datetime.now(timezone.utc) - self.fetched_utc).total_seconds()

    def as_tuples(self) -> List[Tuple[int, str, Satrec]]:
        """The (norad, name, satrec) shape the screening functions expect."""
        return [(e.norad, e.name, e.satrec) for e in self.entries.values()]

    def get(self, norad: int) -> Optional[CatalogEntry]:
        return self.entries.get(int(norad))


def parse_tle_text(text: str) -> Dict[int, CatalogEntry]:
    """
    Parse a three-line-element listing.

    Malformed triples are skipped rather than aborting the parse: a single bad
    record in a 16 000-object feed must not take the whole catalog down.
    """
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    entries: Dict[int, CatalogEntry] = {}
    skipped = 0
    for i in range(0, len(lines) - 2, 3):
        name, line1, line2 = lines[i], lines[i + 1], lines[i + 2]
        if not (line1.startswith("1 ") and line2.startswith("2 ")):
            skipped += 1
            continue
        try:
            satrec = Satrec.twoline2rv(line1, line2)
            norad = int(line2[2:7])
        except (ValueError, IndexError):
            skipped += 1
            continue
        entries[norad] = CatalogEntry(norad=norad, name=name.strip(), satrec=satrec)
    if skipped:
        log.debug("skipped %d malformed TLE records", skipped)
    return entries


def newest_element_epoch(entries: Dict[int, CatalogEntry]) -> Optional[datetime]:
    """
    The most recent element-set epoch in a catalog.

    Used to date the bundled offline catalog. Its file modification time is a
    checkout artefact — cloning the repository stamps it with today's date — so
    reporting that as the catalog's age would tell an operator the data is fresh
    when it is months old. The element sets themselves carry the real date.
    """
    newest = None
    for entry in entries.values():
        try:
            julian = float(entry.satrec.jdsatepoch) + float(entry.satrec.jdsatepochF)
        except (AttributeError, TypeError, ValueError):
            continue
        when = J2000 + timedelta(days=julian - 2451545.0)
        if newest is None or when > newest:
            newest = when
    return newest


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a same-directory temp file so a crash cannot truncate the cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(text)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class CatalogService:
    """Loads, caches and shares the public object catalog."""

    def __init__(self, cache_path: Path, ttl_s: int = 6 * 3600,
                 group: str = DEFAULT_GROUP, fallback_path: Optional[Path] = None):
        self.cache_path = Path(cache_path)
        self.fallback_path = Path(fallback_path) if fallback_path else None
        self.ttl_s = max(60, int(ttl_s))
        self.group = group
        self._snapshot: Optional[CatalogSnapshot] = None
        self._name_index: Optional[Dict[str, Dict[str, int]]] = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------ retrieval
    def _fetch_remote(self) -> Optional[str]:
        try:
            response = requests.get(
                CELESTRAK_GP,
                params={"GROUP": self.group, "FORMAT": "tle"},
                headers={"User-Agent": USER_AGENT},
                timeout=FETCH_TIMEOUT_S,
            )
            response.raise_for_status()
            text = response.text
        except requests.RequestException as exc:
            log.warning("catalog fetch failed: %s", exc)
            return None
        if len(text) < 500 or "1 " not in text[:2000]:
            log.warning("catalog fetch returned an unexpected payload; ignoring")
            return None
        return text

    def _load_cache(self) -> Optional[Tuple[str, datetime]]:
        if not self.cache_path.exists():
            return None
        try:
            text = self.cache_path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            log.warning("could not read catalog cache: %s", exc)
            return None
        stamped = datetime.fromtimestamp(
            self.cache_path.stat().st_mtime, tz=timezone.utc
        )
        return text, stamped

    def refresh(self, force: bool = False) -> Optional[CatalogSnapshot]:
        """
        Return a current snapshot, fetching if the cache is stale.

        Falls back to a stale cache when the network is unavailable — a screen
        against yesterday's elements with the age reported is far more useful to
        an operator than no screen at all — but never invents data.
        """
        with self._lock:
            cached = self._load_cache()
            cache_fresh = (
                cached is not None
                and (datetime.now(timezone.utc) - cached[1]).total_seconds() < self.ttl_s
            )

            if not force and self._snapshot is not None and self._snapshot.age_s < self.ttl_s:
                return self._snapshot

            if not force and cache_fresh:
                entries = parse_tle_text(cached[0])
                if entries:
                    self._snapshot = CatalogSnapshot(entries, cached[1], "disk cache")
                    self._name_index = None
                    return self._snapshot

            text = self._fetch_remote()
            if text is not None:
                entries = parse_tle_text(text)
                if entries:
                    try:
                        _atomic_write_text(self.cache_path, text)
                    except OSError as exc:
                        log.warning("could not persist catalog cache: %s", exc)
                    self._snapshot = CatalogSnapshot(
                        entries, datetime.now(timezone.utc), "CelesTrak"
                    )
                    self._name_index = None
                    log.info("catalog refreshed: %d objects", len(entries))
                    return self._snapshot

            if cached is not None:
                entries = parse_tle_text(cached[0])
                if entries:
                    age_h = (datetime.now(timezone.utc) - cached[1]).total_seconds() / 3600
                    log.warning(
                        "serving a stale catalog (%.1f h old): the upstream fetch failed",
                        age_h,
                    )
                    self._snapshot = CatalogSnapshot(
                        entries, cached[1], "stale disk cache"
                    )
                    self._name_index = None
                    return self._snapshot

            # Last resort: the committed offline catalog. It is certainly old,
            # and its age is reported so nothing downstream can mistake it for
            # current data — but a screen against last month's element sets,
            # clearly labelled, beats no screen at all.
            if self._snapshot is None and self.fallback_path \
                    and self.fallback_path.exists():
                try:
                    text = self.fallback_path.read_text(
                        encoding="utf-8", errors="replace")
                    entries = parse_tle_text(text)
                except OSError:
                    entries = {}
                if entries:
                    stamped = newest_element_epoch(entries) or datetime.fromtimestamp(
                        self.fallback_path.stat().st_mtime, tz=timezone.utc)
                    age_days = (datetime.now(timezone.utc) - stamped).days
                    log.warning(
                        "no network and no cache: falling back to the bundled "
                        "offline catalog (%d days old, %d objects)",
                        age_days, len(entries))
                    self._snapshot = CatalogSnapshot(
                        entries, stamped, "bundled offline catalog")
                    self._name_index = None

            return self._snapshot

    def snapshot(self) -> Optional[CatalogSnapshot]:
        """Current snapshot, loading it on first use."""
        if self._snapshot is None:
            return self.refresh()
        if self._snapshot.age_s >= self.ttl_s:
            return self.refresh()
        return self._snapshot

    # -------------------------------------------------------------- lookup
    def _index(self, snapshot: CatalogSnapshot) -> Dict[str, Dict[str, int]]:
        if self._name_index is not None:
            return self._name_index
        by_name: Dict[str, int] = {}
        by_cospar: Dict[str, int] = {}
        for entry in snapshot.entries.values():
            by_name[entry.name.upper()] = entry.norad
            designator = (getattr(entry.satrec, "intldesg", "") or "").strip()
            if len(designator) >= 2 and designator[:2].isdigit():
                yy = int(designator[:2])
                year = 2000 + yy if yy < 57 else 1900 + yy
                by_cospar[f"{year}-{designator[2:]}".upper()] = entry.norad
        self._name_index = {"by_name": by_name, "by_cospar": by_cospar}
        return self._name_index

    def resolve(self, query: str, max_candidates: int = 8) -> dict:
        """
        Resolve a NORAD id, satellite name or COSPAR id to a catalog object.

        Returns either a single match or a candidate list, never a guess between
        two equally good options — picking one silently is how an operator ends
        up screening the wrong asset.
        """
        snapshot = self.snapshot()
        if snapshot is None:
            return {"error": "catalog unavailable"}
        query = query.strip()
        if not query:
            return {"error": "empty query"}
        index = self._index(snapshot)

        if query.isdigit():
            entry = snapshot.get(int(query))
            if entry:
                return {"norad": entry.norad, "name": entry.name, "matched_by": "norad"}

        upper = query.upper()
        for key, matched_by in (("by_cospar", "cospar"), ("by_name", "name")):
            norad = index[key].get(upper)
            if norad is not None:
                return {
                    "norad": norad,
                    "name": snapshot.entries[norad].name,
                    "matched_by": matched_by,
                }

        partial = [
            (name, norad) for name, norad in index["by_name"].items() if upper in name
        ]
        partial.sort(key=lambda item: (len(item[0]), item[0]))
        if len(partial) == 1:
            norad = partial[0][1]
            return {
                "norad": norad,
                "name": snapshot.entries[norad].name,
                "matched_by": "name_partial",
            }
        if partial:
            return {
                "candidates": [
                    {"norad": norad, "name": snapshot.entries[norad].name}
                    for _, norad in partial[:max_candidates]
                ],
                "total_matches": len(partial),
                "matched_by": "ambiguous",
            }
        return {"error": "not found"}


_service: Optional[CatalogService] = None
_service_lock = threading.Lock()


def get_catalog_service(
    cache_path: Optional[Path] = None,
    ttl_s: int = 6 * 3600,
    fallback_path: Optional[Path] = None,
) -> CatalogService:
    global _service
    with _service_lock:
        if _service is None:
            if cache_path is None:
                raise RuntimeError("get_catalog_service() needs cache_path on first call")
            _service = CatalogService(cache_path, ttl_s, fallback_path=fallback_path)
        return _service


def reset_catalog_for_tests() -> None:
    global _service
    with _service_lock:
        _service = None
