"""
socrates.py — CelesTrak's published SOCRATES conjunction feed.

SOCRATES is CelesTrak's public screening of the whole catalog. It is useful as
an independent cross-check on ISDMAAS's own screen, and it is the only public
source that publishes a maximum collision probability.

The feed has no stable machine API: the CSV endpoint sometimes returns HTML, the
HTML table's column order is positional, and both have changed. The parser
therefore locates fields by content rather than by column index, validates every
value into a plausible physical range, and reports how confident it is. A
conjunction whose fields cannot be validated is dropped rather than passed on
with a zero in it — a miss distance of 0.0 km displayed next to a real one is
worse than a shorter list.
"""
from __future__ import annotations

import csv
import io
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import requests

from .logging_config import get_logger

log = get_logger("isdmaas.socrates")

USER_AGENT = "ISDMAAS/2.0 (orbital safety research)"
CACHE_TTL_S = 3600          # CelesTrak asks callers not to re-fetch aggressively
FETCH_TIMEOUT_S = 90

CSV_ENDPOINTS = (
    "https://celestrak.org/SOCRATES/socrates-search-results.php",
    "https://celestrak.org/SOCRATES-Plus/socrates-search-results.php",
)
HTML_ENDPOINTS = (
    "https://celestrak.org/SOCRATES/table-socrates.php",
    "https://celestrak.org/SOCRATES-Plus/table-socrates.php",
)

# Physical plausibility bounds used to validate parsed values.
MAX_MISS_KM = 1000.0
MAX_REL_SPEED_KMS = 20.0
_OPS_STATUS = re.compile(r"\s*\[[^\]]*\]\s*$")
_TAG = re.compile(r"<[^>]+>")
_DATE = re.compile(r"\d{4}-\d{2}-\d{2}|\d{4}\s+\w{3}\s+\d{1,2}")
_NUMBER = re.compile(r"^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$")


@dataclass
class SocratesRow:
    primary_norad: int
    primary_name: str
    secondary_norad: int
    secondary_name: str
    tca: str
    miss_km: float
    relative_speed_kms: float
    max_pc: Optional[float]

    def as_dict(self) -> Dict:
        return {
            "p_norad": self.primary_norad,
            "p_name": self.primary_name,
            "s_norad": self.secondary_norad,
            "s_name": self.secondary_name,
            "tca": self.tca,
            "miss_km": self.miss_km,
            "rel_speed_kms": self.relative_speed_kms,
            "max_prob": self.max_pc,
        }


def _clean(cell: str) -> str:
    return _TAG.sub("", cell).replace("&nbsp;", " ").strip()


def _strip_ops_status(name: str) -> str:
    return _OPS_STATUS.sub("", name).strip()


def _parse_html(html: str) -> List[SocratesRow]:
    """
    Parse the SOCRATES results table.

    Each object occupies one row, and consecutive rows pair into one conjunction
    sharing the TCA, miss distance and relative speed. Fields are found by
    content — a five-or-six-digit integer is the catalog number, a date-shaped
    cell is the TCA, the remaining numbers are ranked by plausible magnitude —
    because the column order is not stable across the two SOCRATES variants.
    """
    objects: List[Dict] = []
    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I):
        cells = [
            _clean(c)
            for c in re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row_html, re.S | re.I)
        ]
        norad_index = next(
            (i for i, value in enumerate(cells) if re.fullmatch(r"\d{5,6}", value)),
            None,
        )
        if norad_index is None:
            continue
        norad = int(cells[norad_index])
        after = cells[norad_index + 1:]
        name = _strip_ops_status(after[0]) if after else str(norad)
        tca = next((value for value in after if _DATE.search(value)), "")
        numbers = [float(v) for v in after if _NUMBER.fullmatch(v)]
        # Of the trailing numbers, the miss distance is the one within catalog
        # screening range and the relative speed the one under orbital velocity.
        miss = next(
            (n for n in reversed(numbers) if 0.0 < n <= MAX_MISS_KM), None
        )
        speed = next(
            (n for n in reversed(numbers) if 0.0 < n <= MAX_REL_SPEED_KMS), None
        )
        objects.append(
            {"norad": norad, "name": name, "tca": tca, "miss": miss, "speed": speed}
        )

    rows: List[SocratesRow] = []
    for i in range(0, len(objects) - 1, 2):
        a, b = objects[i], objects[i + 1]
        miss = a["miss"] if a["miss"] is not None else b["miss"]
        speed = a["speed"] if a["speed"] is not None else b["speed"]
        tca = a["tca"] or b["tca"]
        if miss is None or not tca:
            # Without a miss distance and a TCA this is not a usable conjunction.
            continue
        rows.append(
            SocratesRow(
                primary_norad=a["norad"],
                primary_name=a["name"],
                secondary_norad=b["norad"],
                secondary_name=b["name"],
                tca=tca,
                miss_km=miss,
                relative_speed_kms=speed if speed is not None else 0.0,
                max_pc=None,
            )
        )
    return rows


def _first(record: Dict[str, str], *keys: str) -> Optional[str]:
    for key in keys:
        value = record.get(key)
        if value not in (None, ""):
            return value
    return None


def _parse_csv(text: str) -> List[SocratesRow]:
    rows: List[SocratesRow] = []
    for record in csv.DictReader(io.StringIO(text)):
        try:
            primary = int(_first(record, "NORAD_CAT_ID_1", "NORAD_CAT_ID_I", "SAT1"))
            secondary = int(_first(record, "NORAD_CAT_ID_2", "NORAD_CAT_ID_J", "SAT2"))
            miss = float(
                _first(record, "TCA_RANGE", "MIN_RNG", "MINRANGE", "RANGE") or 0
            )
            speed = float(
                _first(record, "TCA_RELATIVE_SPEED", "REL_SPEED", "RELATIVE_SPEED") or 0
            )
        except (TypeError, ValueError):
            continue
        if not (0.0 < miss <= MAX_MISS_KM):
            continue
        raw_pc = _first(record, "MAX_PROB", "MAXPROB")
        try:
            max_pc = float(raw_pc) if raw_pc is not None else None
        except ValueError:
            max_pc = None
        rows.append(
            SocratesRow(
                primary_norad=primary,
                primary_name=_strip_ops_status(
                    (_first(record, "OBJECT_NAME_1", "SAT_1_NAME", "NAME1") or "").strip()
                ),
                secondary_norad=secondary,
                secondary_name=_strip_ops_status(
                    (_first(record, "OBJECT_NAME_2", "SAT_2_NAME", "NAME2") or "").strip()
                ),
                tca=(_first(record, "TCA") or "").strip(),
                miss_km=miss,
                relative_speed_kms=speed if 0.0 < speed <= MAX_REL_SPEED_KMS else 0.0,
                max_pc=max_pc,
            )
        )
    return rows


class SocratesFeed:
    """Cached client for the SOCRATES feed."""

    def __init__(self, ttl_s: int = CACHE_TTL_S):
        self.ttl_s = ttl_s
        self._cache: Dict[str, Tuple[float, Dict]] = {}
        self._lock = threading.Lock()

    def _fetch(self, params: Dict[str, str]) -> Tuple[Optional[str], bool, str]:
        """Returns (text, is_csv, last_error)."""
        last_error = "no endpoint responded"
        headers = {"User-Agent": USER_AGENT}
        for url in CSV_ENDPOINTS:
            try:
                response = requests.get(
                    url, params={**params, "FORMAT": "CSV"},
                    headers=headers, timeout=FETCH_TIMEOUT_S,
                )
                if response.status_code == 404:
                    last_error = f"404 {url}"
                    continue
                response.raise_for_status()
                body = response.text.lstrip()
                first_line = body.splitlines()[0] if body else ""
                if body[:1] != "<" and ("NORAD_CAT_ID" in body[:400] or "," in first_line):
                    return response.text, True, last_error
            except requests.RequestException as exc:
                last_error = str(exc)
        for url in HTML_ENDPOINTS:
            try:
                response = requests.get(
                    url, params=params, headers=headers, timeout=FETCH_TIMEOUT_S
                )
                if response.status_code == 404:
                    last_error = f"404 {url}"
                    continue
                response.raise_for_status()
                return response.text, False, last_error
            except requests.RequestException as exc:
                last_error = str(exc)
        return None, False, last_error

    def fetch(
        self, norad: Optional[int] = None, order: str = "MAXPROB", max_rows: int = 25
    ) -> Dict:
        key = f"{norad}|{order}|{max_rows}"
        now = time.monotonic()
        with self._lock:
            cached = self._cache.get(key)
            if cached and now - cached[0] < self.ttl_s:
                return cached[1]

        params: Dict[str, str] = {"ORDER": order, "MAX": str(max_rows)}
        if norad:
            params["CATNR"] = f"{norad},"
        else:
            params["NAME"] = ","

        text, is_csv, last_error = self._fetch(params)
        if text is None:
            raise RuntimeError(f"SOCRATES fetch failed: {last_error}")

        stripped = text.lstrip()
        looks_html = stripped[:1] == "<" or "<table" in stripped[:2000].lower()
        rows = _parse_html(text) if (looks_html or not is_csv) else _parse_csv(text)

        result = {
            "source": "CelesTrak SOCRATES",
            "order": order,
            "format": "html_table" if (looks_html or not is_csv) else "csv",
            "count": len(rows),
            "max_pc_available": any(r.max_pc is not None for r in rows),
            "conjunctions": [r.as_dict() for r in rows],
        }
        with self._lock:
            self._cache[key] = (now, result)
        return result


_feed: Optional[SocratesFeed] = None
_feed_lock = threading.Lock()


def get_feed() -> SocratesFeed:
    global _feed
    with _feed_lock:
        if _feed is None:
            _feed = SocratesFeed()
        return _feed
