"""
logging_config.py — structured logging with per-request correlation ids.

Every log line carries the id of the request that produced it, so a slow screen
or a failed maneuver plan can be traced end to end from a single grep. In
production the formatter emits one JSON object per line for log shippers; in
development it emits a compact human-readable line.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from typing import Any, Dict

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "isdmaas_request_id", default="-"
)

_RESERVED = set(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"message", "asctime", "taskName"}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "request_id": request_id_var.get(),
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = request_id_var.get()
        prefix = f"{self.formatTime(record, '%H:%M:%S')} {record.levelname:<7}"
        suffix = f" [{rid[:8]}]" if rid != "-" else ""
        base = f"{prefix} {record.name}{suffix}: {record.getMessage()}"
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Install the root handler. Safe to call more than once."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if json_output else _TextFormatter())
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # uvicorn installs its own noisy access logger; route it through ours.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


def new_request_id() -> str:
    return uuid.uuid4().hex


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
