"""Centralized logging configuration.

Call ``setup_logging("name", level="DEBUG")`` once from an entry point. All
modules use ``logging.getLogger(__name__)`` so the formatter shows which
component a line came from. Logs are written to both stdout and a timestamped
file under ``logs/`` so the user can copy/paste them when something misfires.
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_FMT = "%(asctime)s.%(msecs)03d  %(levelname)-7s  %(name)-18s  %(message)s"
_DATEFMT = "%H:%M:%S"


def setup_logging(name: str = "run", level: str = "INFO",
                  log_dir: str | Path = "logs", to_file: bool = True) -> Path | None:
    """Configure the root logger. Returns the path to the .log file, or None."""
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(_FMT, datefmt=_DATEFMT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    log_path: Path | None = None
    if to_file:
        d = Path(log_dir)
        d.mkdir(parents=True, exist_ok=True)
        log_path = d / f"{name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Quiet down noisy third-party loggers by default.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    return log_path
