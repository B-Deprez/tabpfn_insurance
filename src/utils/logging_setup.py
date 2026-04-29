"""Shared logging configuration for experiment scripts.

Each script calls ``setup_logging(experiment_id)`` once at startup to attach
both a console handler (INFO) and a file handler (DEBUG) that writes to
``logs/<experiment_id>.log``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def setup_logging(experiment_id: str, level_console: int = logging.INFO) -> None:
    """Configure root logger with console + file handlers.

    Args:
        experiment_id: used as the log filename stem (``logs/<id>.log``).
        level_console: log level for the console handler (default INFO).
    """
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{experiment_id}.log"

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level_console)
    ch.setFormatter(fmt)

    # File handler
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    # Avoid duplicate handlers if called multiple times
    if not root.handlers:
        root.addHandler(ch)
        root.addHandler(fh)
    else:
        root.handlers.clear()
        root.addHandler(ch)
        root.addHandler(fh)

    logging.getLogger(__name__).info(
        "Logging initialised — console: %s, file: %s",
        logging.getLevelName(level_console),
        log_path,
    )
