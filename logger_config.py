"""
logger_config.py
-----------------
Centralised logging setup used across all modules.

Non-functional requirement addressed: Logging & Monitoring.
Every pipeline run writes a timestamped entry to `output/pipeline.log`
as well as to the console, so failures and performance can be traced
after the fact without re-running the whole pipeline.
"""

import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
LOG_FILE = os.path.join(LOG_DIR, "pipeline.log")


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger that writes to both console and a
    persistent log file. Safe to call repeatedly (handlers are only
    attached once per logger name).
    """
    os.makedirs(LOG_DIR, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured (e.g. imported from multiple modules)
        return logger

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def run_timestamp() -> str:
    """Return a filesystem-safe timestamp string for naming output files."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
