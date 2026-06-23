"""
logger.py — Project-wide logging setup.

All modules call `get_logger(__name__)` to get a named logger that writes to
both the console and a rotating file.  Using __name__ means log lines
automatically show which module emitted them, which is invaluable during
debugging.
"""

import logging
import logging.handlers
from pathlib import Path
from config.settings import cfg


def get_logger(name: str) -> logging.Logger:
    """
    Return a configured logger for the given module name.

    We use RotatingFileHandler (not FileHandler) so that log files don't
    grow unbounded during long indexing sessions.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        A Logger instance with both console and file handlers attached.
    """
    logger = logging.getLogger(name)

    # Guard: don't add duplicate handlers if this function is called twice
    # for the same name (common in Streamlit's hot-reload environment).
    if logger.handlers:
        return logger

    logger.setLevel(cfg.logging.level)

    formatter = logging.Formatter(cfg.logging.format)

    # ── Console handler ────────────────────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler ──────────────────────────────────────────────
    # maxBytes=5MB, backupCount=3 → keeps at most 15 MB of logs
    log_path = Path(cfg.logging.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
