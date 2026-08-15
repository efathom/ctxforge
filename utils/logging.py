"""
Logging setup helpers for the ctxforge framework.

The framework follows the standard-library convention of *not* configuring
logging globally on import (libraries should be silent by default). Instead:

- A ``NullHandler`` is attached to the ``ctxforge`` logger in
  ``ctxforge/__init__.py`` so importing the package never emits
  "No handlers found" warnings.
- Applications can call :func:`setup_logging` to honor ``ObservabilityConfig``
  (level, format, file output) and route all ``ctxforge`` loggers accordingly.
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

from ctxforge.config.base import ObservabilityConfig

_LEVELS = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_DEFAULT_FORMAT = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"


def _configured() -> bool:
    """Return True if the ctxforge logger already has handlers."""
    return bool(logging.getLogger("ctxforge").handlers)


def setup_logging(
    config: Optional[ObservabilityConfig] = None,
    *,
    level: Optional[str] = None,
    log_format: Optional[str] = None,
    log_to_file: Optional[bool] = None,
    log_file_path: Optional[str] = None,
) -> None:
    """
    Configure logging for the ``ctxforge`` package.

    Args:
        config: An optional :class:`ObservabilityConfig` used to source defaults.
        level: Override log level name (DEBUG/INFO/WARNING/ERROR/CRITICAL).
        log_format: Override the log record format string.
        log_to_file: Whether to also write to a file.
        log_file_path: Destination file path (defaults to ``ctxforge.log``).
    """
    cfg = config or ObservabilityConfig()

    resolved_level = level or cfg.log_level.value
    resolved_format = log_format or cfg.log_format or _DEFAULT_FORMAT
    resolved_to_file = cfg.log_to_file if log_to_file is None else log_to_file
    resolved_file = log_file_path or cfg.log_file_path or "ctxforge.log"

    logger = logging.getLogger("ctxforge")
    logger.setLevel(_LEVELS.get(resolved_level.upper(), logging.INFO))
    logger.propagate = False

    # Avoid stacking duplicate handlers on repeated calls.
    if _configured():
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(resolved_format)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if resolved_to_file:
        try:
            file_handler = logging.FileHandler(resolved_file)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError:
            logger.exception("Failed to open log file: %s", resolved_file)
