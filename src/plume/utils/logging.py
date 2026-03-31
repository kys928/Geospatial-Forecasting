from __future__ import annotations

import logging

_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure root logging once with a consistent console formatter."""
    root_logger = logging.getLogger()

    if not any(getattr(handler, "_plume_handler", False) for handler in root_logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_FORMAT))
        handler._plume_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(handler)

    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


# Backward-compatible alias for existing imports.
def configure_logger(name: str, level: int = logging.INFO):
    configure_logging(logging.getLevelName(level))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
