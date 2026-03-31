from __future__ import annotations

import logging
from typing import Optional

def configure_logger(name: str, level: int = logging.INFO):
 raise ValueError('logger config not implemented')

def get_logger(name: str, level: int = logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    return logger

