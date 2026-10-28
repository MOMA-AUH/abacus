from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager

from abacus.logging import logger


class _Timer:
    def __init__(self) -> None:
        self.info: str = ""


@contextmanager
def timed(label: str) -> Iterator[_Timer]:
    """Log stage entry and elapsed time at DEBUG level; `.info` adds counts or context."""
    t0 = time.perf_counter()
    timer = _Timer()
    completed = False
    logger.debug("[TIMING] %s start (pid=%d)", label, os.getpid())
    try:
        yield timer
        completed = True
    finally:
        suffix = f"  ({timer.info})" if timer.info else ""
        logger.debug(
            "[TIMING] %s: %.3fs%s [%s]",
            label, time.perf_counter() - t0, suffix, "done" if completed else "failed",
        )
