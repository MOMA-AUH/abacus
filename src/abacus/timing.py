from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager

from abacus.logging import logger


class _Timer:
    def __init__(self) -> None:
        self.info: str = ""


@contextmanager
def timed(label: str) -> Iterator[_Timer]:
    """Log elapsed time for a block at DEBUG level, prefixed with [TIMING].

    Set `.info` on the yielded handle to append extra context (e.g. counts) to the log line.
    """
    t0 = time.perf_counter()
    timer = _Timer()
    try:
        yield timer
    finally:
        suffix = f"  ({timer.info})" if timer.info else ""
        logger.debug(f"[TIMING] {label}: {time.perf_counter() - t0:.3f}s{suffix}")
