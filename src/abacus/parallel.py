"""Serial and bounded process-pool execution with process-local alignment and reference inputs."""

import os
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack, contextmanager
from functools import partial
from itertools import chain, islice
from logging import Filter, LogRecord
from logging.handlers import QueueHandler, QueueListener
from multiprocessing import get_context
from multiprocessing.util import Finalize
from pathlib import Path

from pyfaidx import Fasta
from pysam import AlignmentFile

from abacus.config import config
from abacus.locus import Locus
from abacus.logging import logger
from abacus.preprocess import open_alignment_file
from abacus.utils import Sex

_worker_bamfile: AlignmentFile | None = None
_worker_reference: Fasta | None = None
_locus_id = ""


class _LocusFilter(Filter):
    def filter(self, record: LogRecord) -> bool:
        if _locus_id and _locus_id not in record.getMessage():
            record.msg = f"[{_locus_id}] {record.msg}"
        return True


def _apply_settings(settings: dict) -> None:
    for key, value in settings.items():
        setattr(config, key, value)


def _worker_init(settings: dict, log_queue, log_level: int, bam: Path, ref: Path) -> None:
    global _worker_bamfile, _worker_reference

    logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    logger.setLevel(log_level)
    logger.addFilter(_LocusFilter())
    _apply_settings(settings)
    _worker_bamfile = open_alignment_file(bam, ref)
    Finalize(None, _worker_bamfile.close, exitpriority=10)
    _worker_reference = Fasta(ref)
    Finalize(None, _worker_reference.close, exitpriority=10)
    logger.debug("Worker %d ready", os.getpid())


def _call_locus(process_locus: Callable, locus: Locus, ref: Fasta, sex: Sex, bamfile: AlignmentFile) -> dict:
    global _locus_id

    _locus_id = locus.id
    try:
        return process_locus(locus, ref=ref, sex=sex, bamfile=bamfile)
    except Exception:
        logger.exception("Locus failed")
        raise
    finally:
        _locus_id = ""


def _worker_call(process_locus: Callable, locus: Locus, sex: Sex) -> dict:
    assert _worker_bamfile is not None
    assert _worker_reference is not None
    return _call_locus(process_locus, locus, _worker_reference, sex, _worker_bamfile)


def _completed_results(pool, process_locus, loci, pending: set[Future], limit: int) -> Iterator[dict]:
    while True:
        # A pending task may have failed while the caller was writing the previous batch.
        for future in pending:
            if future.done() and future.exception() is not None:
                future.result()
        for locus in islice(loci, limit - len(pending)):
            pending.add(pool.submit(process_locus, locus))
        if not pending:
            return

        completed, _ = wait(pending, return_when=FIRST_COMPLETED)
        # Detect any failure in this batch before writing results or submitting more work.
        for future in completed:
            if future.exception() is not None:
                future.result()
        for future in completed:
            pending.remove(future)
            yield future.result()
        completed.clear()


@contextmanager
def locus_results(
    loci: Iterable[Locus],
    process_locus: Callable,
    *,
    bam: Path,
    ref: Path,
    settings: dict,
    sex: Sex,
    workers: int,
) -> Iterator[Iterator[dict]]:
    """Yield locus results; cancel queued work on exit, then wait for running tasks.

    Parallel execution keeps at most 2 * workers tasks pending. The caller writes results.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    source = iter(loci)
    first = next(source, None)
    if first is None:
        yield iter(())
        return
    source = chain((first,), source)

    if workers == 1:
        previous_settings = config.to_dict()
        context_filter = _LocusFilter()
        logger.addFilter(context_filter)
        try:
            _apply_settings(settings)
            with open_alignment_file(bam, ref) as bamfile, Fasta(ref) as reference:
                results = (_call_locus(process_locus, locus, reference, sex, bamfile) for locus in source)
                try:
                    yield results
                finally:
                    results.close()
        finally:
            logger.removeFilter(context_filter)
            _apply_settings(previous_settings)
        return

    context = get_context("spawn")
    pending: set[Future] = set()
    with ExitStack() as cleanup:
        log_queue = context.Queue()
        cleanup.callback(log_queue.join_thread)
        cleanup.callback(log_queue.close)
        listener = QueueListener(log_queue, *logger.handlers, respect_handler_level=True)
        listener.start()
        cleanup.callback(listener.stop)
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=_worker_init,
            initargs=(settings, log_queue, logger.level, bam, ref),
            max_tasks_per_child=100,
        )
        # Keep logging alive until workers have exited and flushed their records.
        cleanup.callback(pool.shutdown, wait=True, cancel_futures=True)
        process_fn = partial(_worker_call, process_locus, sex=sex)
        results = _completed_results(pool, process_fn, source, pending, 2 * workers)
        try:
            yield results
        except BrokenProcessPool:
            logger.error("Worker pool failed during startup or execution; see traceback.")
            raise
        finally:
            results.close()
            for future in pending:
                future.cancel()
