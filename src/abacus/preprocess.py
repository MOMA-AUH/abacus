import logging
import random
from pathlib import Path
from typing import Literal

import pysam

from abacus.config import config
from abacus.locus import Locus
from abacus.read import Read
from abacus.timing import timed

logger = logging.getLogger()


def open_alignment_file(bam: Path, ref: Path) -> pysam.AlignmentFile:
    mode: Literal["rb", "rc"] = "rc" if str(bam).lower().endswith(".cram") else "rb"
    with timed("open alignment input"):
        return pysam.AlignmentFile(str(bam), mode, reference_filename=str(ref))


def get_reads_in_locus(bamfile: pysam.AlignmentFile, locus: Locus) -> list[Read]:
    """Extract locus reads; the caller owns the alignment handle."""
    with timed("fetch locus alignments") as t:
        # Finish this iterator before supplementary fetches reuse the file position.
        alignments = list(bamfile.fetch(locus.location.chrom, locus.location.start, locus.location.end))
        t.info = f"{len(alignments)} alignments"

    if not alignments:
        return []

    alignments = [ali for ali in alignments if not ali.is_secondary]
    primary_alignments = [read for read in alignments if not read.is_supplementary]
    supplementary_alignments = [read for read in alignments if read.is_supplementary]

    # Recover only reads whose primary alignment was outside the locus.
    primary_ids = [read.query_name for read in primary_alignments]
    supplementary_alignments = [ali for ali in supplementary_alignments if ali.query_name not in primary_ids]

    with timed("recover supplementary alignments") as t:
        for ali in supplementary_alignments:
            chrom, pos = str(ali.get_tag("SA")).split(",")[0:2]
            primary_candidates = bamfile.fetch(chrom, int(pos), int(pos) + 1)
            primary_alignment = [cand for cand in primary_candidates if cand.query_name == ali.query_name]
            primary_alignments.extend(primary_alignment)
        t.info = f"{len(supplementary_alignments)} lookups"

    # Downsample if coverage exceeds threshold
    if config.downsample > 0 and len(primary_alignments) > config.downsample:
        logger.warning(f"High coverage at {locus.id} ({len(primary_alignments)} reads). Downsampling to {config.downsample}.")
        primary_alignments = random.Random(config.downsample_seed).sample(primary_alignments, config.downsample)

    # Convert primary alignments to Read objects
    with timed("convert locus reads") as t:
        reads = [Read.from_alignment(alignment=alignment, locus=locus) for alignment in primary_alignments]
        t.info = f"{len(reads)} reads"

    # Trim soft-clipped bases
    reads = [trim_soft_clipped_bases(read, config.max_trim) for read in reads]

    # Trim low quality bases
    reads = [trim_low_quality_end_bases(read) for read in reads]

    # Make sure reads have unique names
    read_names = set()
    for read in reads:
        # Check if the read name is already in the set
        if read.name in read_names:
            # If it is, append a suffix to make it unique
            suffix = 1
            while f"{read.name}_{suffix}" in read_names:
                suffix += 1
            read.name = f"{read.name}_{suffix}"
        # Add the read name to the set
        read_names.add(read.name)

    return reads


def trim_low_quality_end_bases(read: Read) -> Read:
    # If no read quality is available, return the read
    if not read.qualities:
        return read

    # Trim low quality bases
    trim_start_index = next((i for i in range(config.max_trim) if min(read.qualities[i : i + config.trim_window_size]) >= config.min_end_qual), 0)
    trim_end_index = next((j for j in range(config.max_trim) if min(read.qualities[-j - 1 - config.trim_window_size : -j - 1]) >= config.min_end_qual), 0)

    # If no trimming is needed, return the read
    if trim_start_index == 0 and trim_end_index == 0:
        return read

    # Check if trimming would remove the entire read
    if trim_start_index + trim_end_index >= len(read.sequence):
        return read

    # Trim start and end of the read
    read.sequence = read.sequence[trim_start_index:]
    read.qualities = read.qualities[trim_start_index:]
    read.mod_5mc_probs = read.mod_5mc_probs[trim_start_index:]
    if trim_end_index > 0:
        read.sequence = read.sequence[:-trim_end_index]
        read.qualities = read.qualities[:-trim_end_index]
        read.mod_5mc_probs = read.mod_5mc_probs[:-trim_end_index]

    return read


def trim_soft_clipped_bases(read: Read, max_trim: int) -> Read:
    # If no soft-clipping is needed, return the read
    if max_trim == 0:
        return read

    # Trim start of the read
    trim_start_index = min(read.n_soft_clipped_left, max_trim)
    if trim_start_index > 0:
        read.sequence = read.sequence[trim_start_index:]
        read.qualities = read.qualities[trim_start_index:]
        read.mod_5mc_probs = read.mod_5mc_probs[trim_start_index:]
        read.n_soft_clipped_left = read.n_soft_clipped_left - trim_start_index

    # Trim end of the read
    trim_end_index = min(read.n_soft_clipped_right, max_trim)

    if trim_end_index > 0:
        read.sequence = read.sequence[:-trim_end_index]
        read.qualities = read.qualities[:-trim_end_index]
        read.mod_5mc_probs = read.mod_5mc_probs[:-trim_end_index]
        read.n_soft_clipped_right = read.n_soft_clipped_right - trim_end_index

    return read
