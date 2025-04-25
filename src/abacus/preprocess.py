from pathlib import Path

import pysam

from abacus.config import config
from abacus.locus import Locus
from abacus.read import Read


def get_reads_in_locus(bam: Path, locus: Locus) -> list[Read]:
    # Get alignments overlapping the region
    with pysam.AlignmentFile(str(bam), "rb") as bamfile:
        alignments = list(bamfile.fetch(locus.location.chrom, locus.location.start, locus.location.end))

        # Return empty list if no alignments found
        if not alignments:
            return []

        # Remove secondary alignments
        alignments = [ali for ali in alignments if not ali.is_secondary]

        # Split alignments into primary and supplementary
        primary_alignments = [read for read in alignments if not read.is_supplementary]
        supplementary_alignments = [read for read in alignments if read.is_supplementary]

        # Filter supplementary alignments that are in primary alignments already
        primary_ids = [read.query_name for read in primary_alignments]
        supplementary_alignments = [ali for ali in supplementary_alignments if ali.query_name not in primary_ids]

        # Find primary alignments of left supplementary alignments
        for ali in supplementary_alignments:
            # Get the primary alignment position
            chrom, pos = str(ali.get_tag("SA")).split(",")[0:2]

            # Load all alignments at the primary alignment position
            primary_candidates = bamfile.fetch(chrom, int(pos), int(pos) + 1)

            # Find the primary alignment and add it to the primary alignments
            primary_alignment = [cand for cand in primary_candidates if cand.query_name == ali.query_name]
            primary_alignments.extend(primary_alignment)

    # Convert primary alignments to Read objects
    reads = [Read.from_alignment(alignment=alignment, locus=locus) for alignment in primary_alignments]

    # Trim soft-clipped bases
    reads = [trim_soft_clipped_bases(read, config.max_trim) for read in reads]

    # Filter reads with low quality bases
    reads = [trim_low_quality_end_bases(read) for read in reads]

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
