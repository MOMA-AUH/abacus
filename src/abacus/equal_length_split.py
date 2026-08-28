import random
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from spoa import poa

from abacus.config import config
from abacus.graph import ReadCall
from abacus.logging import logger
from abacus.utils import AlignmentType, Haplotype

# Printable single-byte ASCII (38-126) minus '-' (45, poa's gap char) and 33-37 (reserved for
# anchors/missing-end char). Capped below 128: spoa encodes str as UTF-8, so higher code points
# go multi-byte and desync its alignment-matrix indexing, causing a segfault.
_RESERVED_CHARS = {chr(i) for i in (33, 34, 35, 36, 37, 45)}
_USABLE_KMER_CHARS = [chr(i) for i in range(38, 127) if chr(i) not in _RESERVED_CHARS]
_OTHER_KMER_CHAR = _USABLE_KMER_CHARS[-1]


def build_kmer_char_maps(sequences: list[list[str]]) -> tuple[dict[str, str], dict[str, str]]:
    """Build bidirectional kmer <-> unique character translation maps.

    Kmers beyond len(_USABLE_KMER_CHARS) are bucketed into one "other" character
    (least-observed first), with a warning logged.
    """
    read_counts: dict[str, int] = defaultdict(int)
    occurrence_counts: dict[str, int] = defaultdict(int)
    for seq in sequences:
        for kmer in set(seq):
            read_counts[kmer] += 1
        for kmer in seq:
            occurrence_counts[kmer] += 1

    all_observed_kmers = sorted(read_counts)
    if len(all_observed_kmers) <= len(_USABLE_KMER_CHARS):
        kmer_to_unique_char = dict(zip(all_observed_kmers, _USABLE_KMER_CHARS, strict=False))
        unique_char_to_kmer = {v: k for k, v in kmer_to_unique_char.items()}
        return kmer_to_unique_char, unique_char_to_kmer

    # Most-observed kmers first, so the tail is what gets bucketed into "other"
    importance = lambda k: (read_counts[k], occurrence_counts[k])  # noqa: E731
    ranked_kmers = sorted(all_observed_kmers, key=importance, reverse=True)

    num_keepable = len(_USABLE_KMER_CHARS) - 1  # reserve one character for the "other" bucket
    kept_kmers = sorted(ranked_kmers[:num_keepable])
    bucketed_kmers = ranked_kmers[num_keepable:]

    logger.warning(
        f"{len(all_observed_kmers)} distinct kmers observed, exceeding the {len(_USABLE_KMER_CHARS)} "
        f"safe characters available for spoa. Grouping the {len(bucketed_kmers)} least-observed "
        "kmers into a single 'other' character for alignment purposes.",
    )

    # "other" resolves back to whichever bucketed kmer was observed the most
    kept_pairs = list(zip(kept_kmers, _USABLE_KMER_CHARS[:-1], strict=False))
    kmer_to_unique_char = dict(kept_pairs) | dict.fromkeys(bucketed_kmers, _OTHER_KMER_CHAR)
    unique_char_to_kmer = {char: kmer for kmer, char in kept_pairs} | {_OTHER_KMER_CHAR: max(bucketed_kmers, key=importance)}

    return kmer_to_unique_char, unique_char_to_kmer


def find_most_variable_msa_position(
    msa: list[str],
    missing_end_char: str,
    ignore_chars: list[str],
) -> tuple[int, dict[str, int]]:
    """Find the MSA column most useful for splitting reads into two groups.

    For each column, counts character frequencies (excluding gaps '-',
    missing_end_char, ignore_chars, and the "other" bucket char). Returns the
    column index where the count of the 2nd most common character is highest.

    Returns (position_index, char_counts_at_that_position).
    Returns (-1, {}) if no suitable position exists.
    """
    if not msa:
        return -1, {}

    # "other" bucket lumps unrelated rare kmers together, so skip it too
    skip_chars = set(ignore_chars) | {missing_end_char, "-", _OTHER_KMER_CHAR}

    # Initialize tracking variables
    best_pos = -1
    best_second_count = -1
    best_char_counts: dict[str, int] = {}

    # Iterate over columns in MSA
    for col in range(len(msa[0])):
        char_counts: dict[str, int] = defaultdict(int)

        # Count characters in this column, skipping specified characters
        for row in msa:
            c = row[col]
            if c not in skip_chars:
                char_counts[c] += 1

        # Need at least 2 different characters to consider this position for splitting
        if len(char_counts) < 2:
            continue

        # Get counts of characters sorted by frequency
        sorted_counts = sorted(char_counts.values(), reverse=True)
        second_count = sorted_counts[1]

        # Check if this column has a higher 2nd most common character count
        if second_count > best_second_count:
            best_second_count = second_count
            best_pos = col
            best_char_counts = dict(char_counts)

    return best_pos, best_char_counts


# Global alignment (poa algorithm 0) with a high gap-open penalty, so that positionally shifted
# interruptions (same k-mers at different offsets) appear as mismatches rather than a
# gap-shifted alignment that would hide the sequence difference. The only caller needs exactly
# these settings, so they're fixed here rather than threaded through as parameters.
_POA_ALGORITHM = 0
_POA_GAP_OPEN = -20


def generate_msa(
    spanning_sequences: list[str],
    left_flanking_sequences: list[str],
    right_flanking_sequences: list[str],
    missing_end_char: str,
) -> list[str]:
    # Combine all sequences
    all_translated_sequences = spanning_sequences + left_flanking_sequences + right_flanking_sequences
    _, msa = poa(all_translated_sequences, algorithm=_POA_ALGORITHM, g=_POA_GAP_OPEN)

    # Split MSA
    spanning_msa: list[str] = msa[: len(spanning_sequences)]
    left_flanking_msa: list[str] = msa[len(spanning_sequences) : len(spanning_sequences) + len(left_flanking_sequences)]
    right_flanking_msa: list[str] = msa[len(spanning_sequences) + len(left_flanking_sequences) :]

    # For flankings change missing end character "-" to missing_end_char
    # Left flanking: Remove from right end
    for i, seq in enumerate(left_flanking_msa):
        end_stripped_seq = seq.rstrip("-")
        left_flanking_msa[i] = end_stripped_seq + missing_end_char * (len(seq) - len(end_stripped_seq))

    # Right flanking: Remove from left end
    for i, seq in enumerate(right_flanking_msa):
        start_stripped_seq = seq.lstrip("-")
        right_flanking_msa[i] = missing_end_char * (len(seq) - len(start_stripped_seq)) + start_stripped_seq

    return spanning_msa + left_flanking_msa + right_flanking_msa


def _get_kmer_index_at_msa_col(msa_row: str, col: int, anchor_chars: set[str], missing_end_char: str) -> int:
    """Return the 1-based kmer index at MSA column col for a single MSA row.

    Kmer chars are those that are not anchor chars, not '-', and not missing_end_char.
    Counts kmer characters up to and including col, giving a 1-based position.
    """
    return sum(1 for c in msa_row[: col + 1] if c not in anchor_chars and c not in {"-", missing_end_char})


def _make_empty_backup_summary(run: bool, significant: bool = False) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "backup_test_run": run,
            "backup_test_p_value": np.nan,
            "backup_test_significant": significant,
            "backup_test_position_min": -1,
            "backup_test_position_max": -1,
            "backup_test_kmer_h1": None,
            "backup_test_kmer_h2": None,
            "backup_test_n_h1": 0,
            "backup_test_n_h2": 0,
        },
        index=[0],
    )


def run_equal_length_backup_test(
    read_calls: list[ReadCall],
    alpha: float,
) -> tuple[list[ReadCall], list[ReadCall], pd.DataFrame]:
    """Sequence-based split test for equal-length haplotypes.

    When the primary length-based heterozygosity test is not significant, this
    test uses the multiple sequence alignment to find the most variable kmer
    position and runs a binomial test (H0: p = 0.5) on the counts of the two
    most common kmers at that position.

    A NOT significant result (p >= alpha) means the observed split is
    consistent with a 50/50 ratio, indicating two alleles with the same length
    but different sequences — in that case reads are tagged H1/H2.
    A significant result (p < alpha) means the ratio deviates from 50/50, so
    no split is made.

    Returns (updated_read_calls, new_outliers, backup_summary_df).

    Columns added to backup_summary_df:
      backup_test_run, backup_test_p_value, backup_test_significant,
      backup_test_position_min, backup_test_position_max,
      backup_test_kmer_h1, backup_test_kmer_h2,
      backup_test_n_h1, backup_test_n_h2

    Note: backup_test_significant=True means p < alpha (split NOT 50/50, no
    split made). Split is made when backup_test_significant=False (p >= alpha).
    """
    _empty = _make_empty_backup_summary(run=False)

    # Get spanning and flanking reads
    spanning_reads = [r for r in read_calls if r.alignment.type == AlignmentType.SPANNING]
    left_flanking_reads = [r for r in read_calls if r.alignment.type == AlignmentType.LEFT_FLANKING]
    right_flanking_reads = [r for r in read_calls if r.alignment.type == AlignmentType.RIGHT_FLANKING]

    # If not enough spanning reads, skip the sequence-based split test
    if len(spanning_reads) < config.min_haplotyping_depth:
        return read_calls, [], _empty

    # Build kmer sequences (same logic as create_consensus_calls)
    spanning_sequences: list[list[str]] = [r.obs_kmer_string.split("|") for r in spanning_reads]
    left_flanking_sequences: list[list[str]] = [r.obs_kmer_string.split("|")[:-1] for r in left_flanking_reads]
    right_flanking_sequences: list[list[str]] = [r.obs_kmer_string.split("|")[1:] for r in right_flanking_reads]

    all_sequences = spanning_sequences + left_flanking_sequences + right_flanking_sequences
    kmer_to_char, char_to_kmer = build_kmer_char_maps(all_sequences)

    # Translate to unique characters
    translated_spanning = ["".join(kmer_to_char[k] for k in seq) for seq in spanning_sequences]
    translated_left = ["".join(kmer_to_char[k] for k in seq) for seq in left_flanking_sequences]
    translated_right = ["".join(kmer_to_char[k] for k in seq) for seq in right_flanking_sequences]

    # Add anchors (same as in create_consensus_calls)
    left_anchor_chars = [chr(i) for i in [33, 34]]
    right_anchor_chars = [chr(i) for i in [35, 36]]
    missing_end_char = chr(37)
    anchor_chars: set[str] = set(left_anchor_chars + right_anchor_chars)

    # Use fixed random seed to ensure reproducibility of the test results
    random.seed(42)
    anchor_len = 100
    random_left_anchor = "".join([random.choice(left_anchor_chars) for _ in range(anchor_len)])
    random_right_anchor = "".join([random.choice(right_anchor_chars) for _ in range(anchor_len)])

    # Add anchors to sequences
    translated_spanning = [random_left_anchor + s + random_right_anchor for s in translated_spanning]
    translated_left = [random_left_anchor + s for s in translated_left]
    translated_right = [s + random_right_anchor for s in translated_right]

    # Generate MSA
    msa = generate_msa(translated_spanning, translated_left, translated_right, missing_end_char)

    # Find most variable position
    ignore_chars = left_anchor_chars + right_anchor_chars
    best_col, char_counts = find_most_variable_msa_position(msa, missing_end_char, ignore_chars)
    if best_col == -1:
        return read_calls, [], _make_empty_backup_summary(run=True, significant=True)

    # Get top 2 chars and map back to kmers
    sorted_chars = sorted(char_counts, key=lambda c: char_counts[c], reverse=True)
    char_h1, char_h2 = sorted_chars[0], sorted_chars[1]
    kmer_h1 = char_to_kmer[char_h1]
    kmer_h2 = char_to_kmer[char_h2]

    # Count spanning + flanking reads with kmer_h1 vs kmer_h2 at best_col
    count_h1 = sum(1 for row in msa if row[best_col] != missing_end_char and row[best_col] == char_h1)
    count_h2 = sum(1 for row in msa if row[best_col] != missing_end_char and row[best_col] == char_h2)

    # Binomial test
    result = binomtest(count_h1, count_h1 + count_h2, p=0.5, alternative="two-sided")
    p_value = float(result.pvalue)
    is_significant = p_value < alpha

    # Compute position interval from spanning reads (first len(spanning_reads) rows in MSA)
    spanning_msa = msa[: len(spanning_reads)]
    kmer_indices = [_get_kmer_index_at_msa_col(row, best_col, anchor_chars, missing_end_char) for row in spanning_msa if row[best_col] != missing_end_char]
    position_min = int(min(kmer_indices)) if kmer_indices else -1
    position_max = int(max(kmer_indices)) if kmer_indices else -1

    summary_df = pd.DataFrame(
        {
            "backup_test_run": True,
            "backup_test_p_value": p_value,
            "backup_test_significant": is_significant,
            "backup_test_position_min": position_min,
            "backup_test_position_max": position_max,
            "backup_test_kmer_h1": kmer_h1,
            "backup_test_kmer_h2": kmer_h2,
            "backup_test_n_h1": count_h1,
            "backup_test_n_h2": count_h2,
        },
        index=[0],
    )

    # If significant, the ratio is NOT approximately 50/50 and the split is likely due to noise
    if is_significant:
        return read_calls, [], summary_df

    # Re-tag reads based on which kmer they carry at best_col
    new_outliers: list[ReadCall] = []
    updated_reads: list[ReadCall] = []
    all_read_calls_ordered = spanning_reads + left_flanking_reads + right_flanking_reads

    for read_call, msa_row in zip(all_read_calls_ordered, msa, strict=False):
        # Get character at best MSA column for this read
        c = msa_row[best_col]

        # Split into H1 vs H2 based on character
        if c == char_h1:
            read_call.set_haplotype(Haplotype.H1)
            updated_reads.append(read_call)
        elif c == char_h2:
            read_call.set_haplotype(Haplotype.H2)
            updated_reads.append(read_call)
        # If character is missing_end_char or something else, mark as qc_filtered (could not be classified based on sequence)
        else:
            read_call.add_qc_filter_reason("not_split_base")
            new_outliers.append(read_call)

    # Cancel the split if any haplotype cluster would be a singleton — a singleton after a
    # sequence split indicates noise rather than a real second allele.
    haplotype_counts = {h: sum(1 for r in updated_reads if r.haplotype == h) for h in (Haplotype.H1, Haplotype.H2)}
    if any(count == 1 for count in haplotype_counts.values()):
        for read_call in updated_reads:
            read_call.set_haplotype(Haplotype.HOM)
        # Move all updated reads to new_outliers since the split is not valid
        updated_reads.extend(new_outliers)
        new_outliers = []

    return updated_reads, new_outliers, summary_df
