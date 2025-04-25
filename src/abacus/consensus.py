import random
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from Levenshtein import distance as levenshtein_distance
from spoa import poa

from abacus.graph import AlignmentType, Read, ReadCall, get_read_calls
from abacus.locus import Locus


@dataclass
class ConsensusCall(ReadCall):
    spanning_reads: int = 0
    flanking_reads: int = 0

    # TODO: When assembly is used this should be a list (maybe one for left and right flanking, if not spanning available)
    consensus_string: str = field(init=False)
    # TODO: Add EM "count", i.e. estimated means from EM algorithm

    def __post_init__(self: "ConsensusCall") -> None:
        self.consensus_string = contract_kmer_string(self.obs_kmer_string)

    def to_dict(self) -> dict:
        return super().to_dict() | {
            "consensus_strings": self.consensus_string,
            "spanning_reads": self.spanning_reads,
            "flanking_reads": self.flanking_reads,
        }

    @classmethod
    def from_read_call(cls, read_call: ReadCall, spanning_reads: int, flanking_reads: int) -> "ConsensusCall":
        return cls(
            locus=read_call.locus,
            alignment=read_call.alignment,
            haplotype=read_call.haplotype,
            outlier_reasons=read_call.outlier_reasons,
            satellite_count=read_call.satellite_count,
            kmer_count_str=read_call.kmer_count_str,
            obs_kmer_string=read_call.obs_kmer_string,
            ref_kmer_string=read_call.ref_kmer_string,
            mod_5mc_kmer_string=read_call.mod_5mc_kmer_string,
            spanning_reads=spanning_reads,
            flanking_reads=flanking_reads,
        )


def contract_kmer_string(kmer_string: str) -> str:
    # Split kmer string
    kmer_list = kmer_string.split("-")

    # Contract kmer string
    contracted_kmer = ""

    # Initialize
    prev_kmer = kmer_list[0]
    count = 1

    # Iterate
    for kmer in kmer_list[1:]:
        if kmer == prev_kmer:
            count += 1
        else:
            contracted_kmer += f"{count}({prev_kmer})-"
            prev_kmer = kmer
            count = 1

    contracted_kmer += f"{count}({prev_kmer})"

    return contracted_kmer


def create_consensus_calls(read_calls: list[ReadCall], haplotype: str) -> list[ConsensusCall]:
    locus = read_calls[0].alignment.locus

    # Split read calls by alignment type
    spanning_read_calls = [r for r in read_calls if r.alignment.type == AlignmentType.SPANNING]
    left_flanking_read_calls = [r for r in read_calls if r.alignment.type == AlignmentType.LEFT_FLANKING]
    right_flanking_read_calls = [r for r in read_calls if r.alignment.type == AlignmentType.RIGHT_FLANKING]

    # Group sequences by haplotype
    spanning_sequences: list[list[str]] = [s.obs_kmer_string.split("-") for s in spanning_read_calls]
    left_flanking_sequences: list[list[str]] = [s.obs_kmer_string.split("-") for s in left_flanking_read_calls]
    right_flanking_sequences: list[list[str]] = [s.obs_kmer_string.split("-") for s in right_flanking_read_calls]

    spanning_count: int = len(spanning_sequences)
    # TODO: Should this be split?
    flanking_count: int = len(left_flanking_sequences) + len(right_flanking_sequences)

    # For flanking reads remove last kmer
    left_flanking_sequences = [seq[:-1] for seq in left_flanking_sequences]
    right_flanking_sequences = [seq[1:] for seq in right_flanking_sequences]

    # Get dictionaries to translate kmers <-> unique characters
    # Add dictionary for anchor kmers
    all_sequences = spanning_sequences + left_flanking_sequences + right_flanking_sequences
    all_observed_kmers = sorted({kmer for seq in all_sequences for kmer in seq})
    # Start from 46 to avoid special characters - esp. "-" used by poa!
    kmer_to_unique_char = {kmer: chr(47 + i) for i, kmer in enumerate(all_observed_kmers)}
    uniqe_char_to_kmer = {v: k for k, v in kmer_to_unique_char.items()}

    # Translate kmer strings to unique character strings
    translated_spanning_sequences: list[str] = ["".join(kmer_to_unique_char[kmer] for kmer in seq) for seq in spanning_sequences]
    translated_left_flanking_sequences: list[str] = ["".join(kmer_to_unique_char[kmer] for kmer in seq) for seq in left_flanking_sequences]
    translated_right_flanking_sequences: list[str] = ["".join(kmer_to_unique_char[kmer] for kmer in seq) for seq in right_flanking_sequences]

    # Reserve special characters for anchors and missing end character
    left_anchor_chars = [chr(i) for i in [33, 34]]
    right_anchor_chars = [chr(i) for i in [35, 36]]
    missing_end_char = chr(37)

    # Add random anchors to sequences
    random.seed(42)
    anchor_len = 100
    random_left_anchor = "".join([random.choice(left_anchor_chars) for _ in range(anchor_len)])
    random_right_anchor = "".join([random.choice(right_anchor_chars) for _ in range(anchor_len)])

    translated_spanning_sequences = [random_left_anchor + seq + random_right_anchor for seq in translated_spanning_sequences]
    translated_left_flanking_sequences = [random_left_anchor + seq for seq in translated_left_flanking_sequences]
    translated_right_flanking_sequences = [seq + random_right_anchor for seq in translated_right_flanking_sequences]

    # Use poa to create consensus sequences
    spanning_consensus_sequence = ""
    left_flanking_consensus_sequence = ""
    right_flanking_consensus_sequence = ""
    algorithm = 0
    # If there are spanning sequences, use all sequences to create consensus
    # If not, create consensus for left and right flanking sequences separately
    # Translate unique character sequences back to kmers
    if spanning_sequences:
        msa = generate_msa(translated_spanning_sequences, translated_left_flanking_sequences, translated_right_flanking_sequences, missing_end_char, algorithm)
        spanning_consensus_sequence = generate_majority_consensus(msa, missing_end_char, left_anchor_chars + right_anchor_chars)
    else:
        if left_flanking_sequences:
            msa = generate_msa([], translated_left_flanking_sequences, [], missing_end_char, algorithm)
            left_flanking_consensus_sequence = generate_majority_consensus(msa, missing_end_char, left_anchor_chars + right_anchor_chars)
        if right_flanking_sequences:
            msa = generate_msa([], [], translated_right_flanking_sequences, missing_end_char, algorithm)
            right_flanking_consensus_sequence = generate_majority_consensus(msa, missing_end_char, left_anchor_chars + right_anchor_chars)

    # Translate unique character sequences back to kmers
    spanning_consensus_sequence = "".join([uniqe_char_to_kmer[char] for char in spanning_consensus_sequence])
    left_flanking_consensus_sequence = "".join([uniqe_char_to_kmer[char] for char in left_flanking_consensus_sequence])
    right_flanking_consensus_sequence = "".join([uniqe_char_to_kmer[char] for char in right_flanking_consensus_sequence])

    # Create reads for consensus sequences
    consensus_read_calls: list[ReadCall] = []
    if spanning_consensus_sequence:
        spanning_consensus_read_calls = get_consensus_read_call(locus, spanning_consensus_sequence, AlignmentType.SPANNING, haplotype)
        consensus_read_calls.append(spanning_consensus_read_calls)
    if left_flanking_consensus_sequence:
        left_flanking_consensus_read_calls = get_consensus_read_call(locus, left_flanking_consensus_sequence, AlignmentType.LEFT_FLANKING, haplotype)
        consensus_read_calls.append(left_flanking_consensus_read_calls)
    if right_flanking_consensus_sequence:
        right_flanking_consensus_read_calls = get_consensus_read_call(locus, right_flanking_consensus_sequence, AlignmentType.RIGHT_FLANKING, haplotype)
        consensus_read_calls.append(right_flanking_consensus_read_calls)

    # Add haplotype to read calls - use read name
    for read_call in consensus_read_calls:
        read_call.haplotype = haplotype

    # Create consensus calls
    return [
        ConsensusCall.from_read_call(
            read_call=consensus_read_call,
            spanning_reads=spanning_count,  # TODO: These counts are not right!
            flanking_reads=flanking_count,
        )
        for consensus_read_call in consensus_read_calls
    ]


def generate_msa(
    spanning_sequences: list[str],
    left_flanking_sequences: list[str],
    right_flanking_sequences: list[str],
    missing_end_char: str,
    algorithm: int,
) -> list[str]:
    # Combine all sequences
    all_translated_sequences = spanning_sequences + left_flanking_sequences + right_flanking_sequences
    _, msa = poa(all_translated_sequences, algorithm=algorithm)

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


def generate_majority_consensus(msa: list[str], missing_end_char: str, remove_chars: list[str]) -> str:
    consensus_sequence = ""
    for i in range(len(msa[0])):
        char_votes: dict[str, int] = defaultdict(int)
        for seq in msa:
            char = seq[i]
            # Skip missing end character
            if char == missing_end_char:
                continue
            # Add vote
            char_votes[char] += 1
        majority_char = max(char_votes, key=lambda k: char_votes[k])
        consensus_sequence += majority_char

    # Remove "-" and other characters from consensus sequence
    for char in [*remove_chars, "-"]:
        consensus_sequence = consensus_sequence.replace(char, "")

    return consensus_sequence


def get_consensus_read_call(locus: Locus, sequence: str, alignment_type: AlignmentType, haplotype: str) -> ReadCall:
    if alignment_type == AlignmentType.SPANNING:
        full_sequence = locus.left_anchor + sequence + locus.right_anchor
    elif alignment_type == AlignmentType.LEFT_FLANKING:
        full_sequence = locus.left_anchor + sequence
    elif alignment_type == AlignmentType.RIGHT_FLANKING:
        full_sequence = sequence + locus.right_anchor

    # Create read name
    consensus_read_name = f"consensus_{haplotype}"
    # Add alignment type to name
    if alignment_type == AlignmentType.LEFT_FLANKING:
        consensus_read_name += "_left"
    elif alignment_type == AlignmentType.RIGHT_FLANKING:
        consensus_read_name += "_right"

    # Create read
    sequence_length = len(full_sequence)
    consensus_read = Read(
        name=consensus_read_name,
        sequence=full_sequence,
        qualities=[60] * sequence_length,
        mod_5mc_probs="0" * sequence_length,
        strand="+",
        locus=locus,
        n_soft_clipped_left=0,
        n_soft_clipped_right=0,
    )

    # Get consensus read calls
    consensus_read_calls, _ = get_read_calls([consensus_read], locus)
    return consensus_read_calls[0]


def get_heterozygote_labels_seq(
    read_calls: list[ReadCall],
    consensus_read_calls: list[ConsensusCall],
) -> list[ReadCall]:
    # Get unique consensus haplotypes
    unique_haplotypes = {read_call.haplotype for read_call in consensus_read_calls}
    for read_call in read_calls:
        # Skip if read call is spanning
        if read_call.alignment.type == AlignmentType.SPANNING:
            continue

        # Find closest consensus and use this as haplotype
        # Initialize
        closest_consensus = ""
        dist_to_closest = np.inf
        for haplotype in unique_haplotypes:
            # Get consensus read calls for this haplotype
            haplotype_consensus_read_calls = [x for x in consensus_read_calls if x.haplotype == haplotype]

            # Get group probabilities using string distance
            dist_to_consensus = calc_dist_to_consensus(
                read_call=read_call,
                consensus_read_calls=haplotype_consensus_read_calls,
            )

            # Check if this is the closest consensus
            if dist_to_consensus < dist_to_closest:
                dist_to_closest = dist_to_consensus
                closest_consensus = haplotype

            read_call.haplotype = closest_consensus

    return read_calls


def calc_dist_to_consensus(
    read_call: ReadCall,
    consensus_read_calls: list[ConsensusCall],
) -> float:
    # Extract information from reads
    seq = read_call.alignment.str_sequence
    seq_type = read_call.alignment.type

    best_dist = np.inf
    for consensus_read_call in consensus_read_calls:
        consensus_seq = consensus_read_call.alignment.str_sequence
        consensus_type = consensus_read_call.alignment.type

        # Skip if both are flanking and on different sides
        if AlignmentType.SPANNING not in (seq_type, consensus_type) and seq_type != consensus_type:
            continue

        seq_trimmed, consensus_seq_trimmed = trim_sequences_for_comparison(seq, seq_type, consensus_seq, consensus_type)
        dist = levenshtein_distance(seq_trimmed, consensus_seq_trimmed)

        # Check if distance is better than best distance
        best_dist = min(best_dist, dist)

    return best_dist


def trim_sequences_for_comparison(seq1: str, type1: AlignmentType, seq2: str, type2: AlignmentType) -> tuple[str, str]:
    """Trim two sequences for appropriate comparison based on their alignment types.

    Args:
        seq1: First sequence string
        type1: Alignment type of the first sequence
        seq2: Second sequence string
        type2: Alignment type of the second sequence

    Returns:
        tuple[str, str]: Trimmed versions of seq1 and seq2

    """
    seq1_start, seq1_end = 0, len(seq1)
    seq2_start, seq2_end = 0, len(seq2)

    # LEFT_FLANKING vs LEFT_FLANKING
    if type1 == AlignmentType.LEFT_FLANKING and type2 == AlignmentType.LEFT_FLANKING:
        min_length = min(len(seq1), len(seq2))
        seq1_end = seq2_end = min_length

    # LEFT_FLANKING vs SPANNING
    elif type1 == AlignmentType.LEFT_FLANKING and type2 == AlignmentType.SPANNING:
        trim_length = min(len(seq1), len(seq2))
        seq2_end = trim_length
    elif type1 == AlignmentType.SPANNING and type2 == AlignmentType.LEFT_FLANKING:
        trim_length = min(len(seq1), len(seq2))
        seq1_end = trim_length

    # RIGHT_FLANKING vs SPANNING
    elif type1 == AlignmentType.RIGHT_FLANKING and type2 == AlignmentType.SPANNING:
        trim_length = min(len(seq1), len(seq2))
        seq2_start = len(seq2) - trim_length
    elif type1 == AlignmentType.SPANNING and type2 == AlignmentType.RIGHT_FLANKING:
        trim_length = min(len(seq1), len(seq2))
        seq1_start = len(seq1) - trim_length

    # RIGHT_FLANKING vs RIGHT_FLANKING
    elif type1 == AlignmentType.RIGHT_FLANKING and type2 == AlignmentType.RIGHT_FLANKING:
        min_length = min(len(seq1), len(seq2))
        seq1_start = len(seq1) - min_length
        seq2_start = len(seq2) - min_length

    # LEFT_FLANKING vs RIGHT_FLANKING
    # No meaningful overlap, trim the beginning of the longer sequence
    elif (type1, type2) == (AlignmentType.LEFT_FLANKING, AlignmentType.RIGHT_FLANKING):
        min_length = min(len(seq1), len(seq2))
        seq1_end = min_length
        seq2_start = len(seq2) - min_length

    elif (type1, type2) == (AlignmentType.RIGHT_FLANKING, AlignmentType.LEFT_FLANKING):
        min_length = min(len(seq1), len(seq2))
        seq1_start = len(seq1) - min_length
        seq2_end = min_length

    # Apply trimming
    seq1_trimmed = seq1[seq1_start:seq1_end]
    seq2_trimmed = seq2[seq2_start:seq2_end]

    return seq1_trimmed, seq2_trimmed
