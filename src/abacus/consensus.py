from dataclasses import dataclass, field

import numpy as np
from Levenshtein import distance as levenshtein_distance

from abacus.consensus_search import hill_climb
from abacus.graph import AlignmentType, Read, ReadCall, get_read_calls
from abacus.locus import Locus
from abacus.timing import timed
from abacus.utils import Haplotype, trim_sequences_for_comparison


@dataclass
class ConsensusCall(ReadCall):
    spanning_reads: int = 0
    flanking_reads: int = 0

    consensus_string: str = field(init=False)

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
            obs_kmer_string=read_call.obs_kmer_string,
            ref_kmer_string=read_call.ref_kmer_string,
            mod_5mc_kmer_string=read_call.mod_5mc_kmer_string,
            qual_kmer_string=read_call.qual_kmer_string,
            spanning_reads=spanning_reads,
            flanking_reads=flanking_reads,
            str_error_rate=0.0,
        )


def contract_kmer_string(kmer_string: str) -> str:
    # Split kmer string
    kmer_list = kmer_string.split("|")

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


def build_consensus_for_locus(grouped_read_calls: list[ReadCall]) -> list[ConsensusCall]:
    """Build final per-haplotype consensus calls for a locus.

    Two passes are required: a raw consensus is built first and used to re-group flanking read
    calls (a flanking read only ever moves positions its own alignment reaches, so grouping must
    be settled against a real consensus before the final one is built). `grouped_read_calls` is
    relabeled in place by the re-grouping step.
    """
    with timed("consensus"):
        raw_consensus_calls = consensus_calls_by_haplotype(grouped_read_calls)
        update_flanking_labels_based_on_consensus(read_calls=grouped_read_calls, consensus_read_calls=raw_consensus_calls)
        return consensus_calls_by_haplotype(grouped_read_calls)


def consensus_calls_by_haplotype(read_calls: list[ReadCall]) -> list[ConsensusCall]:
    consensus_calls: list[ConsensusCall] = []
    for haplotype in {r.haplotype for r in read_calls}:
        haplotyped_read_calls = [r for r in read_calls if r.haplotype == haplotype]
        consensus_calls.extend(create_consensus_calls(read_calls=haplotyped_read_calls, haplotype=haplotype))
    return consensus_calls


def create_consensus_calls(read_calls: list[ReadCall], haplotype: Haplotype) -> list[ConsensusCall]:
    locus = read_calls[0].alignment.locus

    # Split read calls by alignment type
    spanning_read_calls = [r for r in read_calls if r.alignment.type == AlignmentType.SPANNING]
    left_flanking_read_calls = [r for r in read_calls if r.alignment.type == AlignmentType.LEFT_FLANKING]
    right_flanking_read_calls = [r for r in read_calls if r.alignment.type == AlignmentType.RIGHT_FLANKING]

    # Group sequences by haplotype
    spanning_sequences: list[list[str]] = [s.obs_kmer_string.split("|") for s in spanning_read_calls]
    left_flanking_sequences: list[list[str]] = [s.obs_kmer_string.split("|") for s in left_flanking_read_calls]
    right_flanking_sequences: list[list[str]] = [s.obs_kmer_string.split("|") for s in right_flanking_read_calls]

    spanning_count: int = len(spanning_sequences)
    flanking_count: int = len(left_flanking_sequences) + len(right_flanking_sequences)

    # With spanning reads, search one consensus over all reads together - a flanking read only
    # ever moves positions its own alignment reaches, so it can't corrupt the rest. With none,
    # search left and right consensuses separately: there's no shared coordinate system to
    # combine them in.
    spanning_consensus_sequence = ""
    left_flanking_consensus_sequence = ""
    right_flanking_consensus_sequence = ""
    if spanning_sequences:
        reads = (
            [(seq, "spanning", r.alignment.strand) for seq, r in zip(spanning_sequences, spanning_read_calls, strict=True)]
            + [(seq, "left", r.alignment.strand) for seq, r in zip(left_flanking_sequences, left_flanking_read_calls, strict=True)]
            + [(seq, "right", r.alignment.strand) for seq, r in zip(right_flanking_sequences, right_flanking_read_calls, strict=True)]
        )
        spanning_consensus_sequence = "".join(hill_climb(reads))
    else:
        if left_flanking_sequences:
            left_reads = [(seq, "left", r.alignment.strand) for seq, r in zip(left_flanking_sequences, left_flanking_read_calls, strict=True)]
            left_flanking_consensus_sequence = "".join(hill_climb(left_reads))
        if right_flanking_sequences:
            right_reads = [(seq, "right", r.alignment.strand) for seq, r in zip(right_flanking_sequences, right_flanking_read_calls, strict=True)]
            right_flanking_consensus_sequence = "".join(hill_climb(right_reads))

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
        read_call.set_haplotype(haplotype)

    # Create consensus calls
    return [
        ConsensusCall.from_read_call(
            read_call=consensus_read_call,
            spanning_reads=spanning_count,
            flanking_reads=flanking_count,
        )
        for consensus_read_call in consensus_read_calls
    ]


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


def update_flanking_labels_based_on_consensus(
    read_calls: list[ReadCall],
    consensus_read_calls: list[ConsensusCall],
) -> None:
    # Get unique consensus haplotypes
    unique_haplotypes = {read_call.haplotype for read_call in consensus_read_calls}
    for read_call in read_calls:
        # Skip if read call is spanning
        if read_call.alignment.type == AlignmentType.SPANNING:
            continue

        # Find closest consensus and use this as haplotype
        # Initialize
        current_haplotype = read_call.haplotype
        closest_consensus_haplotype = Haplotype.NONE
        dist_to_closest = np.inf
        for haplotype in unique_haplotypes:
            # Get consensus read calls for this haplotype
            haplotype_consensus_read_calls = [x for x in consensus_read_calls if x.haplotype == haplotype]

            # Get group probabilities using string distance
            dist_to_consensus = calc_dist_to_consensus(
                read_call=read_call,
                consensus_read_calls=haplotype_consensus_read_calls,
            )

            # Check if this is the closest consensus (or tied for closest, in which case prefer current haplotype to avoid unnecessary label changes)
            if dist_to_consensus < dist_to_closest or (dist_to_closest == dist_to_consensus and haplotype == current_haplotype):
                dist_to_closest = dist_to_consensus
                closest_consensus_haplotype = haplotype

            read_call.set_haplotype(closest_consensus_haplotype)


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
