from dataclasses import dataclass, field

import Levenshtein
from spoa import poa

from abacus.graph import AlignmentType, Read, ReadCall, get_read_calls, graph_align_reads_to_locus
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
            em_haplotype=read_call.em_haplotype,
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


def create_consensus_calls_per_haplotype_old(read_calls: list[ReadCall]) -> list[ConsensusCall]:
    locus = read_calls[0].alignment.locus

    # Group sequences by haplotype
    sequences: dict[str, list[str]] = {}
    spanning_count: dict[str, int] = {}
    flanking_count: dict[str, int] = {}
    for read_call in read_calls:
        # Get haplotype
        haplotype = read_call.em_haplotype

        # Initialize counts
        if haplotype not in spanning_count:
            spanning_count[haplotype] = 0
        if haplotype not in flanking_count:
            flanking_count[haplotype] = 0

        # Skip if flanking
        if read_call.alignment.alignment_type in [AlignmentType.LEFT_FLANKING, AlignmentType.RIGHT_FLANKING]:
            flanking_count[haplotype] += 1
            continue

        # Get sequence
        s = read_call.obs_kmer_string

        # Add sequence to haplotype
        if haplotype not in sequences:
            sequences[haplotype] = []
        sequences[haplotype].append(s)

    # Get observed kmers
    observed_kmers = {kmer for seqs in sequences.values() for seq in seqs for kmer in seq.split("-")}

    # Get dictionaries to translate kmers <-> unique characters
    kmer_to_uniqe_char = {kmer: chr(33 + i) for i, kmer in enumerate(observed_kmers)}
    uniqe_char_to_kmer = {v: k for k, v in kmer_to_uniqe_char.items()}

    # Translate kmer strings to unique character strings
    translated_sequences: dict[str, list[str]] = {}
    for haplotype, seqs in sequences.items():
        translated_sequences[haplotype] = ["".join([kmer_to_uniqe_char[kmer] for kmer in seq.split("-")]) for seq in seqs]

    # Use Levenshtein median to create unique character consensus sequences
    consensus_unique_char_sequences = {}
    for haplotype, seqs in translated_sequences.items():
        consensus_unique_char_sequences[haplotype] = Levenshtein.median([s.encode() for s in seqs])

    # Translate unique character sequences back to kmers
    consensus_sequences = {}
    for haplotype, consensus_sequence in consensus_unique_char_sequences.items():
        consensus_sequences[haplotype] = "".join([uniqe_char_to_kmer[char] for char in consensus_sequence])

    # Create reads for consensus sequences
    consensus_reads = []
    for haplotype, consensus_sequence in consensus_sequences.items():
        full_sequence = locus.left_anchor + consensus_sequence + locus.right_anchor
        sequence_length = len(full_sequence)

        consensus_reads.append(
            Read(
                name=haplotype,
                sequence=full_sequence,
                qualities=[60] * sequence_length,
                mod_5mc_probs="0" * sequence_length,
                strand="+",
                locus=locus,
                n_soft_clipped_left=0,
                n_soft_clipped_right=0,
            ),
        )

    # Align
    consensus_alignments, _ = graph_align_reads_to_locus(consensus_reads, locus)

    # Get consensus read calls
    consensus_read_calls = get_read_calls(consensus_alignments, locus)

    # Add haplotype to read calls - use read name
    for read_call in consensus_read_calls:
        read_call.em_haplotype = read_call.alignment.name

    # Create consensus calls
    return [
        ConsensusCall.from_read_call(
            read_call=consensus_read_call,
            spanning_reads=spanning_count[consensus_read_call.em_haplotype],
            flanking_reads=flanking_count[consensus_read_call.em_haplotype],
        )
        for consensus_read_call in consensus_read_calls
    ]


def create_consensus_calls(read_calls: list[ReadCall], haplotype: str) -> list[ConsensusCall]:
    locus = read_calls[0].alignment.locus

    # Split read calls by alignment type
    spanning_read_calls = [r for r in read_calls if r.alignment.alignment_type == AlignmentType.SPANNING]
    left_flanking_read_calls = [r for r in read_calls if r.alignment.alignment_type == AlignmentType.LEFT_FLANKING]
    right_flanking_read_calls = [r for r in read_calls if r.alignment.alignment_type == AlignmentType.RIGHT_FLANKING]

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
    all_observed_kmers = {kmer for seq in all_sequences for kmer in seq}
    kmer_to_unique_char = {kmer: chr(33 + i) for i, kmer in enumerate(all_observed_kmers)}
    uniqe_char_to_kmer = {v: k for k, v in kmer_to_unique_char.items()}

    # Translate kmer strings to unique character strings
    translated_spanning_sequences: list[str] = ["".join(kmer_to_unique_char[kmer] for kmer in seq) for seq in spanning_sequences]
    translated_left_flanking_sequences: list[str] = ["".join(kmer_to_unique_char[kmer] for kmer in seq) for seq in left_flanking_sequences]
    translated_right_flanking_sequences: list[str] = ["".join(kmer_to_unique_char[kmer] for kmer in seq) for seq in right_flanking_sequences]

    # Use poa to create consensus sequences
    spanning_consensus_sequence = ""
    left_flanking_consensus_sequence = ""
    right_flanking_consensus_sequence = ""
    algorithm = 1
    # If there are spanning sequences, use all sequences to create consensus
    # If not, create consensus for left and right flanking sequences separately
    # Translate unique character sequences back to kmers
    if spanning_sequences:
        all_translated_sequences = translated_spanning_sequences + translated_left_flanking_sequences + translated_right_flanking_sequences
        cons, _ = poa(all_translated_sequences, algorithm=algorithm)
        spanning_consensus_sequence = "".join([uniqe_char_to_kmer[char] for char in cons])
    else:
        if left_flanking_sequences:
            cons, _ = poa(translated_left_flanking_sequences, algorithm=algorithm)
            left_flanking_consensus_sequence = "".join([uniqe_char_to_kmer[char] for char in cons])
        if right_flanking_sequences:
            cons, _ = poa(translated_right_flanking_sequences, algorithm=algorithm)
            right_flanking_consensus_sequence = "".join([uniqe_char_to_kmer[char] for char in cons])

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
        print(right_flanking_consensus_read_calls)
        consensus_read_calls.append(right_flanking_consensus_read_calls)

    # Add haplotype to read calls - use read name
    for read_call in consensus_read_calls:
        read_call.em_haplotype = haplotype

    # Create consensus calls
    return [
        ConsensusCall.from_read_call(
            read_call=consensus_read_call,
            spanning_reads=spanning_count,  # TODO: These counts are not right!
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

    # Align
    consensus_alignments, _ = graph_align_reads_to_locus([consensus_read], locus)

    # Get consensus read calls
    consensus_read_calls = get_read_calls(consensus_alignments, locus)
    return consensus_read_calls[0]
