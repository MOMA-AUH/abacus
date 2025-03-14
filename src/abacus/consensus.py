from dataclasses import dataclass, field

import Levenshtein

from abacus.graph import AlignmentType, Read, ReadCall, get_read_calls, graph_align_reads_to_locus


@dataclass
class ConsensusCall(ReadCall):
    spanning_reads: int = 0
    flanking_reads: int = 0

    # TODO: When assembly is used this shpuld be a list (maybe one for left and right flanking, if not spanning available)
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


def create_consensus_calls_per_haplotype(read_calls: list[ReadCall]) -> list[ConsensusCall]:
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
