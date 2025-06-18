from enum import StrEnum, auto
from pathlib import Path

import Levenshtein

REPORT_TEMPLATE = Path(__file__).parent / "report.Rmd"

# Link: https://www.dnabaser.com/articles/IUPAC%20ambiguity%20codes.html
AMBIGUOUS_BASES_DICT = {
    "N": ["A", "T", "C", "G"],
    "R": ["A", "G"],
    "Y": ["T", "C"],
    "S": ["G", "C"],
    "W": ["A", "T"],
    "K": ["G", "T"],
    "M": ["A", "C"],
    "B": ["C", "G", "T"],
    "D": ["A", "G", "T"],
    "H": ["A", "C", "T"],
    "V": ["A", "C", "G"],
}


# Enum for alignment type
class AlignmentType(StrEnum):
    SPANNING = auto()
    LEFT_FLANKING = auto()
    RIGHT_FLANKING = auto()
    NO_ANCHORS = auto()


class Sex(StrEnum):
    XX = "XX"
    XY = "XY"


class Haplotype(StrEnum):
    H1 = auto()
    H2 = auto()
    HOM = auto()
    OUTLIER = auto()
    NONE = auto()


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


def compute_levenshtein_rate(s1: str, s2: str, indel_cost: float = 1, replace_cost: float = 1) -> float:
    # If reference and sequence are empty, return 0
    if not s1 and not s2:
        return 0

    # Get edit operations
    ops = Levenshtein.editops(s1, s2)

    # Calculate cost
    cost = 0.0
    for op in ops:
        if op[0] in ["delete", "insert"]:
            cost += indel_cost
        elif op[0] == "replace":
            cost += replace_cost

    # Normalize cost
    return cost / max(len(s2), len(s1))


def compute_ref_divergence(cigar: str, indel_cost: float = 1, replace_cost: float = 1) -> float:
    # If reference and sequence are empty, return 0
    if not cigar:
        return 0

    # Calculate cost
    cost = 0.0
    for op in cigar:
        # Skip matches and equalities
        if op[0] in ["M", "="]:
            continue

        # Insertions + deletions
        if op[0] == "I" or op[0] == "D":
            cost += indel_cost
        # Replacements
        elif op[0] == "X":
            cost += replace_cost
        else:
            error_msg = f"Unknown CIGAR operation: {op}"
            raise ValueError(error_msg)

    # Normalize cost
    return cost / len(cigar)
