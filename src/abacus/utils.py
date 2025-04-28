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


def compute_error_rate(s1: str, s2: str, indel_cost: float = 1, replace_cost: float = 1) -> float:
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
