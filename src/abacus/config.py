# This file contains the configuration parameters for the abacus module

from dataclasses import dataclass


@dataclass
class Config:
    """Class for keeping configuration parameters"""

    # Graph parameters
    ANCHOR_LEN: int = 500
    MIN_ANCHOR_OVERLAP: int = 3
    MIN_STR_READ_QUAL: int = 15

    # Old
    MAX_UNLINK_DIST = 50
    GAP_OPEN_PENALTY = 4
    GAP_EXTENSION_PENALTY = 2
    MISMATCH_PENALTY = 4
    MATCH_SCORE = 1
    LONG_GAP_OPEN_PENALTY = 24
    LONG_GAP_EXTENSION_PENALTY = 1
    N_MISMATCH_PENALTY = 1


config = Config()
