# This file contains the configuration parameters for the abacus module

from dataclasses import dataclass


@dataclass
class Config:
    """Class for keeping configuration parameters"""

    # Graph parameters
    anchor_len: int = 500
    min_anchor_overlap: int = 200
    min_str_read_qual: int = 17
    max_trim: int = 50
    error_rate_threshold: float = 0.2


config = Config()
