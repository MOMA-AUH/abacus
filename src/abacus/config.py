# This file contains the configuration parameters for the abacus module

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Class for keeping configuration parameters"""

    # Graph parameters
    anchor_len: int = 500
    min_anchor_overlap: int = 200
    min_str_qual: int = 17
    min_end_qual: int = 15
    trim_window_size: int = 10
    max_trim: int = 50
    error_rate_threshold: float = 0.2

    # Haplotype parameters
    min_sd: float = 0.05
    min_var: float = field(init=False)
    het_alpha: float = 0.05
    split_alpha: float = 0.01
    min_haplotype_depth: int = 3

    def __post_init__(self):
        self.min_var = self.min_sd**2

    # Output files
    log_file: Path = Path("abacus.log")


config = Config()
