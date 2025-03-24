# This file contains the configuration parameters for the abacus module

from dataclasses import dataclass, field


@dataclass
class Config:
    """Class for keeping configuration parameters"""

    # Graph parameters
    anchor_len: int = 500
    min_anchor_overlap: int = 200
    min_str_read_qual: int = 17
    max_trim: int = 50
    error_rate_threshold: float = 0.2

    # Haplotype parameters
    min_sd: float = 0.33
    min_var: float = field(init=False)

    def __post_init__(self):
        self.min_var = self.min_sd**2


config = Config()
