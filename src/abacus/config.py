# This file contains the configuration parameters for the abacus module

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    """Class for keeping configuration parameters"""

    # Graph parameters
    anchor_len: int = 500
    min_anchor_overlap: int = 200
    min_end_qual: int = 17
    trim_window_size: int = 10
    max_trim: int = 50

    # Filtering parameters
    min_mean_str_quality: int = 20
    tol_mean_str_quality: int = 30
    min_q10_str_quality: int = 15
    tol_q10_str_quality: int = 30
    max_error_rate: float = 0.01
    tol_error_rate: float = 0.005
    max_ref_divergence: float = 0.34

    # Outlier detection parameters
    min_n_outlier_detection: int = 10

    # Output parameters
    add_consensus_to_vcf: bool = False
    add_contracted_consensus_to_vcf: bool = False

    # Haplotype parameters
    min_haplotyping_depth: int = 10
    min_sd: float = 0.05
    min_var: float = field(init=False)
    het_alpha: float = 0.05

    def __post_init__(self):
        self.min_var = self.min_sd**2

    # Output files
    log_file: Path = Path("abacus.log")


config = Config()
