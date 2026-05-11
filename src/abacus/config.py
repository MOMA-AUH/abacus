# This file contains the configuration parameters for the abacus module

from dataclasses import asdict, dataclass, field
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

    # QC filtering parameters
    min_n_qc_filtering: int = 5
    min_mean_str_quality: int = 17
    tol_mean_str_quality: int = 35
    min_q10_str_quality: int = 7
    tol_q10_str_quality: int = 25
    max_error_rate: float = 0.01
    tol_error_rate: float = 0.005
    max_ref_divergence: float = 0.34

    # Length outlier detection parameters
    min_n_length_outlier_detection: int = 5
    tol_length_outlier_bases: int = 3  # tolerance in bases; reads within this range are always kept
    tol_length_outlier_pct: float = 0.10  # tolerance in % of median; reads within this range are always kept

    # Output parameters
    add_consensus_to_vcf: bool = False
    add_contracted_consensus_to_vcf: bool = False

    # Haplotype parameters
    min_haplotyping_depth: int = 10
    min_sd: float = 0.05
    min_var: float = field(init=False)
    het_alpha: float = 0.05
    equal_length_alpha: float = 0.05

    def __post_init__(self):
        self.min_var = self.min_sd**2

    def to_dict(self) -> dict:
        """Convert the Config dataclass to a dictionary."""
        return asdict(self)

    # Output files
    log_file: Path = Path("abacus.log")


config = Config()
