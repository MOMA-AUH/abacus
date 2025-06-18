from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from abacus.config import config
from abacus.graph import ReadCall
from abacus.parameter_estimation import (
    HeterozygousParameters,
    HomozygousParameters,
    calculate_grouping_probabilities_flanking,
    calculate_grouping_probabilities_spanning,
    calculate_log_likelihood_heterozygous,
    calculate_log_likelihood_homozygous,
    estimate_heterozygous_parameters,
    estimate_homozygous_parameters,
    unpack_read_calls,
)
from abacus.utils import AlignmentType, Haplotype


def run_haplotyping(
    read_calls: list[ReadCall],
    ploidy: int,
) -> tuple[list[ReadCall], list[ReadCall], HeterozygousParameters, HomozygousParameters, pd.DataFrame]:
    # Handle empty read_calls
    if not read_calls:
        return create_empty_results()

    if ploidy not in (1, 2):
        error_msg = f"Unsupported ploidy: {ploidy}. Only 1 and 2 are supported."
        raise ValueError(error_msg)

    # Initialize
    all_outlier_read_calls: list[ReadCall] = []
    hom_params = estimate_homozygous_parameters(read_calls)
    het_params = estimate_heterozygous_parameters(read_calls)

    # Initialize grouping
    grouped_read_calls = group_read_calls(read_calls, het_params, ploidy)

    # Check for singleton clusters, and keep removing them until there are none left or the number of read calls is below the minimum threshold
    singleton_read_calls = check_for_singleton_clusters(grouped_read_calls)
    while len(singleton_read_calls) > 0 and len(grouped_read_calls) > config.min_n_outlier_detection:
        # If there are singleton read calls, they are outliers - remove them
        for outlier in singleton_read_calls:
            all_outlier_read_calls.append(outlier)
            grouped_read_calls.remove(outlier)

        # Re-estimate parameters
        hom_params = estimate_homozygous_parameters(grouped_read_calls)
        het_params = estimate_heterozygous_parameters(grouped_read_calls)

        # Re-group read calls
        grouped_read_calls = group_read_calls(grouped_read_calls, het_params, ploidy)

        # Check for singleton clusters again
        singleton_read_calls = check_for_singleton_clusters(grouped_read_calls)

    if ploidy == 1:
        het_params_nan = HeterozygousParameters(
            mean_h1=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            mean_h2=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            unit_var=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            mean_h1_ci_low=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            mean_h1_ci_high=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            mean_h2_ci_low=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            mean_h2_ci_high=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
        )

        test_summary_df = summarize_test_statistics(
            log_lik_hetero=np.float64(np.nan),
            log_lik_hom=np.float64(np.nan),
            n_par_hetero=-1,
            n_par_hom=-1,
            test_statistic=np.float64(0),
            df=-1,
            heterozygosity_p_value=np.float64(1),
            is_significant=False,
        )

        return grouped_read_calls, all_outlier_read_calls, het_params_nan, hom_params, test_summary_df

    # Test for heterozygosity
    log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, df, heterozygosity_p_value = test_heterozygosity(
        grouped_read_calls,
        het_params,
        hom_params,
    )

    # Set haplotype to "hom" if heterozygosity test is not significant
    heterozygosity_test_significant = bool(heterozygosity_p_value < config.het_alpha)
    if not heterozygosity_test_significant:
        for read in grouped_read_calls:
            read.set_haplotype(Haplotype.HOM)

    # Summarize test statistics
    test_summary_df = summarize_test_statistics(
        log_lik_hetero=log_lik_hetero,
        log_lik_hom=log_lik_hom,
        n_par_hetero=n_par_hetero,
        n_par_hom=n_par_hom,
        test_statistic=test_statistic,
        df=df,
        heterozygosity_p_value=heterozygosity_p_value,
        is_significant=heterozygosity_test_significant,
    )

    return grouped_read_calls, all_outlier_read_calls, het_params, hom_params, test_summary_df


def create_empty_results() -> tuple[list[ReadCall], list[ReadCall], HeterozygousParameters, HomozygousParameters, pd.DataFrame]:
    summary_res_df = summarize_test_statistics(
        log_lik_hom=np.float64(np.nan),
        log_lik_hetero=np.float64(np.nan),
        n_par_hom=-1,
        n_par_hetero=-1,
        test_statistic=np.float64(0),
        df=-1,
        heterozygosity_p_value=np.float64(1),
        is_significant=False,
    )

    het_params = HeterozygousParameters(
        mean_h1=np.full(1, np.nan, dtype=np.float64),
        mean_h2=np.full(1, np.nan, dtype=np.float64),
        unit_var=np.full(1, np.nan, dtype=np.float64),
        mean_h1_ci_low=np.full(1, np.nan, dtype=np.float64),
        mean_h1_ci_high=np.full(1, np.nan, dtype=np.float64),
        mean_h2_ci_low=np.full(1, np.nan, dtype=np.float64),
        mean_h2_ci_high=np.full(1, np.nan, dtype=np.float64),
    )

    hom_params = HomozygousParameters(
        mean=np.array([np.nan], dtype=np.float64),
        unit_var=np.array([np.nan], dtype=np.float64),
        mean_ci_low=np.array([np.nan], dtype=np.float64),
        mean_ci_high=np.array([np.nan], dtype=np.float64),
    )

    return [], [], het_params, hom_params, summary_res_df


def get_is_left_flanking_bool(flanking_reads: list[ReadCall]) -> list[bool]:
    return [read.alignment.type == AlignmentType.LEFT_FLANKING for read in flanking_reads]


def group_read_calls(
    read_calls: list[ReadCall],
    het_params: HeterozygousParameters,
    ploidy: int = 2,
) -> list[ReadCall]:
    if ploidy == 1:
        return [read.set_haplotype(Haplotype.HOM) for read in read_calls]

    # Split read calls into spanning and flanking reads
    spanning_reads = [read for read in read_calls if read.is_spanning()]
    flanking_reads = [read for read in read_calls if not read.is_spanning()]

    # Add labels to spanning reads
    spanning_counts = np.array([read.satellite_count for read in spanning_reads])

    # Calculate grouping probabilities
    p_group_h1_spanning, p_group_h2_spanning = calculate_grouping_probabilities_spanning(
        spanning_counts,
        het_params.mean_h1,
        het_params.mean_h2,
        het_params.unit_var,
    )

    for i, read in enumerate(spanning_reads):
        # Assign labels based on maximum probability
        read.set_haplotype(Haplotype.H1 if p_group_h1_spanning[i] > p_group_h2_spanning[i] else Haplotype.H2)

    # Add labels to flanking reads
    flanking_counts = np.array([read.satellite_count for read in flanking_reads])
    is_left_flanking = get_is_left_flanking_bool(flanking_reads)

    # Calculate grouping probabilities for flanking reads
    p_group_h1_flanking, p_group_h2_flanking = calculate_grouping_probabilities_flanking(
        flanking_counts,
        is_left_flanking,
        het_params.mean_h1,
        het_params.mean_h2,
        het_params.unit_var,
    )

    for i, read in enumerate(flanking_reads):
        # Assign labels based on maximum probability
        read.set_haplotype(Haplotype.H1 if p_group_h1_flanking[i] > p_group_h2_flanking[i] else Haplotype.H2)

    return spanning_reads + flanking_reads


def check_for_singleton_clusters(
    grouped_read_calls: list[ReadCall],
) -> list[ReadCall]:
    if not len(grouped_read_calls):
        return []

    # Initialize lists
    outlier_read_calls: list[ReadCall] = []

    # Evaluate haplotypes
    # Get unique haplotypes
    haplotypes = {rc.haplotype for rc in grouped_read_calls}
    for haplotype in haplotypes:
        # Get read calls for haplotype
        haplotype_read_calls = [rc for rc in grouped_read_calls if rc.haplotype == haplotype]

        # If haplotype has only one read call, mark as outlier and continue
        if len(haplotype_read_calls) == 1:
            rc = haplotype_read_calls[0]
            rc.add_outlier_reason("caused_singleton_in_haplotyping")
            outlier_read_calls.append(rc)
            continue

    return outlier_read_calls


def test_heterozygosity(
    read_calls: list[ReadCall],
    par_het: HeterozygousParameters,
    par_hom: HomozygousParameters,
) -> tuple[np.float64, np.float64, int, int, np.float64, int, np.float64]:
    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

    # Calculate log likelihoods for homozygous and heterozygous models
    log_lik_hetero = calculate_log_likelihood_heterozygous(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        par_het.mean_h1,
        par_het.mean_h2,
        par_het.unit_var,
    )
    log_lik_hom = calculate_log_likelihood_homozygous(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        par_hom.mean,
        par_hom.unit_var,
    )

    # Calculate test statistic
    test_statistic = -2 * (log_lik_hom - log_lik_hetero)
    n_par_hetero = len(par_het.mean_h1) + len(par_het.mean_h2) + len(par_het.unit_var)
    n_par_hom = len(par_hom.mean) + len(par_hom.unit_var)
    deg_freedom = n_par_hetero - n_par_hom

    p_value = 1 - np.float64(chi2.cdf(test_statistic, deg_freedom))

    return log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, deg_freedom, p_value


def summarize_test_statistics(
    log_lik_hom: np.float64,
    log_lik_hetero: np.float64,
    n_par_hom: int,
    n_par_hetero: int,
    test_statistic: np.float64,
    df: int,
    heterozygosity_p_value: np.float64,
    is_significant: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_lik_hom": log_lik_hom,
            "log_lik_hetero": log_lik_hetero,
            "n_par_hom": n_par_hom,
            "n_par_hetero": n_par_hetero,
            "statistic": test_statistic,
            "df": df,
            "p_value": heterozygosity_p_value,
            "is_significant": is_significant,
        },
        index=[0],
    )


def summarize_parameter_estimates(
    het_params: HeterozygousParameters,
    hom_params: HomozygousParameters,
) -> pd.DataFrame:
    # Gather results
    result_data = [
        {
            "haplotype": "h1",
            "mean": het_params.mean_h1,
            "mean_lower": het_params.mean_h1_ci_low,
            "mean_upper": het_params.mean_h1_ci_high,
            "unit_var": het_params.unit_var,
            "idx": list(range(len(het_params.mean_h1))),
        },
        {
            "haplotype": "h2",
            "mean": het_params.mean_h2,
            "mean_lower": het_params.mean_h2_ci_low,
            "mean_upper": het_params.mean_h2_ci_high,
            "unit_var": het_params.unit_var,
            "idx": list(range(len(het_params.mean_h2))),
        },
        {
            "haplotype": "hom",
            "mean": hom_params.mean,
            "mean_lower": hom_params.mean_ci_low,
            "mean_upper": hom_params.mean_ci_high,
            "unit_var": hom_params.unit_var,
            "idx": list(range(len(hom_params.mean))),
        },
    ]

    return pd.concat([pd.DataFrame(data) for data in result_data])
