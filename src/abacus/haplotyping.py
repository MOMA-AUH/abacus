from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import binom, chi2

from abacus.config import config
from abacus.graph import ReadCall
from abacus.parameter_estimation import (
    HeterozygousParameters,
    HomozygousParameters,
    calculate_grouping_probabilities_flanking,
    calculate_grouping_probabilities_spanning,
    calculate_log_likelihood_heterozygous,
    calculate_log_likelihood_homozygous,
    estimate_confidence_intervals_heterozygous,
    estimate_confidence_intervals_homozygous,
    estimate_heterozygous_parameters,
    estimate_homozygous_parameters,
    unpack_read_calls,
)
from abacus.utils import AlignmentType


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

    # TODO: This should be a parameter in config
    haplotyping_reruns = 2
    all_outlier_read_calls: list[ReadCall] = []
    for _ in range(haplotyping_reruns):
        # Assign labels to reads
        if ploidy == 1:
            # Assign labels to reads
            grouped_read_calls = [read.set_haplotype("hom") for read in read_calls]
        else:
            # Assign labels to reads
            het_params = estimate_heterozygous_parameters(read_calls)
            grouped_read_calls = add_heterozygote_labels(
                read_calls,
                het_params,
            )

        # Remove outliers
        grouped_read_calls, outlier_read_calls = remove_haplotype_outliers(grouped_read_calls)

        # If no outliers,
        if not outlier_read_calls:
            break

        # Add outlier reads to the list
        all_outlier_read_calls.extend(outlier_read_calls)

        # Update read calls for next iteration
        read_calls = [read for read in read_calls if read not in outlier_read_calls]

    # Estimate model parameters and confidence intervals for heterozygous means
    hom_params = estimate_homozygous_parameters(read_calls)

    if ploidy == 1:
        het_params_nan = HeterozygousParameters(
            mean_h1=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            mean_h2=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            unit_var=np.full_like(hom_params.mean, np.nan, dtype=np.float64),
            pi=np.float64(np.nan),
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
            split_p_value=np.float64(1),
        )

        return grouped_read_calls, all_outlier_read_calls, het_params_nan, hom_params, test_summary_df

    # TODO: Pack test results nicely to be passed to the next step
    # Test for heterozygosity
    log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, df, heterozygosity_p_value = test_heterozygosity(
        read_calls,
        het_params,
        hom_params,
    )

    # Test if split ratio is significant
    split_p_value = test_split_ratio(read_calls, het_params)

    # Set haplotype to "hom" if either heterozygosity test is not significant or split ratio is not significant
    heterozygosity_test_significant = bool(heterozygosity_p_value < config.het_alpha)
    split_test_significant = bool(split_p_value < config.split_alpha)
    if not heterozygosity_test_significant or split_test_significant:
        for read in grouped_read_calls:
            read.set_haplotype("hom")

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
        split_p_value=split_p_value,
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
        split_p_value=np.float64(1),
    )

    het_params = HeterozygousParameters(
        mean_h1=np.full(1, np.nan, dtype=np.float64),
        mean_h2=np.full(1, np.nan, dtype=np.float64),
        unit_var=np.full(1, np.nan, dtype=np.float64),
        pi=np.float64(np.nan),
    )

    hom_params = HomozygousParameters(
        mean=np.full(1, np.nan, dtype=np.float64),
        unit_var=np.full(1, np.nan, dtype=np.float64),
    )

    return [], [], het_params, hom_params, summary_res_df


def get_is_left_flanking_bool(flanking_reads: list[ReadCall]) -> list[bool]:
    return [read.alignment.type == AlignmentType.LEFT_FLANKING for read in flanking_reads]


def add_heterozygote_labels(
    read_calls: list[ReadCall],
    het_params: HeterozygousParameters,
) -> list[ReadCall]:
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
        np.float64(0.5),  # Assuming equal probability for both haplotypes
    )

    for i, read in enumerate(spanning_reads):
        # Assign labels based on maximum probability
        read.set_haplotype("h1" if p_group_h1_spanning[i] > p_group_h2_spanning[i] else "h2")

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
        np.float64(0.5),  # Assuming equal probability for both haplotypes
    )

    for i, read in enumerate(flanking_reads):
        # Assign labels based on maximum probability
        read.set_haplotype("h1" if p_group_h1_flanking[i] > p_group_h2_flanking[i] else "h2")

    return spanning_reads + flanking_reads


def remove_haplotype_outliers(grouped_read_calls: list[ReadCall]) -> tuple[list[ReadCall], list[ReadCall]]:
    # If few reads, skip outlier removal
    if len(grouped_read_calls) <= config.min_haplotype_depth:
        return grouped_read_calls, []

    # Initialize lists
    good_read_calls: list[ReadCall] = []
    outlier_read_calls: list[ReadCall] = []

    # Evaluate haplotypes
    # Get unique haplotypes
    haplotypes = {rc.haplotype for rc in grouped_read_calls}
    for haplotype in haplotypes:
        # Get read calls for haplotype
        haplotype_read_calls = [rc for rc in grouped_read_calls if rc.haplotype == haplotype]

        # If haplotype has only one read call, mark as outlier
        if len(haplotype_read_calls) == 1:
            rc = haplotype_read_calls[0]
            rc.add_outlier_reason("caused_singleton_in_haplotyping")
            outlier_read_calls.append(rc)
            continue

        # Split read calls by type
        spanning_read_calls = [rc for rc in haplotype_read_calls if rc.is_spanning()]
        flanking_read_calls = [rc for rc in haplotype_read_calls if not rc.is_spanning()]

        # Flanking read calls are always good
        good_read_calls.extend(flanking_read_calls)

        # Skip if too few read calls
        if len(spanning_read_calls) <= 3:  # Need at least 3 reads for meaningful statistics
            good_read_calls.extend(spanning_read_calls)
            continue

        # Extract counts data
        counts_data = np.array([rc.satellite_count for rc in spanning_read_calls])

        # Calculate distances
        mahalanobis_distances = calculate_mahalanobis_distances(counts_data)
        max_count_differences = calculate_max_count_differences(counts_data)

        # Classify reads using chi-square threshold
        # For n independent variables, Mahalanobis distance follows chi-square with n degrees of freedom
        # Using 97.5th percentile of chi-square distribution with n dimensions
        degrees_of_freedom = counts_data.shape[1]  # number of dimensions
        mahalanobis_cutoff = np.sqrt(chi2.ppf(0.99, degrees_of_freedom))

        # Remove outliers: reads with high Mahalanobis distance AND counts further than 1 from median
        for rc, m_dist, max_count_diff in zip(haplotype_read_calls, mahalanobis_distances, max_count_differences):
            if m_dist > mahalanobis_cutoff and max_count_diff > 1:
                rc.add_outlier_reason("high_cluster_distance")
                outlier_read_calls.append(rc)
            else:
                good_read_calls.append(rc)

    return good_read_calls, outlier_read_calls


def calculate_mahalanobis_distances(counts_data: np.ndarray) -> np.ndarray:
    haplotype_mean = np.mean(counts_data, axis=0)
    haplotype_variance = np.var(counts_data, axis=0)
    haplotype_variance = np.maximum(haplotype_variance, config.min_var)  # Ensure variance is not too small

    # Calculate Mahalanobis distance for each read
    # For independent dimensions, this simplifies to normalized Euclidean distance
    return np.sqrt(
        np.sum(
            ((counts_data - haplotype_mean) ** 2) / (haplotype_variance + 1e-10),  # add small constant to avoid division by zero
            axis=1,
        ),
    )


def calculate_max_count_differences(counts_data: np.ndarray) -> np.ndarray:
    mean = np.median(counts_data, axis=0)

    return np.max(np.abs(counts_data - mean), axis=1)


def test_split_ratio(
    grouped_reads: list[ReadCall],
    het_params: HeterozygousParameters,
) -> np.float64:
    unique_haplotypes = {x.haplotype for x in grouped_reads}

    if len(unique_haplotypes) == 1:
        return np.float64(1.0)
    # Else assume ploidy == 2
    # Split reads into spanning and flanking reads
    spanning_reads = [read for read in grouped_reads if read.is_spanning()]
    flanking_reads = [read for read in grouped_reads if not read.is_spanning()]

    # Calculate grouping probabilities
    spanning_counts = np.array([read.satellite_count for read in spanning_reads])
    p_group_h1_spanning, p_group_h2_spanning = calculate_grouping_probabilities_spanning(
        spanning_counts,
        het_params.mean_h1,
        het_params.mean_h2,
        het_params.unit_var,
        np.float64(0.5),  # Assuming equal probability for both haplotypes
    )

    # Calculate grouping probabilities for flanking reads
    flanking_counts = np.array([read.satellite_count for read in flanking_reads])
    is_left_flanking = get_is_left_flanking_bool(flanking_reads)
    p_group_h1_flanking, p_group_h2_flanking = calculate_grouping_probabilities_flanking(
        flanking_counts,
        is_left_flanking,
        het_params.mean_h1,
        het_params.mean_h2,
        het_params.unit_var,
        np.float64(0.5),  # Assuming equal probability for both haplotypes
    )

    # Get counts for each haplotype
    h1_count = np.int64(np.round(np.sum(p_group_h1_spanning) + np.sum(p_group_h1_flanking)))
    h2_count = np.int64(np.round(np.sum(p_group_h2_spanning) + np.sum(p_group_h2_flanking)))
    total_count = h1_count + h2_count

    # Calculate p-value for binomial test (testing if p=0.5)
    # If any of the counts are zero, return 1.0
    if h1_count == 0 or h2_count == 0:
        return np.float64(1.0)

    # If both groups are sufficiently large, skip the test
    if h1_count >= config.min_haplotype_depth and h2_count >= config.min_haplotype_depth:
        return np.float64(1.0)

    # Use binomial test with p=0.5
    # Get the probability of observing a deviation at least as extreme as observed
    left_p_value = binom.pmf(range(min(h1_count, h2_count) + 1), total_count, 0.5).sum()
    right_p_value = binom.pmf(range(max(h1_count, h2_count), total_count + 1), total_count, 0.5).sum()
    p_value = 2 * min(left_p_value, right_p_value)
    # Make sure p-value is between 0 and 1
    return np.clip(p_value, 0, 1)


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
        par_het.pi,
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
    n_par_hetero = len(par_het.mean_h1) + len(par_het.mean_h2) + len(par_het.unit_var) + 1  # +1 for pi
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
    split_p_value: np.float64,
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
            "split_p_value": split_p_value,
        },
        index=[0],
    )


def summarize_parameter_estimates(
    read_calls: list[ReadCall],
    het_params: HeterozygousParameters,
    hom_params: HomozygousParameters,
) -> pd.DataFrame:
    # If nan in het_params, set confidence intervals to nan
    if np.isnan(het_params.mean_h1).any() or np.isnan(het_params.mean_h2).any():
        het_conf_mean_h1_lower = np.full_like(hom_params.mean, np.nan, dtype=np.float64)
        het_conf_mean_h1_upper = np.full_like(hom_params.mean, np.nan, dtype=np.float64)
        het_conf_mean_h2_lower = np.full_like(hom_params.mean, np.nan, dtype=np.float64)
        het_conf_mean_h2_upper = np.full_like(hom_params.mean, np.nan, dtype=np.float64)
    # Estimate confidence intervals means
    else:
        het_conf_mean_h1_lower, het_conf_mean_h1_upper, het_conf_mean_h2_lower, het_conf_mean_h2_upper = estimate_confidence_intervals_heterozygous(
            read_calls,
            het_params.mean_h1,
            het_params.mean_h2,
            het_params.unit_var,
            het_params.pi,
        )

    # If nan in hom_params, set confidence intervals to nan
    if np.isnan(hom_params.mean).any():
        hom_conf_mean_lower = np.full_like(hom_params.mean, np.nan, dtype=np.float64)
        hom_conf_mean_upper = np.full_like(hom_params.mean, np.nan, dtype=np.float64)
    else:
        # Estimate confidence intervals means
        hom_conf_mean_lower, hom_conf_mean_upper = estimate_confidence_intervals_homozygous(read_calls, hom_params.mean, hom_params.unit_var)

    # Gather results
    result_data = [
        {
            "haplotype": "h1",
            "mean": het_params.mean_h1,
            "mean_lower": het_conf_mean_h1_lower,
            "mean_upper": het_conf_mean_h1_upper,
            "unit_var": het_params.unit_var,
            "pi": het_params.pi,
            "idx": list(range(len(het_params.mean_h1))),
        },
        {
            "haplotype": "h2",
            "mean": het_params.mean_h2,
            "mean_lower": het_conf_mean_h2_lower,
            "mean_upper": het_conf_mean_h2_upper,
            "unit_var": het_params.unit_var,
            "pi": 1 - het_params.pi,
            "idx": list(range(len(het_params.mean_h2))),
        },
        {
            "haplotype": "hom",
            "mean": hom_params.mean,
            "mean_lower": hom_conf_mean_lower,
            "mean_upper": hom_conf_mean_upper,
            "unit_var": hom_params.unit_var,
            "pi": 1,
            "idx": list(range(len(hom_params.mean))),
        },
    ]

    return pd.concat([pd.DataFrame(data) for data in result_data])
