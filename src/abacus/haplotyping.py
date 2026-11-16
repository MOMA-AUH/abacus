from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from abacus.config import config
from abacus.equal_length_split import _make_empty_backup_summary, run_equal_length_backup_test
from abacus.filtering import detect_length_outliers, filter_unsupported_haplotypes
from abacus.graph import ReadCall
from abacus.logging import logger
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
from abacus.timing import timed
from abacus.utils import AlignmentType, Haplotype


def run_haplotyping(
    read_calls: list[ReadCall],
    ploidy: int,
) -> tuple[list[ReadCall], list[ReadCall], HeterozygousParameters, HomozygousParameters, dict[Haplotype, HomozygousParameters], pd.DataFrame]:
    # Handle empty read_calls
    if not read_calls:
        return create_empty_results()

    if ploidy not in (1, 2):
        error_msg = f"Unsupported ploidy: {ploidy}. Only 1 and 2 are supported."
        raise ValueError(error_msg)

    # PHASE 1 + 2: Initial grouping, then iterative outlier detection and re-grouping
    grouped_read_calls, outlier_read_calls, het_params_test = group_and_remove_outliers(read_calls, ploidy)

    if not grouped_read_calls:
        _, _, het_params, hom_params, final_params, summary = create_empty_results()
        return [], outlier_read_calls, het_params, hom_params, final_params, summary

    # PHASE 2b: If more reads were removed than the smaller haplotype holds (e.g. it is a few contaminating reads),
    # regroup without it
    smaller_haplotype_read_calls = get_smaller_haplotype(grouped_read_calls) if ploidy == 2 else []
    if smaller_haplotype_read_calls and len(outlier_read_calls) > len(smaller_haplotype_read_calls):
        smaller_haplotype_names = {rc.alignment.name for rc in smaller_haplotype_read_calls}
        remaining_read_calls = [rc for rc in read_calls if rc.alignment.name not in smaller_haplotype_names]
        regrouped_read_calls, regrouped_outlier_read_calls, regrouped_het_params = group_and_remove_outliers(remaining_read_calls, ploidy)

        # Keep the regrouping if it, with the smaller haplotype's reads as outliers, leaves fewer reads unexplained
        if len(regrouped_outlier_read_calls) + len(smaller_haplotype_read_calls) < len(outlier_read_calls):
            for rc in smaller_haplotype_read_calls:
                rc.add_outlier_reason("caused_unexplained_reads_in_haplotyping")
            grouped_read_calls = regrouped_read_calls
            outlier_read_calls = regrouped_outlier_read_calls + smaller_haplotype_read_calls
            het_params_test = regrouped_het_params

    # PHASE 3: Test for heterozygosity

    # Estimate hom_params once after all outlier removal — only needed for the LRT below
    with timed("haplotyping estimate_homozygous_parameters"):
        hom_params_test = estimate_homozygous_parameters(grouped_read_calls)

    # Too few surviving reads cannot support every chromosome copy independently.
    if ploidy == 1 or len(grouped_read_calls) < config.min_reads_per_haplotype * ploidy:
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
        test_summary_df = pd.concat(
            [test_summary_df.reset_index(drop=True), _make_empty_backup_summary(run=False).reset_index(drop=True)],
            axis=1,
        )
        het_params_test = _make_empty_het_par_estimate(dim=len(hom_params_test.mean))
        # Ensure all haplotypes are included in final_params
        final_params = _estimate_final_parameters(grouped_read_calls)
        return grouped_read_calls, outlier_read_calls, het_params_test, hom_params_test, final_params, test_summary_df

    # Test for heterozygosity
    with timed("haplotyping test_heterozygosity"):
        log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, df, heterozygosity_p_value = test_heterozygosity(
            grouped_read_calls,
            het_params_test,
            hom_params_test,
        )

    # Check if heterozygosity test is significant
    heterozygosity_test_significant = bool(heterozygosity_p_value < config.het_alpha)

    # If not significant -> Haplotypes are not well separated by length -> Run backup test to see if they can be separated by sequence
    if not heterozygosity_test_significant:
        # Start by tagging all reads as Homozygous, then re-tag based on sequence if backup test finds a split
        for read in grouped_read_calls:
            read.set_haplotype(Haplotype.HOM)

        # Run backup sequence test — updates read labels only, no parameter estimation here
        grouped_read_calls, sequence_split_outliers, backup_summary_df = _run_sequence_split_test_and_update_params(
            grouped_read_calls,
        )
        outlier_read_calls.extend(sequence_split_outliers)
    else:
        backup_summary_df = _make_empty_backup_summary(run=False)

    # Phase 4: Final parameter estimation and summarization

    grouped_read_calls, unsupported = filter_unsupported_haplotypes(grouped_read_calls)
    outlier_read_calls.extend(unsupported)

    # Always use homozygous model per group for final estimates, regardless of how grouping was determined
    final_params = _estimate_final_parameters(grouped_read_calls)

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
    # Combine with sequence split backup test summary
    test_summary_df = pd.concat([test_summary_df.reset_index(drop=True), backup_summary_df.reset_index(drop=True)], axis=1)

    return grouped_read_calls, outlier_read_calls, het_params_test, hom_params_test, final_params, test_summary_df


def _make_empty_het_par_estimate(dim: int) -> HeterozygousParameters:
    nan_like = np.full(dim, np.nan, dtype=np.float64)
    return HeterozygousParameters(
        mean_h1=nan_like.copy(),
        mean_h2=nan_like.copy(),
        unit_var=nan_like.copy(),
        mean_h1_ci_low=nan_like.copy(),
        mean_h1_ci_high=nan_like.copy(),
        mean_h2_ci_low=nan_like.copy(),
        mean_h2_ci_high=nan_like.copy(),
    )


def _make_empty_hom_par_estimate(dim: int) -> HomozygousParameters:
    nan_like = np.full(dim, np.nan, dtype=np.float64)
    return HomozygousParameters(
        mean=nan_like.copy(),
        unit_var=nan_like.copy(),
        mean_ci_low=nan_like.copy(),
        mean_ci_high=nan_like.copy(),
    )


def create_empty_results() -> tuple[
    list[ReadCall],
    list[ReadCall],
    HeterozygousParameters,
    HomozygousParameters,
    dict[Haplotype, HomozygousParameters],
    pd.DataFrame,
]:
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
    summary_res_df = pd.concat(
        [summary_res_df.reset_index(drop=True), _make_empty_backup_summary(run=False).reset_index(drop=True)],
        axis=1,
    )

    het_params = _make_empty_het_par_estimate(dim=1)
    hom_params = _make_empty_hom_par_estimate(dim=1)

    final_params = {Haplotype.HOM: hom_params}

    return [], [], het_params, hom_params, final_params, summary_res_df


def group_and_remove_outliers(
    read_calls: list[ReadCall],
    ploidy: int,
) -> tuple[list[ReadCall], list[ReadCall], HeterozygousParameters]:
    # Work on copies, so the minority-group retry can reconsider the original reads.
    remaining = [read.copy() for read in read_calls]
    removed: list[ReadCall] = []
    dim = len(read_calls[0].satellite_count) if read_calls else 1
    het_params = _make_empty_het_par_estimate(dim)

    while remaining:
        fit_ploidy = ploidy if len(remaining) >= config.min_reads_per_haplotype * ploidy else 1
        if fit_ploidy == 2:
            with timed("haplotyping estimate_heterozygous_parameters"):
                het_params = estimate_heterozygous_parameters(remaining)
        else:
            het_params = _make_empty_het_par_estimate(dim)
        grouped = group_read_calls(remaining, het_params, fit_ploidy)
        remaining, unsupported = filter_unsupported_haplotypes(grouped)
        if unsupported:
            removed.extend(unsupported)
            continue

        # Length filtering retains its own per-group sample-size requirement.
        outliers = detect_length_outliers(remaining)
        if not outliers:
            break
        removed.extend(outliers)
        remaining = [read for read in remaining if read not in outliers]

    return remaining, removed, het_params


def get_smaller_haplotype(grouped_read_calls: list[ReadCall]) -> list[ReadCall]:
    h1_read_calls = [rc for rc in grouped_read_calls if rc.haplotype == Haplotype.H1]
    h2_read_calls = [rc for rc in grouped_read_calls if rc.haplotype == Haplotype.H2]

    # Equal-sized haplotypes have no smaller one
    if len(h1_read_calls) == len(h2_read_calls):
        return []

    return min(h1_read_calls, h2_read_calls, key=len)


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


def _run_sequence_split_test_and_update_params(
    grouped_read_calls: list[ReadCall],
) -> tuple[list[ReadCall], list[ReadCall], pd.DataFrame]:
    """Run the equal-length sequence split test and update read haplotype labels if a split is detected.

    Returns (updated_grouped_reads, new_outliers, backup_summary_df).

    Parameter re-estimation is not done here; it is the responsibility of
    _estimate_final_parameters(), which is called after this function returns.
    """
    with timed("haplotyping equal_length_backup_test"):
        grouped_read_calls, backup_outliers, backup_summary_df = run_equal_length_backup_test(
            grouped_read_calls,
            config.equal_length_alpha,
        )

    backup_split_made = backup_summary_df["backup_test_run"].iloc[0] and not backup_summary_df["backup_test_significant"].iloc[0]
    if backup_split_made:
        logger.debug(
            "Equal-length sequence split test: split detected "
            f"(p={backup_summary_df['backup_test_p_value'].iloc[0]:.4g}, not significant — consistent with 50/50): "
            f"kmer_h1={backup_summary_df['backup_test_kmer_h1'].iloc[0]}, "
            f"kmer_h2={backup_summary_df['backup_test_kmer_h2'].iloc[0]}",
        )

    return grouped_read_calls, backup_outliers, backup_summary_df


def _estimate_final_parameters(grouped_read_calls: list[ReadCall]) -> dict[Haplotype, HomozygousParameters]:
    """Fit a homozygous model independently to each H1, H2, and HOM group.

    Return a parameter dictionary containing all three haplotypes, with NaN
    estimates for groups without reads. Each populated group has its own mean,
    unit variance, and confidence intervals.
    """
    # Grouped reads go on to consensus, which only handles H1, H2 and HOM
    invalid_labels = {r.haplotype for r in grouped_read_calls} - {Haplotype.H1, Haplotype.H2, Haplotype.HOM}
    if invalid_labels:
        raise ValueError(f"Locus {grouped_read_calls[0].locus.id}: grouped reads have invalid haplotype labels {sorted(h.value for h in invalid_labels)}")

    res = {}

    # For each haplotype, estimate a homozygous model independently
    for h in [Haplotype.H1, Haplotype.H2, Haplotype.HOM]:
        reads = [r for r in grouped_read_calls if r.haplotype == h]
        if not reads:
            dim = len(grouped_read_calls[0].locus.satellites) if grouped_read_calls else 1
            res[h] = _make_empty_hom_par_estimate(dim=dim)
        else:
            res[h] = estimate_homozygous_parameters(reads)

    return res


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


def summarize_final_parameter_estimates(
    params: dict[Haplotype, HomozygousParameters],
) -> pd.DataFrame:
    # Gather results
    result_data = []
    for h, par in params.items():
        result_data.append(
            {
                "haplotype": h.value,
                "mean": par.mean,
                "mean_lower": par.mean_ci_low,
                "mean_upper": par.mean_ci_high,
                "unit_var": par.unit_var,
                "idx": list(range(len(par.mean))),
            },
        )

    return pd.concat([pd.DataFrame(data) for data in result_data])


def summarize_test_parameter_estimates(
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
