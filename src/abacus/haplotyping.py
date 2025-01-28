from itertools import product

import Levenshtein
import numpy as np
import pandas as pd
from scipy.stats import chi2, multivariate_normal, norm
from sklearn.cluster import DBSCAN

from abacus.config import config
from abacus.graph import AlignmentType, GroupedReadCall, ReadCall


def discrete_multivariate_normal_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Set var to minimum 1e-6
    var = np.maximum(var, 1e-6)
    sd = np.sqrt(var)

    # Initialize logpdf with -Inf
    logpdf = np.zeros(x.shape[0])

    # Loop through dimensions and calculate logpdf
    for i in range(x.shape[1]):
        # Extract mean and sd for the current dimension
        m = mean[i]
        s = sd[i]

        # Calculate logpdf
        logcdf_lower = norm.logcdf(x=x[:, i] - 0.5, loc=m, scale=s)
        logcdf_higher = norm.logcdf(x=x[:, i] + 0.5, loc=m, scale=s)

        # Get difference between cdf values using exp-sum trick
        print(x[:, i])
        print(m)
        print(s)
        logpdf_i = logcdf_higher + np.log(1 - np.exp(logcdf_lower - logcdf_higher))

        # Add to logpdf
        logpdf += logpdf_i

    return logpdf


# TOOO: Remove if this does not work
def discrete_multivariate_normal_logpdf_old(x, mean, var):
    # Set var to 0.04 if it is less than 0.04
    var = np.maximum(var, 0.04)

    # Create a grid around the mean
    ranges = []
    for m, v in zip(mean, var):
        sd = np.sqrt(v)
        start = int(m - 10 * sd)
        end = int(m + 10 * sd) + 1
        ranges.append(range(start, end))

    grid = [list(c) for c in product(*ranges)]

    # Calculate logpdf for x and points in the grid
    covariance = var * np.eye(len(mean))
    logpdf_x = multivariate_normal.logpdf(x, mean, covariance)
    logpdf_grid = multivariate_normal.logpdf(grid, mean, covariance)

    # Normalize with exp-sum trick
    sum_logpdf_grid = np.logaddexp.reduce(logpdf_grid)

    return logpdf_x - sum_logpdf_grid


# TODO: Go through filters and make sure they are working as expected - also for flanking reads
def filter_read_calls(read_calls: list[ReadCall]) -> tuple[list[GroupedReadCall], list[ReadCall]]:
    if not read_calls:
        return [], []

    read_calls = list(read_calls)

    # Check STR match ratios
    str_match_ratios = np.array([rc.alignment.str_match_ratio for rc in read_calls])
    read_error_mask = str_match_ratios < 0.80

    # Check STR quality
    str_qualities = np.array([rc.alignment.str_median_quality for rc in read_calls])
    has_left_anchor = np.array([rc.alignment.has_left_anchor for rc in read_calls])
    has_right_anchor = np.array([rc.alignment.has_right_anchor for rc in read_calls])
    str_quality_mask = np.logical_and.reduce([str_qualities < config.MIN_STR_READ_QUAL, has_left_anchor, has_right_anchor])

    # String outliers with pairwise Levenshtein distances
    read_str_sequences = [r.alignment.str_sequence for r in read_calls]
    pairwise_sequence_dist = np.array(
        [
            [Levenshtein.distance(str1, str2) / np.maximum(len(str1), len(str2)) if str1 and str2 else 1.0 for str2 in read_str_sequences]
            for str1 in read_str_sequences
        ],
    )
    string_clustering = DBSCAN(eps=0.33, min_samples=3, metric="precomputed").fit(pairwise_sequence_dist)
    # TODO: Check if this is necessary - disabled for now
    # string_outlier_mask = string_clustering.labels_ == -1
    string_outlier_mask = string_clustering.labels_ == 100

    # Kmer count outliers
    kmer_counts = np.array([r.satellite_count for r in read_calls])
    kmer_pct_dist = np.array([[np.max([np.abs(k1 - k2) / (np.maximum(k1, k2) + 1)]) for k2 in kmer_counts] for k1 in kmer_counts])
    kmer_clustering = DBSCAN(eps=0.33, min_samples=3, metric="precomputed").fit(kmer_pct_dist)
    # TODO: Check if this is necessary - disabled for now
    # kmer_outlier_mask = kmer_clustering.labels_ == -1
    kmer_outlier_mask = kmer_clustering.labels_ == 100

    # Identify and annotate outliers
    outlier_read_calls = []
    good_read_calls = []
    for i, read in enumerate(read_calls):
        # Annotate outliers
        outlier_reasons = []
        if read_error_mask[i]:
            outlier_reasons.append("read_error")
        if str_quality_mask[i]:
            outlier_reasons.append("low_quality_str_region")
        if string_outlier_mask[i]:
            outlier_reasons.append("string_outlier")
        if kmer_outlier_mask[i]:
            outlier_reasons.append("kmer_outlier")

        # Add to outlier list
        outlier_reason_str = ", ".join(outlier_reasons)
        if outlier_reason_str:
            outlier_read_calls.append(GroupedReadCall.from_read_call(read_call=read, group="outlier", outlier_reason=outlier_reason_str))
            continue

        # If not an outlier, add to good list
        good_read_calls.append(read)

    return outlier_read_calls, good_read_calls


def group_read_calls(read_calls: list[ReadCall], kmer_dim: int) -> tuple[pd.DataFrame, pd.DataFrame, list[GroupedReadCall], list[int], list[int]]:
    if not read_calls:
        return create_empty_results(kmer_dim)

    # Split read calls into flanking and spanning reads
    spanning_reads = [r for r in read_calls if r.alignment.alignment_type == AlignmentType.SPANNING]
    flanking_reads = [r for r in read_calls if r.alignment.alignment_type != AlignmentType.SPANNING]

    # Get counts from reads
    spanning_counts = get_counts(spanning_reads)
    flanking_counts = get_counts(flanking_reads)

    het_mean_h1, het_mean_h2, var_h1, var_h2, pi = em_algorithm(spanning_counts, flanking_counts)

    hom_mean, hom_var = estimate_homzygous_parameters(spanning_counts)

    log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, df, p_value, is_significant = test_heterozygosity(
        spanning_counts,
        het_mean_h1,
        het_mean_h2,
        var_h1,
        var_h2,
        pi,
        hom_mean,
        hom_var,
    )

    summary_res_df = summarize_em_results(
        log_lik_hetero=log_lik_hetero,
        log_lik_hom=log_lik_hom,
        n_par_hetero=n_par_hetero,
        n_par_hom=n_par_hom,
        pi=pi,
        het_mean_h1=het_mean_h1.tolist(),
        het_sd_h1=np.sqrt(var_h1).tolist(),
        het_mean_h2=het_mean_h2.tolist(),
        het_sd_h2=np.sqrt(var_h2).tolist(),
        hom_mean=hom_mean.tolist(),
        hom_sd=np.sqrt(hom_var).tolist(),
        test_statistic=test_statistic,
        df=df,
        p_value=p_value,
        is_significant=is_significant,
    )

    grouping_prob_threshold = 0.1
    labels = assign_labels(spanning_counts, het_mean_h1, het_mean_h2, var_h1, var_h2, pi, grouping_prob_threshold, is_significant)

    grouped_read_calls = add_labels_to_read_calls(read_calls, labels)

    if flanking_reads:
        flanking_labels = assign_labels(flanking_counts, het_mean_h1, het_mean_h2, var_h1, var_h2, pi, grouping_prob_threshold, is_significant)
        grouped_read_calls.extend(add_labels_to_read_calls(flanking_reads, flanking_labels))

    haplotyping_res_df = calculate_final_group_summaries(spanning_counts, labels, kmer_dim)

    return haplotyping_res_df, summary_res_df, grouped_read_calls, [int(x) for x in het_mean_h1], [int(x) for x in het_mean_h2]


def create_empty_results(kmer_dim: int) -> tuple[pd.DataFrame, pd.DataFrame, list[GroupedReadCall], list[int], list[int]]:
    haplotyping_res_df = pd.DataFrame(
        {
            "em_haplotype": "none",
            "mean": pd.NA,
            "sd": pd.NA,
            "median": pd.NA,
            "iqr": pd.NA,
            "n": pd.NA,
            "idx": list(range(kmer_dim)),
        },
    )

    summary_res_df = pd.DataFrame(
        {
            "em_haplotype": "overall",
            "log_lik_hom": pd.NA,
            "log_lik_hetero": pd.NA,
            "n_par_hetero": pd.NA,
            "n_par_hom": pd.NA,
            "df": pd.NA,
            "statistic": pd.NA,
            "p_value": pd.NA,
            "is_significant": pd.NA,
        },
        index=[0],
    )
    return haplotyping_res_df, summary_res_df, [], [], []


def get_counts(read_calls: list[ReadCall]) -> np.ndarray:
    return np.array([r.satellite_count for r in read_calls])


def initialize_parameters(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    mean_h1 = np.percentile(counts, 25, axis=0) - 1
    mean_h2 = np.percentile(counts, 75, axis=0) + 1
    var_h1 = var_h2 = np.var(counts, axis=0) + 1
    pi = 0.51
    return mean_h1, mean_h2, var_h1, var_h2, pi


def weighted_median(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    res = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        idx = np.argsort(x[:, i])
        cs = np.cumsum(w[idx])
        cs /= cs[-1]
        res[i] = x[idx, i][np.searchsorted(cs, 0.5)]
    return res


def em_algorithm(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    # Get initial parameters from spanning reads only
    mean_h1, mean_h2, var_h1, var_h2, pi = initialize_parameters(spanning_counts)

    # Run EM algorithm
    for _ in range(500):
        # Get grouping probabilities
        logpdf_h1 = np.log(pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h1, var_h1)
        logpdf_h2 = np.log(1 - pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h2, var_h2)

        total_logpdf = np.logaddexp(logpdf_h1, logpdf_h2)

        gamma_h1 = np.exp(logpdf_h1 - total_logpdf)
        gamma_h2 = np.exp(logpdf_h2 - total_logpdf)

        # Estimate new parameters
        # Mean
        # TODO: Check if this is correct
        # new_mean_h1 = np.average(counts, axis=0, weights=gamma_h1) * 0.5 + weighted_median(counts, gamma_h1) * 0.5
        # new_mean_h2 = np.average(counts, axis=0, weights=gamma_h2) * 0.5 + weighted_median(counts, gamma_h2) * 0.5
        new_mean_h1 = np.average(spanning_counts, axis=0, weights=gamma_h1)
        new_mean_h2 = np.average(spanning_counts, axis=0, weights=gamma_h2)

        # pi
        new_pi = np.average(gamma_h1)

        # Variance
        unit_var_h1 = np.average(((spanning_counts - new_mean_h1) ** 2) / (new_mean_h1 + 1e-5), axis=0, weights=gamma_h1)
        unit_var_h2 = np.average(((spanning_counts - new_mean_h2) ** 2) / (new_mean_h2 + 1e-5), axis=0, weights=gamma_h2)

        unit_var = new_pi * unit_var_h1 + (1 - new_pi) * unit_var_h2

        new_var_h1 = unit_var * new_mean_h1
        new_var_h2 = unit_var * new_mean_h2

        if check_convergence(mean_h1, new_mean_h1, mean_h2, new_mean_h2, var_h1, new_var_h1, var_h2, new_var_h2, pi, new_pi):
            break

        # Update parameters
        mean_h1, mean_h2, var_h1, var_h2, pi = new_mean_h1, new_mean_h2, new_var_h1, new_var_h2, new_pi

    return mean_h1, mean_h2, var_h1, var_h2, pi


def check_convergence(
    mean_h1: np.ndarray,
    new_mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    new_mean_h2: np.ndarray,
    var_h1: np.ndarray,
    new_var_h1: np.ndarray,
    var_h2: np.ndarray,
    new_var_h2: np.ndarray,
    pi: float,
    new_pi: float,
) -> bool:
    return (
        np.allclose(mean_h1, new_mean_h1)
        and np.allclose(mean_h2, new_mean_h2)
        and np.allclose(var_h1, new_var_h1)
        and np.allclose(var_h2, new_var_h2)
        and np.allclose(pi, new_pi)
    )


def estimate_homzygous_parameters(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom_mean = np.average(counts, axis=0)
    hom_var = np.average((counts - hom_mean) ** 2, axis=0)
    return hom_mean, hom_var


def test_heterozygosity(
    counts: np.ndarray,
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    var_h1: np.ndarray,
    var_h2: np.ndarray,
    pi: float,
    hom_mean: np.ndarray,
    hom_var: np.ndarray,
) -> tuple[float, float, int, int, float, int, float, bool]:
    logpdf_h1 = np.log(pi) + discrete_multivariate_normal_logpdf(counts, mean_h1, var_h1)
    logpdf_h2 = np.log(1 - pi) + discrete_multivariate_normal_logpdf(counts, mean_h2, var_h2)
    total_logpdf = np.logaddexp(logpdf_h1, logpdf_h2)
    log_lik_hetero = float(np.sum(total_logpdf))

    log_lik_hom = float(np.sum(discrete_multivariate_normal_logpdf(counts, hom_mean, hom_var)))

    test_statistic = -2 * (log_lik_hom - log_lik_hetero)
    n_par_hetero = len(mean_h1) + len(var_h2) + len(var_h1) + len(var_h2) + 1
    n_par_hom = len(hom_mean) + len(hom_var)
    deg_freedom = n_par_hetero - n_par_hom

    p_value = 1 - float(chi2.cdf(test_statistic, deg_freedom))
    is_significant = p_value < 0.05

    return log_lik_hom, log_lik_hetero, n_par_hetero, n_par_hom, test_statistic, deg_freedom, p_value, is_significant


def summarize_em_results(
    log_lik_hom: float,
    log_lik_hetero: float,
    n_par_hom: int,
    n_par_hetero: int,
    pi: float,
    het_mean_h1: list[float],
    het_sd_h1: list[float],
    het_mean_h2: list[float],
    het_sd_h2: list[float],
    hom_mean: list[float],
    hom_sd: list[float],
    test_statistic: float,
    df: int,
    p_value: float,
    is_significant: bool,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "log_lik_hom": log_lik_hom,
            "log_lik_hetero": log_lik_hetero,
            "pi": pi,
            "het_mean_h1": ";".join([str(x) for x in het_mean_h1]),
            "het_sd_h1": ";".join([str(x) for x in het_sd_h1]),
            "het_mean_h2": ";".join([str(x) for x in het_mean_h2]),
            "het_sd_h2": ";".join([str(x) for x in het_sd_h2]),
            "hom_mean": ";".join([str(x) for x in hom_mean]),
            "hom_sd": ";".join([str(x) for x in hom_sd]),
            "n_par_hom": n_par_hom,
            "n_par_hetero": n_par_hetero,
            "statistic": test_statistic,
            "df": df,
            "p_value": p_value,
            "is_significant": is_significant,
        },
        index=[0],
    )


def assign_labels(
    counts: np.ndarray,
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    var_h1: np.ndarray,
    var_h2: np.ndarray,
    pi: float,
    grouping_prob_threshold: float,
    is_significant: bool,
) -> np.ndarray:
    if is_significant:
        logpdf_h1 = np.log(pi) + discrete_multivariate_normal_logpdf(counts, mean_h1, var_h1)
        logpdf_h2 = np.log(1 - pi) + discrete_multivariate_normal_logpdf(counts, mean_h2, var_h2)
        total_logpdf = np.logaddexp(logpdf_h1, logpdf_h2)
        gamma_h1 = np.exp(logpdf_h1 - total_logpdf)
        gamma_h2 = np.exp(logpdf_h2 - total_logpdf)
        pi_arr = np.array((gamma_h1, gamma_h2, np.full(len(counts), grouping_prob_threshold)))
        em_label_idx = np.argmax(pi_arr, axis=0)
        em_labels = np.array(("h1", "h2", "undeterminded"))[em_label_idx]
    else:
        em_labels = np.full(len(counts), "h1")
    return em_labels


def add_labels_to_read_calls(read_calls: list[ReadCall], labels: np.ndarray) -> list[GroupedReadCall]:
    return [GroupedReadCall.from_read_call(read_call=read, group=em_label, outlier_reason="") for read, em_label in zip(read_calls, labels)]


def calculate_final_group_summaries(counts: np.ndarray, em_labels: np.ndarray, kmer_dim: int) -> pd.DataFrame:
    result_df_list = []
    for h in ["h1", "h2"]:
        h_idx = em_labels == h
        good_data_h = counts[h_idx,]

        if len(good_data_h) > 0:
            mean_h = np.mean(good_data_h, axis=0)
            sd_h = np.std(good_data_h, axis=0)
            median_h = np.median(good_data_h, axis=0)
            q1_h = np.percentile(good_data_h, 25, axis=0)
            q3_h = np.percentile(good_data_h, 75, axis=0)
            iqr_h = q3_h - q1_h

            result_dict = {
                "em_haplotype": h,
                "mean": mean_h,
                "sd": sd_h,
                "median": median_h,
                "iqr": iqr_h,
                "n": len(good_data_h),
                "idx": list(range(kmer_dim)),
            }

            result_df_list.append(pd.DataFrame(result_dict))

    return pd.concat(result_df_list)
