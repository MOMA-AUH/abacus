from itertools import product
from typing import List, Tuple

import Levenshtein as ls
import numpy as np
import pandas as pd
from scipy.stats import chi2, multivariate_normal
from sklearn.cluster import DBSCAN

from abacus.config import config
from abacus.graph import GroupedReadCall, ReadCall


def discrete_multivariate_normal_logpdf(x, mean, var):
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
def filter_read_calls(read_calls: List[ReadCall]) -> Tuple[List[GroupedReadCall], List[ReadCall]]:
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
        [[ls.distance(str1, str2) / np.maximum(len(str1), len(str2)) for str2 in read_str_sequences] for str1 in read_str_sequences]
    )
    string_clustering = DBSCAN(eps=0.33, min_samples=3, metric="precomputed").fit(pairwise_sequence_dist)
    # string_outlier_mask = string_clustering.labels_ == -1
    string_outlier_mask = string_clustering.labels_ == 100

    # Kmer count outliers
    kmer_counts = np.array([r.satellite_count for r in read_calls])
    kmer_pct_dist = np.array([[np.max([np.abs(k1 - k2) / (np.maximum(k1, k2) + 1)]) for k2 in kmer_counts] for k1 in kmer_counts])
    kmer_clustering = DBSCAN(eps=0.33, min_samples=3, metric="precomputed").fit(kmer_pct_dist)
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


def group_read_calls(read_calls: List[ReadCall], kmer_dim: int) -> Tuple[pd.DataFrame, pd.DataFrame, List[GroupedReadCall], List[int], List[int]]:
    if not read_calls:
        # Create a list to hold results

        haplotyping_res_df = pd.DataFrame(
            {
                "em_haplotype": "none",
                "mean": pd.NA,
                "sd": pd.NA,
                "median": pd.NA,
                "iqr": pd.NA,
                "n": pd.NA,
                "idx": list(range(kmer_dim)),
            }
        )

        summary_res_df = pd.DataFrame(
            {
                "em_haplotype": "overall",
                "log_lik_homo": pd.NA,
                "log_lik_hetero": pd.NA,
                "n_par_hetero": pd.NA,
                "n_par_homo": pd.NA,
                "df": pd.NA,
                "statistic": pd.NA,
                "p_value": pd.NA,
                "is_significant": pd.NA,
            },
            index=[0],
        )
        return haplotyping_res_df, summary_res_df, [], [], []

    counts = np.array([r.satellite_count for r in read_calls])

    # Initialize parameters
    # Use median for initial values of mean and variances for robustness
    mean_h1 = np.percentile(counts, 25, axis=0) - 1
    mean_h2 = np.percentile(counts, 75, axis=0) + 1  # Add 1 to make sure mean_h1 != mean_h2

    # Use population variance
    var_h1 = var_h2 = np.var(counts, axis=0)

    pi = 0.51

    # EM algorithm
    for _ in range(500):
        # E-step

        # Calcualte probabilities for each haplotype for each read
        logpdf_h1 = np.log(pi) + discrete_multivariate_normal_logpdf(counts, mean_h1, var_h1)
        logpdf_h2 = np.log(1 - pi) + discrete_multivariate_normal_logpdf(counts, mean_h2, var_h2)

        total_logpdf = np.logaddexp(logpdf_h1, logpdf_h2)

        gamma_h1 = np.exp(logpdf_h1 - total_logpdf)
        gamma_h2 = np.exp(logpdf_h2 - total_logpdf)

        # M-step

        def weighted_median(x, w):
            res = np.zeros(x.shape[1])
            for i in range(x.shape[1]):
                idx = np.argsort(x[:, i])
                cs = np.cumsum(w[idx])
                cs /= cs[-1]
                res[i] = x[idx, i][np.searchsorted(cs, 0.5)]
            return res

        # Estimate parameters
        # Use average of weighted mean and weighted median for robustness
        new_mean_h1 = np.average(counts, axis=0, weights=gamma_h1) * 0.5 + weighted_median(counts, gamma_h1) * 0.5
        new_mean_h2 = np.average(counts, axis=0, weights=gamma_h2) * 0.5 + weighted_median(counts, gamma_h2) * 0.5

        # Use average of weights as the new pi
        new_pi = np.average(gamma_h1)

        # Calculate the variance of a unit and extrapolate to the variance of the haplotypes i.e. N ~ N(m, m*v), m = unit count, v = unit variance
        unit_var = new_pi * np.average(((counts - new_mean_h1) ** 2) / (new_mean_h1 + 1e-5), axis=0, weights=gamma_h1) + (1 - new_pi) * np.average(
            ((counts - new_mean_h2) ** 2) / (new_mean_h2 + 1e-5), axis=0, weights=gamma_h2
        )
        new_var_h1 = unit_var * new_mean_h1
        new_var_h2 = unit_var * new_mean_h2

        if (
            any(new_var_h2 < 0)
            or any(new_var_h1 < 0)
            or any(np.isnan(new_mean_h1))
            or any(np.isnan(new_mean_h2))
            or any(np.isnan(new_var_h1))
            or any(np.isnan(new_var_h2))
        ):
            exit()

        # Check for convergence
        has_converged = (
            np.allclose(mean_h1, new_mean_h1)
            and np.allclose(mean_h2, new_mean_h2)
            and np.allclose(var_h1, new_var_h1)
            and np.allclose(var_h2, new_var_h2)
            and np.allclose(pi, new_pi)
        )

        # Print parameters
        print("\n\nIteration: ", _)
        print("mean_h1: ", new_mean_h1)
        print("mean_h2: ", new_mean_h2)
        print("var_h1: ", new_var_h1)
        print("var_h2: ", new_var_h2)
        print("pi: ", new_pi)
        print("Log pdf h1: ", logpdf_h1)
        print("Log pdf h2: ", logpdf_h2)
        print("Gamma_h1: ", gamma_h1)
        print("Gamma_h2: ", gamma_h2)

        # Update parameters
        mean_h1 = new_mean_h1
        mean_h2 = new_mean_h2
        var_h1 = new_var_h1
        var_h2 = new_var_h2
        pi = new_pi

        # Break if converged
        if has_converged:
            break

    # Estimate parameters for homozygous model
    homo_mean = np.average(counts, axis=0)
    homo_var = np.average((counts - homo_mean) ** 2, axis=0)

    # Test for heterozygosity
    # Calculate log likelihood for both models
    # Heterozygote model
    het_mean_h1 = np.average(counts, axis=0, weights=gamma_h1)
    het_mean_h2 = np.average(counts, axis=0, weights=gamma_h2)
    het_var_h1 = np.average((counts - het_mean_h1) ** 2, axis=0, weights=gamma_h1)
    het_var_h2 = np.average((counts - het_mean_h2) ** 2, axis=0, weights=gamma_h2)
    logpdf_h1 = np.log(pi) + discrete_multivariate_normal_logpdf(counts, het_mean_h1, het_var_h1)
    logpdf_h2 = np.log(1 - pi) + discrete_multivariate_normal_logpdf(counts, het_mean_h2, het_var_h2)
    total_logpdf = np.logaddexp(logpdf_h1, logpdf_h2)
    log_lik_hetero = np.sum(total_logpdf)

    # Homozygote model
    log_lik_homo = np.sum(discrete_multivariate_normal_logpdf(counts, homo_mean, homo_var))

    # The test statistic
    test_statistic = -2 * (log_lik_homo - log_lik_hetero)

    # Degrees of freedom is the difference in the number of parameters between the two models
    n_par_hetero = len(mean_h1) + len(var_h2) + len(var_h1) + len(var_h2) + 1
    n_par_homo = len(homo_mean) + len(homo_var)
    df = n_par_hetero - n_par_homo

    # Calculate the p-value using the chi-square distribution
    p_value = 1 - chi2.cdf(test_statistic, df)
    is_significant = p_value < 0.05

    # Summarize the results
    summary_res_df = pd.DataFrame(
        {
            "log_lik_homo": log_lik_homo,
            "log_lik_hetero": log_lik_hetero,
            "n_par_hetero": n_par_hetero,
            "n_par_homo": n_par_homo,
            "statistic": test_statistic,
            "df": df,
            "p_value": p_value,
            "is_significant": is_significant,
        },
        index=[0],
    )

    # Decide EM labels
    if is_significant:
        pi_arr = np.array((gamma_h1, gamma_h2))
        em_label_idx = np.argmax(pi_arr, axis=0)
        em_labels = np.array(("h1", "h2"))[em_label_idx]
    else:
        em_labels = np.full(len(read_calls), "h1")

    # Update original data with EM grouping
    grouped_read_calls = [
        GroupedReadCall.from_read_call(read_call=read, group=em_label, outlier_reason="") for read, em_label in zip(read_calls, em_labels)
    ]

    # Calculate summaries for final groups
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

            # Create a list to hold results
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

    haplotyping_res_df = pd.concat(result_df_list)

    return (
        haplotyping_res_df,
        summary_res_df,
        grouped_read_calls,
        [int(x) for x in mean_h1],
        [int(x) for x in mean_h2],
    )
