from itertools import compress, product
from typing import List, Tuple

import Levenshtein as ls
import numpy as np
import pandas as pd
from scipy.stats import chi2, multivariate_normal
from sklearn.cluster import DBSCAN

from abacus.graph import Read_Call


def discrete_multivariate_normal_pdf(x, mean, var):
    ranges = []
    for m, v in zip(mean, var):
        sd = np.sqrt(v)
        start = int(m - 10 * sd)
        end = int(m + 10 * sd) + 1
        ranges.append(range(start, end))

    grid = [list(c) for c in product(*ranges)]

    covariance = np.maximum(var, 0.04) * np.eye(len(mean))
    pdf_x = multivariate_normal.pdf(x, mean, covariance)
    pdf_grid = multivariate_normal.pdf(grid, mean, covariance)

    return pdf_x / np.sum(pdf_grid)


def call_haplotypes(read_calls: List[Read_Call]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_satellites = len(read_calls[0].locus.satellites)

    # DBSCAN for initial outlier detection
    good_read_calls = read_calls
    outlier_read_calls = []

    # String outliers with pairwise Levenshtein distances
    read_str_sequences = [r.alignment.str_sequence for r in good_read_calls]
    pairwise_sequence_dist = np.array(
        [[ls.distance(str1, str2) / max(len(str1), len(str2)) for str2 in read_str_sequences] for str1 in read_str_sequences]
    )
    string_clustering = DBSCAN(eps=0.05, min_samples=2, metric="precomputed").fit(pairwise_sequence_dist)
    string_outlier_mask = string_clustering.labels_ == -1

    # Identify and annotate outliers
    outlier_read_calls.extend(
        [read.to_dict() | {"em_haplotype": "outlier", "outlier_reason": "sequence_errors"} for read in list(compress(good_read_calls, string_outlier_mask))]
    )
    good_read_calls = list(compress(good_read_calls, ~string_outlier_mask))

    # Kmer count outliers
    kmer_counts = np.array([r.kmer_count for r in good_read_calls])
    kmer_dist = np.array([[np.sum(np.abs(k1 - k2)) for k2 in kmer_counts] for k1 in kmer_counts])
    kmer_clustering = DBSCAN(eps=n_satellites + 1, min_samples=2, metric="precomputed").fit(kmer_dist)
    kmer_outlier_mask = kmer_clustering.labels_ == -1

    # Identify and annotate outliers
    outlier_read_calls.extend(
        [read.to_dict() | {"em_haplotype": "outlier", "outlier_reason": "unusual_kmer_count"} for read in list(compress(good_read_calls, kmer_outlier_mask))]
    )
    good_read_calls = list(compress(good_read_calls, ~kmer_outlier_mask))
    good_read_calls = [read.to_dict() | {"em_haplotype": pd.NA, "outlier_reason": pd.NA} for read in good_read_calls]

    # TODO: good_read_calls are updated in the function - make this explicit!
    haplotyping_df, test_summary_df = run_em_algo(good_read_calls, n_satellites)

    # Merge data frames
    read_calls_df = pd.concat([pd.DataFrame(good_read_calls), pd.DataFrame(outlier_read_calls)])

    return haplotyping_df, read_calls_df, test_summary_df


def run_em_algo(good_read_calls: List, kmer_dim: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not good_read_calls:
        # Create a list to hold results

        haplotyping_res_df = pd.DataFrame(
            {
                "em_haplotype": "none",
                "mean": pd.NA,
                "sd": pd.NA,
                "median": pd.NA,
                "iqr": pd.NA,
                "n": pd.NA,
                "idx": [i for i in range(kmer_dim)],
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
        return haplotyping_res_df, summary_res_df

    good_data = np.array([r["kmer_count"] for r in good_read_calls])

    # Use median for initial values of mean and variances for robustness
    mean_h1 = np.percentile(good_data, 25, axis=0) - 1
    mean_h2 = np.percentile(good_data, 75, axis=0) + 1

    var_h1 = var_h2 = np.var(good_data, axis=0)

    # Initialize parameters
    pi = 0.51

    gamma_h1 = np.full(good_data.shape[0], pi)
    gamma_h2 = np.full(good_data.shape[0], (1 - pi))

    # EM algorithm
    for _ in range(500):
        # E-step

        # Calcualte probabilities for each haplotype for each read
        pdf_h1 = pi * discrete_multivariate_normal_pdf(good_data, mean_h1, var_h1)
        pdf_h2 = (1 - pi) * discrete_multivariate_normal_pdf(good_data, mean_h2, var_h2)

        total_pdf = pdf_h1 + pdf_h2

        gamma_h1 = pdf_h1 / total_pdf
        gamma_h2 = pdf_h2 / total_pdf

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
        new_mean_h1 = weighted_median(good_data, gamma_h1)
        new_mean_h2 = weighted_median(good_data, gamma_h2)
        new_var_h1 = np.average((good_data - mean_h1) ** 2, axis=0, weights=gamma_h1)
        new_var_h2 = np.average((good_data - mean_h2) ** 2, axis=0, weights=gamma_h2)

        new_pi = np.average(gamma_h1)

        # Check for convergence
        has_converged = (
            np.allclose(mean_h1, new_mean_h1)
            and np.allclose(mean_h2, new_mean_h2)
            and np.allclose(var_h1, new_var_h1)
            and np.allclose(var_h2, new_var_h2)
            and np.allclose(pi, new_pi)
        )

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
    homo_mean = np.median(good_data, axis=0)
    homo_var = np.average((good_data - homo_mean) ** 2, axis=0)

    # Test for heterozygosity
    # Calculate log likelihood for both models
    log_lik_hetero = np.sum(
        np.log(
            pi * discrete_multivariate_normal_pdf(good_data, mean_h1, var_h1)
            + (1 - pi) * discrete_multivariate_normal_pdf(good_data, mean_h2, var_h2)
        )
    )
    log_lik_homo = np.sum(np.log(discrete_multivariate_normal_pdf(good_data, homo_mean, homo_var)))

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
        em_label = np.array(("h1", "h2"))[em_label_idx]
    else:
        em_label = np.full(len(good_read_calls), "h1")

    # Update original data with EM grouping
    for i, read in enumerate(good_read_calls):
        read.update({"em_haplotype": em_label[i]})

    # Calculate summaries for final groups
    result_df_list = []
    for h in ["h1", "h2"]:
        h_idx = em_label == h
        good_data_h = good_data[h_idx,]

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
                "idx": [i for i in range(kmer_dim)],
            }

            result_df_list.append(pd.DataFrame(result_dict))

    haplotyping_res_df = pd.concat(result_df_list)

    return haplotyping_res_df, summary_res_df
