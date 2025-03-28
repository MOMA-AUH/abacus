from __future__ import annotations

import numpy as np
import pandas as pd
from Levenshtein import distance as levenshtein_distance
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, norm

from abacus.config import config
from abacus.graph import AlignmentType, ReadCall


def safe_log(x: np.ndarray | np.float64) -> np.ndarray | np.float64:
    return np.log(np.maximum(x, 1e-100))


def discrete_multivariate_normal_logpdf_old(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Cap sd at minimum
    sd = np.sqrt(mean * unit_var)
    sd = np.maximum(sd, config.min_sd)

    # Initialize logpdf with 0 (log(1))
    logpdf = np.zeros(x.shape[0])

    # Loop through dimensions and calculate logpdf
    for i in range(mean.shape[0]):
        # Extract mean and sd for the current dimension
        x_i = np.round(x[:, i])
        m = mean[i]
        s = sd[i]

        # Get logpdf
        logpdf_i = norm.logpdf(x_i, loc=m, scale=s)

        # Normalize
        lower_bound = np.floor(m - 5 * s) - 1
        higher_bound = np.ceil(m + 5 * s) + 1

        # Bound should be >0
        lower_bound = np.maximum(lower_bound, 0)
        higher_bound = np.maximum(higher_bound, 0)

        # Sum pdf over range
        range_array = np.arange(lower_bound, higher_bound + 1)
        norm_const = np.logaddexp.reduce(norm.logpdf(range_array, loc=m, scale=s), axis=0)

        # Add to logpdf
        logpdf += logpdf_i - norm_const

    return logpdf


def discrete_multivariate_normal_logpdf(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Set var to minimum 1e-6
    sd = np.sqrt(mean * unit_var)
    sd = np.maximum(sd, config.min_sd)

    # Initialize logpdf with 0 (log(1))
    logpdf = np.zeros(x.shape[0])

    # Loop through dimensions and calculate logpdf
    for i in range(mean.shape[0]):
        # Extract mean and sd for the current dimension
        x_i = np.round(x[:, i])
        m = mean[i]
        s = sd[i]

        # Get logpdf as "mass" between integers
        cdf_lower = norm.cdf(x_i - 0.5, loc=m, scale=s)
        cdf_upper = norm.cdf(x_i + 0.5, loc=m, scale=s)

        # Find diff in log space
        logpdf_i = np.log(np.maximum(cdf_upper - cdf_lower, 1e-100))

        # Add to logpdf
        logpdf += logpdf_i

    return logpdf


def flanking_logpdf_old(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray, is_left_flank: list[bool]) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Initialize logpdf as array of same size as x
    logpdf = np.zeros(x.shape)

    # Loop through dimensions and calculate logpdf
    for i in range(mean.shape[0]):
        # Extract mean and variance for the current dimension
        x_i = x[:, i : i + 1]
        v_i = unit_var[i : i + 1]

        # TODO: We only need to do this for the "cut" dimension - not all! In case of more complex repeats
        # We need to sum the logpdf over m : 0 -> mean, but we only need m "close" to the x_i value
        logpdf_i = np.full(x.shape[0], -np.inf)
        min_x_i = np.min(x_i)
        max_x_i = np.max(x_i)
        range_start = np.maximum(min_x_i - 10 * np.sqrt(v_i), 0)
        range_end = np.minimum(max_x_i + 10 * np.sqrt(v_i), mean[i])
        for m in range(int(range_start), int(range_end) + 2):
            # Calculate logpdf
            m_ij = np.array([m])
            logpdf_ij = discrete_multivariate_normal_logpdf(x_i, m_ij, v_i)

            # Add to logpdf
            logpdf_i = np.logaddexp(logpdf_i, logpdf_ij)

        # Add to logpdf
        logpdf[:, i] = logpdf_i

    # Handle not reached repeats
    logpdf = adjust_logpdf_for_flanks(x, is_left_flank, logpdf)

    # Sum logpdf over repeat dimensions
    logpdf = np.sum(logpdf, axis=1)

    # Normalize with uniform distribution constant
    m_prod = np.prod(mean + 1)
    logpdf += np.log(1 / m_prod)

    return logpdf


def flanking_logpdf(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray, is_left_flank: list[bool]) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Initialize logpdf as array of same size as x
    logpdf = np.zeros(x.shape)

    # Loop through dimensions and calculate logpdf
    for i in range(mean.shape[0]):
        # Extract mean and variance for the current dimension
        x_i = x[:, i : i + 1]
        m_i = mean[i : i + 1]
        v_i = unit_var[i : i + 1]

        # TODO: We only need to do this for the "cut" dimension - not all! In case of more complex repeats
        # We need to sum the logpdf over m : 0 -> mean, but we only need m "close" to the x_i value
        logpdf_i = np.full(x.shape[0], -np.inf)

        # Calculate logpdf at mean
        pdf_m_i = np.float64(np.exp(discrete_multivariate_normal_logpdf(np.array([m_i]), m_i, v_i)))

        const_norm = 2 / (2 * m_i * pdf_m_i + 1)
        const_uniform = 2 * m_i * pdf_m_i / (2 * m_i * pdf_m_i + 1)

        # For x > mean: Use normal distribution
        if np.any(x_i > m_i):
            x_i_subset = x_i[x_i.flatten() > m_i]
            logpdf_i[x_i.flatten() > m_i] = discrete_multivariate_normal_logpdf(x_i_subset, m_i, v_i) + np.log(const_norm)

        # For x < mean: Use uniform distribution
        if np.any(x_i <= m_i):
            logpdf_i[x_i.flatten() <= m_i] = np.log(1 / m_i) + np.log(const_uniform)

        # Add to logpdf
        logpdf[:, i] = logpdf_i

    # Handle not reached repeats
    logpdf = adjust_logpdf_for_flanks(x, is_left_flank, logpdf)

    # Sum logpdf over repeat dimensions
    return np.sum(logpdf, axis=1)


def adjust_logpdf_for_flanks(x: np.ndarray, is_left_flank: list[bool], logpdf: np.ndarray) -> np.ndarray:
    res = logpdf
    for n in range(x.shape[0]):
        is_left = is_left_flank[n]
        x_n = x[n, :]
        for j, _ in enumerate(x_n):
            # If this and all following/preceding (depending on is_left) repeats are 0, set logpdf to 0 (Log(1))
            if (is_left and all(x_n[: j + 1] == 0)) or (not is_left and all(x_n[j:] == 0)):
                res[n, j] = 0

    return res


def group_read_calls(read_calls: list[ReadCall], ploidy: int) -> tuple[list[ReadCall], pd.DataFrame, pd.DataFrame]:
    # Handle empty read_calls
    if not read_calls:
        return create_empty_results(0)

    # Split read calls into flanking and spanning reads
    spanning_reads = [r for r in read_calls if r.alignment.alignment_type == AlignmentType.SPANNING]
    flanking_reads = [r for r in read_calls if r.alignment.alignment_type != AlignmentType.SPANNING]
    is_left_flank = [r.alignment.alignment_type == AlignmentType.LEFT_FLANKING for r in flanking_reads]

    # Get counts from reads
    spanning_counts = get_counts(spanning_reads)
    flanking_counts = get_counts(flanking_reads)

    # TODO: Add flanking_counts to homzygous estimation
    hom_mean, hom_unit_var = estimate_homzygous_parameters(spanning_counts)

    # Estimate confidence intervals for heterozygous means
    hom_conf_mean_lower, hom_conf_mean_upper = estimate_confidence_intervals_homozygous(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        hom_mean,
        hom_unit_var,
    )

    if ploidy == 1:
        # Assign homzygous labels to reads
        grouped_read_calls = []
        grouped_read_calls.extend([read.add_em_haplotype("hom") for read in spanning_reads])
        grouped_read_calls.extend([read.add_em_haplotype("hom") for read in flanking_reads])

        # For ploidy=1, we'll create a summary with NaN values since we don't perform heterozygosity testing
        test_summary_df = summarize_test_statistics(
            log_lik_hetero=np.float64(np.nan),
            log_lik_hom=np.float64(np.nan),
            n_par_hetero=-1,
            n_par_hom=-1,
            test_statistic=np.float64(0),
            df=-1,
            p_value=np.float64(1),
            is_significant=False,
        )

        # Summarize parameter estimates
        parameter_summary_df = summarize_parameter_estimates(
            # Fill heterozygous parameters with NaN
            het_mean_h1=np.full_like(hom_mean, np.nan),
            het_mean_h1_conf_lower=np.full_like(hom_mean, np.nan),
            het_mean_h1_conf_upper=np.full_like(hom_mean, np.nan),
            het_mean_h2=np.full_like(hom_mean, np.nan),
            het_mean_h2_conf_lower=np.full_like(hom_mean, np.nan),
            het_mean_h2_conf_upper=np.full_like(hom_mean, np.nan),
            het_unit_var=np.full_like(hom_mean, np.nan),
            pi=np.float64(np.nan),
            # Add homozygous parameters
            hom_mean=hom_mean,
            hom_mean_conf_lower=hom_conf_mean_lower,
            hom_mean_conf_upper=hom_conf_mean_upper,
            hom_unit_var=hom_unit_var,
        )

        return read_calls, test_summary_df, parameter_summary_df

    # Estimate parameters for heterozygous and homozygous models
    het_mean_h1, het_mean_h2, unit_var, pi = estimate_heterozygous_parameters(spanning_reads, flanking_reads)

    # Estimate confidence intervals for heterozygous means
    het_conf_mean_h1_lower, het_conf_mean_h1_upper, het_conf_mean_h2_lower, het_conf_mean_h2_upper = estimate_confidence_intervals(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        het_mean_h1,
        het_mean_h2,
        unit_var,
        pi,
    )

    # Test for heterozygosity
    log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, df, p_value, is_significant = test_heterozygosity(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        het_mean_h1,
        het_mean_h2,
        unit_var,
        pi,
        hom_mean,
        hom_unit_var,
    )

    # Summarize test statistics
    test_summary_df = summarize_test_statistics(
        log_lik_hetero=log_lik_hetero,
        log_lik_hom=log_lik_hom,
        n_par_hetero=n_par_hetero,
        n_par_hom=n_par_hom,
        test_statistic=test_statistic,
        df=df,
        p_value=p_value,
        is_significant=is_significant,
    )

    # Summarize parameter estimates
    parameter_summary_df = summarize_parameter_estimates(
        het_mean_h1=het_mean_h1,
        het_mean_h1_conf_lower=het_conf_mean_h1_lower,
        het_mean_h1_conf_upper=het_conf_mean_h1_upper,
        het_mean_h2=het_mean_h2,
        het_mean_h2_conf_lower=het_conf_mean_h2_lower,
        het_mean_h2_conf_upper=het_conf_mean_h2_upper,
        het_unit_var=unit_var,
        pi=pi,
        hom_mean=hom_mean,
        hom_mean_conf_lower=hom_conf_mean_lower,
        hom_mean_conf_upper=hom_conf_mean_upper,
        hom_unit_var=hom_unit_var,
    )

    # Assign labels to reads
    spanning_labels, flanking_labels = assign_labels(
        spanning_reads,
        flanking_reads,
        het_mean_h1,
        het_mean_h2,
        unit_var,
        pi,
        is_significant,
    )

    # Create grouped read calls from spanning and flanking reads with their assigned labels
    grouped_read_calls = []
    grouped_read_calls.extend([read.add_em_haplotype(label) for read, label in zip(spanning_reads, spanning_labels)])
    grouped_read_calls.extend([read.add_em_haplotype(label) for read, label in zip(flanking_reads, flanking_labels)])

    return grouped_read_calls, test_summary_df, parameter_summary_df


def create_empty_results(kmer_dim: int) -> tuple[list[ReadCall], pd.DataFrame, pd.DataFrame]:
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

    empty_params = pd.DataFrame(
        {
            "em_haplotype": "none",
            "mean": pd.NA,
            "unit_var": pd.NA,
            "pi": pd.NA,
            "n": pd.NA,
            "idx": list(range(kmer_dim)),
        },
    )

    return ([], summary_res_df, empty_params)


def get_counts(read_calls: list[ReadCall]) -> np.ndarray:
    return np.array([r.satellite_count for r in read_calls])


def initialize_parameters(spanning_counts: np.ndarray, flanking_counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float64]:
    counts = spanning_counts
    # If any flanking reads are longer than the longest spanning read, we need to extend the spanning reads
    if flanking_counts.size > 0:
        max_spanning_counts = np.max(spanning_counts, axis=0)
        long_flanking_reads = np.array([x for x in flanking_counts if any(x > max_spanning_counts)])

        # Add long flanking reads to spanning reads
        if long_flanking_reads.size > 0:
            counts = np.concatenate((spanning_counts, long_flanking_reads))

    # Sort counts by max count
    max_counts = np.max(counts, axis=1)
    idx = np.argsort(max_counts)
    sorted_counts = counts[idx, :]
    # Split counts into two halves
    counts_len = len(sorted_counts)
    counts_half = counts_len // 2
    counts_h1 = sorted_counts[:counts_half, :]
    counts_h2 = sorted_counts[counts_half:, :]

    # Mean
    # Calculate mean for each half
    mean_h1 = np.mean(counts_h1, axis=0)
    mean_h2 = np.mean(counts_h2, axis=0)

    # Variance
    unit_var_h1 = np.var(counts_h1, axis=0) / (mean_h1 + 1e-5)
    unit_var_h2 = np.var(counts_h2, axis=0) / (mean_h2 + 1e-5)
    unit_var = np.average(np.array([unit_var_h1, unit_var_h2]), axis=0)
    unit_var = np.maximum(unit_var, 1e-5)

    # Pi
    pi = np.float64(0.51)

    # Make sure mean is at least 0.1
    mean_h1 = np.maximum(mean_h1, 0.1)
    mean_h2 = np.maximum(mean_h2, 0.1)

    # Make sure init means are not too close - if so, move them apart
    means_too_close = abs(mean_h1 - mean_h2) < 1
    mean_h1[means_too_close & (mean_h1 <= mean_h2)] *= 0.9
    mean_h1[means_too_close & (mean_h1 >= mean_h2)] *= 1.1
    mean_h2[means_too_close & (mean_h1 >= mean_h2)] *= 0.9
    mean_h2[means_too_close & (mean_h1 <= mean_h2)] *= 1.1

    return mean_h1, mean_h2, unit_var, pi


def weighted_median(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    res = np.zeros(x.shape[1])
    for i in range(x.shape[1]):
        idx = np.argsort(x[:, i])
        cs = np.cumsum(w[idx])
        cs /= cs[-1]
        res[i] = x[idx, i][np.searchsorted(cs, 0.5)]
    return res


def estimate_mean_spanning(counts: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    # Handle empty arrays
    if not counts.size:
        return np.array([])

    # Handle gamma = 0
    if np.all(gamma == 0):
        return np.zeros(counts.shape[1])

    return np.average(counts, axis=0, weights=gamma)


def estimate_mean_flanking(counts: np.ndarray, gamma: np.ndarray, is_left_flank: list[bool]) -> np.ndarray:
    # Handle empty arrays
    if not counts.size:
        return np.array([])

    # Loop through dimensions and calculate mean
    res = np.zeros(counts.shape[1])
    for i in range(counts.shape[1]):
        counts_i = counts[:, i]

        # Remove zeros from flanking reads
        # Initialize updated counts empty
        updated_counts_i = np.array([])
        for j in range(len(counts_i)):
            # TODO: Use NAN instead of 0 for overhangs
            is_left = is_left_flank[j]
            if (is_left and all(counts_i[: j + 1] == 0)) or (not is_left and all(counts_i[j:] == 0)):
                continue
            updated_counts_i = np.append(updated_counts_i, counts_i[j])

        # Sort counts and gamma
        idx = np.argsort(updated_counts_i)
        updated_counts_i = updated_counts_i[idx]
        gamma_i = gamma[idx]

        # Calculate maximum likelihood estimateas max with probability
        est = 0
        for j in range(len(updated_counts_i)):
            est += updated_counts_i[j] * gamma_i[j] * np.prod(1 - gamma_i[j + 1 :])

        # Add to result
        res[i] = est

    return res


def estimate_par(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    gamma_h1_spanning: np.ndarray,
    gamma_h2_spanning: np.ndarray,
    gamma_h1_flanking: np.ndarray,
    gamma_h2_flanking: np.ndarray,
    pi: np.float64,
    is_left_flank: list[bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float64, np.float64]:
    # Get initial estimates
    if spanning_counts.size and flanking_counts.size:
        # Get initial estimates from spanning reads
        mean_h1_spanning_init = estimate_mean_spanning(spanning_counts, gamma_h1_spanning)
        mean_h2_spanning_init = estimate_mean_spanning(spanning_counts, gamma_h2_spanning)

        # Get initial estimates from flanking reads
        mean_h1_flanking_init = estimate_mean_flanking(flanking_counts, gamma_h1_flanking, is_left_flank)
        mean_h2_flanking_init = estimate_mean_flanking(flanking_counts, gamma_h2_flanking, is_left_flank)

        # Use max as initial estimates
        mean_h1_init = np.maximum(mean_h1_spanning_init, mean_h1_flanking_init)
        mean_h2_init = np.maximum(mean_h2_spanning_init, mean_h2_flanking_init)

    elif spanning_counts.size:
        # Get initial estimates from spanning reads only
        mean_h1_init = estimate_mean_spanning(spanning_counts, gamma_h1_spanning)
        mean_h2_init = estimate_mean_spanning(spanning_counts, gamma_h2_spanning)
    elif flanking_counts.size:
        # Get initial estimates from flanking reads only
        mean_h1_init = estimate_mean_flanking(flanking_counts, gamma_h1_flanking, is_left_flank)
        mean_h2_init = estimate_mean_flanking(flanking_counts, gamma_h2_flanking, is_left_flank)
    else:
        # No reads - return empty arrays
        return np.array([]), np.array([]), np.array([]), np.float64(0), np.float64(0)

    # Get initial estimates for unit variance
    unit_var_init = estimate_unit_variance(spanning_counts, gamma_h1_spanning, gamma_h2_spanning, mean_h1_init, mean_h2_init)

    # Get initial estimate for pi
    pi_init = np.float64(np.average(np.append(gamma_h1_spanning, gamma_h1_flanking)))

    # Define bounds for optimization
    mean_bound = (0, None)
    unit_var_bound = (1e-5, None)
    pi_bound = (1e-5, 1 - 1e-5)

    # Make sure initial estimates are within bounds
    mean_h1_init = np.clip(mean_h1_init, mean_bound[0], mean_bound[1])
    mean_h2_init = np.clip(mean_h2_init, mean_bound[0], mean_bound[1])
    unit_var_init = np.clip(unit_var_init, unit_var_bound[0], unit_var_bound[1])
    pi_init = np.clip(pi_init, pi_bound[0], pi_bound[1])

    # Optimize log likelihood to find best mean
    dim = spanning_counts.shape[1]
    optim_res = minimize(
        fun=lambda x: -calculate_log_likelihood_heterozygous(
            spanning_counts=spanning_counts,
            flanking_counts=flanking_counts,
            is_left_flank=is_left_flank,
            mean_h1=np.array(x[:dim]),
            mean_h2=np.array(x[dim : 2 * dim]),
            unit_var=np.array(x[2 * dim : 3 * dim]),
            pi=np.float64(x[-1]),
        ),
        x0=np.concatenate((mean_h1_init, mean_h2_init, unit_var_init, np.array([pi_init]))),
        method="L-BFGS-B",
        bounds=[mean_bound] * dim * 2 + [unit_var_bound] * dim + [pi_bound],
    )

    # Split optimized result into mean_h1, mean_h2, and unit_var
    mean_h1 = np.array(optim_res.x[:dim])
    mean_h2 = np.array(optim_res.x[dim : 2 * dim])
    unit_var = np.array(optim_res.x[2 * dim : 3 * dim])
    pi = np.float64(optim_res.x[-1])

    print("------------")
    print("Mean H1:", mean_h1)
    print("Mean H2:", mean_h2)
    print("Unit Var:", unit_var)
    print("Pi:", pi)
    print("Log Likelihood:", -optim_res.fun)

    log_likelihood = -optim_res.fun

    return mean_h1, mean_h2, unit_var, pi, log_likelihood


def estimate_heterozygous_parameters(
    spanning_reads: list[ReadCall],
    flanking_reads: list[ReadCall],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.float64]:
    # Extract information from reads
    spanning_counts, _ = extract_info_from_spanning_reads(spanning_reads)
    flanking_counts, _, is_left_flank = extract_info_from_flanking_reads(flanking_reads)

    # Get initial parameters from spanning reads only
    mean_h1, mean_h2, unit_var, pi = initialize_parameters(spanning_counts, flanking_counts)

    # Initialize log likelihood
    log_likelihood = np.float64(-np.inf)

    # Run EM algorithm
    for i in range(500):
        # Get grouping probabilities
        gamma_h1_spanning, gamma_h2_spanning, gamma_h1_flanking, gamma_h2_flanking = calculate_grouping_probabilities(
            spanning_reads,
            flanking_reads,
            mean_h1,
            mean_h2,
            unit_var,
            pi,
        )

        # Estimate new parameters
        # Mean
        # Spanning
        new_mean_h1, new_mean_h2, new_unit_variance, new_pi, new_log_likelihood = estimate_par(
            spanning_counts=spanning_counts,
            flanking_counts=flanking_counts,
            gamma_h1_spanning=gamma_h1_spanning,
            gamma_h2_spanning=gamma_h2_spanning,
            gamma_h1_flanking=gamma_h1_flanking,
            gamma_h2_flanking=gamma_h2_flanking,
            pi=pi,
            is_left_flank=is_left_flank,
        )

        # Check convergence
        all_params_close = (
            np.allclose(mean_h1, new_mean_h1) and np.allclose(mean_h2, new_mean_h2) and np.allclose(unit_var, new_unit_variance) and np.allclose(pi, new_pi)
        )
        log_likelihood_has_decreased = new_log_likelihood < log_likelihood
        run_for_min_iterations = i > 10

        if (all_params_close or log_likelihood_has_decreased) and run_for_min_iterations:
            break

        # Update parameters
        mean_h1 = new_mean_h1
        mean_h2 = new_mean_h2
        unit_var = new_unit_variance
        pi = new_pi

        # Update log likelihood
        log_likelihood = new_log_likelihood

    return mean_h1, mean_h2, unit_var, pi


def calculate_grouping_probabilities(
    spanning_reads: list[ReadCall],
    flanking_reads: list[ReadCall],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Extract information from reads
    spanning_counts, spanning_seqs = extract_info_from_spanning_reads(spanning_reads)
    flanking_counts, flanking_seqs, is_left_flank = extract_info_from_flanking_reads(flanking_reads)

    # Get logpdf for counts
    # Spanning reads
    logpdf_h1_spanning = safe_log(pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h1, unit_var)
    logpdf_h2_spanning = safe_log(1 - pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h2, unit_var)

    # Flanking reads
    logpdf_h1_flanking = safe_log(pi) + flanking_logpdf(flanking_counts, mean_h1, unit_var, is_left_flank)
    logpdf_h2_flanking = safe_log(1 - pi) + flanking_logpdf(flanking_counts, mean_h2, unit_var, is_left_flank)

    # Get group probabilities using string distance
    seq_log_prob_h1_spanning, seq_log_prob_h2_spanning, seq_log_prob_h1_flanking, seq_log_prob_h2_flanking = calc_sequence_log_probs(
        spanning_seqs=spanning_seqs,
        gamma_h1_spanning=np.exp(logpdf_h1_spanning),
        gamma_h2_spanning=np.exp(logpdf_h2_spanning),
        flanking_seqs=flanking_seqs,
        gamma_h1_flanking=np.exp(logpdf_h1_flanking),
        gamma_h2_flanking=np.exp(logpdf_h2_flanking),
        is_left_flank=is_left_flank,
    )

    # Calculate grouping probabilities
    gamma_h1_spanning = logpdf_h1_spanning + seq_log_prob_h1_spanning
    gamma_h2_spanning = logpdf_h2_spanning + seq_log_prob_h2_spanning

    gamma_h1_flanking = logpdf_h1_flanking + seq_log_prob_h1_flanking
    gamma_h2_flanking = logpdf_h2_flanking + seq_log_prob_h2_flanking

    # Normalize
    total_logp_spanning = np.logaddexp(gamma_h1_spanning, gamma_h2_spanning)
    total_logp_flanking = np.logaddexp(gamma_h1_flanking, gamma_h2_flanking)

    gamma_h1_spanning = np.exp(gamma_h1_spanning - total_logp_spanning)
    gamma_h2_spanning = np.exp(gamma_h2_spanning - total_logp_spanning)

    gamma_h1_flanking = np.exp(gamma_h1_flanking - total_logp_flanking)
    gamma_h2_flanking = np.exp(gamma_h2_flanking - total_logp_flanking)

    return gamma_h1_spanning, gamma_h2_spanning, gamma_h1_flanking, gamma_h2_flanking


def extract_info_from_spanning_reads(reads: list[ReadCall]) -> tuple[np.ndarray, list[str]]:
    """Extract information from spanning reads.

    Args:
        reads: List of ReadCall objects

    Returns:
        tuple containing:
        - spanning_counts: np.ndarray of counts for spanning reads
        - spanning_seqs: list of sequences for spanning reads

    """
    spanning_reads = [r for r in reads if r.alignment.alignment_type == AlignmentType.SPANNING]
    spanning_counts = get_counts(spanning_reads)
    spanning_seqs = [r.alignment.str_sequence for r in spanning_reads]

    return spanning_counts, spanning_seqs


def extract_info_from_flanking_reads(reads: list[ReadCall]) -> tuple[np.ndarray, list[str], list[bool]]:
    """Extract information from flanking reads.

    Args:
        reads: List of ReadCall objects

    Returns:
        tuple containing:
        - flanking_counts: np.ndarray of counts for flanking reads
        - flanking_seqs: list of sequences for flanking reads
        - is_left_flank: list of booleans indicating if flanking reads are left

    """
    flanking_reads = []
    is_left_flank = []

    for read in reads:
        if read.alignment.alignment_type != AlignmentType.SPANNING:
            flanking_reads.append(read)
            is_left_flank.append(read.alignment.alignment_type == AlignmentType.LEFT_FLANKING)

    flanking_counts = get_counts(flanking_reads)
    flanking_seqs = [r.alignment.str_sequence for r in flanking_reads]

    return flanking_counts, flanking_seqs, is_left_flank


def calc_sequence_log_probs(
    spanning_seqs: list[str],
    gamma_h1_spanning: np.ndarray,
    gamma_h2_spanning: np.ndarray,
    flanking_seqs: list[str],
    gamma_h1_flanking: np.ndarray,
    gamma_h2_flanking: np.ndarray,
    is_left_flank: list[bool],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Initialize arrays for mean distances
    spanning_h1_mean_dists = np.zeros(len(spanning_seqs))
    spanning_h2_mean_dists = np.zeros(len(spanning_seqs))
    flanking_h1_mean_dists = np.zeros(len(flanking_seqs))
    flanking_h2_mean_dists = np.zeros(len(flanking_seqs))

    # Handle spanning reads
    for i, seq1 in enumerate(spanning_seqs):
        seq1_type = AlignmentType.SPANNING
        # Compare to spanning reads
        for j, seq2 in enumerate(spanning_seqs):
            if i == j:
                continue
            seq2_type = AlignmentType.SPANNING
            seq1_trimmed, seq2_trimmed = trim_sequences_for_comparison(seq1, seq1_type, seq2, seq2_type)
            dist = levenshtein_distance(seq1_trimmed, seq2_trimmed)
            spanning_h1_mean_dists[i] += dist * gamma_h1_spanning[j]
            spanning_h2_mean_dists[i] += dist * gamma_h2_spanning[j]

        # Compare to flanking reads
        for j, seq2 in enumerate(flanking_seqs):
            seq2_type = AlignmentType.LEFT_FLANKING if is_left_flank[j] else AlignmentType.RIGHT_FLANKING
            seq1_trimmed, seq2_trimmed = trim_sequences_for_comparison(seq1, seq1_type, seq2, seq2_type)
            dist = levenshtein_distance(seq1_trimmed, seq2_trimmed)
            spanning_h1_mean_dists[i] += dist * gamma_h1_flanking[j]
            spanning_h2_mean_dists[i] += dist * gamma_h2_flanking[j]

        # Normalize
        spanning_h1_mean_dists[i] /= np.sum(gamma_h1_spanning) + np.sum(gamma_h1_flanking)
        spanning_h2_mean_dists[i] /= np.sum(gamma_h2_spanning) + np.sum(gamma_h2_flanking)

    # Handle flanking reads
    for i, seq1 in enumerate(flanking_seqs):
        seq1_type = AlignmentType.LEFT_FLANKING if is_left_flank[i] else AlignmentType.RIGHT_FLANKING

        # Compare to spanning reads
        for j, seq2 in enumerate(spanning_seqs):
            seq2_type = AlignmentType.SPANNING
            seq1_trimmed, seq2_trimmed = trim_sequences_for_comparison(seq1, seq1_type, seq2, seq2_type)
            dist = levenshtein_distance(seq1_trimmed, seq2_trimmed)
            flanking_h1_mean_dists[i] += dist * gamma_h1_spanning[j]
            flanking_h2_mean_dists[i] += dist * gamma_h2_spanning[j]

        # Compare to flanking reads
        for j, seq2 in enumerate(flanking_seqs):
            # Skip self-comparison
            if i == j:
                continue
            seq2_type = AlignmentType.LEFT_FLANKING if is_left_flank[j] else AlignmentType.RIGHT_FLANKING

            # Skip if flanking reads are on different sides
            if seq1_type != seq2_type:
                continue

            seq1_trimmed, seq2_trimmed = trim_sequences_for_comparison(seq1, seq1_type, seq2, seq2_type)
            dist = levenshtein_distance(seq1_trimmed, seq2_trimmed)
            flanking_h1_mean_dists[i] += dist * gamma_h1_flanking[j]
            flanking_h2_mean_dists[i] += dist * gamma_h2_flanking[j]

        # Normalize
        flanking_h1_mean_dists[i] /= np.sum(gamma_h1_spanning) + np.sum(gamma_h1_flanking)
        flanking_h2_mean_dists[i] /= np.sum(gamma_h2_spanning) + np.sum(gamma_h2_flanking)

    # Turn distances into probabilities
    spanning_h1_log_probs = np.log(1 / (1 + spanning_h1_mean_dists))
    spanning_h2_log_probs = np.log(1 / (1 + spanning_h2_mean_dists))
    flanking_h1_log_probs = np.log(1 / (1 + flanking_h1_mean_dists))
    flanking_h2_log_probs = np.log(1 / (1 + flanking_h2_mean_dists))

    return spanning_h1_log_probs, spanning_h2_log_probs, flanking_h1_log_probs, flanking_h2_log_probs


def trim_sequences_for_comparison(seq1: str, type1: AlignmentType, seq2: str, type2: AlignmentType) -> tuple[str, str]:
    """Trim two sequences for appropriate comparison based on their alignment types.

    Args:
        seq1: First sequence string
        type1: Alignment type of the first sequence
        seq2: Second sequence string
        type2: Alignment type of the second sequence

    Returns:
        tuple[str, str]: Trimmed versions of seq1 and seq2

    """
    seq1_start, seq1_end = 0, len(seq1)
    seq2_start, seq2_end = 0, len(seq2)

    # LEFT_FLANKING vs LEFT_FLANKING
    if type1 == AlignmentType.LEFT_FLANKING and type2 == AlignmentType.LEFT_FLANKING:
        min_length = min(len(seq1), len(seq2))
        seq1_end = seq2_end = min_length

    # LEFT_FLANKING vs SPANNING
    elif type1 == AlignmentType.LEFT_FLANKING and type2 == AlignmentType.SPANNING:
        trim_length = min(len(seq1), len(seq2))
        seq2_end = trim_length
    elif type1 == AlignmentType.SPANNING and type2 == AlignmentType.LEFT_FLANKING:
        trim_length = min(len(seq1), len(seq2))
        seq1_end = trim_length

    # RIGHT_FLANKING vs SPANNING
    elif type1 == AlignmentType.RIGHT_FLANKING and type2 == AlignmentType.SPANNING:
        trim_length = min(len(seq1), len(seq2))
        seq2_start = len(seq2) - trim_length
    elif type1 == AlignmentType.SPANNING and type2 == AlignmentType.RIGHT_FLANKING:
        trim_length = min(len(seq1), len(seq2))
        seq1_start = len(seq1) - trim_length

    # RIGHT_FLANKING vs RIGHT_FLANKING
    elif type1 == AlignmentType.RIGHT_FLANKING and type2 == AlignmentType.RIGHT_FLANKING:
        min_length = min(len(seq1), len(seq2))
        seq1_start = len(seq1) - min_length
        seq2_start = len(seq2) - min_length

    # LEFT_FLANKING vs RIGHT_FLANKING
    # No meaningful overlap, trim the beginning of the longer sequence
    elif (type1, type2) == (AlignmentType.LEFT_FLANKING, AlignmentType.RIGHT_FLANKING):
        min_length = min(len(seq1), len(seq2))
        seq1_end = min_length
        seq2_start = len(seq2) - min_length

    elif (type1, type2) == (AlignmentType.RIGHT_FLANKING, AlignmentType.LEFT_FLANKING):
        min_length = min(len(seq1), len(seq2))
        seq1_start = len(seq1) - min_length
        seq2_end = min_length

    # Apply trimming
    seq1_trimmed = seq1[seq1_start:seq1_end]
    seq2_trimmed = seq2[seq2_start:seq2_end]

    return seq1_trimmed, seq2_trimmed


def estimate_unit_variance(
    spanning_counts: np.ndarray,
    gamma_h1_spanning: np.ndarray,
    gamma_h2_spanning: np.ndarray,
    new_mean_h1_spanning: np.ndarray,
    new_mean_h2_spanning: np.ndarray,
) -> np.ndarray:
    # Handle empty arrays
    if not spanning_counts.size:
        return np.array([])

    # Estimate unit variance
    # H1
    # Handle gamma = 0
    if np.all(gamma_h1_spanning == 0):
        unit_var_h1 = np.zeros(spanning_counts.shape[1])
    else:
        unit_var_h1 = np.average(((spanning_counts - new_mean_h1_spanning) ** 2) / (new_mean_h1_spanning + 1e-5), axis=0, weights=gamma_h1_spanning)

    # H2
    # Handle gamma = 0
    if np.all(gamma_h2_spanning == 0):
        unit_var_h2 = np.zeros(spanning_counts.shape[1])
    else:
        unit_var_h2 = np.average(((spanning_counts - new_mean_h2_spanning) ** 2) / (new_mean_h2_spanning + 1e-5), axis=0, weights=gamma_h2_spanning)

    # Combine variance estimates
    h1_ratio = np.average(gamma_h1_spanning)
    return h1_ratio * unit_var_h1 + (1 - h1_ratio) * unit_var_h2


def estimate_homzygous_parameters(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hom_mean = np.average(counts, axis=0)
    hom_uni_var = np.average((counts - hom_mean) ** 2, axis=0) / (hom_mean + 1e-5)
    return hom_mean, hom_uni_var


def estimate_confidence_intervals(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Handle empty arrays
    if not spanning_counts.size and not flanking_counts.size:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Find where log likelihood ratio is equal to chi2(0.95, 1)
    chi2_val = np.float64(chi2.ppf(0.95, 1))

    def find_confidence_interval(mean: np.ndarray, pi: np.float64, is_h1: bool) -> tuple[np.ndarray, np.ndarray]:
        conf_lower = np.full(mean.shape, np.nan)
        conf_upper = np.full(mean.shape, np.nan)

        if (is_h1 and pi > 0.01) or (not is_h1 and pi < 0.99):
            for i in range(mean.shape[0]):
                # Define objective function
                def objective(x: np.float64, i: int) -> np.float64:
                    # Alternative hypothesis
                    ll_alt = calculate_log_likelihood_heterozygous(spanning_counts, flanking_counts, is_left_flank, mean_h1, mean_h2, unit_var, pi)
                    # Null hypothesis
                    mean_null = mean.copy()
                    mean_null[i] = x
                    ll_null = calculate_log_likelihood_heterozygous(
                        spanning_counts,
                        flanking_counts,
                        is_left_flank,
                        mean_null if is_h1 else mean_h1,
                        mean_h2 if is_h1 else mean_null,
                        unit_var,
                        pi,
                    )
                    # Test statistic
                    q_val = -2 * (ll_null - ll_alt)

                    return q_val - chi2_val

                # Find lower bound
                if objective(np.float64(0), i) < 0:
                    conf_lower[i] = 0
                else:
                    a = mean[i]
                    while objective(a, i) < 0:
                        a = a / 2
                    b = mean[i]
                    conf_lower[i] = brentq(objective, a, b, args=(i,))

                # Find upper bound
                a = mean[i]
                b = a
                conf_upper[i] = np.inf
                for j in range(1, 20):
                    b = a + 2**j
                    if objective(b, i) > 0:
                        conf_upper[i] = brentq(objective, a, b, args=(i,))
                        break

        return conf_lower, conf_upper

    conf_mean_h1_lower, conf_mean_h1_upper = find_confidence_interval(mean_h1, pi, True)
    conf_mean_h2_lower, conf_mean_h2_upper = find_confidence_interval(mean_h2, 1 - pi, False)

    return conf_mean_h1_lower, conf_mean_h1_upper, conf_mean_h2_lower, conf_mean_h2_upper


def estimate_confidence_intervals_homozygous(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean: np.ndarray,
    unit_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Handle empty arrays
    if not spanning_counts.size and not flanking_counts.size:
        return np.array([]), np.array([])

    # Find where log likelihood ratio is equal to chi2(0.95, 1)
    chi2_val = np.float64(chi2.ppf(0.95, 1))

    def find_confidence_interval(mean: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        conf_lower = np.full(mean.shape, np.nan)
        conf_upper = np.full(mean.shape, np.nan)

        for i in range(mean.shape[0]):
            # Define objective function
            def objective(x: np.float64, i: int) -> np.float64:
                # Alternative hypothesis
                ll_alt = calculate_log_likelihood_homozygous(
                    spanning_counts,
                    flanking_counts,
                    is_left_flank,
                    mean,
                    unit_var,
                )
                # Null hypothesis
                mean_null = mean.copy()
                mean_null[i] = x
                ll_null = calculate_log_likelihood_homozygous(
                    spanning_counts,
                    flanking_counts,
                    is_left_flank,
                    mean_null,
                    unit_var,
                )
                # Test statistic
                q_val = -2 * (ll_null - ll_alt)

                return q_val - chi2_val

            # Find lower bound
            if objective(np.float64(0), i) < 0:
                conf_lower[i] = 0
            else:
                a = mean[i]
                while objective(a, i) < 0:
                    a = a / 2
                b = mean[i]
                conf_lower[i] = brentq(objective, a, b, args=(i,))

            # Find upper bound
            a = mean[i]
            b = a
            conf_upper[i] = np.inf
            for j in range(1, 20):
                b = a + 2**j
                if objective(b, i) > 0:
                    conf_upper[i] = brentq(objective, a, b, args=(i,))
                    break

        return conf_lower, conf_upper

    # Calculate heterozygous confidence intervals
    conf_mean_lower, conf_mean_upper = find_confidence_interval(mean)

    return conf_mean_lower, conf_mean_upper


def test_heterozygosity(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
    hom_mean: np.ndarray,
    hom_var: np.ndarray,
) -> tuple[np.float64, np.float64, int, int, np.float64, int, np.float64, bool]:
    # Calculate log likelihoods for homozygous and heterozygous models
    log_lik_hetero = calculate_log_likelihood_heterozygous(spanning_counts, flanking_counts, is_left_flank, mean_h1, mean_h2, unit_var, pi)
    log_lik_hom = calculate_log_likelihood_homozygous(spanning_counts, flanking_counts, is_left_flank, hom_mean, hom_var)

    # Calculate test statistic
    test_statistic = -2 * (log_lik_hom - log_lik_hetero)
    n_par_hetero = len(mean_h1) + len(mean_h2) + len(unit_var) + 1  # +1 for pi
    n_par_hom = len(hom_mean) + len(hom_var)
    deg_freedom = n_par_hetero - n_par_hom

    p_value = 1 - np.float64(chi2.cdf(test_statistic, deg_freedom))
    is_significant = bool(p_value < 0.05)

    return log_lik_hom, log_lik_hetero, n_par_hom, n_par_hetero, test_statistic, deg_freedom, p_value, is_significant


def calculate_log_likelihood_homozygous(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    hom_mean: np.ndarray,
    hom_var: np.ndarray,
) -> np.float64:
    logpdf_spanning = discrete_multivariate_normal_logpdf(spanning_counts, hom_mean, hom_var)
    logpdf_flanking = flanking_logpdf(flanking_counts, hom_mean, hom_var, is_left_flank)
    return np.float64(np.sum(logpdf_spanning) + np.sum(logpdf_flanking))


def calculate_log_likelihood_heterozygous(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> np.float64:
    # Calculate logpdf for spanning reads
    logpdf_h1_spanning = safe_log(pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h1, unit_var)
    logpdf_h2_spanning = safe_log(1 - pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h2, unit_var)

    logpdf_spanning = np.logaddexp(logpdf_h1_spanning, logpdf_h2_spanning)

    # Calculate logpdf for flanking reads
    logpdf_h1_flanking = safe_log(pi) + flanking_logpdf(flanking_counts, mean_h1, unit_var, is_left_flank)
    logpdf_h2_flanking = safe_log(1 - pi) + flanking_logpdf(flanking_counts, mean_h2, unit_var, is_left_flank)

    logpdf_flanking = np.logaddexp(logpdf_h1_flanking, logpdf_h2_flanking)

    # Sum logpdfs
    return np.float64(np.sum(logpdf_spanning) + np.sum(logpdf_flanking))


def summarize_test_statistics(
    log_lik_hom: np.float64,
    log_lik_hetero: np.float64,
    n_par_hom: int,
    n_par_hetero: int,
    test_statistic: np.float64,
    df: int,
    p_value: np.float64,
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
            "p_value": p_value,
            "is_significant": is_significant,
        },
        index=[0],
    )


def summarize_parameter_estimates(
    het_mean_h1: np.ndarray,
    het_mean_h1_conf_lower: np.ndarray,
    het_mean_h1_conf_upper: np.ndarray,
    het_mean_h2: np.ndarray,
    het_mean_h2_conf_lower: np.ndarray,
    het_mean_h2_conf_upper: np.ndarray,
    het_unit_var: np.ndarray,
    pi: np.float64,
    hom_mean: np.ndarray,
    hom_mean_conf_lower: np.ndarray,
    hom_mean_conf_upper: np.ndarray,
    hom_unit_var: np.ndarray,
) -> pd.DataFrame:
    result_data = [
        {
            "em_haplotype": "h1",
            "mean": het_mean_h1,
            "mean_lower": het_mean_h1_conf_lower,
            "mean_upper": het_mean_h1_conf_upper,
            "unit_var": het_unit_var,
            "pi": pi,
            "idx": list(range(len(het_mean_h1))),
        },
        {
            "em_haplotype": "h2",
            "mean": het_mean_h2,
            "mean_lower": het_mean_h2_conf_lower,
            "mean_upper": het_mean_h2_conf_upper,
            "unit_var": het_unit_var,
            "pi": 1 - pi,
            "idx": list(range(len(het_mean_h2))),
        },
        {
            "em_haplotype": "hom",
            "mean": hom_mean,
            "mean_lower": hom_mean_conf_lower,
            "mean_upper": hom_mean_conf_upper,
            "unit_var": hom_unit_var,
            "pi": 1,
            "idx": list(range(len(hom_mean))),
        },
    ]

    return pd.concat([pd.DataFrame(data) for data in result_data])


def assign_labels(
    spanning_reads: list[ReadCall],
    flanking_reads: list[ReadCall],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
    is_significant: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if is_significant:
        # Calculate grouping probabilities
        gamma_h1_spanning, gamma_h2_spanning, gamma_h1_flanking, gamma_h2_flanking = calculate_grouping_probabilities(
            spanning_reads,
            flanking_reads,
            mean_h1,
            mean_h2,
            unit_var,
            pi,
        )

        # Labels
        em_labels_spanning = np.array(("h1", "h2"))[np.argmax(np.array((gamma_h1_spanning, gamma_h2_spanning)), axis=0)]
        em_labels_flanking = np.array(("h1", "h2"))[np.argmax(np.array((gamma_h1_flanking, gamma_h2_flanking)), axis=0)]
    else:
        em_labels_spanning = np.full(len(spanning_reads), "hom")
        em_labels_flanking = np.full(len(flanking_reads), "hom")
    return em_labels_spanning, em_labels_flanking
