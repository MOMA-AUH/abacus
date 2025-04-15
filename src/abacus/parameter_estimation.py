from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.stats import chi2, norm

from abacus.config import config
from abacus.graph import ReadCall
from abacus.utils import AlignmentType


@dataclass
class HeterozygousParameters:
    mean_h1: np.ndarray
    mean_h2: np.ndarray
    unit_var: np.ndarray
    pi: np.float64


@dataclass
class HomozygousParameters:
    mean: np.ndarray
    unit_var: np.ndarray


def safe_log(x: np.ndarray | np.float64) -> np.ndarray | np.float64:
    return np.log(np.maximum(x, 1e-100))


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
        logcdf_lower = norm.logcdf(x_i - 0.5, loc=m, scale=s)
        logcdf_upper = norm.logcdf(x_i + 0.5, loc=m, scale=s)

        # Substract logcdf_lower from logcdf_upper in log space
        # log(cdf_upper - cdf_lower) = logcdf_upper + log(1 - exp(logcdf_lower - logcdf_upper))
        logpdf_i = logcdf_upper + np.log1p(-np.exp(logcdf_lower - logcdf_upper))

        # Make sure logpdf_i is not -Inf
        logpdf_i = np.where(logpdf_i == -np.inf, -1000, logpdf_i)

        # Add to logpdf
        logpdf += logpdf_i

    return logpdf


def flanking_logpdf(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray, is_left_flank: list[bool]) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Initialize logpdf as array of same size as x
    logpdf = np.zeros_like(x, dtype=np.float64)

    # Loop through individual reads
    for i in range(x.shape[0]):
        # Extract counts for the current read
        x_i = x[i, :]
        # Loop through dimensions
        for d in range(x.shape[1]):
            is_left = is_left_flank[i]

            # If left flank and this and all following repeats are 0, skip
            if is_left and all(x_i[d:] == 0):
                continue
            # If right flank and this and all preceding repeats are 0, skip
            if not is_left and all(x_i[: d + 1] == 0):
                continue

            # Figure out if this is the cut dimension i.e. last dimension with usable count info
            is_cut_dim = all(x_i[d + 1 :] == 0) if is_left else all(x_i[:d] == 0)

            # Extract mean and variance for the current dimension
            x_id = np.array([[x[i, d]]])
            m = np.array([mean[d]])
            v = np.array([unit_var[d]])

            # If this is NOT the cut dimension, simply use the normal distribution
            if not is_cut_dim:
                logpdf[i, d] = discrete_multivariate_normal_logpdf(x_id, m, v)
                continue

            # If this is the cut dimension, we need to use the uniform distribution

            # Calculate logpdf at mean

            # Calculate sum of unnormalized pdf over support
            split_point = int(np.round(m[0]))
            upper_bound = int(np.ceil(m[0] + 5 * np.sqrt(m[0] * v[0])))

            # 0 to split_point: Use uniform distribution
            # split_point to upper bound: Use normal distribution
            normal_support = np.array([[x] for x in range(split_point, upper_bound + 1)])
            norm_const = np.sum(np.exp(discrete_multivariate_normal_logpdf(normal_support, m, v)))

            # Add uniform part:
            norm_pdf_at_split = np.exp(discrete_multivariate_normal_logpdf(np.array([[split_point]]), m, v))
            if split_point > 0:
                norm_const += norm_pdf_at_split * (split_point - 1)

            # For x > mean: Use normal distribution
            if x_id >= split_point:
                logpdf[i, d] = discrete_multivariate_normal_logpdf(x_id, m, v) - np.log(norm_const)
            # For x < mean: Use uniform distribution
            else:
                logpdf[i, d] = np.log(norm_pdf_at_split / norm_const)

    # Sum logpdf over repeat dimensions
    return np.sum(logpdf, axis=1)


def flanking_logpdf_old(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray, is_left_flank: list[bool]) -> np.ndarray:
    # Handle empty arrays
    if not x.size:
        return np.array([])

    # Initialize logpdf as array of same size as x
    logpdf = np.ones_like(x)

    # Loop through individual reads
    for i in range(x.shape[0]):
        # Extract counts for the current read
        x_i = x[i, :]
        # Loop through dimensions
        for d in range(x.shape[1]):
            is_left = is_left_flank[i]

            # If left flank and this and all following repeats are 0, skip
            if is_left and all(x_i[d:] == 0):
                continue
            # If right flank and this and all preceding repeats are 0, skip
            if not is_left and all(x_i[: d + 1] == 0):
                continue

            # Figure out if this is the cut dimension i.e. last dimension with usable count info
            is_cut_dim = all(x_i[d + 1 :] == 0) if is_left else all(x_i[:d] == 0)

            # Extract mean and variance for the current dimension
            x_id = np.array([[x[i, d]]])
            m = np.array([mean[d]])
            v = np.array([unit_var[d]])

            # If this is NOT the cut dimension, simply use the normal distribution
            if not is_cut_dim:
                logpdf[i, d] = discrete_multivariate_normal_logpdf(x_id, m, v)
                continue

            # If this is the cut dimension, we need to use the uniform distribution

            # Calculate logpdf at mean
            pdf_m_i = np.float64(np.exp(discrete_multivariate_normal_logpdf(np.array([m]), m, v)))

            # Calculate constants
            const_norm = 2 / (2 * m * pdf_m_i + 1)
            const_uniform = 2 * m * pdf_m_i / (2 * m * pdf_m_i + 1)

            # For x > mean: Use normal distribution
            if x_id > m:
                logpdf[i, d] = discrete_multivariate_normal_logpdf(x_id, m, v) + np.log(const_norm)
            # For x < mean: Use uniform distribution
            else:
                logpdf[i, d] = np.log(1 / m) + np.log(const_uniform)

    # Sum logpdf over repeat dimensions
    return np.sum(logpdf, axis=1)


def calculate_initial_estimates(read_calls: list[ReadCall]) -> HeterozygousParameters:
    # Extract counts
    spanning_counts, flanking_counts, _ = unpack_read_calls(read_calls)

    # Initialize counts
    counts = spanning_counts.copy()

    # If any flanking reads are longer than the median spanning read, add them to the counts
    if flanking_counts.size > 0:
        median_spanning_counts = np.median(spanning_counts, axis=0)
        long_flanking_reads = np.array([x for x in flanking_counts if any(x > median_spanning_counts)])

        # Add long flanking reads to spanning reads
        if long_flanking_reads.size > 0:
            counts = np.concatenate((spanning_counts, long_flanking_reads))

    # Sort counts by max count
    max_counts = np.max(counts, axis=1)
    idx = np.argsort(max_counts)
    sorted_counts = counts[idx, :]

    # Split counts into two halves. In case of odd number of counts, the middle count is included in both halves (also in the case of 1 count)
    counts_len = len(sorted_counts)
    counts_half = (counts_len + 1) // 2
    counts_h1 = sorted_counts[:counts_half, :]
    counts_h2 = sorted_counts[-counts_half:, :]

    # Mean
    # Calculate mean for each half
    mean_h1 = np.mean(counts_h1, axis=0)
    mean_h2 = np.mean(counts_h2, axis=0)

    median_h1 = np.median(counts_h1, axis=0)
    median_h2 = np.median(counts_h2, axis=0)

    robust_mean_h1 = (mean_h1 + median_h1) / 2
    robust_mean_h2 = (mean_h2 + median_h2) / 2

    # Variance
    sd_h1 = np.sqrt(np.average((counts_h1 - robust_mean_h1) ** 2, axis=0))
    sd_h2 = np.sqrt(np.average((counts_h2 - robust_mean_h2) ** 2, axis=0))

    mad_h1 = np.median(np.abs(counts_h1 - robust_mean_h1), axis=0)
    mad_h2 = np.median(np.abs(counts_h2 - robust_mean_h2), axis=0)

    robust_sd_h1 = (sd_h1 + mad_h1) / 2
    robust_sd_h2 = (sd_h2 + mad_h2) / 2

    # Calculate robust variance
    unit_var_h1 = robust_sd_h1**2 / (robust_mean_h1 + 1e-5)
    unit_var_h2 = robust_sd_h2**2 / (robust_mean_h2 + 1e-5)

    unit_var = np.average(np.array([unit_var_h1, unit_var_h2]), axis=0)
    unit_var = np.maximum(unit_var, config.min_var)

    # Pi
    pi = np.float64(0.51)

    # Make sure mean is at least 0.1
    robust_mean_h1 = np.maximum(robust_mean_h1, 0.1)
    robust_mean_h2 = np.maximum(robust_mean_h2, 0.1)

    # Make sure init means are not too close - if so, move them apart
    means_too_close = abs(robust_mean_h1 - robust_mean_h2) < 1
    robust_mean_h1[means_too_close & (robust_mean_h1 <= robust_mean_h2)] *= 0.9
    robust_mean_h1[means_too_close & (robust_mean_h1 >= robust_mean_h2)] *= 1.1
    robust_mean_h2[means_too_close & (robust_mean_h1 >= robust_mean_h2)] *= 0.9
    robust_mean_h2[means_too_close & (robust_mean_h1 <= robust_mean_h2)] *= 1.1

    return HeterozygousParameters(
        mean_h1=robust_mean_h1,
        mean_h2=robust_mean_h2,
        unit_var=unit_var,
        pi=pi,
    )


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


def estimate_heterozygous_parameters(
    read_calls: list[ReadCall],
) -> HeterozygousParameters:
    # No reads - return empty parameters
    if not read_calls:
        return HeterozygousParameters(
            mean_h1=np.array([]),
            mean_h2=np.array([]),
            unit_var=np.array([]),
            pi=np.float64(0),
        )

    par_init = calculate_initial_estimates(read_calls)

    par_refined = refine_initial_estimates(
        read_calls,
        par_init.mean_h1,
        par_init.mean_h2,
        par_init.unit_var,
        par_init.pi,
    )

    par_optim = optimize_estimates(
        read_calls,
        par_refined.mean_h1,
        par_refined.mean_h2,
        par_refined.unit_var,
        par_refined.pi,
    )

    return optimize_estimates_integers(
        read_calls,
        par_optim.mean_h1,
        par_optim.mean_h2,
        par_optim.unit_var,
        par_optim.pi,
    )


def optimize_estimates_integers(
    read_calls: list[ReadCall],
    mean_h1_optim: np.ndarray,
    mean_h2_optim: np.ndarray,
    unit_var_optim: np.ndarray,
    pi_optim: np.float64,
) -> HeterozygousParameters:
    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

    # Get initial estimates
    dim = spanning_counts.shape[1]

    ranges_h1 = []
    for i in range(dim):
        start = np.floor(mean_h1_optim[i]) - 1
        start = np.maximum(start, 0)
        end = np.ceil(mean_h1_optim[i]) + 1
        ranges_h1.append(np.arange(start, end, 1))

    ranges_h2 = []
    for i in range(dim):
        start = np.floor(mean_h2_optim[i]) - 1
        start = np.maximum(start, 0)
        end = np.ceil(mean_h2_optim[i]) + 1
        ranges_h2.append(np.arange(start, end, 1))

    # Create grids - all combinations of min and max values
    mean_grid_h1 = np.array(np.meshgrid(*ranges_h1)).T.reshape(-1, dim)
    mean_grid_h2 = np.array(np.meshgrid(*ranges_h2)).T.reshape(-1, dim)

    # Calculate optimized log likelihood for each grid point combination
    best_log_likelihood = -np.inf
    best_mean_h1 = np.zeros(dim)
    best_mean_h2 = np.zeros(dim)
    best_unit_var = np.zeros(dim)
    best_pi = np.float64(0)

    # Define bounds for optimization
    unit_var_bound = (config.min_var, None)
    pi_bound = (1e-5, 1 - 1e-5)

    # Loop through all combinations of mean_h1 and mean_h2
    for mean_h1_int, mean_h2_int in product(mean_grid_h1, mean_grid_h2):
        # Optimize variance and pi while keeping integer means fixed
        dim = spanning_counts.shape[1]
        optim_res = minimize(
            fun=lambda x, mean_h1_int=mean_h1_int, mean_h2_int=mean_h2_int, dim=dim: -calculate_log_likelihood_heterozygous(
                spanning_counts=spanning_counts,
                flanking_counts=flanking_counts,
                is_left_flank=is_left_flank,
                mean_h1=mean_h1_int,
                mean_h2=mean_h2_int,
                unit_var=np.array(x[:dim]),
                pi=np.float64(x[-1]),
            ),
            x0=np.concatenate((unit_var_optim, np.array([pi_optim]))),
            method="L-BFGS-B",
            bounds=[unit_var_bound] * dim + [pi_bound],
        )

        # Update best estimates
        if -optim_res.fun > best_log_likelihood:
            best_log_likelihood = -optim_res.fun
            best_mean_h1 = mean_h1_int
            best_mean_h2 = mean_h2_int
            best_unit_var = np.array(optim_res.x[:dim])
            best_pi = np.float64(optim_res.x[-1])

    return HeterozygousParameters(
        mean_h1=best_mean_h1,
        mean_h2=best_mean_h2,
        unit_var=best_unit_var,
        pi=best_pi,
    )


def optimize_estimates(
    read_calls: list[ReadCall],
    mean_h1_init: np.ndarray,
    mean_h2_init: np.ndarray,
    unit_var_init: np.ndarray,
    pi_init: np.float64,
) -> HeterozygousParameters:
    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

    # Define bounds for optimization
    mean_bound = (0, None)
    unit_var_bound = (config.min_var, None)
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
    return HeterozygousParameters(
        mean_h1=np.array(optim_res.x[:dim]),
        mean_h2=np.array(optim_res.x[dim : 2 * dim]),
        unit_var=np.array(optim_res.x[2 * dim : 3 * dim]),
        pi=np.float64(optim_res.x[-1]),
    )


def refine_initial_estimates(
    read_calls: list[ReadCall],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> HeterozygousParameters:
    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

    # Get grouping probabilities
    gamma_h1_spanning, gamma_h2_spanning = calculate_grouping_probabilities_spanning(
        spanning_counts,
        mean_h1,
        mean_h2,
        unit_var,
        pi,
    )
    gamma_h1_flanking, gamma_h2_flanking = calculate_grouping_probabilities_flanking(
        flanking_counts,
        is_left_flank,
        mean_h1,
        mean_h2,
        unit_var,
        pi,
    )

    # Get initial estimates from spanning reads
    mean_h1_spanning_init = estimate_mean_spanning(spanning_counts, gamma_h1_spanning)
    mean_h2_spanning_init = estimate_mean_spanning(spanning_counts, gamma_h2_spanning)

    # Get initial estimates from flanking reads
    mean_h1_flanking_init = estimate_mean_flanking(flanking_counts, gamma_h1_flanking, is_left_flank)
    mean_h2_flanking_init = estimate_mean_flanking(flanking_counts, gamma_h2_flanking, is_left_flank)

    # Get initial estimates
    if spanning_counts.size and flanking_counts.size:
        # Use max as initial estimates
        mean_h1_init = np.maximum(mean_h1_spanning_init, mean_h1_flanking_init)
        mean_h2_init = np.maximum(mean_h2_spanning_init, mean_h2_flanking_init)
    elif spanning_counts.size:
        # Get initial estimates from spanning reads only
        mean_h1_init = mean_h1_spanning_init
        mean_h2_init = mean_h2_spanning_init
    else:
        # Get initial estimates from flanking reads only
        mean_h1_init = mean_h1_flanking_init
        mean_h2_init = mean_h2_flanking_init

    # Get initial estimates for unit variance
    unit_var_init = estimate_unit_variance(spanning_counts, gamma_h1_spanning, gamma_h2_spanning, mean_h1_init, mean_h2_init)

    # Get initial estimate for pi
    pi_init = np.float64(np.average(np.append(gamma_h1_spanning, gamma_h1_flanking)))
    return HeterozygousParameters(
        mean_h1=mean_h1_init,
        mean_h2=mean_h2_init,
        unit_var=unit_var_init,
        pi=pi_init,
    )


def calculate_grouping_probabilities_spanning(
    spanning_counts: np.ndarray,
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    gamma_h1_spanning = safe_log(pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h1, unit_var)
    gamma_h2_spanning = safe_log(1 - pi) + discrete_multivariate_normal_logpdf(spanning_counts, mean_h2, unit_var)

    total_logp_spanning = np.logaddexp(gamma_h1_spanning, gamma_h2_spanning)

    gamma_h1_spanning = np.exp(gamma_h1_spanning - total_logp_spanning)
    gamma_h2_spanning = np.exp(gamma_h2_spanning - total_logp_spanning)

    return gamma_h1_spanning, gamma_h2_spanning


def calculate_grouping_probabilities_flanking(
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> tuple[np.ndarray, np.ndarray]:
    gamma_h1_flanking = safe_log(pi) + flanking_logpdf(flanking_counts, mean_h1, unit_var, is_left_flank)
    gamma_h2_flanking = safe_log(1 - pi) + flanking_logpdf(flanking_counts, mean_h2, unit_var, is_left_flank)

    total_logp_flanking = np.logaddexp(gamma_h1_flanking, gamma_h2_flanking)

    gamma_h1_flanking = np.exp(gamma_h1_flanking - total_logp_flanking)
    gamma_h2_flanking = np.exp(gamma_h2_flanking - total_logp_flanking)

    return gamma_h1_flanking, gamma_h2_flanking


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
        var_h1 = np.average(((spanning_counts - new_mean_h1_spanning) ** 2), axis=0, weights=gamma_h1_spanning)
        unit_var_h1 = var_h1 / (new_mean_h1_spanning + 1e-5)

    # H2
    # Handle gamma = 0
    if np.all(gamma_h2_spanning == 0):
        unit_var_h2 = np.zeros(spanning_counts.shape[1])
    else:
        var_h2 = np.average(((spanning_counts - new_mean_h2_spanning) ** 2), axis=0, weights=gamma_h2_spanning)
        unit_var_h2 = var_h2 / (new_mean_h2_spanning + 1e-5)

    # Combine variance estimates
    h1_ratio = np.average(gamma_h1_spanning)
    return h1_ratio * unit_var_h1 + (1 - h1_ratio) * unit_var_h2


def unpack_read_calls(
    read_calls: list[ReadCall],
) -> tuple[np.ndarray, np.ndarray, list[bool]]:
    # Split read calls into flanking and spanning reads
    spanning_reads = [r for r in read_calls if r.alignment.type == AlignmentType.SPANNING]
    flanking_reads = [r for r in read_calls if r.alignment.type != AlignmentType.SPANNING]

    # Get counts from reads
    spanning_counts = np.array([r.satellite_count for r in spanning_reads])
    flanking_counts = np.array([r.satellite_count for r in flanking_reads])

    # Get is_left_flank boolean
    is_left_flank = [r.alignment.type == AlignmentType.LEFT_FLANKING for r in flanking_reads]

    return spanning_counts, flanking_counts, is_left_flank


def estimate_homozygous_parameters(
    read_calls: list[ReadCall],
) -> HomozygousParameters:
    if not read_calls:
        return HomozygousParameters(
            mean=np.array([]),
            unit_var=np.array([]),
        )

    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

    # Combine spanning and flanking counts
    if flanking_counts.size > 0:
        max_spanning_counts = np.max(spanning_counts, axis=0)
        long_flanking_reads = np.array([x for x in flanking_counts if any(x > max_spanning_counts)])

        # Add long flanking reads to spanning reads
        if long_flanking_reads.size > 0:
            spanning_counts = np.concatenate((spanning_counts, long_flanking_reads))

    # Calculate initial estimates
    mean_init = np.average(spanning_counts, axis=0)
    unit_var_init = np.average((spanning_counts - mean_init) ** 2, axis=0) / (mean_init + 1e-5)

    # Optimize estimates
    mean_optim, unit_var_optim = optimize_homozygous_estimates(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        mean_init,
        unit_var_init,
    )

    # Optimize estimates with integer means
    mean_int, unit_var_int = optimize_estimates_integers_homozygous(
        spanning_counts,
        flanking_counts,
        is_left_flank,
        mean_optim,
        unit_var_optim,
    )

    return HomozygousParameters(
        mean=mean_int,
        unit_var=unit_var_int,
    )


def optimize_estimates_integers_homozygous(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean_optim: np.ndarray,
    unit_var_optim: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    dim = spanning_counts.shape[1]

    mean_ranges = []
    for i in range(dim):
        start = np.floor(mean_optim[i]) - 1
        start = np.maximum(start, 0)
        end = np.ceil(mean_optim[i]) + 1
        mean_ranges.append(np.arange(start, end, 1))

    # Create grids - all combinations of min and max values
    mean_grid = np.array(np.meshgrid(*mean_ranges)).T.reshape(-1, dim)

    # Calculate optimized log likelihood for each grid point combination
    best_log_likelihood = -np.inf
    best_mean = np.zeros(dim)
    best_unit_var = np.zeros(dim)

    # Define bounds for optimization
    unit_var_bound = (config.min_var, None)

    # Loop through all combinations of mean
    for mean_int in mean_grid:
        # Optimize variance while keeping integer means fixed
        dim = spanning_counts.shape[1]
        optim_res = minimize(
            fun=lambda x, mean_int=mean_int, dim=dim: -calculate_log_likelihood_homozygous(
                spanning_counts=spanning_counts,
                flanking_counts=flanking_counts,
                is_left_flank=is_left_flank,
                mean=mean_int,
                unit_var=np.array(x[:dim]),
            ),
            x0=unit_var_optim,
            method="L-BFGS-B",
            bounds=[unit_var_bound] * dim,
        )

        # Update best estimates
        if -optim_res.fun > best_log_likelihood:
            best_log_likelihood = -optim_res.fun
            best_mean = mean_int
            best_unit_var = np.array(optim_res.x[:dim])

    return best_mean, best_unit_var


def optimize_homozygous_estimates(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean_init: np.ndarray,
    unit_var_init: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Define bounds for optimization
    mean_bound = (0, None)
    unit_var_bound = (config.min_var, None)

    # Make sure initial estimates are within bounds
    mean_init = np.clip(mean_init, mean_bound[0], mean_bound[1])
    unit_var_init = np.clip(unit_var_init, unit_var_bound[0], unit_var_bound[1])

    # Optimize log likelihood to find best mean
    dim = spanning_counts.shape[1]
    optim_res = minimize(
        fun=lambda x: -calculate_log_likelihood_homozygous(
            spanning_counts=spanning_counts,
            flanking_counts=flanking_counts,
            is_left_flank=is_left_flank,
            mean=np.array(x[:dim]),
            unit_var=np.array(x[dim : 2 * dim]),
        ),
        x0=np.concatenate((mean_init, unit_var_init)),
        method="L-BFGS-B",
        bounds=[mean_bound] * dim + [unit_var_bound] * dim,
    )

    # Split optimized result into mean_h1, mean_h2, and unit_var
    mean = np.array(optim_res.x[:dim])
    unit_var = np.array(optim_res.x[dim : 2 * dim])
    return mean, unit_var


def estimate_confidence_intervals_heterozygous(
    read_calls: list[ReadCall],
    mean_h1: np.ndarray,
    mean_h2: np.ndarray,
    unit_var: np.ndarray,
    pi: np.float64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Handle empty arrays
    if not read_calls:
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

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
    read_calls: list[ReadCall],
    mean: np.ndarray,
    unit_var: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Handle empty arrays
    if not read_calls:
        return np.array([]), np.array([])

    # Unpack read calls
    spanning_counts, flanking_counts, is_left_flank = unpack_read_calls(read_calls)

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


def calculate_log_likelihood_homozygous(
    spanning_counts: np.ndarray,
    flanking_counts: np.ndarray,
    is_left_flank: list[bool],
    mean: np.ndarray,
    unit_var: np.ndarray,
) -> np.float64:
    logpdf_spanning = discrete_multivariate_normal_logpdf(spanning_counts, mean, unit_var)
    logpdf_flanking = flanking_logpdf(flanking_counts, mean, unit_var, is_left_flank)
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
