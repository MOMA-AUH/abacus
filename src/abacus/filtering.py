import numpy as np
from scipy.stats import chi2

from abacus.config import config
from abacus.haplotyping import ReadCall
from abacus.utils import compute_error_rate


def filter_read_calls(read_calls: list[ReadCall]) -> tuple[list[ReadCall], list[ReadCall]]:
    if not read_calls:
        return [], []

    # Check alignment parameters and add reasons for each read
    for rc in read_calls:
        check_alignment_parameters(rc)

    # Split read calls into good and filtered reads
    good_read_calls: list[ReadCall] = []
    outlier_read_calls: list[ReadCall] = []
    for rc in read_calls:
        if rc.outlier_reasons:
            outlier_read_calls.append(rc)
        else:
            good_read_calls.append(rc)

    return good_read_calls, outlier_read_calls


def check_alignment_parameters(rc: ReadCall) -> None:
    # Check STR quality
    if rc.alignment.str_median_quality < config.min_str_read_qual:
        rc.add_outlier_reason("low_str_median_quality")

    # Check STR error rates
    error_rate = compute_error_rate(rc.alignment.str_reference, rc.alignment.str_sequence)
    if error_rate > config.error_rate_threshold:
        rc.add_outlier_reason("high_str_error_rate")

    # Check STR window error rates
    # Find highest base error rate in STR region
    window_length = 50
    str_end = len(rc.alignment.str_sequence_synced)
    str_window_base_rate = 0.0
    str_window_indel_rate = 0.0
    if window_length < str_end:
        for i in range(str_end - window_length):
            window_ref = rc.alignment.str_reference[i : i + window_length]
            window_seq = "".join(rc.alignment.str_sequence_synced[i : i + window_length])
            str_window_base_rate = max(str_window_base_rate, compute_error_rate(window_ref, window_seq, indel_cost=0))
            str_window_indel_rate = max(str_window_indel_rate, compute_error_rate(window_ref, window_seq, replace_cost=0))

    # Add outlier reasons
    # Base error rate
    if str_window_base_rate > config.error_rate_threshold:
        rc.add_outlier_reason("has_window_with_high_base_error_rate")

    # Indel error rate
    if str_window_indel_rate > config.error_rate_threshold:
        rc.add_outlier_reason("has_window_with_high_indel_error_rate")


def evaluate_haplotyping(grouped_read_calls: list[ReadCall]) -> tuple[list[ReadCall], list[ReadCall]]:
    # Initialize lists
    good_read_calls: list[ReadCall] = []
    outlier_read_calls: list[ReadCall] = []

    # Evaluate haplotypes
    # Get unique haplotypes
    haplotypes = {rc.em_haplotype for rc in grouped_read_calls}
    for haplotype in haplotypes:
        # Get read calls for haplotype
        haplotype_read_calls = [rc for rc in grouped_read_calls if rc.em_haplotype == haplotype]

        # If haplotype has only one read call, mark as outlier
        if len(haplotype_read_calls) == 1:
            rc = haplotype_read_calls[0]
            rc.add_outlier_reason("single_read_haplotype")
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
        distances = calculate_mahalanobis_distances(counts_data)
        mean_distances = calculate_mean_dist(counts_data)

        # Classify reads using chi-square threshold
        # For n independent variables, Mahalanobis distance follows chi-square with n degrees of freedom
        # Using 97.5th percentile of chi-square distribution with n dimensions
        df = counts_data.shape[1]  # number of dimensions
        threshold = np.sqrt(chi2.ppf(0.99, df))

        # Remove outliers: reads with high Mahalanobis distance AND counts further than 1 from mean
        for read_call, distance, mean_distance in zip(haplotype_read_calls, distances, mean_distances):
            if distance > threshold and mean_distance > 1.5:
                read_call.add_outlier_reason("high_cluster_distance")
                outlier_read_calls.append(read_call)
            else:
                good_read_calls.append(read_call)

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


def calculate_mean_dist(counts_data: np.ndarray) -> np.ndarray:
    mean = np.mean(counts_data, axis=0)

    return np.max(np.abs(counts_data - mean), axis=1)
