from numpy import median, percentile

from abacus.config import config
from abacus.graph import ReadCall


def filter_read_calls(read_calls: list[ReadCall]) -> tuple[list[ReadCall], list[ReadCall]]:
    if not read_calls:
        return [], []

    # Step 0: Initialize read calls
    good_read_calls = read_calls.copy()

    # Step 1: Check individual read calls
    outlier_read_calls: list[ReadCall] = []
    for rc in good_read_calls.copy():
        qc_check(rc)
        # Check if read call is an outlier
        if rc.outlier_reasons:
            # Add it to the outlier list
            outlier_read_calls.append(rc)
            # Remove it from the good read calls
            good_read_calls.remove(rc)

    # Step 2: Check if enough read calls are left for outlier detection
    if len(good_read_calls) < config.min_n_outlier_detection:
        # If not, return all remaining read calls as good read calls
        return good_read_calls, outlier_read_calls

    # Step 3: Find outliers
    mark_outliers(good_read_calls)
    for rc in good_read_calls.copy():
        # Check if read call is an outlier
        if rc.outlier_reasons:
            # Add it to the outlier list
            outlier_read_calls.append(rc)
            # Remove it from the good read calls
            good_read_calls.remove(rc)

    return good_read_calls, outlier_read_calls


def qc_check(rc: ReadCall) -> None:
    # Check STR quality
    if rc.alignment.mean_str_quality < config.min_mean_str_quality:
        rc.add_outlier_reason("filtered_low_mean_str_quality")

    if rc.alignment.q10_str_quality < config.min_q10_str_quality:
        rc.add_outlier_reason("filtered_low_q10_str_quality")

    # Check STR error rates
    if rc.str_error_rate > config.max_error_rate:
        rc.add_outlier_reason("filtered_high_str_error_rate")

    # Check STR reference divergence
    if rc.alignment.str_ref_divergence > config.max_ref_divergence:
        rc.add_outlier_reason("filtered_high_str_ref_divergence")


def mark_outliers(read_calls: list[ReadCall]) -> None:
    if not read_calls:
        return

    # STR quality
    str_qualities = [rc.alignment.mean_str_quality for rc in read_calls]
    str_qual_low, _ = compute_robust_thresholds(str_qualities)
    for rc in read_calls:
        # If the STR quality is below the outlier threshold, and below the tolerance, mark it as an outlier
        if rc.alignment.mean_str_quality < str_qual_low and rc.alignment.mean_str_quality < config.tol_mean_str_quality:
            rc.add_outlier_reason("outlier_str_quality")

    # Q10 STR quality
    q10_str_qualities = [float(rc.alignment.q10_str_quality) for rc in read_calls]
    q10_str_qual_low, _ = compute_robust_thresholds(q10_str_qualities)
    for rc in read_calls:
        # If the Q10 STR quality is below the outlier threshold, and below the tolerance, mark it as an outlier
        if rc.alignment.q10_str_quality < q10_str_qual_low and rc.alignment.q10_str_quality < config.tol_q10_str_quality:
            rc.add_outlier_reason("outlier_q10_str_quality")

    # Error rates
    error_rates = [rc.str_error_rate for rc in read_calls]
    _, error_rate_high = compute_robust_thresholds(error_rates)
    for rc in read_calls:
        error_rate = rc.str_error_rate
        # If the error rate is above the outlier threshold, and above the tolerance, mark it as an outlier
        if error_rate > error_rate_high and error_rate > config.tol_error_rate:
            rc.add_outlier_reason("outlier_error_rate")


def compute_robust_thresholds(x: list[float]) -> tuple[float, float]:
    # Median and IQR
    x_median = median(x)
    q1 = percentile(x, 25)
    q3 = percentile(x, 75)
    iqr = q3 - q1
    lower_bound = x_median - 1.5 * iqr
    upper_bound = x_median + 1.5 * iqr

    return float(lower_bound), float(upper_bound)
