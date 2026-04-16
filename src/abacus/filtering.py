from numpy import median, percentile

from abacus.config import config
from abacus.graph import ReadCall
from abacus.utils import Haplotype


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
    if len(good_read_calls) < config.min_n_qc_filtering:
        # If not, return all remaining read calls as good read calls
        return good_read_calls, outlier_read_calls

    # Step 3: Find outliers
    mark_qc_outliers(good_read_calls)
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
        rc.add_qc_filter_reason("filtered_low_mean_str_quality")

    if rc.alignment.q10_str_quality < config.min_q10_str_quality:
        rc.add_qc_filter_reason("filtered_low_q10_str_quality")

    # Check STR error rates
    if rc.str_error_rate > config.max_error_rate:
        rc.add_qc_filter_reason("filtered_high_str_error_rate")

    # Check STR reference divergence
    if rc.alignment.str_ref_divergence > config.max_ref_divergence:
        rc.add_qc_filter_reason("filtered_high_str_ref_divergence")


def mark_qc_outliers(read_calls: list[ReadCall]) -> None:
    if not read_calls:
        return

    # STR quality
    str_qualities = [rc.alignment.mean_str_quality for rc in read_calls]
    str_qual_low, _ = compute_robust_thresholds(str_qualities)
    for rc in read_calls:
        # If the STR quality is below the outlier threshold, and below the tolerance, mark it as qc_filtered
        if rc.alignment.mean_str_quality < str_qual_low and rc.alignment.mean_str_quality < config.tol_mean_str_quality:
            rc.add_qc_filter_reason("outlier_str_quality")

    # Q10 STR quality
    q10_str_qualities = [float(rc.alignment.q10_str_quality) for rc in read_calls]
    q10_str_qual_low, _ = compute_robust_thresholds(q10_str_qualities)
    for rc in read_calls:
        # If the Q10 STR quality is below the outlier threshold, and below the tolerance, mark it as qc_filtered
        if rc.alignment.q10_str_quality < q10_str_qual_low and rc.alignment.q10_str_quality < config.tol_q10_str_quality:
            rc.add_qc_filter_reason("outlier_q10_str_quality")

    # Error rates
    error_rates = [rc.str_error_rate for rc in read_calls]
    _, error_rate_high = compute_robust_thresholds(error_rates)
    for rc in read_calls:
        error_rate = rc.str_error_rate
        # If the error rate is above the outlier threshold, and above the tolerance, mark it as qc_filtered
        if error_rate > error_rate_high and error_rate > config.tol_error_rate:
            rc.add_qc_filter_reason("outlier_error_rate")


def detect_length_outliers(
    grouped_read_calls: list[ReadCall],
) -> tuple[list[ReadCall], list[ReadCall]]:
    """Per-haplotype length outlier detection after first round of haplotyping.

    For each haplotype group (h1, h2, hom), computes the median total base pair count
    (sum of satellite_count[i] * len(satellite.sequences[0]) for each satellite i)
    and robust Tukey-fence thresholds. Reads outside those bounds AND outside
    median * (1 ± tolerance_pct/100) are tagged as length outliers (Haplotype.OUTLIER).
    """
    clean: list[ReadCall] = []
    length_outliers: list[ReadCall] = []
    tolerance = config.tol_length_outlier_pct

    haplotype_groups = {rc.haplotype for rc in grouped_read_calls if rc.haplotype in (Haplotype.H1, Haplotype.H2, Haplotype.HOM)}

    # Pass through reads with other haplotypes (e.g. OUTLIER-tagged singletons) unchanged
    clean.extend(rc for rc in grouped_read_calls if rc.haplotype not in haplotype_groups)

    for haplotype in haplotype_groups:
        group = [rc for rc in grouped_read_calls if rc.haplotype == haplotype]

        # Only spanning reads have full STR coverage; flanking reads have partial str_sequence
        # and cannot be meaningfully compared on length
        spanning = [rc for rc in group if rc.is_spanning()]
        non_spanning = [rc for rc in group if not rc.is_spanning()]

        # Remove at most one outlier per haplotype per call; the caller re-estimates and
        # re-groups before calling again, mirroring the singleton removal loop
        if len(spanning) > config.min_n_length_outlier_detection:
            lengths = [float(len(rc.alignment.str_sequence)) for rc in spanning]
            med = float(median(lengths))
            lower_robust, upper_robust = compute_robust_thresholds(lengths)
            lower_tol = med * (1.0 - tolerance)
            upper_tol = med * (1.0 + tolerance)

            candidates = [
                (rc, length)
                for rc, length in zip(spanning, lengths)
                if (length < lower_robust or length > upper_robust) and (length < lower_tol or length > upper_tol)
            ]

            if candidates:
                worst_rc, _ = max(candidates, key=lambda x: abs(x[1] - med))
                worst_rc.add_outlier_reason("outlier_length")
                length_outliers.append(worst_rc)
                spanning.remove(worst_rc)

        # Spanning before non-spanning — preserves the order emitted by group_read_calls
        clean.extend(spanning)
        clean.extend(non_spanning)

    return clean, length_outliers


def compute_robust_thresholds(x: list[float]) -> tuple[float, float]:
    # Median and IQR
    x_median = median(x)
    q1 = percentile(x, 25)
    q3 = percentile(x, 75)
    iqr = q3 - q1
    lower_bound = x_median - 1.5 * iqr
    upper_bound = x_median + 1.5 * iqr

    return float(lower_bound), float(upper_bound)
