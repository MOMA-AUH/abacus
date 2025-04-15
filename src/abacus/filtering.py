from abacus.config import config
from abacus.graph import ReadCall
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
