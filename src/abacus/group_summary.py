import numpy as np
import pandas as pd

from abacus.haplotyping import ReadCall


def calculate_final_group_summaries(grouped_read_calls: list[ReadCall]) -> pd.DataFrame:
    kmer_dim = len(grouped_read_calls[0].satellite_count)

    # Hanlde empty data
    if len(grouped_read_calls) == 0:
        return pd.DataFrame(
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

    result_df_list = []
    # Get unique haplotypes from labels
    unique_haplotypes = np.unique([rc.em_haplotype for rc in grouped_read_calls])
    for h in unique_haplotypes:
        # Get data for each haplotype
        counts_h = np.array([rc.satellite_count for rc in grouped_read_calls if rc.em_haplotype == h])

        # Calculate summary statistics
        mean_h = np.mean(counts_h, axis=0)
        sd_h = np.std(counts_h, axis=0)
        median_h = np.median(counts_h, axis=0)
        q1_h = np.percentile(counts_h, 25, axis=0)
        q3_h = np.percentile(counts_h, 75, axis=0)
        iqr_h = q3_h - q1_h

        result_dict = {
            "em_haplotype": h,
            "mean": mean_h,
            "sd": sd_h,
            "median": median_h,
            "iqr": iqr_h,
            "n": len(counts_h),
            "idx": list(range(kmer_dim)),
        }

        result_df_list.append(pd.DataFrame(result_dict))

    return pd.concat(result_df_list)
