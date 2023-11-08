import argparse
import json
import re
from ast import Continue, arg
from dataclasses import dataclass
from itertools import combinations, compress, product
from math import ceil, floor
from statistics import median
from tracemalloc import start
from turtle import st

import mappy as mp
import numpy as np
import pandas as pd
import pysam
from pyfaidx import Fasta
from scipy.stats import multivariate_normal, trim_mean
from sklearn.cluster import DBSCAN


@dataclass
class STR:
    """Class for keeping dat for STR"""

    locus_id: str
    chrom: str
    start: int
    end: int
    satelites: list[str]
    breaks: list[str]


# Define constants
ANCHOR_LEN = 250
MIN_ANCHOR_OVERLAP = 125

SATELITE_PATTERN = r"(?<=\()(?:[GCATN]+)(?=\)[\*\+])"
BREAK_PATTERN = r"(?<=\)[\*\+])(?:[GCATN]*)(?=\()"


def check_read(
    min_anchor_overlap, anchor_len, max_unlink_dist, min_read_quality, left_anchor_mapping, right_anchor_mapping, str_start, str_end, read
):
    read_error_list = []
    if str_start - read.reference_start < min_anchor_overlap:
        read_error_list.append("too_short_left_anchor")
    if read.reference_end - str_end < min_anchor_overlap:
        read_error_list.append("too_short_right_anchor")

    # If any anchor mapping -> Check validity of anchor mapping
    if left_anchor_mapping:
        if left_anchor_mapping.mlen < min_anchor_overlap:
            read_error_list.append("bad_left_anchor_mapping")
        if anchor_len - left_anchor_mapping.q_en > max_unlink_dist:
            read_error_list.append("unlinked_left_anchor_mapping")
    else:
        read_error_list.append("no_left_anchor_mapping")

    # Check rights anchor mapping
    if right_anchor_mapping:
        if right_anchor_mapping.mlen < min_anchor_overlap:
            read_error_list.append("bad_right_anchor_mapping")
        if right_anchor_mapping.q_st > max_unlink_dist:
            read_error_list.append("unlinked_right_anchor_mapping")
    else:
        read_error_list.append("no_right_anchor_mapping")

    # Check quality and location of left and right mapping
    if left_anchor_mapping and right_anchor_mapping:
        if left_anchor_mapping.r_en < right_anchor_mapping.r_st:
            median_str_qual = median(read.query_qualities[left_anchor_mapping.r_en : right_anchor_mapping.r_st])
            if median_str_qual < min_read_quality:
                read_error_list.append("low_read_quality")
        else:
            read_error_list.append("overlapping_anchors")
    return ";".join(read_error_list)


def get_mapping(seq, aligner):
    mappings = [m for m in aligner.map(seq=seq)]

    if not mappings:
        return None

    # Return best mapping
    return max(mappings, key=lambda x: x.mapq)


# TODO: Include insertions in kmer string!
def get_kmer_string(satelites, breaks, kmer_count, seq, mapping, ref):
    # Convert reference with cigar
    converted_seq = convert_ref_with_cigar(
        ref=ref[(mapping.r_st) : (mapping.r_en)],
        cig=mapping.cigar_str,
    )

    # Trim region up and downstrem for str region
    trim_start = max(ANCHOR_LEN - mapping.q_st, 0)
    trim_end = max(ANCHOR_LEN - (len(seq) - mapping.q_en), 0)
    seq_in_region = converted_seq[trim_start:-trim_end]

    # Create kmer string
    kmers = []

    # Add case for easy looping
    breaks = breaks + [""]
    for satelite, count, br in zip(satelites, kmer_count, breaks):
        # Extract kmers
        kmers.extend([seq_in_region[i : i + len(satelite)] for i in range(0, len(satelite) * count, len(satelite))])

        # Clip kmers
        seq_in_region = seq_in_region[len(satelite) * count :]

        # Handle break
        if br != "":
            # Extract break
            kmers.append(seq_in_region[: len(br)])

            # Clip break
            seq_in_region = seq_in_region[len(br) :]

    kmer_string = "-".join(kmers)

    return kmer_string


def convert_ref_with_cigar(ref, cig):
    # Translate seq to reference index using cigar
    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cig)

    out_str = ""

    for item in cigar_matches:
        cigar_len = int(item[0])
        cigar_ops = item[1]

        # Note: I and D are swapped when looking at reference not read
        if cigar_ops in ["M", "=", "X"]:
            out_str += ref[:cigar_len]
            ref = ref[cigar_len:]
        elif cigar_ops == "D":
            ref = ref[cigar_len:]
        elif cigar_ops == "I":
            out_str += "D" * cigar_len

    return out_str


def update_cigar(ref, seq, cigar_string):
    updated_cigar = ""

    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cigar_string)

    ref_index = 0
    seq_index = 0

    for item in cigar_matches:
        count = int(item[0])
        op = item[1]

        if op == "M":
            # Split M into matchs (=) and mismatches (X)
            raw_cigar = ""
            for _ in range(count):
                if ref[ref_index] == seq[seq_index]:
                    raw_cigar += "="
                else:
                    raw_cigar += "X"

                # Move index
                ref_index += 1
                seq_index += 1

            # Compress the raw string XX=== -> 2X3= and update cigar
            compressed_cigar = compress_raw_cigar(raw_cigar)

            # Add to updated cigar
            updated_cigar += compressed_cigar

        else:
            updated_cigar += f"{count}{op}"
            # Move index
            if op == "I":
                seq_index += count
            elif op == "D":
                ref_index += count

    return "".join(updated_cigar)


def compress_raw_cigar(raw_cigar):
    compressed_cigar = ""
    count = 1
    for i in range(len(raw_cigar) - 1):
        current_base = raw_cigar[i]
        next_base = raw_cigar[i + 1]
        if current_base == next_base:
            count += 1
        else:
            compressed_cigar += f"{count}{current_base}"
            count = 1

    compressed_cigar += f"{count}{raw_cigar[-1]}"
    return compressed_cigar


# Get mapping score from cigar string
def get_mapping_score(ref, seq, cigar_string):
    matches = re.findall(r"(\d+)([MIDNSHP=X])", cigar_string)

    cigar_list = [[int(match[0]), match[1]] for match in matches]

    gap_open_penanlty = 1
    gap_extension_penanlty = 2
    mismatch_penanlty = 2
    match_score = 1

    # Get score by +1 for matches (=) and -1 mismathes (X). Affine penalty for gaps (indels)
    score = 0
    for count, operation in cigar_list:
        # Affine gap penalty: -1 for each gap open, -1 for each gap extension
        if operation == "I" or operation == "D":
            score -= gap_open_penanlty + count * gap_extension_penanlty
        elif operation == "=":
            score += count * match_score
        elif operation == "X":
            score -= count * mismatch_penanlty

    return score


def levenshtein_pct_dist(str1, str2):
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    gap_cost = 1
    mismatch_cost = 1
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if str1[i - 1] == str2[j - 1] else mismatch_cost
            dp[i][j] = min(dp[i - 1][j] + gap_cost, dp[i][j - 1] + gap_cost, dp[i - 1][j - 1] + cost)

    # Calculate as pct difference, and allow for one mismatch
    score = dp[m][n]
    score_pct = max(0, score - 1) / max(n, m)
    return score_pct


def discrete_multivariate_normal_pdf(x, mean, cov):
    grid_ranges = []
    for i in range(len(mean)):
        m = mean[i]
        sd = cov[i][i] ** 0.5
        range_min = floor(m - 5 * sd)
        range_max = ceil(m + 5 * sd) + 1
        grid_ranges.append(range(range_min, range_max))

    grid = [list(c) for c in product(*grid_ranges)]

    norm_pdf = multivariate_normal.pdf(x, mean, cov) / np.sum(multivariate_normal.pdf(grid, mean, cov))
    return norm_pdf


def call_str(read_seq, seq_aligner, ref_list, kmer_counts, s: STR, read_str_substring):
    kmer_length = np.array([len(s) for s in s.satelites])
    mappings_res = []

    best_score = -np.inf
    mapped_kmers_dict = {}

    for kmer_count, altered_ref in sorted(zip(kmer_counts, ref_list), key=lambda pair: abs(len(pair[1]) - 2 * ANCHOR_LEN - len(read_str_substring))):
        kmer_count = np.array(kmer_count)

        if mapped_kmers_dict:
            # Check if mapping is neccesary
            # Find nearest mapped kmer count
            mapped_kmers = np.array(list(mapped_kmers_dict.keys()))
            distances = np.sum(np.abs(kmer_count - mapped_kmers), axis=1)
            nearest_mapped_kmer = mapped_kmers[np.argmin(distances)]
            nearest_mapped_kmer_score = mapped_kmers_dict[tuple(nearest_mapped_kmer)]

            # NOTE: 3 is the best ber base improvment
            best_possible_improvment = np.dot(abs(nearest_mapped_kmer - kmer_count), kmer_length) * 3

            if best_score > nearest_mapped_kmer_score + best_possible_improvment:
                # If not, skip mapping
                continue

        # Map altered reference to read
        mapping = get_mapping(aligner=seq_aligner, seq=altered_ref)

        if not mapping:
            print("altered reference not mapping to read")
            continue

        # Get scores
        mapping_read_seq = read_seq[(mapping.r_st) : (mapping.r_en)]
        mapping_altered_ref = altered_ref[(mapping.q_st) : (mapping.q_en)]
        score = get_mapping_score(ref=mapping_read_seq, seq=mapping_altered_ref, cigar_string=mapping.cigar_str)

        if best_score < score:
            best_score = score

        mapped_kmers_dict[tuple(kmer_count)] = score

        res = {
            "kmer_count": kmer_count,
            "altered_ref": altered_ref,
            "mapping": mapping,
            "score": score,
        }

        # Add to list
        mappings_res.append(res)

    str_call = max(mappings_res, key=lambda x: x["score"])
    return str_call


def process_str_pattern(str_pattern):
    satelites = re.findall(SATELITE_PATTERN, str_pattern)
    breaks = re.findall(BREAK_PATTERN, str_pattern)

    # Check alternative satelite pattern (e.g. for single STR)
    if not satelites:
        alternative_satelite_pattern = r"[GCATN]+"
        satelites = re.findall(alternative_satelite_pattern, str_pattern)

    return satelites, breaks


def process_reads_in_str_region(bamfile, ref, s: STR):
    # Get region around STR
    left_anchor = str(ref[s.chrom][(s.start - ANCHOR_LEN) : s.start])
    right_anchor = str(ref[s.chrom][s.end : (s.end + ANCHOR_LEN)])

    # Get reads overlapping the region
    read_calls = []
    filterd_reads = []
    for read in bamfile.fetch(s.chrom, s.start, s.end):
        # Extract read properties
        read_seq = read.query_sequence
        qual = read.query_qualities
        phase = read.get_tag("HP") if "HP" in [tag[0] for tag in read.get_tags()] else 0

        # Create aligner from read. Note for extra_flags: 0x4000000 -> M replaced by X/= in cigar and 0x100000 -> Only forward mapping
        seq_aligner = mp.Aligner(seq=read_seq, preset="map-ont", extra_flags=0x4000000 + 0x100000)

        # Map anchors to read
        left_anchor_mapping = get_mapping(aligner=seq_aligner, seq=left_anchor)
        right_anchor_mapping = get_mapping(aligner=seq_aligner, seq=right_anchor)

        # Check if read has full STR and sufficient anchors
        read_error_flags = check_read(
            min_anchor_overlap=MIN_ANCHOR_OVERLAP,
            anchor_len=ANCHOR_LEN,
            max_unlink_dist=10,
            min_read_quality=15,
            left_anchor_mapping=left_anchor_mapping,
            right_anchor_mapping=right_anchor_mapping,
            str_start=s.start,
            str_end=s.end,
            read=read,
        )

        if read_error_flags:
            filterd_reads.append({"read_name": read.query_name, "error_flags": read_error_flags})
            continue

        # Infer str start and end in read:
        read_str_start = left_anchor_mapping.r_en + ANCHOR_LEN - left_anchor_mapping.q_en
        read_str_end = right_anchor_mapping.r_st - right_anchor_mapping.q_st

        # Extract sequence and qual from read in STR region
        read_str_substring = read_seq[read_str_start:read_str_end]
        median_str_qual = median(qual[read_str_start:read_str_end])

        # Heuristic for max: Assume the str region consists of only the satelite
        max_str_count = [ceil(len(read_str_substring) * 1.025 / len(satelite)) + 1 for satelite in s.satelites]

        # Make range for each satelite count
        str_count_ranges = []
        for i in range(0, len(max_str_count)):
            str_count_ranges.append(range(max_str_count[i] + 1))
        kmer_counts = [list(combination) for combination in product(*str_count_ranges)]

        # Heuristic for min: Look for exact occurences of satelite
        # Avoid subpatterns by making at least equal length: [CA,CAC,CACG] -> [CACA,CACCAC,CACG]
        unique_satelites = [
            {
                "unique_satelite": satelite * ceil(len(max(s.satelites, key=len)) / len(satelite)),
                "satelite": satelite,
                "repeats": ceil(len(max(s.satelites, key=len)) / len(satelite)),
            }
            for satelite in s.satelites
        ]

        unique_satelites = [dict(t) for t in {tuple(d.items()) for d in unique_satelites}]

        for us in unique_satelites:
            min_str_count = max(0, read_str_substring.count(us["unique_satelite"]) * us["repeats"] - 1)
            satelite_index = [us["satelite"] == s for s in s.satelites]
            min_filter_idx = [sum(list(compress(kmer_count, satelite_index))) >= min_str_count for kmer_count in kmer_counts]
            kmer_counts = list(compress(kmer_counts, min_filter_idx))

        # Create altered references for each kmer count combination
        altered_ref_list = [
            # Start with the left anchor
            left_anchor +
            # The satelites and breaks
            "".join(s * k + b for s, k, b in zip(s.satelites[:-1], kmer_count[:-1], s.breaks)) +
            # The last satelite and the right anchor
            s.satelites[-1] * kmer_count[-1] + right_anchor
            for kmer_count in kmer_counts
        ]

        # Filter unsensiply short/long altered references
        max_sat_len = max(len(s) for s in s.satelites)
        estimated_str_length = len(read_str_substring)
        keep_idx = [
            estimated_str_length * 0.975 < ref_str_len < estimated_str_length * 1.025
            or estimated_str_length - max_sat_len <= ref_str_len <= estimated_str_length + max_sat_len
            for ref_str_len in [len(ref) - ANCHOR_LEN * 2 for ref in altered_ref_list]
        ]
        filtered_altered_ref_list = list(compress(altered_ref_list, keep_idx))
        filtered_kmer_counts = list(compress(kmer_counts, keep_idx))

        # Filter duplicate altered refs
        new_filtered_altered_ref_list = []
        new_filtered_kmer_counts = []
        seen_alt_refs = set()

        for kmer_count, alt_ref in sorted(zip(filtered_kmer_counts, filtered_altered_ref_list), key=lambda pair: pair[0], reverse=True):
            if alt_ref not in seen_alt_refs:
                seen_alt_refs.add(alt_ref)
                new_filtered_altered_ref_list.append(alt_ref)
                new_filtered_kmer_counts.append(kmer_count)

        str_call = call_str(
            read_seq=read_seq,
            seq_aligner=seq_aligner,
            ref_list=new_filtered_altered_ref_list,
            kmer_counts=new_filtered_kmer_counts,
            s=s,
            read_str_substring=read_str_substring,
        )

        # Create kmer string
        kmer_string = get_kmer_string(
            satelites=s.satelites,
            breaks=s.breaks,
            kmer_count=str_call["kmer_count"],
            seq=str_call["altered_ref"],
            mapping=str_call["mapping"],
            ref=read_seq,
        )

        alignment_data = {
            "query_name": read.query_name,
            "locus_id": s.locus_id,
            "chrom": s.chrom,
            "str_start": s.start,
            "str_end": s.end,
            "satelite": "-".join(s.satelites),
            "reference_start": str_call["mapping"].r_st,
            "reference_end": str_call["mapping"].r_en,
            "phase": phase,
            "median_str_qual": median_str_qual,
            "kmer_count": str_call["kmer_count"],
            "kmer_count_str": "-".join([str(x) for x in str_call["kmer_count"]]),
            "score": str_call["score"],
            "kmers": kmer_string,
            "read_str_substring": read_str_substring,
        }

        # Find best call
        read_calls.append(alignment_data)
    return read_calls, filterd_reads


def call_str_haplotypes(read_calls):
    read_str_substrings = [r["read_str_substring"] for r in read_calls]
    kmer_counts = np.array([r["kmer_count"] for r in read_calls])

    # DBSCAN for initial outlier detection
    # Kmer count outliers
    clustering = DBSCAN(eps=1, min_samples=2).fit(kmer_counts)
    kmer_outlier_idx = clustering.labels_ == -1

    # String outliers
    # Create a distance matrix with pairwise Levenshtein distances
    distance_matrix = np.array([[levenshtein_pct_dist(str1, str2) for str2 in read_str_substrings] for str1 in read_str_substrings])

    # Perform clustering using DBSCAN
    string_clustering = DBSCAN(eps=0.05, min_samples=2, metric="precomputed").fit(distance_matrix)
    string_outlier_idx = string_clustering.labels_ == -1

    # Identify and annotate outliers
    good_read_calls = []
    bad_read_calls = []
    for i, read in enumerate(read_calls):
        out_list = []
        if kmer_outlier_idx[i]:
            out_list.append("kmer_count")
        if string_outlier_idx[i]:
            out_list.append("sequence")
        outlier_reasons = ",".join(out_list)

        if outlier_reasons:
            bad_read_calls.append(read | {"em_haplotype": "outlier", "outlier_reason": outlier_reasons})
        else:
            good_read_calls.append(read)

    if good_read_calls:
        good_data = np.array([r["kmer_count"] for r in good_read_calls])

        # Use median for initial values of mean and variances for robustness
        mean_h1 = np.percentile(good_data, 25, axis=0)
        mean_h2 = np.percentile(good_data, 75, axis=0)

        var_h1 = var_h2 = np.var(good_data, axis=0)

        # Set a minimum/maximum standard deviation
        min_var = 0.2**2
        cov_h1 = np.maximum(var_h1, min_var) * np.eye(len(mean_h1))
        cov_h2 = np.maximum(var_h2, min_var) * np.eye(len(mean_h2))

        pi_h1, pi_h2 = 0.51, 0.49

        # EM algorithm
        for _ in range(100):
            # E-step

            # Haplotype clusters
            pdf_h1 = pi_h1 * discrete_multivariate_normal_pdf(good_data, mean_h1, cov_h1)
            pdf_h2 = pi_h2 * discrete_multivariate_normal_pdf(good_data, mean_h2, cov_h2)

            total_pdf = pdf_h1 + pdf_h2

            gamma_h1 = pdf_h1 / total_pdf
            gamma_h2 = pdf_h2 / total_pdf

            # M-step
            n_h1 = np.sum(gamma_h1)
            n_h2 = np.sum(gamma_h2)

            pi_h1 = n_h1 / len(good_data)
            pi_h2 = n_h2 / len(good_data)

            mean_h1 = np.dot(gamma_h1, good_data) / n_h1
            mean_h2 = np.dot(gamma_h2, good_data) / n_h2
            var_h1 = np.dot(gamma_h1, (good_data - mean_h1) ** 2) / n_h1
            var_h2 = np.dot(gamma_h2, (good_data - mean_h2) ** 2) / n_h2

            # Set a minimum standard deviation
            var_h1 = np.maximum(var_h1, min_var)
            var_h2 = np.maximum(var_h2, min_var)

            # Make Covariance matrix
            cov_h1 = var_h1 * np.eye(len(mean_h1))
            cov_h2 = var_h2 * np.eye(len(mean_h2))

        # Create a list to hold results
        result_dict_h1 = {
            "em_haplotype": "h1",
            "mean": mean_h1,
            "var": var_h1,
            "n": n_h1,
            "gamma": pi_h1,
            "index": [i + 1 for i in range(len(kmer_counts[0]))],
        }

        result_dict_h2 = {
            "em_haplotype": "h2",
            "mean": mean_h2,
            "var": var_h2,
            "n": n_h2,
            "gamma": pi_h2,
            "index": [i + 1 for i in range(len(kmer_counts[0]))],
        }

        result_df = pd.concat([pd.DataFrame(result_dict_h1), pd.DataFrame(result_dict_h2)])

        # EM outliers
        pi_arr = np.array((gamma_h1, gamma_h2))
        em_label_idx = np.argmax(pi_arr, axis=0)
        em_label = np.array(("h1", "h2"))[em_label_idx]

        # Add original data, group from EM algorithm, and outlier reason
        for i, read in enumerate(good_read_calls):
            read.update({"em_haplotype": em_label[i]})
    else:
        # Create a list to hold results
        result_dict_h1 = {
            "em_haplotype": "h1",
            "mean": pd.NA,
            "var": pd.NA,
            "n": pd.NA,
            "gamma": pd.NA,
            "index": [i + 1 for i in range(len(kmer_counts[0]))],
        }

        result_dict_h2 = {
            "em_haplotype": "h2",
            "mean": pd.NA,
            "var": pd.NA,
            "n": pd.NA,
            "gamma": pd.NA,
            "index": [i + 1 for i in range(len(kmer_counts[0]))],
        }

        result_df = pd.concat([pd.DataFrame(result_dict_h1), pd.DataFrame(result_dict_h2)])

    # Create DataFrames
    df1 = pd.DataFrame(good_read_calls)
    df2 = pd.DataFrame(bad_read_calls)

    # Merge data frames
    read_result_df = pd.concat([df1, df2])

    return result_df, read_result_df


def main(args):
    # Open reference
    ref_seqs = Fasta(args.ref)

    # Open BAM file
    bamfile = pysam.AlignmentFile(args.bam, "rb")

    # Loop over JSON file
    results = pd.DataFrame()
    read_results = pd.DataFrame()

    with open(args.json, "r") as f:
        json_file = json.load(f)

    for region in json_file:
        # Unpack JSON fields

        locus_id = region["LocusId"]
        locus_structure = region["LocusStructure"]
        reference_region = region["ReferenceRegion"]
        locus_id = region["LocusId"]

        if not locus_id == "STARD7":
            True
            continue

        chrom, start, end = process_region(reference_region)

        print(f"Processing {locus_id} {locus_structure}...")

        satelites, breaks = process_str_pattern(locus_structure)
        if not satelites:
            print("No satelite pattern found in STR definition")
            continue
        s = STR(locus_id=locus_id, chrom=chrom, start=int(start), end=int(end), satelites=satelites, breaks=breaks)
        # Call STR in individual reads
        read_calls, filtered_reads = process_reads_in_str_region(bamfile, ref_seqs, s=s)

        # Call STR haplotypes
        str_result_df, str_read_result_df = call_str_haplotypes(read_calls)

        # Annotate results
        str_result_df["locus_id"] = locus_id
        str_result_df["chrom"] = s.chrom
        str_result_df["str_start"] = s.start
        str_result_df["str_end"] = s.end
        str_result_df["satelite"] = "-".join(s.satelites)

        results = pd.concat([results, str_result_df], axis=0, ignore_index=True)
        read_results = pd.concat([read_results, str_read_result_df], axis=0, ignore_index=True)

    # Write output
    results.to_csv(args.out, index=False)
    read_results.to_csv(args.reads, index=False)


def process_region(reference_region):
    if not isinstance(reference_region, list):
        reference_region = [reference_region]

    chroms, starts, ends = [], [], []
    for region in [re.match(r"(.+)\:(\d+)-(\d+)", r).groups() for r in reference_region]:
        chroms.append(region[0])
        starts.append(int(region[1]))
        ends.append(int(region[2]))

    chrom = chroms[0]
    start = min(starts)
    end = max(ends)
    return chrom, start, end


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("--ref", required=True, help="Path to the reference genome FASTA file")
    parser.add_argument("--bam", required=True, help="Path to the BAM file")
    parser.add_argument("--json", required=True, help="Path to the JSON file")
    parser.add_argument("--out", required=True, help="Output csv file og haplotype calls")
    parser.add_argument("--reads", required=True, help="Output csv of individual read info")

    args = parser.parse_args()
    main(args)
