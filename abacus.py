import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from itertools import compress, product
from math import ceil
from statistics import median
from typing import List, Tuple

import Levenshtein as ls
import mappy as mp
import numpy as np
import pandas as pd
import pysam
from pyfaidx import Fasta
from pyinstrument import Profiler
from scipy.stats import chi2, multivariate_normal
from sklearn.cluster import DBSCAN

# Constants
ANCHOR_LEN = 500
MIN_ANCHOR_OVERLAP = 125
MIN_STR_READ_QUAL = 15
MAX_UNLINK_DIST = 50

# For mapping (Minimap2)
GAP_OPEN_PENALTY = 4
GAP_EXTENSION_PENALTY = 2
MISMATCH_PENALTY = 4
MATCH_SCORE = 2

# For mapping (my implementation)
# GAP_OPEN_PENALTY = 1
# GAP_EXTENSION_PENALTY = 2
# MISMATCH_PENALTY = 2
# MATCH_SCORE = 1


@dataclass
class Locus:
    """Class for keeping data for a STR locus"""

    id: str
    chrom: str
    start: int
    end: int
    left_anchor: str
    right_anchor: str
    structure: str
    satelites: list[str]
    breaks: list[str]

    @classmethod
    def from_json_item(cls, item, ref):
        chrom, start, end = process_region(item["ReferenceRegion"])
        satelites, breaks = process_str_pattern(item["LocusStructure"])

        # Get region around STR
        left_anchor = str(ref[chrom][(start - ANCHOR_LEN) : start])
        right_anchor = str(ref[chrom][end : (end + ANCHOR_LEN)])

        return cls(
            id=item["LocusId"],
            chrom=chrom,
            start=start,
            end=end,
            structure=item["LocusStructure"],
            left_anchor=left_anchor,
            right_anchor=right_anchor,
            satelites=satelites,
            breaks=breaks,
        )

    def to_dict(self):
        return {
            "locus_id": self.id,
            "locus_chrom": self.chrom,
            "locus_start": self.start,
            "locus_end": self.end,
            "satelites_str": "-".join(self.satelites),
            "structure": self.structure,
        }


@dataclass
class STR_Read:
    """Class for keeping data for read covering a STR locus"""

    query_name: str
    query_sequence: str
    query_qualities: List[float]
    strand: str
    phase: int

    left_anchor: str
    right_anchor: str

    aligner: mp.Aligner = field(init=False)

    left_anchor_map: mp.Alignment = field(init=False)
    right_anchor_map: mp.Alignment = field(init=False)

    left_anchor_unmapped_end_length: int = field(init=False)
    right_anchor_unmapped_end_length: int = field(init=False)

    str_start: int = field(init=False)
    str_end: int = field(init=False)
    str_sequence: str = field(init=False)
    str_qualities: List[float] = field(init=False)
    median_str_quality: float = field(init=False)

    error_flags: str = field(init=False)

    @classmethod
    def from_aligment(cls, alignment: pysam.AlignedSegment, left_anchor: str, right_anchor: str):
        query_name = alignment.query_name if alignment.query_name is not None else ""
        query_sequence = alignment.query_alignment_sequence if alignment.query_alignment_sequence is not None else ""
        query_qualities = list(alignment.query_alignment_qualities) if alignment.query_alignment_qualities is not None else []
        strand = ("rev" if alignment.is_reverse else "fwd") if alignment.is_reverse is not None else ""
        phase = int(alignment.get_tag("HP")) if "HP" in [tag[0] for tag in alignment.get_tags()] else 0

        return cls(
            query_name=query_name,
            query_sequence=query_sequence,
            query_qualities=query_qualities,
            strand=strand,
            phase=phase,
            left_anchor=left_anchor,
            right_anchor=right_anchor,
        )

    def __post_init__(self):
        # Create aligner from read. Note for extra_flags: 0x4000000 -> M replaced by X/= in cigar and 0x100000 -> Only forward mapping
        self.aligner = mp.Aligner(
            seq=self.query_sequence,
            preset="map-ont",
            extra_flags=0x4000000 + 0x100000,
            best_n=1,
            scoring=[MATCH_SCORE, MISMATCH_PENALTY, GAP_OPEN_PENALTY, GAP_EXTENSION_PENALTY],
        )

        # Map anchors
        self.left_anchor_map = self.map(self.left_anchor)
        self.right_anchor_map = self.map(self.right_anchor)

        # Get start and end of STR
        left_anchor_end = self.left_anchor_map.r_en if self.left_anchor_map is not None else 0
        self.left_anchor_unmapped_end_length = ANCHOR_LEN - (self.left_anchor_map.q_en if self.left_anchor_map is not None else 0)

        right_anchor_start = self.right_anchor_map.r_st if self.right_anchor_map is not None else 0
        self.right_anchor_unmapped_end_length = self.right_anchor_map.q_st if self.right_anchor_map is not None else 0

        self.str_start = left_anchor_end + self.left_anchor_unmapped_end_length
        self.str_end = right_anchor_start - self.right_anchor_unmapped_end_length
        self.str_sequence = self.query_sequence[self.str_start : self.str_end]
        self.str_qualities = self.query_qualities[self.str_start : self.str_end]
        self.median_str_quality = median(self.str_qualities) if self.str_qualities else 0

        # Check if read has full STR and sufficient anchors
        self.check_read()

    def map(self, seq):
        mapping = next(self.aligner.map(seq=seq), None)
        return mapping

    def check_read(self):
        errors = []

        if not self.left_anchor_map:
            errors.append("missing_left_anchor")
        else:
            if self.left_anchor_map.mlen < MIN_ANCHOR_OVERLAP:
                errors.append("bad_left_anchor")
            if ANCHOR_LEN - self.left_anchor_map.q_en > MAX_UNLINK_DIST:
                errors.append("unlinked_right_anchor")

        if not self.right_anchor_map:
            errors.append("missing_right_anchor")
        else:
            if self.right_anchor_map.mlen < MIN_ANCHOR_OVERLAP:
                errors.append("bad_right_anchor")
            if self.right_anchor_map.q_st > MAX_UNLINK_DIST:
                errors.append("unlinked_right_anchor")

        if self.left_anchor_map and self.right_anchor_map:
            if self.right_anchor_map.r_st < self.left_anchor_map.r_en:
                errors.append("overlapping_anchors")
            else:
                read_str_start = self.left_anchor_map.r_en + ANCHOR_LEN - self.left_anchor_map.q_en
                read_str_end = self.right_anchor_map.r_st - self.right_anchor_map.q_st
                str_qual = self.query_qualities[read_str_start:read_str_end]
                median_str_qual = median(str_qual)
                if median_str_qual < MIN_STR_READ_QUAL:
                    errors.append("low_read_quality")

        self.error_flags = ",".join(errors)

    def to_dict(self):
        return {
            "query_name": self.query_name,
            "strand": self.strand,
            "phase": self.phase,
            "str_start": self.str_start,
            "str_end": self.str_end,
            "left_anchor_unmapped_end_length": self.left_anchor_unmapped_end_length,
            "right_anchor_unmapped_end_length": self.right_anchor_unmapped_end_length,
            "read_str_sequence": self.str_sequence,
            "read_str_qualities": self.str_qualities,
            "error_flags": self.error_flags,
            "median_str_qual": self.median_str_quality,
        }


@dataclass
class STR_Candidate:
    """Class for keeping data for STR candidate"""

    kmer_count: list[int]
    altered_ref: str


# TODO: Include insertions in kmer string!
def get_kmer_string(satelites, breaks, kmer_count, seq, mapping, ref) -> Tuple[str, str]:
    # Convert reference with cigar
    synced_seq_list = sync_ref_with_cigar(
        ref=ref[(mapping.r_st) : (mapping.r_en)],
        cig=mapping.cigar_str,
    )

    # Trim region up and downstrem for str region
    trim_start = max(ANCHOR_LEN - mapping.q_st, 0)
    trim_end = max(ANCHOR_LEN - (len(seq) - mapping.q_en), 0)
    seq_in_region = synced_seq_list[trim_start:-trim_end]

    # Create kmer string
    obs_kmers = []
    exp_kmers = []

    # Add case for easy looping
    satelites_loop = satelites + [""]
    kmer_count_loop = np.concatenate([kmer_count, np.array([0])])
    for sat, cnt, brk in zip(satelites_loop, kmer_count_loop, breaks):
        if brk != "":
            # Add observed break
            obs_kmers.append("".join(seq_in_region[: len(brk)]))

            # Add expected break
            exp_kmers.append(brk)

            # Clip break
            seq_in_region = seq_in_region[len(brk) :]

        if sat != "":
            # Add observed kmers
            obs_kmers.extend(["".join(seq_in_region[i : i + len(sat)]) for i in range(0, len(sat) * cnt, len(sat))])

            # Add expected break
            exp_kmers.extend([sat] * cnt)

            # Clip kmers
            seq_in_region = seq_in_region[len(sat) * cnt :]

    return "-".join(obs_kmers), "-".join(exp_kmers)


def sync_ref_with_cigar(ref: str, cig: str) -> List[str]:
    # Translate seq to reference index using cigar
    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cig)

    res = []

    for item in cigar_matches:
        cigar_len = int(item[0])
        cigar_ops = item[1]

        # Note: I and D are swapped when looking at reference not read
        if cigar_ops in ["M", "=", "X"]:
            res.extend(list(ref[:cigar_len]))
            ref = ref[cigar_len:]
        elif cigar_ops == "D":
            res[-1] += ref[:cigar_len]
            ref = ref[cigar_len:]
        elif cigar_ops == "I":
            res.extend([""] * cigar_len)

    return res


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
def get_mapping_score(cigar_string):
    matches = re.findall(r"(\d+)([MIDNSHP=X])", cigar_string)

    cigar_list = [[int(match[0]), match[1]] for match in matches]

    # Get score by +1 for matches (=) and -1 mismathes (X). Affine penalty for gaps (indels)
    score = 0
    for count, operation in cigar_list:
        if operation in ["I", "D"]:
            # Affine gap penalty
            score -= GAP_OPEN_PENALTY + count * GAP_EXTENSION_PENALTY
        elif operation == "=":
            score += count * MATCH_SCORE
        elif operation == "X":
            score -= count * MISMATCH_PENALTY

    return score


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


def get_best_str_candidate(read: STR_Read, candidate_list: List[STR_Candidate], locus: Locus):
    kmer_length = np.array([len(s) for s in locus.satelites])

    best_candidate = None
    mapped_kmers_dict = {}

    for c in sorted(candidate_list, key=lambda c: abs((len(c.altered_ref) - 2 * ANCHOR_LEN) - len(read.str_sequence))):
        kmer_count_array = np.array(c.kmer_count)

        # Check if mapping is neccesary
        if best_candidate and mapped_kmers_dict:
            # Check if mapping is neccesary
            # Find nearest mapped kmer count
            mapped_kmers = np.array(list(mapped_kmers_dict.keys()))
            distances = np.sum(np.abs(kmer_count_array - mapped_kmers), axis=1)
            nearest_mapped_kmer = mapped_kmers[np.argmin(distances)]
            nearest_mapped_kmer_score = mapped_kmers_dict[tuple(nearest_mapped_kmer)]

            # NOTE: 4 is the best ber base improvment
            best_possible_improvment = np.dot(abs(nearest_mapped_kmer - kmer_count_array), kmer_length) * 4

            if best_candidate["score"] > nearest_mapped_kmer_score + best_possible_improvment:
                # If not, skip mapping
                continue

        # Map altered reference to read
        mapping = read.map(c.altered_ref)

        if not mapping:
            print("altered reference not mapping to read")
            continue

        # Get scores
        score = get_mapping_score(cigar_string=mapping.cigar_str)

        # Add to dict
        mapped_kmers_dict[tuple(kmer_count_array)] = score

        # Update best mapping
        if not best_candidate or best_candidate["score"] <= score and len(best_candidate["altered_ref"]) > len(c.altered_ref):
            best_candidate = {
                "kmer_count": c.kmer_count,
                "altered_ref": c.altered_ref,
                "mapping": mapping,
                "score": score,
            }

    return best_candidate


def process_str_pattern(str_pattern):
    SATELITE_PATTERN = r"(?<=\()(?:[GCATN]+)(?=\)[\*\+])"
    BREAK_PATTERN = r"(?<=\)[\*\+])(?:[GCATN]*)(?=\()"

    satelites = re.findall(SATELITE_PATTERN, str_pattern)
    breaks = re.findall(BREAK_PATTERN, str_pattern)
    prefix = str_pattern[: str_pattern.find("(")]
    suffix = str_pattern[str_pattern.rfind(")") + 2 :]

    return satelites, [prefix] + breaks + [suffix]


def process_reads_in_str_region(bamfile, locus: Locus):
    # Get reads overlapping the region
    read_calls = []
    filtered_reads = []
    for alignment in bamfile.fetch(locus.chrom, locus.start, locus.end):
        read = STR_Read.from_aligment(
            alignment=alignment,
            left_anchor=locus.left_anchor,
            right_anchor=locus.right_anchor,
        )

        if read.error_flags:
            filtered_reads.append(read.to_dict() | locus.to_dict())
            continue

        candidate_list = get_str_candidates(locus, read.str_sequence)

        str_call = get_best_str_candidate(
            read=read,
            candidate_list=candidate_list,
            locus=locus,
        )

        # Create kmer string
        observed_kmer_string, expected_kmer_string = get_kmer_string(
            satelites=locus.satelites,
            breaks=locus.breaks,
            kmer_count=str_call["kmer_count"],
            seq=str_call["altered_ref"],
            mapping=str_call["mapping"],
            ref=read.query_sequence,
        )

        alignment_data = (
            read.to_dict()
            | locus.to_dict()
            | {
                "kmer_count": str_call["kmer_count"],
                "kmer_count_str": "-".join([str(x) for x in str_call["kmer_count"]]),
                "score": str_call["score"],
                "kmer_str": observed_kmer_string,
                "expected_kmer_str": expected_kmer_string,
            }
        )

        # Find best call
        read_calls.append(alignment_data)
    return read_calls, filtered_reads


def get_str_candidates(locus, read_str_sequence) -> List[STR_Candidate]:
    # STEP 1: Heuristics for min and max kmer count

    # Heuristic for max: Assume the str region consists of only the satelite
    max_str_count = [ceil(len(read_str_sequence) * 1.025 / len(satelite)) + 1 for satelite in locus.satelites]

    # Make range for each satelite count
    str_count_ranges = [range(x + 1) for x in max_str_count]
    kmer_counts = [list(combination) for combination in product(*str_count_ranges)]

    # Heuristic for min: Look for exact occurences of satelite
    # Avoid subpatterns by making at least equal length: [CA,CAC,CACG] -> [CACA,CACCAC,CACG]
    # Make more robust by copying: [CA,CAC,CACG] -> [CACACACA,CACCACCACCAC,CACGCACG] (2x)
    # Replace "N" -> "." for regex
    unique_satelites = [
        {
            "satelite": satelite,
            "unique_satelite_regex": (satelite * n).replace("N", "."),
            "copies": n,
        }
        for satelite in locus.satelites
        if (n := ceil(len(max(locus.satelites, key=len)) / len(satelite)) * 2)
    ]

    for us in unique_satelites:
        unique_satelite_count = len(re.findall(us["unique_satelite_regex"], read_str_sequence))
        min_str_count = unique_satelite_count * us["copies"]
        satelite_index = [us["satelite"] == s for s in locus.satelites]
        min_filter_mask = [sum(list(compress(kmer_count, satelite_index))) >= min_str_count for kmer_count in kmer_counts]
        kmer_counts = list(compress(kmer_counts, min_filter_mask))

    # STEP 2: Create altered references
    altered_ref_list = [
        # Start with the left anchor
        locus.left_anchor +
        # The satelites and breaks
        "".join(b + s * k for b, s, k in zip(locus.breaks[:-1], locus.satelites, kmer_count)) +
        # The last satelite and the right anchor
        locus.breaks[-1] + locus.right_anchor
        for kmer_count in kmer_counts
    ]

    candidate_list = [STR_Candidate(k, a) for k, a in zip(kmer_counts, altered_ref_list)]

    # STEP 3: Filtering

    # Filter out unsensible short/long altered references
    max_sat_len = max(len(s) for s in locus.satelites)
    estimated_str_length = len(read_str_sequence)
    keep_idx = [
        estimated_str_length * 0.95 < ref_str_len < estimated_str_length * 1.05
        or estimated_str_length - 2 * max_sat_len <= ref_str_len <= estimated_str_length + 2 * max_sat_len
        for ref_str_len in [len(c.altered_ref) - ANCHOR_LEN * 2 for c in candidate_list]
    ]
    candidate_list = list(compress(candidate_list, keep_idx))

    # Filter duplicate altered refs
    filtered_candidate_list = []
    seen_alt_refs = set()

    for c in sorted(candidate_list, key=lambda c: c.kmer_count, reverse=True):
        if c.altered_ref not in seen_alt_refs:
            seen_alt_refs.add(c.altered_ref)
            filtered_candidate_list.append(c)

    return filtered_candidate_list


def call_haplotypes(read_calls: List):
    kmer_dim = len(read_calls[0]["kmer_count"])

    # DBSCAN for initial outlier detection
    good_read_calls = read_calls
    outlier_read_calls = []

    # String outliers with pairwise Levenshtein distances
    read_str_sequences = [r["read_str_sequence"] for r in good_read_calls]
    pairwise_sequence_dist = np.array(
        [[ls.distance(str1, str2) / max(len(str1), len(str2)) for str2 in read_str_sequences] for str1 in read_str_sequences]
    )
    string_clustering = DBSCAN(eps=0.05, min_samples=2, metric="precomputed").fit(pairwise_sequence_dist)
    string_outlier_mask = string_clustering.labels_ == -1

    # Identify and annotate outliers
    outlier_read_calls.extend(
        [read | {"em_haplotype": "outlier", "outlier_reason": "sequence_errors"} for read in list(compress(good_read_calls, string_outlier_mask))]
    )
    good_read_calls = list(compress(good_read_calls, ~string_outlier_mask))

    # Kmer count outliers
    kmer_counts = np.array([r["kmer_count"] for r in good_read_calls])
    kmer_dist = np.array([[np.sum(np.abs(k1 - k2)) for k2 in kmer_counts] for k1 in kmer_counts])
    kmer_clustering = DBSCAN(eps=kmer_dim + 1, min_samples=2, metric="precomputed").fit(kmer_dist)
    kmer_outlier_mask = kmer_clustering.labels_ == -1

    # Identify and annotate outliers
    outlier_read_calls.extend(
        [read | {"em_haplotype": "outlier", "outlier_reason": "unusual_kmer_count"} for read in list(compress(good_read_calls, kmer_outlier_mask))]
    )
    good_read_calls = list(compress(good_read_calls, ~kmer_outlier_mask))
    good_read_calls = [read | {"em_haplotype": pd.NA, "outlier_reason": pd.NA} for read in good_read_calls]

    haplotyping_df, test_summary_df = run_em_algo(good_read_calls, kmer_dim)

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


def process_region(reference_region) -> Tuple[str, int, int]:
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


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Your script description.")
    parser.add_argument("--ref", required=True, help="Path to the reference genome FASTA file")
    parser.add_argument("--bam", required=True, help="Path to the BAM file")
    parser.add_argument("--json", required=True, help="Path to the JSON file")
    parser.add_argument("--out", required=True, help="Output csv file og haplotype calls")
    parser.add_argument("--report_html", required=True, help="HTML report")
    parser.add_argument("--reads", required=True, help="Output csv of individual read info")
    parser.add_argument("--filtered_reads", required=True, help="Output csv of individual read info for filtered reads")
    parser.add_argument("--summary", required=True, help="Output csv of summary stats")
    args = parser.parse_args()

    # Load reference FASTA
    ref = Fasta(args.ref)

    # Load loci data from JSON
    with open(args.json, "r") as json_file:
        loci_data = json.load(json_file)

    # Open BAM file
    bamfile = pysam.AlignmentFile(args.bam, "rb")

    # Initialize output dataframes
    haplotyping_df = pd.DataFrame()
    read_calls_df = pd.DataFrame()
    filtered_reads_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    # Process each locus
    with Profiler(interval=0.1) as profiler:
        for locus_data in loci_data:
            # Process locus
            locus = Locus.from_json_item(locus_data, ref)

            # if locus.id not in ["AR", "HTT", "RFC1_alt", "ATXN8OS", "CNBP"]:
            # if locus.id not in ["AR", "HTT", "CNBP", "FMR1", "FGF14", "DMPK"]:
            if locus.id not in ["FGF14"]:
                var = True
                continue

            print(f"Processing {locus.id} {locus.structure}...")

            if not locus.satelites:
                print("No valid satelite pattern found in STR definition")
                continue

            # Call STR in individual reads
            read_calls, filtered_reads = process_reads_in_str_region(bamfile=bamfile, locus=locus)
            filtered_reads_res_df = pd.DataFrame(filtered_reads)

            # Call STR haplotypes
            haplotyping_res_df, read_calls_res_df, test_summary_res_df = call_haplotypes(read_calls)

            # Annotate results with locus info
            for df in [haplotyping_res_df, test_summary_res_df, filtered_reads_res_df]:
                for k, v in locus.to_dict().items():
                    df[k] = v

            # Add satelite information to haplotype results
            haplotyping_res_df["satelite"] = haplotyping_res_df["idx"].apply(lambda x: locus.satelites[x])

            # Concatenate results
            haplotyping_df = pd.concat([haplotyping_df, haplotyping_res_df], axis=0, ignore_index=True)
            read_calls_df = pd.concat([read_calls_df, read_calls_res_df], axis=0, ignore_index=True)
            summary_df = pd.concat([summary_df, test_summary_res_df], axis=0, ignore_index=True)
            filtered_reads_df = pd.concat([filtered_reads_df, filtered_reads_res_df], axis=0, ignore_index=True)

    profiler.print()

    # Write output
    with open(args.reads, "w") as f:
        read_calls_df.to_csv(f, index=False)
    with open(args.filtered_reads, "w") as f:
        filtered_reads_df.to_csv(f, index=False)
    with open(args.out, "w") as f:
        haplotyping_df.to_csv(f, index=False)
    with open(args.summary, "w") as f:
        summary_df.to_csv(f, index=False)

    # Render report
    print(f"Render report...")

    report_name = os.path.basename(args.report_html)
    report_dir = os.path.realpath(os.path.dirname(args.report_html))

    report_proc = subprocess.run(
        [
            "Rscript",
            "-e",
            f"""
                rmarkdown::render('report.Rmd', \
                    output_file='{report_name}', \
                    output_dir='{report_dir}', \
                    intermediates_dir='{report_dir}', \
                    params=list( \
                        reads_csv = '{args.reads}', \
                        filtered_reads_csv = '{args.filtered_reads}', \
                        clustering_summary_csv = '{args.out}', \
                        test_summary_csv = '{args.summary}' \
                    ))""",
        ],
        capture_output=True,
        text=True,
    )

    print(report_proc.stdout)
    print(report_proc.stderr)


if __name__ == "__main__":
    main()
