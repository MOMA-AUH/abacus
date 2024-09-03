import re
from dataclasses import dataclass, field
from itertools import compress, product
from math import ceil
from statistics import median
from typing import List, Tuple

import mappy as mp
import numpy as np

from abacus.constants import (
    ANCHOR_LEN,
    GAP_EXTENSION_PENALTY,
    GAP_OPEN_PENALTY,
    LONG_GAP_EXTENSION_PENALTY,
    LONG_GAP_OPEN_PENALTY,
    MATCH_SCORE,
    MAX_UNLINK_DIST,
    MIN_ANCHOR_OVERLAP,
    MIN_STR_READ_QUAL,
    MISMATCH_PENALTY,
    N_MISMATCH_PENALTY,
)
from abacus.locus import Locus


@dataclass
class STR_Candidate:
    """Class for keeping data for STR candidate"""

    kmer_count: list[int]
    altered_ref: str
    left_linker_length: int
    right_linker_length: int
    str_start: int
    str_end: int


@dataclass
class STR_Call(STR_Candidate):
    """Class for keeping data for STR call"""

    mapping: mp.Alignment
    score: float


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
    def from_aligment(cls, alignment, left_anchor: str, right_anchor: str):
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
            scoring=[
                MATCH_SCORE,
                MISMATCH_PENALTY,
                GAP_OPEN_PENALTY,
                GAP_EXTENSION_PENALTY,
                LONG_GAP_OPEN_PENALTY,
                LONG_GAP_EXTENSION_PENALTY,
                N_MISMATCH_PENALTY,
            ],
        )

        # Map anchors
        self.left_anchor_map = self.map(self.left_anchor)
        self.right_anchor_map = self.map(self.right_anchor)

        # Get start and end of STR
        left_anchor_end = self.left_anchor_map.r_en if self.left_anchor_map is not None else 0
        self.left_anchor_unmapped_end_length = ANCHOR_LEN - (self.left_anchor_map.q_en if self.left_anchor_map is not None else 0)

        right_anchor_start = self.right_anchor_map.r_st if self.right_anchor_map is not None else 0
        self.right_anchor_unmapped_end_length = self.right_anchor_map.q_st if self.right_anchor_map is not None else 0

        # self.str_start = left_anchor_end + self.left_anchor_unmapped_end_length
        # self.str_end = right_anchor_start - self.right_anchor_unmapped_end_length
        self.str_start = left_anchor_end
        self.str_end = right_anchor_start
        self.str_sequence = self.query_sequence[self.str_start : self.str_end]
        self.str_qualities = self.query_qualities[self.str_start : self.str_end]
        self.median_str_quality = median(self.str_qualities) if self.str_qualities else 0

        # Check if read has full STR and sufficient anchors
        self.check_read()

    def map(self, seq):
        return next(self.aligner.map(seq=seq), None)

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


# TODO: Include insertions in kmer string!
def get_kmer_string(locus: Locus, read: STR_Read, call: STR_Call) -> Tuple[str, str]:
    satellites = locus.satellites
    breaks = locus.breaks
    kmer_count = call.kmer_count
    seq = call.altered_ref

    mapping = call.mapping
    ref = read.query_sequence

    # Convert reference with cigar
    synced_seq_list = sync_ref_with_cigar(
        ref=ref[(mapping.r_st) : (mapping.r_en)],
        cig=mapping.cigar_str,
    )

    # Trim region up and downstrem for str region
    # TODO: Make this more intuitive
    seq_in_region = synced_seq_list[
        (call.str_start - mapping.q_st - call.left_linker_length) : (call.str_end - mapping.q_st + call.right_linker_length)
    ]

    # Create kmer string
    obs_kmers = []
    exp_kmers = []

    # Take care of left linker
    if call.left_linker_length > 0:
        obs_kmers.extend(seq_in_region[: call.left_linker_length])
        exp_kmers.extend(["N"] * call.left_linker_length)

        seq_in_region = seq_in_region[call.left_linker_length :]

    # Add case for easy looping
    satellites_loop = satellites + [""]
    kmer_count_loop = np.concatenate([kmer_count, np.array([0])])
    for sat, cnt, brk in zip(satellites_loop, kmer_count_loop, breaks):
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

    # Take care of right linker
    if call.right_linker_length > 0:
        obs_kmers.extend(seq_in_region[: call.right_linker_length])
        exp_kmers.extend(["N"] * call.right_linker_length)

        seq_in_region = seq_in_region[call.right_linker_length :]

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


def trim_anchor_for_messy_bases(seq, cigar, min_matching_bases, trim_start, trim_end):
    matches = re.findall(r"(\d+)([MIDNSHP=X])", cigar)

    cigar_list = [[int(match[0]), match[1]] for match in matches]

    if trim_end:
        for count, operation in reversed(cigar_list):
            if operation == "=" and count >= min_matching_bases:
                break
            seq = seq[:-count]

    if trim_start:
        for count, operation in cigar_list:
            if operation == "=" and count >= min_matching_bases:
                break
            seq = seq[count:]

    return seq


def get_str_candidates(locus: Locus, read: STR_Read) -> List[STR_Candidate]:
    # STEP 0: Get N's for linker region
    # Trim
    trimmed_left_anchor = trim_anchor_for_messy_bases(
        locus.left_anchor[: read.left_anchor_map.q_en],
        read.left_anchor_map.cigar_str,
        min_matching_bases=10,
        trim_start=False,
        trim_end=True,
    )
    trimmed_right_anchor = trim_anchor_for_messy_bases(
        locus.right_anchor[read.right_anchor_map.q_st :],
        read.right_anchor_map.cigar_str,
        min_matching_bases=10,
        trim_start=True,
        trim_end=False,
    )

    trimmed_left_anchor_map = read.map(trimmed_left_anchor)
    trimmed_right_anchor_map = read.map(trimmed_right_anchor)

    if not trimmed_left_anchor_map or not trimmed_right_anchor_map:
        return []

    # Estimate STR sequence
    estimated_str_sequence = read.query_sequence[trimmed_left_anchor_map.r_en : trimmed_right_anchor_map.r_st]

    # Find first occurence of first satellite after left anchor
    left_n_linker = ""
    if len(trimmed_left_anchor) < ANCHOR_LEN or trimmed_left_anchor_map.q_en < ANCHOR_LEN:
        seq_minus_left_anchor = read.query_sequence[trimmed_left_anchor_map.r_en :]
        start_pos_of_first_satellite = seq_minus_left_anchor.find(locus.satellites[0])
        start_pos_of_first_break = seq_minus_left_anchor.find(locus.breaks[0]) if locus.breaks[0] != "" else 1000000
        left_n_linker = "N" * (min(start_pos_of_first_satellite, start_pos_of_first_break))

    # Find last occurence of last satellite before right anchor
    right_n_linker = ""
    if len(trimmed_right_anchor) < ANCHOR_LEN or trimmed_right_anchor_map.q_st > 0:
        seq_minus_right_anchor = read.query_sequence[: trimmed_right_anchor_map.r_st]
        end_pos_of_last_satellite = seq_minus_right_anchor.rfind(locus.satellites[-1]) + len(locus.satellites[-1])
        end_pos_of_last_break = seq_minus_right_anchor.rfind(locus.breaks[-1]) if locus.breaks[-1] != "" else -1
        right_n_linker = "N" * (len(seq_minus_right_anchor) - max(end_pos_of_last_satellite, end_pos_of_last_break) + 1)

    # STEP 1: Heuristics for min and max kmer count

    # Heuristic for max: Assume the str region consists of only the satellite
    max_str_count = [ceil(len(estimated_str_sequence) * 1.025 / len(satellite)) + 1 for satellite in locus.satellites]

    # Make range for each satellite count
    str_count_ranges = [range(x + 1) for x in max_str_count]
    kmer_counts = [list(combination) for combination in product(*str_count_ranges)]

    # Heuristic for min: Look for exact occurences of satellite
    # Avoid subpatterns by making at least equal length: [CA,CAC,CACG] -> [CACA,CACCAC,CACG]
    # Make more robust by copying: [CA,CAC,CACG] -> [CACACACA,CACCACCACCAC,CACGCACG] (2x)
    # Replace "N" -> "." for regex
    unique_satellites = [
        {
            "satellite": satellite,
            "unique_satellite_regex": (satellite * n).replace("N", "."),
            "copies": n,
        }
        for satellite in locus.satellites
        if (n := ceil(len(max(locus.satellites, key=len)) / len(satellite)) * 2)
    ]

    for us in unique_satellites:
        unique_satellite_count = len(re.findall(us["unique_satellite_regex"], estimated_str_sequence))
        min_str_count = unique_satellite_count * us["copies"]
        satellite_index = [us["satellite"] == s for s in locus.satellites]
        min_filter_mask = [sum(list(compress(kmer_count, satellite_index))) >= min_str_count for kmer_count in kmer_counts]
        kmer_counts = list(compress(kmer_counts, min_filter_mask))

    # STEP 2: Create altered references
    altered_ref_list = [
        # Start with the left anchor
        trimmed_left_anchor
        +
        # N linker
        left_n_linker
        +
        # The satellites and breaks
        "".join(b + s * k for b, s, k in zip(locus.breaks[:-1], locus.satellites, kmer_count))
        +
        # The last satellite and the right anchor
        locus.breaks[-1]
        +
        # N linker
        right_n_linker
        +
        # End
        trimmed_right_anchor
        for kmer_count in kmer_counts
    ]

    candidate_list = [
        STR_Candidate(
            kmer_count=k,
            altered_ref=a,
            left_linker_length=len(left_n_linker),
            right_linker_length=len(right_n_linker),
            str_start=len(trimmed_left_anchor) + len(left_n_linker),
            str_end=len(a) - len(trimmed_right_anchor) - len(right_n_linker),
        )
        for k, a in zip(kmer_counts, altered_ref_list)
    ]

    # STEP 3: Filtering

    # Filter out unsensible short/long altered references
    max_sat_len = max(len(s) for s in locus.satellites)
    estimated_str_length = len(estimated_str_sequence)
    keep_idx = [
        estimated_str_length * 0.90 < ref_str_len < estimated_str_length * 1.10
        or estimated_str_length - 2 * max_sat_len <= ref_str_len <= estimated_str_length + 2 * max_sat_len
        for ref_str_len in [len(c.altered_ref) - len(trimmed_left_anchor) - len(trimmed_right_anchor) for c in candidate_list]
    ]
    candidate_list_1 = list(compress(candidate_list, keep_idx))

    # Filter duplicate altered refs
    filtered_candidate_list = []
    seen_alt_refs = set()

    for c in sorted(candidate_list_1, key=lambda c: c.kmer_count, reverse=True):
        if c.altered_ref not in seen_alt_refs:
            seen_alt_refs.add(c.altered_ref)
            filtered_candidate_list.append(c)

    return filtered_candidate_list


def get_best_str_candidate(read: STR_Read, candidate_list: List[STR_Candidate], locus: Locus) -> STR_Call:
    kmer_length = np.array([len(s) for s in locus.satellites])

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

            # Calculate best possible improvment
            best_per_base_improvemnt = max(
                MATCH_SCORE + GAP_EXTENSION_PENALTY + GAP_OPEN_PENALTY,
                MATCH_SCORE + MISMATCH_PENALTY,
            )
            best_possible_improvment = np.dot(abs(nearest_mapped_kmer - kmer_count_array), kmer_length) * best_per_base_improvemnt

            # Skip mapping if a better mapping is not possible
            if best_candidate.score > nearest_mapped_kmer_score + best_possible_improvment:
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
        if (
            not best_candidate  # If no candiate has been found
            or best_candidate.score < score  # If score is better
            or (
                best_candidate.score == score and len(best_candidate.altered_ref) > len(c.altered_ref)
            )  # If score is equal and mapping is shorter (tie break)
        ):
            best_candidate = STR_Call(
                altered_ref=c.altered_ref,
                kmer_count=c.kmer_count,
                str_start=c.str_start,
                str_end=c.str_end,
                left_linker_length=c.left_linker_length,
                right_linker_length=c.right_linker_length,
                mapping=mapping,
                score=score,
            )

    if best_candidate is None:
        raise Exception("No candidate found")

    return best_candidate


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

        candidate_list = get_str_candidates(locus, read)

        call = get_best_str_candidate(
            read=read,
            candidate_list=candidate_list,
            locus=locus,
        )

        # Create kmer string
        observed_kmer_string, expected_kmer_string = get_kmer_string(
            locus=locus,
            read=read,
            call=call,
        )

        alignment_data = (
            read.to_dict()
            | locus.to_dict()
            | {
                "kmer_count": call.kmer_count,
                "kmer_count_str": "-".join([str(x) for x in call.kmer_count]),
                "score": call.score,
                "kmer_str": observed_kmer_string,
                "expected_kmer_str": expected_kmer_string,
            }
        )

        # Find best call
        read_calls.append(alignment_data)
    return read_calls, filtered_reads
