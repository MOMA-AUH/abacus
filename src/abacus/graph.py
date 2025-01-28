from __future__ import annotations

import itertools
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from statistics import median

import numpy as np
import pysam

from abacus.config import config
from abacus.constants import AMBIGUOUS_BASES_DICT
from abacus.locus import Locus
from abacus.logging import logger


def sync_cigar(cigar: str) -> list[str]:
    """Sync CIGAR string to a list of operations"""
    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cigar)

    res_cigar: list[str] = []

    for item in cigar_matches:
        cigar_len = int(item[0])
        cigar_ops = item[1]

        if cigar_ops in ["M", "=", "X", "D"]:
            # For match, equal, mismatch, and deletion: add operations 1:1
            res_cigar.extend([cigar_ops] * cigar_len)
        elif cigar_ops == "I":
            # For insertion: mark the previous position with insertion
            res_cigar[-1] = cigar_ops

    return res_cigar


def sync_with_cigar(input_list: list, cig: str) -> list[list]:
    """Sync input list with CIGAR string"""
    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cig)

    res_list: list[list] = []

    for item in cigar_matches:
        cigar_len = int(item[0])
        cigar_ops = item[1]

        if cigar_ops in ["M", "=", "X"]:
            # For match, equal and mismatch: add input list 1:1
            res_list.extend([[itm] for itm in input_list[:cigar_len]])
            # Trim "used" input list
            input_list = input_list[cigar_len:]
        elif cigar_ops == "D":
            # For deletion: add empty elements
            res_list.extend([[] for _ in range(cigar_len)])
        elif cigar_ops == "I":
            # For insertion: add elements to the previous position
            res_list[-1].extend(input_list[:cigar_len])
            # Trim "used" input list
            input_list = input_list[cigar_len:]

    return res_list


def get_5mc_modification_probs(alignment: pysam.AlignedSegment, sequence: str) -> str:
    mod_5mc_probs = [0.0] * len(sequence)

    mod_5mc = next((v for k, v in alignment.modified_bases.items() if k[0] == "C" and k[2] == "m"), None) if alignment.modified_bases else None
    if mod_5mc is not None:
        for pos, scaled_prob in mod_5mc:
            # Scale the probability to 0-1 (0-255 -> 0-1)
            prob = scaled_prob / 255
            mod_5mc_probs[pos] = prob

    # Convert methylation to string using 10 bins (printable ASCII characters, 33 and up)
    return "".join([chr(int(m * 10) + 33) for m in mod_5mc_probs])


@dataclass
class Read:
    name: str
    sequence: str
    qualities: list[int]
    mod_5mc_probs: str

    strand: str

    locus: Locus

    def quality_string(self) -> str:
        return "".join([chr(q + 33) for q in self.qualities])

    def to_fastq(self) -> str:
        return f"@{self.name}\n{self.sequence}\n+\n{self.quality_string()}\n"

    @classmethod
    def from_aligment(cls, alignment: pysam.AlignedSegment, locus: Locus) -> Read:
        name = alignment.query_name or ""
        sequence = alignment.query_sequence or ""
        qualities = [int(q) for q in alignment.query_qualities or []]
        strand = "-" if alignment.is_reverse else "+"
        mod_5mc_probs = get_5mc_modification_probs(alignment, sequence)

        return cls(
            name=name,
            sequence=sequence,
            qualities=qualities,
            mod_5mc_probs=mod_5mc_probs,
            strand=strand,
            locus=locus,
        )


@dataclass
class FilteredRead(Read):
    error_flags: str

    def to_dict(self) -> dict:
        return {
            "query_name": self.name,
            "strand": self.strand,
            "read_str_sequence": self.sequence,
            "read_str_qualities": self.qualities,
            "error_flags": self.error_flags,
        } | self.locus.to_dict()

    @classmethod
    def from_read(cls, read: Read, error_flags: str) -> FilteredRead:
        return cls(
            name=read.name,
            sequence=read.sequence,
            qualities=read.qualities,
            mod_5mc_probs=read.mod_5mc_probs,
            strand=read.strand,
            locus=read.locus,
            error_flags=error_flags,
        )


# Enum for alignment type
class AlignmentType(StrEnum):
    SPANNING = "spanning"
    LEFT_FLANKING = "left_flanking"
    RIGHT_FLANKING = "right_flanking"
    OTHER = "other"


@dataclass
class GraphAlignment(Read):
    # Properties of alignment
    query_length: int
    query_start: int
    query_end: int

    path_length: int
    path_start: int
    path_end: int

    path: list[str]

    cigar: str

    # Properties of STR region
    str_sequence: str = field(init=False)
    str_sequence_synced: list[str] = field(init=False)
    str_cigar_synced: list[str] = field(init=False)
    str_mod_5mc_synced: list[str] = field(init=False)
    str_qualities: list[int] = field(init=False)
    str_median_quality: float = field(init=False)

    # Sequence divergence
    str_match_ratio: float = field(init=False)
    upstream_match_ratio: float = field(init=False)
    downstream_match_ratio: float = field(init=False)

    # Properties of flanking regions
    has_left_anchor: bool = field(init=False)
    has_right_anchor: bool = field(init=False)

    # Alignment type
    alignment_type: str = field(init=False)

    @classmethod
    def from_gaf_line(cls, read: Read, gaf_line: str, locus: Locus) -> GraphAlignment:
        # GAF format: https://github.com/lh3/gfatools/blob/master/doc/rGFA.md#the-graph-alignment-format-gaf
        fields = gaf_line.split("\t")

        query_length = int(fields[1])
        query_start = int(fields[2])
        query_end = int(fields[3])

        path_str = fields[5]

        path_length = int(fields[6])
        path_start = int(fields[7])
        path_end = int(fields[8])

        # Get CIGAR string from tags
        tags = fields[12:]
        cigar = next(tag.split(":")[-1] for tag in tags if tag.startswith("cg"))

        # Count satellites from the mapping
        path = path_str.split(">")[1:]

        return cls(
            # Properties from Read
            name=read.name,
            sequence=read.sequence,
            qualities=read.qualities,
            mod_5mc_probs=read.mod_5mc_probs,
            strand=read.strand,
            locus=read.locus,
            # Properties from GAF
            query_length=query_length,
            query_start=query_start,
            query_end=query_end,
            path=path,
            path_length=path_length,
            path_start=path_start,
            path_end=path_end,
            cigar=cigar,
        )

    def __post_init__(self) -> None:
        # Get sequence and qualities of the STR region
        # Remove unmapped regions outside of the alignment
        locus_sequence = self.sequence[self.query_start : self.query_end]
        locus_quals = self.qualities[self.query_start : self.query_end]
        locus_mod_5mc_probs = self.mod_5mc_probs[self.query_start : self.query_end]

        locus_cigar_synced = sync_cigar(self.cigar)

        locus_sequence_synced = ["".join(ls) for ls in sync_with_cigar(list(locus_sequence), self.cigar)]
        locus_mod_5mc_synced = ["".join(ls) for ls in sync_with_cigar(list(locus_mod_5mc_probs), self.cigar)]
        locus_quals_synced = sync_with_cigar(locus_quals, self.cigar)

        # Determine the number of bases to trim from the left and right of the STR region
        trim_start = (
            config.ANCHOR_LEN - self.path_start
            if "left_anchor" in self.path
            else config.MIN_ANCHOR_OVERLAP - self.path_start
            if "left_anchor_overlap" in self.path
            else 0
        )
        trim_end = (
            config.ANCHOR_LEN - (self.path_length - self.path_end)
            if "right_anchor" in self.path
            else config.MIN_ANCHOR_OVERLAP - (self.path_length - self.path_end)
            if "right_anchor_overlap" in self.path
            else 0
        )

        # Trim the sequence, qualities, CIGAR, mod 5mC string
        self.str_sequence_synced = locus_sequence_synced[trim_start:]
        self.str_qualities_synced = locus_quals_synced[trim_start:]
        self.str_cigar_synced = locus_cigar_synced[trim_start:]
        self.str_mod_5mc_synced = locus_mod_5mc_synced[trim_start:]

        if trim_end > 0:
            self.str_sequence_synced = self.str_sequence_synced[:-trim_end]
            self.str_qualities_synced = self.str_qualities_synced[:-trim_end]
            self.str_cigar_synced = self.str_cigar_synced[:-trim_end]
            self.str_mod_5mc_synced = self.str_mod_5mc_synced[:-trim_end]

        # Get sequence and qualities of the STR region
        self.str_sequence = "".join(self.str_sequence_synced)
        self.str_qualities = [q for sublist in self.str_qualities_synced for q in sublist]

        # Calculate median quality
        self.str_median_quality = median(self.str_qualities) if self.str_qualities else 0

        # Calculate str sequence divergence using the CIGAR string
        self.str_match_ratio = get_match_ratio(self.str_cigar_synced, self.str_sequence_synced)

        # Upstream and downstream match ratios
        self.upstream_match_ratio = get_match_ratio(locus_cigar_synced[:trim_start], locus_sequence_synced[:trim_start])
        self.downstream_match_ratio = get_match_ratio(locus_cigar_synced[-trim_end:], locus_sequence_synced[-trim_end:])

        # Check if read has full STR and sufficient anchors
        self.has_left_anchor = "left_anchor" in self.path and "left_anchor_overlap" in self.path
        self.has_right_anchor = "right_anchor" in self.path and "right_anchor_overlap" in self.path

        # TODO: REMOVE or change filtering behavior. This "disables" min_anchor_overlap
        self.has_left_anchor = "left_anchor_overlap" in self.path
        self.has_right_anchor = "right_anchor_overlap" in self.path

        # Determine alignment type
        if self.has_left_anchor and self.has_right_anchor:
            self.alignment_type = AlignmentType.SPANNING
        elif self.has_left_anchor:
            self.alignment_type = AlignmentType.LEFT_FLANKING
        elif self.has_right_anchor:
            self.alignment_type = AlignmentType.RIGHT_FLANKING
        else:
            self.alignment_type = AlignmentType.OTHER

    def get_error_flags(self) -> str:
        errors = []

        # Check upstream and downstream match ratios
        if self.has_left_anchor and self.upstream_match_ratio < 0.8:
            errors.append("low_upstream_match_ratio")
        if self.has_right_anchor and self.downstream_match_ratio < 0.8:
            errors.append("low_downstream_match_ratio")

        # TODO: Remove and handle elsewhere - does not allow for filtering before remapping flanking
        # # Missing anchors
        # if not self.has_left_anchor:
        #     errors.append("missing_left_anchor")
        # if not self.has_right_anchor:
        #     errors.append("missing_right_anchor")

        # Set error flags for OTHER alignment type
        if self.alignment_type == AlignmentType.OTHER:
            errors.append("missing_anchors")

        return ",".join(errors)

    def to_dict(self) -> dict:
        return {
            "query_name": self.name,
            "strand": self.strand,
            "read_str_sequence": self.str_sequence,
            "read_str_qualities": self.str_qualities,
            "alignment_type": self.alignment_type,
            "error_flags": self.get_error_flags(),
            "median_str_qual": self.str_median_quality,
        } | self.locus.to_dict()


@dataclass
class ReadCall:
    locus: Locus
    alignment: GraphAlignment
    satellite_count: list[int]
    kmer_count_str: str

    # Kmer strings (for visualization)
    obs_kmer_string: str
    ref_kmer_string: str
    mod_5mc_kmer_string: str

    def to_dict(self) -> dict:
        return self.alignment.to_dict() | {
            "kmer_count": self.satellite_count,
            "kmer_count_str": self.kmer_count_str,
            "obs_kmer_string": self.obs_kmer_string,
            "ref_kmer_string": self.ref_kmer_string,
            "mod_5mc_kmer_string": self.mod_5mc_kmer_string,
        }


@dataclass
class GroupedReadCall(ReadCall):
    group: str
    outlier_reason: str

    def to_dict(self) -> dict:
        # TODO: Change name of group to em_haplotype
        return super().to_dict() | {"em_haplotype": self.group, "outlier_reason": self.outlier_reason}

    @classmethod
    def from_read_call(cls, read_call: ReadCall, group: str, outlier_reason: str) -> GroupedReadCall:
        return cls(
            locus=read_call.locus,
            alignment=read_call.alignment,
            satellite_count=read_call.satellite_count,
            kmer_count_str=read_call.kmer_count_str,
            obs_kmer_string=read_call.obs_kmer_string,
            ref_kmer_string=read_call.ref_kmer_string,
            mod_5mc_kmer_string=read_call.mod_5mc_kmer_string,
            group=group,
            outlier_reason=outlier_reason,
        )


def get_match_ratio(synced_cigar: list[str], synced_sequence: list[str]) -> float:
    # Count number of 1:1 bases
    n_matches = synced_cigar.count("=")
    n_mismatches = synced_cigar.count("X")

    # Count number of deletion and insertion oberations
    n_deletions = synced_cigar.count("D")
    n_insertions = sum(len(subseq) if c == "I" else 0 for c, subseq in zip(synced_cigar, synced_sequence))

    # Total number of oberations
    n_oberations = n_deletions + n_insertions + n_matches + n_mismatches

    # Return match ratio
    return (n_matches + n_mismatches * 0.5) / n_oberations if n_oberations > 0 else 0


def get_satellite_counts_from_path(path: list[str], locus: Locus) -> list[int]:
    # Filter all sub-satellites with j>0 i.e. satellite_i_1, satellite_i_2, ...

    # Count occurrences of each satellite
    return [len([node for node in path if node.startswith(f"satellite_{i}")]) for i in range(len(locus.satellites))]


# TODO: Implement skip connections when creating the graph for alignment of flanking readsm i.e. left flanking needs skip connection for all nodes to the right anchor
def create_repeat_graph_gfa_from_locus(locus: Locus) -> str:
    # TODO: Make this into a parameter so that you can check for anchor -> this mens the overlap is big enough

    left_anchor = locus.left_anchor[: -config.MIN_ANCHOR_OVERLAP]
    left_anchor_overlap = locus.left_anchor[-config.MIN_ANCHOR_OVERLAP :]

    right_anchor = locus.right_anchor[config.MIN_ANCHOR_OVERLAP :]
    right_anchor_overlap = locus.right_anchor[: config.MIN_ANCHOR_OVERLAP]

    # Initialize lists
    nodes = [
        f"S\tleft_anchor\t{left_anchor}\n",
        f"S\tleft_anchor_overlap\t{left_anchor_overlap}\n",
    ]
    edges = ["L\tleft_anchor\t+\tleft_anchor_overlap\t+\t0M\n"]
    previous_nodes = ["left_anchor_overlap"]

    for i in range(len(locus.satellites)):
        # Add break if present
        if locus.breaks[i]:
            for prev_node in previous_nodes:
                edges.append(f"L\t{prev_node}\t+\tbreak_{i}\t+\t0M\n")
            nodes.append(f"S\tbreak_{i}\t{locus.breaks[i]}\n")
            previous_nodes = [f"break_{i}"]

        # Handle satellite
        current_satellite = locus.satellites[i]

        # Check in ambiguous bases in satellite
        if all(base not in AMBIGUOUS_BASES_DICT for base in current_satellite.sequence):
            satellite_id = f"satellite_{i}"
            # Add satellite
            nodes.append(f"S\t{satellite_id}\t{current_satellite.sequence}\n")
            # Add edge from previous nodes to satellite
            for prev_node in previous_nodes:
                edges.append(f"L\t{prev_node}\t+\t{satellite_id}\t+\t0M\n")
            # Connect satellite to itself
            edges.append(f"L\t{satellite_id}\t+\t{satellite_id}\t+\t0M\n")

            if current_satellite.skippable:
                # Add edges from satellite to previous nodes
                previous_nodes.append(f"{satellite_id}")
            else:
                # Reset previous nodes
                previous_nodes = [f"{satellite_id}"]

        else:
            sub_satellites: list[list[str]] = []
            sub_satellite = ""
            for base in current_satellite.sequence:
                if base in AMBIGUOUS_BASES_DICT:
                    if sub_satellite:
                        sub_satellites.append([sub_satellite])
                        sub_satellite = ""
                    sub_satellites.append(AMBIGUOUS_BASES_DICT[base])
                else:
                    sub_satellite += base
            if sub_satellite:
                sub_satellites.append([sub_satellite])

            # Internal edges
            previous_sub_satellites = list(previous_nodes)
            for j, sub_satellite in enumerate(sub_satellites):
                cur_nodes = []
                prefix = "sub_" if j > 0 else ""
                cur_satallite_id = f"{prefix}satellite_{i}_{j}"

                # For ambiguous bases add anchor and edge for each base A, T, C, G
                if len(sub_satellite) > 1:
                    for base in sub_satellite:
                        edges.extend(f"L\t{prev_sub_sat}\t+\t{cur_satallite_id}_{base}\t+\t0M\n" for prev_sub_sat in previous_sub_satellites)
                        nodes.append(f"S\t{cur_satallite_id}_{base}\t{base}\n")

                        cur_nodes.append(f"{cur_satallite_id}_{base}")
                else:
                    edges.extend(f"L\t{prev_sub_sat}\t+\t{cur_satallite_id}\t+\t0M\n" for prev_sub_sat in previous_sub_satellites)

                    nodes.append(f"S\t{cur_satallite_id}\t{sub_satellite[0]}\n")
                    cur_nodes.append(f"{cur_satallite_id}")

                previous_sub_satellites = cur_nodes

            # Get first and last sub satellites
            first_sub_satellites = [f"satellite_{i}_0"]
            if len(sub_satellites[0]) > 1:
                first_sub_satellites = [f"satellite_{i}_0_{base}" for base in sub_satellites[0]]

            prefix = "sub_" if len(sub_satellites) > 1 else ""
            last_sub_satellites = [f"{prefix}satellite_{i}_{len(sub_satellites) - 1}"]
            if len(sub_satellites[-1]) > 1:
                last_sub_satellites = [f"{prefix}satellite_{i}_{len(sub_satellites) - 1}_{base}" for base in sub_satellites[-1]]

            # Connect last sub satellite(s) to first sub satellite(s) ie. to itself
            edges.extend(
                f"L\t{last_sub_satellite}\t+\t{first_sub_satellite}\t+\t0M\n"
                for first_sub_satellite, last_sub_satellite in itertools.product(first_sub_satellites, last_sub_satellites)
            )

            if current_satellite.skippable:
                # Add last sub satellite to previous nodes
                previous_nodes.extend(last_sub_satellites)
            else:
                # Reset previous nodes
                previous_nodes = last_sub_satellites

    # Add last break if present
    if locus.breaks[-1]:
        for prev_node in previous_nodes:
            edges.extend([f"L\t{prev_node}\t+\tbreak_{len(locus.breaks) - 1}\t+\t0M\n"])
        nodes.append(f"S\tbreak_{len(locus.breaks) - 1}\t{locus.breaks[-1]}\n")
        previous_nodes = [f"break_{len(locus.breaks) - 1}"]

    # Add right anchor
    for prev_node in previous_nodes:
        edges.extend([f"L\t{prev_node}\t+\tright_anchor_overlap\t+\t0M\n"])
    nodes.append(f"S\tright_anchor_overlap\t{right_anchor_overlap}\n")

    edges.append("L\tright_anchor_overlap\t+\tright_anchor\t+\t0M\n")
    nodes.append(f"S\tright_anchor\t{right_anchor}\n")

    # Create GFA string
    return "\n".join(nodes) + "\n" + "\n".join(edges)


def create_linear_graph_gfa(locus: Locus, satellite_counts: list[int]) -> str:
    left_anchor = locus.left_anchor[: -config.MIN_ANCHOR_OVERLAP]
    left_anchor_overlap = locus.left_anchor[-config.MIN_ANCHOR_OVERLAP :]

    right_anchor = locus.right_anchor[config.MIN_ANCHOR_OVERLAP :]
    right_anchor_overlap = locus.right_anchor[: config.MIN_ANCHOR_OVERLAP]

    # Initialize lists
    nodes = [
        f"S\tleft_anchor\t{left_anchor}\n",
        f"S\tleft_anchor_overlap\t{left_anchor_overlap}\n",
    ]
    edges = ["L\tleft_anchor\t+\tleft_anchor_overlap\t+\t0M\n"]
    previous_nodes = ["left_anchor_overlap"]

    for i in range(len(locus.satellites)):
        # Add break if present
        if locus.breaks[i]:
            edges.extend(f"L\t{prev_node}\t+\tbreak_{i}\t+\t0M\n" for prev_node in previous_nodes)
            nodes.append(f"S\tbreak_{i}\t{locus.breaks[i]}\n")
            previous_nodes = [f"break_{i}"]

        # Handle satellite
        current_satellite = locus.satellites[i]
        current_satellite_count = satellite_counts[i]

        # Check in ambiguous bases in satellite
        if all(base not in AMBIGUOUS_BASES_DICT for base in current_satellite.sequence):
            for j in range(current_satellite_count):
                satellite_id = f"satellite_{i}_{j}"
                # Add satellite to nodes
                nodes.append(f"S\t{satellite_id}\t{current_satellite.sequence}\n")
                # Add edge from previous nodes to satellite
                edges.extend(f"L\t{prev_node}\t+\t{satellite_id}\t+\t0M\n" for prev_node in previous_nodes)
                # Set previous nodes to satellite
                previous_nodes = [f"{satellite_id}"]

        else:
            sub_satellites: list[list[str]] = []
            sub_satellite = ""
            for base in current_satellite.sequence:
                if base in AMBIGUOUS_BASES_DICT:
                    if sub_satellite:
                        sub_satellites.append([sub_satellite])
                        sub_satellite = ""
                    sub_satellites.append(AMBIGUOUS_BASES_DICT[base])
                else:
                    sub_satellite += base
            if sub_satellite:
                sub_satellites.append([sub_satellite])

            # Internal edges
            previous_sub_satellites = list(previous_nodes)
            for j in range(current_satellite_count):
                for k, sub_satellite in enumerate(sub_satellites):
                    cur_nodes = []
                    prefix = "sub_" if k > 0 else ""
                    cur_satallite_id = f"{prefix}satellite_{i}_{j}_{k}"
                    # For ambiguous bases add anchor and edge for each base, e.g. A, T, C, G for N
                    if len(sub_satellite) > 1:
                        for base in sub_satellite:
                            edges.extend(f"L\t{prev_sub_sat}\t+\t{cur_satallite_id}_{base}\t+\t0M\n" for prev_sub_sat in previous_sub_satellites)
                            nodes.append(f"S\t{cur_satallite_id}_{base}\t{base}\n")

                            cur_nodes.append(f"{cur_satallite_id}_{base}")
                    else:
                        edges.extend(f"L\t{prev_sub_sat}\t+\t{cur_satallite_id}\t+\t0M\n" for prev_sub_sat in previous_sub_satellites)

                        nodes.append(f"S\t{cur_satallite_id}\t{sub_satellite[0]}\n")
                        cur_nodes.append(f"{cur_satallite_id}")

                    previous_sub_satellites = cur_nodes
                # Set previous nodes to last sub satellite
                previous_nodes = cur_nodes

    # Add last break if present
    if locus.breaks[-1]:
        edges.extend(f"L\t{prev_node}\t+\tbreak_{len(locus.breaks) - 1}\t+\t0M\n" for prev_node in previous_nodes)
        nodes.append(f"S\tbreak_{len(locus.breaks) - 1}\t{locus.breaks[-1]}\n")
        previous_nodes = [f"break_{len(locus.breaks) - 1}"]

    # Add right anchor
    edges.extend(f"L\t{prev_node}\t+\tright_anchor_overlap\t+\t0M\n" for prev_node in previous_nodes)
    nodes.append(f"S\tright_anchor_overlap\t{right_anchor_overlap}\n")

    edges.append("L\tright_anchor_overlap\t+\tright_anchor\t+\t0M\n")
    nodes.append(f"S\tright_anchor\t{right_anchor}\n")

    # Create GFA string
    return "\n".join(nodes) + "\n" + "\n".join(edges)


def get_graph_alignments(reads: list[Read], locus: Locus) -> list[GraphAlignment]:
    # Create graph from locus
    graph_str = create_repeat_graph_gfa_from_locus(locus)

    # Create a temporary directory
    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = Path(_temp_dir)

        # Create input fasta file
        input_fastq = temp_dir / "input.fastq"
        with Path.open(input_fastq, "w") as f:
            for read in reads:
                f.write(read.to_fastq())

        # Create graph
        input_graph_gfa = temp_dir / "graph.gfa"

        # Create output file path
        output_gaf = temp_dir / "output.gaf"

        # Write the graph
        input_graph_gfa.write_text(graph_str)

        # Run the command and redirect the output to a log file
        process = subprocess.run(
            [
                "minigraph",
                "-c",
                "-j",
                "0.3",
                "-x",
                "lr",
                "-o",
                output_gaf,
                input_graph_gfa,
                input_fastq,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # process = subprocess.run(
        #     [
        #         "GraphAligner",
        #         "--preset",
        #         "vg",
        #         "--alignments-out",
        #         output_gaf,
        #         "--graph",
        #         input_graph_gfa,
        #         "--reads",
        #         input_fastq,
        #     ],
        #     check=True,
        #     capture_output=True,
        #     text=True,
        # )

        # Log the stdout and stderr
        logger.info("minigraph stdout: %s", process.stdout)
        logger.error("minigraph stderr: %s", process.stderr)

        # Get the output from file
        with Path.open(output_gaf) as f:
            output_string = f.read()

    # Parse the output
    graph_alignments: list[GraphAlignment] = []
    for read in reads:
        # Get the first alignment for the read
        gaf_lines = next((line for line in output_string.split("\n") if line.startswith(read.name)), None)

        # Skip if no alignment found
        if gaf_lines is None:
            continue

        # Add alignment to the list
        graph_alignments.append(GraphAlignment.from_gaf_line(read=read, gaf_line=gaf_lines, locus=locus))

    return graph_alignments


# TODO: Include insertions in kmer string!


def get_ref_kmer_string(locus: Locus, satellite_counts: list[int]) -> str:
    # Get satellite sequences and counts
    satellite_seqs = [sat.sequence for sat in locus.satellites]

    # Get breaks
    breaks = locus.breaks

    # Create kmer string
    kmer = []

    # Add case for easy looping
    satellites_loop = [*satellite_seqs, ""]
    kmer_count_loop = np.concatenate([satellite_counts, np.array([0])])

    for sat, cnt, brk in zip(satellites_loop, kmer_count_loop, breaks):
        if brk != "":
            kmer.append(brk)
        if sat != "":
            kmer.extend([sat] * cnt)

    return "-".join(kmer)


def get_kmer_string(locus: Locus, synced_list: list[str], satellite_counts: list[int]) -> str:
    # Get satellite sequences and counts
    satellite_seqs = [sat.sequence for sat in locus.satellites]

    # Get breaks
    breaks = locus.breaks

    # Create kmer string
    kmers = []

    # Add case for easy looping
    satellites_loop = [*satellite_seqs, ""]
    kmer_count_loop = np.concatenate([satellite_counts, np.array([0])])

    for sat, cnt, brk in zip(satellites_loop, kmer_count_loop, breaks):
        if brk != "":
            # Add observed break
            kmers.append("".join(synced_list[: len(brk)]))

            # Clip break
            synced_list = synced_list[len(brk) :]

        if sat != "":
            # Add observed kmers
            kmers.extend(["".join(synced_list[i : i + len(sat)]) for i in range(0, len(sat) * cnt, len(sat))])

            # Clip kmers
            synced_list = synced_list[len(sat) * cnt :]

    return "-".join(kmers)


def get_reads_in_locus(bam: Path, locus: Locus) -> list[Read]:
    # Open BAM file
    bamfile = pysam.AlignmentFile(str(bam), "rb")

    # Get alignments overlapping the region
    alignments = bamfile.fetch(locus.chrom, locus.start, locus.end)

    # Remove secondary alignments
    alignments = [ali for ali in alignments if not ali.is_secondary]

    # Split alignments into primary and supplementary
    primary_alignments = [read for read in alignments if not read.is_supplementary]
    supplementary_alignments = [read for read in alignments if read.is_supplementary]

    # Filter supplementary alignments that are in primary alignments already
    primary_ids = [read.query_name for read in primary_alignments]
    supplementary_alignments = [ali for ali in supplementary_alignments if ali.query_name not in primary_ids]

    # Find primary alignments of left supplementary alignments
    for ali in supplementary_alignments:
        # Get the primary alignment position
        chrom, pos = str(ali.get_tag("SA")).split(",")[0:2]

        # Load all alignments at the primary alignment position
        primary_candidates = bamfile.fetch(chrom, int(pos), int(pos) + 1)

        # Find the primary alignment and add it to the primary alignments
        primary_alignment = [cand for cand in primary_candidates if cand.query_name == ali.query_name]
        primary_alignments.extend(primary_alignment)

    # Convert primary alignments to Read objects
    return [Read.from_aligment(alignment=alignment, locus=locus) for alignment in primary_alignments]


# Make sure reads have unique names
# TODO: Remove this if not important
# read_names = {read.name for read in alignments}
# for read_name in read_names:
#     read_subset = [read for read in alignments if read.name == read_name]
#     if len(read_subset) > 1:
#         for i, ali in enumerate(read_subset):
#             # Add index to the name
#             ali.name = f"{ali.name}_{i}"

# return alignments


def graph_align_reads_to_locus(
    reads: list[Read],
    locus: Locus,
) -> tuple[list[GraphAlignment], list[FilteredRead]]:
    # Initialize lists
    alignments: list[GraphAlignment] = []
    flanking_alignments: list[GraphAlignment] = []
    filtered_reads: list[FilteredRead] = []

    # Run the tool
    graph_alignments = get_graph_alignments(reads, locus)

    # Mark unmapped reads
    unmapped_reads = [read for read in reads if read.name not in [aln.name for aln in graph_alignments]]
    filtered_reads.extend([FilteredRead.from_read(read=r, error_flags="unmapped") for r in unmapped_reads])

    # Process the results
    for aln in graph_alignments:
        # Filter out reads with errors
        if aln.get_error_flags() or aln.alignment_type == AlignmentType.OTHER:
            filtered_reads.append(
                FilteredRead.from_read(
                    read=aln,
                    error_flags=aln.get_error_flags() if aln.get_error_flags() else "other",
                ),
            )
            continue

        # Flanking reads
        if aln.alignment_type in [AlignmentType.LEFT_FLANKING, AlignmentType.RIGHT_FLANKING]:
            flanking_alignments.append(aln)
            continue

        # Spanning reads
        alignments.append(aln)

    # Remap flanking reads to locus
    remapped_flanking_alignments, filtered_flanking_reads = remap_flanking_alignments_to_locus(flanking_alignments, locus)

    # Add the remapped flanking alignments to the lists
    alignments.extend(remapped_flanking_alignments)
    filtered_reads.extend(filtered_flanking_reads)

    # Remove flanking reads that do not visit the STR region
    non_overlapping_reads = [aln for aln in alignments if aln.str_sequence == "" and aln.alignment_type != AlignmentType.SPANNING]
    filtered_reads.extend([FilteredRead.from_read(read=aln, error_flags="not_overlapping_str") for aln in non_overlapping_reads])

    alignments = [aln for aln in alignments if aln not in non_overlapping_reads]

    return alignments, filtered_reads


def remap_flanking_alignments_to_locus(
    flanking_alignments: list[GraphAlignment],
    locus: Locus,
) -> tuple[list[GraphAlignment], list[FilteredRead]]:
    # Initialize lists
    synthetic_reads: list[Read] = []
    flanking_direction_map: dict[str, AlignmentType] = {}

    # TODO: Implement handling of flanking reads in more complex loci
    # Only handle loci with a single satellite and no breaks
    if len(locus.satellites) != 1 or any(b != "" for b in locus.breaks):
        return [], [FilteredRead.from_read(read=aln, error_flags="complex_locus") for aln in flanking_alignments]

    # Create synthetic reads by adding the anchor to the end where it is missing
    for aln in flanking_alignments:
        # Add anchor to the end where it is missing
        if aln.alignment_type == AlignmentType.LEFT_FLANKING:
            flanking_direction_map[aln.name] = AlignmentType.LEFT_FLANKING
            sequence = aln.sequence + locus.right_anchor
            qualities = aln.qualities + [30] * len(locus.right_anchor)
        else:
            flanking_direction_map[aln.name] = AlignmentType.RIGHT_FLANKING
            sequence = locus.left_anchor + aln.sequence
            qualities = [30] * len(locus.left_anchor) + aln.qualities

        # Create new "synthetic" spanning read
        synthetic_reads.append(
            Read(
                name=aln.name,
                sequence=sequence,
                qualities=qualities,
                mod_5mc_probs=aln.mod_5mc_probs,
                strand=aln.strand,
                locus=aln.locus,
            ),
        )

    # Re-map the synthetic reads
    remapped_flanking_reads = get_graph_alignments(synthetic_reads, locus)

    remapped_alignments: list[GraphAlignment] = []
    filtered_alignments: list[FilteredRead] = []

    # Mark unmapped reads
    unmapped_reads = [read for read in synthetic_reads if read.name not in [aln.name for aln in remapped_flanking_reads]]
    filtered_alignments.extend([FilteredRead.from_read(read=r, error_flags="unmapped") for r in unmapped_reads])

    for aln in remapped_flanking_reads:
        # Reads with errors
        if aln.get_error_flags() or aln.alignment_type != AlignmentType.SPANNING:
            filtered_alignments.append(
                FilteredRead.from_read(
                    read=aln,
                    error_flags=aln.get_error_flags() if aln.get_error_flags() else "other",
                ),
            )
            continue

        # Flanking reads
        # Fix the alignment type
        aln.alignment_type = flanking_direction_map[aln.name]

        # Add to the list
        remapped_alignments.append(aln)

    return remapped_alignments, filtered_alignments


def get_read_calls(spanning_reads: list[GraphAlignment], locus: Locus) -> list[ReadCall]:
    # Initialize lists
    read_calls: list[ReadCall] = []

    for aln in spanning_reads:
        # Count satellites from the mapping
        satellite_counts = get_satellite_counts_from_path(aln.path, locus)

        # Create kmer strings
        ref_kmer_string = get_ref_kmer_string(locus=locus, satellite_counts=satellite_counts)
        obs_kmer_string = get_kmer_string(
            locus=locus,
            synced_list=aln.str_sequence_synced,
            satellite_counts=satellite_counts,
        )
        mod_5mc_kmer_string = get_kmer_string(
            locus=locus,
            synced_list=aln.str_mod_5mc_synced,
            satellite_counts=satellite_counts,
        )

        # Find best call
        read_calls.append(
            ReadCall(
                locus=locus,
                alignment=aln,
                satellite_count=satellite_counts,
                kmer_count_str="-".join(map(str, satellite_counts)),
                obs_kmer_string=obs_kmer_string,
                ref_kmer_string=ref_kmer_string,
                mod_5mc_kmer_string=mod_5mc_kmer_string,
            ),
        )

    return read_calls


def group_flanking_read_calls(flanking_read_calls: list[ReadCall]) -> list[GroupedReadCall]:
    # Initialize lists
    grouped_read_calls: list[GroupedReadCall] = [
        GroupedReadCall.from_read_call(
            read_call=read_call,
            group="flanking",
            outlier_reason="",
        )
        for read_call in flanking_read_calls
    ]

    return grouped_read_calls
