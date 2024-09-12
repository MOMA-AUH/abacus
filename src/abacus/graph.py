import itertools
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import List, Tuple

import numpy as np
import pysam

from abacus.config import config
from abacus.constants import AMBIGUOUS_BASES_DICT
from abacus.locus import Locus
from abacus.logging import logger


def sync_with_cigar(seq: str, quals: List[int], cig: str) -> Tuple[List[str], List[List[int]]]:
    """Sync sequence and qualities with CIGAR string"""

    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cig)

    res_seq: List[str] = []
    res_quals: List[List[int]] = []

    for item in cigar_matches:
        cigar_len = int(item[0])
        cigar_ops = item[1]

        if cigar_ops in ["M", "=", "X"]:
            # For match, equal and mismatch: add sequence and quality 1:1
            res_seq.extend(list(seq[:cigar_len]))
            res_quals.extend([[q] for q in quals[:cigar_len]])
            # Trim "used" sequence and qualities
            seq = seq[cigar_len:]
            quals = quals[cigar_len:]
        elif cigar_ops == "D":
            # For deletion: add empty sequence and quality
            res_seq.extend(["" for _ in range(cigar_len)])
            res_quals.extend([[] for _ in range(cigar_len)])
        elif cigar_ops == "I":
            # For insertion: add sequence and quality to the previous position
            res_seq[-1] += seq[:cigar_len]
            res_quals[-1].extend(quals[:cigar_len])
            # Trim "used" sequence and qualities
            seq = seq[cigar_len:]
            quals = quals[cigar_len:]

    return res_seq, res_quals


@dataclass
class Read:
    name: str
    sequence: str
    qualities: List[int]

    strand: str
    phase: int

    def quality_string(self):
        return "".join([chr(q + 33) for q in self.qualities])

    def to_fastq(self):
        return f"@{self.name}\n{self.sequence}\n+\n{self.quality_string()}\n"

    @classmethod
    def from_aligment(cls, alignment: pysam.AlignedSegment):
        name = alignment.query_name or ""
        sequence = alignment.query_sequence or ""
        qualities = alignment.query_qualities or []
        qualities = [int(q) for q in qualities]

        strand = "-" if alignment.is_reverse else "+"
        phase = int(alignment.get_tag("HP")) if "HP" in [tag[0] for tag in alignment.get_tags()] else 0

        return cls(
            name=name,
            sequence=sequence,
            qualities=qualities,
            strand=strand,
            phase=phase,
        )


@dataclass
class GraphAlignment(Read):
    query_length: int
    query_start: int
    query_end: int

    path: List[str]

    path_length: int
    path_start: int
    path_end: int

    identity: float
    cigar: str

    locus: Locus

    str_sequence: str = field(init=False)
    str_sequence_synced: List[str] = field(init=False)
    str_qualities: List[int] = field(init=False)
    median_str_quality: float = field(init=False)

    error_flags: str = field(init=False)

    @classmethod
    def from_gaf_line(cls, read: Read, gaf_line: str, locus: Locus):
        # GAF format: https://github.com/lh3/gfatools/blob/master/doc/rGFA.md#the-graph-alignment-format-gaf
        fields = gaf_line.split("\t")

        query_length = int(fields[1])
        query_start = int(fields[2])
        query_end = int(fields[3])

        path_str = fields[5]

        path_length = int(fields[6])
        path_start = int(fields[7])
        path_end = int(fields[8])

        # Handle tags
        tags = fields[12:]
        # Identity (id), CIGAR (cg)
        identity = 1.0 - [float(tag.split(":")[-1]) for tag in tags if tag.startswith("dv")][0]
        cigar = [tag.split(":")[-1] for tag in tags if tag.startswith("cg")][0]

        # Count satellites from the mapping
        path = path_str.split(">")[1:]

        return cls(
            # Properties from Read
            name=read.name,
            sequence=read.sequence,
            qualities=read.qualities,
            strand=read.strand,
            phase=read.phase,
            # Properties from GAF
            query_length=query_length,
            query_start=query_start,
            query_end=query_end,
            path=path,
            path_length=path_length,
            path_start=path_start,
            path_end=path_end,
            identity=identity,
            cigar=cigar,
            # Locus
            locus=locus,
        )

    def __post_init__(self):
        # Get sequence and qualities of the STR region
        # Remove unmapped regions outside of the alignment
        locus_sequence = self.sequence[self.query_start : self.query_end]
        locus_quals = self.qualities[self.query_start : self.query_end]

        locus_sequence_synced, locus_quals_synced = sync_with_cigar(
            seq=locus_sequence,
            quals=locus_quals,
            cig=self.cigar,
        )

        # Trim left and right anchors
        n_trim_start = len(self.locus.left_anchor) - self.path_start
        n_trim_end = len(self.locus.right_anchor) - (self.path_length - self.path_end)
        locus_sequence_synced_trimmed = locus_sequence_synced[n_trim_start:-n_trim_end]
        locus_quals_synced_trimmed = locus_quals_synced[n_trim_start:-n_trim_end]

        # Convert to string
        self.str_sequence = "".join(locus_sequence_synced_trimmed)
        self.str_sequence_synced = locus_sequence_synced_trimmed
        self.str_qualities = [q for sublist in locus_quals_synced_trimmed for q in sublist]
        self.str_qualities_synced = locus_quals_synced_trimmed

        self.median_str_quality = median(self.str_qualities) if self.str_qualities else 0

        # Check if read has full STR and sufficient anchors
        errors = []

        has_left_anchor = self.path_start <= config.MIN_ANCHOR_OVERLAP
        has_right_anchor = self.path_length - self.path_end <= config.MIN_ANCHOR_OVERLAP

        if not has_left_anchor:
            errors.append("left_anchor_too_short")

        if not has_right_anchor:
            errors.append("right_anchor_too_short")

        if has_left_anchor and has_right_anchor and self.median_str_quality < config.MIN_STR_READ_QUAL:
            errors.append("low_read_quality")

        self.error_flags = ",".join(errors)

    def to_dict(self):
        return {
            "query_name": self.name,
            "strand": self.strand,
            "phase": self.phase,
            "read_str_sequence": self.str_sequence,
            "read_str_qualities": self.str_qualities,
            "error_flags": self.error_flags,
            "median_str_qual": self.median_str_quality,
        } | self.locus.to_dict()


@dataclass
class Read_Call:
    locus: Locus
    alignment: GraphAlignment
    kmer_count: List[int]
    kmer_count_str: str
    score: float
    satelite_str: str
    expected_satellite_str: str

    def to_dict(self):
        return self.alignment.to_dict() | {
            "kmer_count": self.kmer_count,
            "kmer_count_str": self.kmer_count_str,
            "score": self.score,
            "observed_satellite_str": self.satelite_str,
            "expected_satellite_str": self.expected_satellite_str,
        }


def get_satellite_counts_from_path(path: List[str], locus: Locus) -> List[int]:
    # Filter all sub-satellites with j>0 i.e. satellite_i_1, satellite_i_2, ...
    path = [node for node in path if not re.search(r"satellite_\d+_[1-9]+", node)]

    # Count occurrences of each satellite
    return [len([node for node in path if node.startswith(f"satellite_{i}")]) for i in range(len(locus.satellites))]


def write_graph_gfa_from_locus(locus: Locus, output_gfa: Path) -> None:
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
            # Add satellite
            nodes.append(f"S\tsatellite_{i}\t{current_satellite.sequence}\n")
            # Add edge from previous nodes to satellite
            for prev_node in previous_nodes:
                edges.append(f"L\t{prev_node}\t+\tsatellite_{i}\t+\t0M\n")
            # Connect satellite to itself
            edges.append(f"L\tsatellite_{i}\t+\tsatellite_{i}\t+\t0M\n")

            if current_satellite.skippable:
                # Add edges from satellite to previous nodes
                previous_nodes.append(f"satellite_{i}")
            else:
                # Reset previous nodes
                previous_nodes = [f"satellite_{i}"]

        else:
            sub_satellites = []
            sub_satellite = ""
            for base in current_satellite.sequence:
                if base in AMBIGUOUS_BASES_DICT:
                    if sub_satellite:
                        sub_satellites.append(sub_satellite)
                        sub_satellite = ""
                    sub_satellites.append(AMBIGUOUS_BASES_DICT[base])
                else:
                    sub_satellite += base
            if sub_satellite:
                sub_satellites.append(sub_satellite)

            # Internal edges
            previous_sub_satellites = list(previous_nodes)
            for j, sub_satellite in enumerate(sub_satellites):
                cur_nodes = []
                # For ambiguous bases add anchor and edge for each base A, T, C, G
                if len(sub_satellite) > 1:
                    for base in sub_satellite:
                        for prev_sub_sat in previous_sub_satellites:
                            edges.append(f"L\t{prev_sub_sat}\t+\tsatellite_{i}_{j}_{base}\t+\t0M\n")
                        nodes.append(f"S\tsatellite_{i}_{j}_{base}\t{base}\n")

                        cur_nodes.append(f"satellite_{i}_{j}_{base}")
                else:
                    for prev_sub_sat in previous_sub_satellites:
                        edges.append(f"L\t{prev_sub_sat}\t+\tsatellite_{i}_{j}\t+\t0M\n")

                    nodes.append(f"S\tsatellite_{i}_{j}\t{sub_satellite}\n")
                    cur_nodes.append(f"satellite_{i}_{j}")

                previous_sub_satellites = cur_nodes

            # Get first and last sub satellites
            first_sub_satellites = [f"satellite_{i}_0"]
            if len(sub_satellites[0]) > 1:
                first_sub_satellites = [f"satellite_{i}_0_{base}" for base in sub_satellites[0]]

            last_sub_satellites = [f"satellite_{i}_{len(sub_satellites) - 1}"]
            if len(sub_satellites[-1]) > 1:
                last_sub_satellites = [f"satellite_{i}_{len(sub_satellites) - 1}_{base}" for base in sub_satellites[-1]]

            # Connect last sub satellite(s) to first sub satellite(s) ie. to itself
            for first_sub_satellite, last_sub_satellite in itertools.product(first_sub_satellites, last_sub_satellites):
                edges.append(f"L\t{last_sub_satellite}\t+\t{first_sub_satellite}\t+\t0M\n")

            if current_satellite.skippable:
                # Add last sub satellite to previous nodes
                previous_nodes.extend(last_sub_satellites)
            else:
                # Reset previous nodes
                previous_nodes = last_sub_satellites

    # Add last break if present
    if locus.breaks[-1]:
        for prev_node in previous_nodes:
            edges.append(f"L\t{prev_node}\t+\tbreak_{len(locus.breaks) - 1}\t+\t0M\n")
        nodes.append(f"S\tbreak_{len(locus.breaks) - 1}\t{locus.breaks[-1]}\n")
        previous_nodes = [f"break_{len(locus.breaks) - 1}"]

    # Add right anchor
    for prev_node in previous_nodes:
        edges.append(f"L\t{prev_node}\t+\tright_anchor_overlap\t+\t0M\n")
    nodes.append(f"S\tright_anchor_overlap\t{right_anchor_overlap}\n")

    edges.append("L\tright_anchor_overlap\t+\tright_anchor\t+\t0M\n")
    nodes.append(f"S\tright_anchor\t{right_anchor}\n")

    # Create GFA file with the graph
    with output_gfa.open("w") as f:
        for anchor in nodes:
            f.write(anchor)
        for edge in edges:
            f.write(edge)


def get_graph_alignments(reads: List[Read], locus: Locus) -> List[GraphAlignment]:
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Create input fasta file
        input_fastq = temp_dir / "input.fastq"
        with open(input_fastq, "w") as f:
            for read in reads:
                f.write(read.to_fastq())

        # Create graph
        input_graph_gfa = temp_dir / "graph.gfa"

        # Create output file path
        output_gaf = temp_dir / "output.gaf"

        # Write the graph
        write_graph_gfa_from_locus(locus, input_graph_gfa)

        # Run the command and redirect the output to a log file
        process = subprocess.run(
            [
                "minigraph",
                "-c",
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

        # Log the stdout and stderr
        logger.info("minigraph stdout: %s", process.stdout)
        logger.error("minigraph stderr: %s", process.stderr)

        # Get the output from file
        with open(output_gaf, "r") as f:
            my_output_string = f.read()

    # Parse the output
    output = []
    for read in reads:
        # Get the alignment for the read using the read name
        gaf_line = [line for line in my_output_string.split("\n") if line.startswith(read.name)][0]
        output.append(GraphAlignment.from_gaf_line(read=read, gaf_line=gaf_line, locus=locus))

    return output


# TODO: Include insertions in kmer string!
def get_satellite_strings(locus: Locus, synced_sequence: List[str], satellite_counts: List[int]) -> Tuple[str, str]:
    # Get satellite sequences and counts
    satellite_seqs = [sat.sequence for sat in locus.satellites]

    # Get breaks
    breaks = locus.breaks

    # Create kmer string
    obs_kmers = []
    exp_kmers = []

    # Add case for easy looping
    satellites_loop = satellite_seqs + [""]
    kmer_count_loop = np.concatenate([satellite_counts, np.array([0])])
    for sat, cnt, brk in zip(satellites_loop, kmer_count_loop, breaks):
        if brk != "":
            # Add observed break
            obs_kmers.append("".join(synced_sequence[: len(brk)]))

            # Add expected break
            exp_kmers.append(brk)

            # Clip break
            synced_sequence = synced_sequence[len(brk) :]

        if sat != "":
            # Add observed kmers
            obs_kmers.extend(["".join(synced_sequence[i : i + len(sat)]) for i in range(0, len(sat) * cnt, len(sat))])

            # Add expected break
            exp_kmers.extend([sat] * cnt)

            # Clip kmers
            synced_sequence = synced_sequence[len(sat) * cnt :]

    return "-".join(obs_kmers), "-".join(exp_kmers)


def process_reads_in_str_region(bamfile: pysam.AlignmentFile, locus: Locus) -> Tuple[List[Read_Call], List[GraphAlignment]]:
    # Get reads overlapping the region
    read_calls: List[Read_Call] = []
    filtered_reads: List[GraphAlignment] = []

    # Get reads overlapping the region
    reads = [Read.from_aligment(alignment=alignment) for alignment in bamfile.fetch(locus.chrom, locus.start, locus.end)]

    # Make sure reads have unique names
    read_names = {read.name for read in reads}
    for read_name in read_names:
        read_subset = [read for read in reads if read.name == read_name]
        if len(read_subset) > 1:
            for i, read in enumerate(read_subset):
                # Add index to the name
                read.name = f"{read.name}_{i}"

    # Run the tool
    graph_alignments = get_graph_alignments(reads, locus)

    # Process the results
    for aln in graph_alignments:
        # Filter reads with errors
        if aln.error_flags:
            filtered_reads.append(aln)
            continue

        # Count satellites from the mapping
        satellite_counts = get_satellite_counts_from_path(aln.path, locus)

        # Create kmer string
        observed_kmer_string, expected_kmer_string = get_satellite_strings(
            locus=locus,
            synced_sequence=aln.str_sequence_synced,
            satellite_counts=satellite_counts,
        )

        # Find best call
        read_calls.append(
            Read_Call(
                locus=locus,
                alignment=aln,
                kmer_count=satellite_counts,
                kmer_count_str="-".join(map(str, satellite_counts)),
                score=aln.identity,
                satelite_str=observed_kmer_string,
                expected_satellite_str=expected_kmer_string,
            )
        )
    return read_calls, filtered_reads
