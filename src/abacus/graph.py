from __future__ import annotations

import itertools
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import numpy as np

from abacus.config import config
from abacus.locus import Locus
from abacus.logging import logger
from abacus.read import FilteredRead, Read
from abacus.utils import AMBIGUOUS_BASES_DICT, AlignmentType, Haplotype, compute_levenshtein_rate, compute_ref_divergence


def sync_cigar(cigar: str) -> list[str]:
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
            # For insertion: Add the operation to the last position
            res_cigar[-1] += cigar_ops * cigar_len

    return res_cigar


def sync_with_cigar(input_list: list, cig: str) -> list[list]:
    if not input_list:
        return []

    cigar_pattern = r"(\d+)([MIDNSHPX=])"
    cigar_matches = re.findall(cigar_pattern, cig)

    res_list: list[list] = []

    print("Res list before")
    print(input_list)

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
            print("Res list after")
            print(res_list)
            print("input list")
            print(input_list)
            res_list[-1].extend(input_list[:cigar_len])
            # Trim "used" input list
            input_list = input_list[cigar_len:]

    return res_list


# TODO: Add test for this and extract into function
def get_reference_sequence_from_path(path: list[str], locus: Locus) -> str:
    # Add left anchor
    reference = locus.left_anchor
    # Run through the path and add breaks and satellites
    for node in path:
        # Breaks are simple, just add the break sequence and move on
        if node.startswith("break"):
            break_idx = int(node.split("_")[1])
            break_seq = locus.breaks[break_idx]
            reference += break_seq
            continue

        # Satellites and sub-satellites
        # Add the satellite sequence
        if node.startswith("satellite"):
            satellite_idx = int(node.split("_")[1])
            satellite_seq = locus.satellites[satellite_idx].sequence
            reference += satellite_seq

        # Handle ambiguous bases
        if node.startswith(("satellite", "sub_satellite")) and any(node.endswith(s) for s in ["_A", "_T", "_C", "_G"]):
            # Ambiguous base
            base = node[-1]
            index = node.removeprefix("sub_").removeprefix("satellite_")
            satellite_idx = int(index.split("_")[0])
            satellite_length = len(locus.satellites[satellite_idx].sequence)
            # Look at the last bases of the reference sequence and replace the first ambiguous base
            for i in range(len(reference) - satellite_length, len(reference)):
                c = reference[i]
                if c in AMBIGUOUS_BASES_DICT:
                    reference = reference[:i] + base + reference[i + 1 :]
                    break
    # Add right anchor
    return reference + locus.right_anchor


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
    mean_str_quality: float = field(init=False)
    q10_str_quality: int = field(init=False)

    reference: str = field(init=False)
    str_reference: str = field(init=False)

    str_ref_divergence: float = field(init=False)

    # Properties of flanking regions
    has_left_anchor: bool = field(init=False)
    has_right_anchor: bool = field(init=False)

    # Alignment type
    type: AlignmentType = field(init=False)

    @classmethod
    def from_gaf_line(cls, read: Read, gaf_line: str) -> GraphAlignment:
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
            n_soft_clipped_left=read.n_soft_clipped_left,
            n_soft_clipped_right=read.n_soft_clipped_right,
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
        # Check if read has full STR and sufficient anchors
        self.has_left_anchor = "left_anchor" in self.path and "left_anchor_overlap" in self.path
        self.has_right_anchor = "right_anchor" in self.path and "right_anchor_overlap" in self.path

        # TODO: REMOVE or change filtering behavior. This "disables" min_anchor_overlap
        self.has_left_anchor = "left_anchor_overlap" in self.path
        self.has_right_anchor = "right_anchor_overlap" in self.path

        # Determine alignment type
        if self.has_left_anchor and self.has_right_anchor:
            self.type = AlignmentType.SPANNING
        elif self.has_left_anchor:
            self.type = AlignmentType.LEFT_FLANKING
        elif self.has_right_anchor:
            self.type = AlignmentType.RIGHT_FLANKING
        else:
            self.type = AlignmentType.NO_ANCHORS

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
            config.anchor_len - self.path_start
            if "left_anchor" in self.path
            else config.min_anchor_overlap - self.path_start
            if "left_anchor_overlap" in self.path
            else 0
        )
        trim_end = (
            config.anchor_len - (self.path_length - self.path_end)
            if "right_anchor" in self.path
            else config.min_anchor_overlap - (self.path_length - self.path_end)
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
        self.mean_str_quality = mean(self.str_qualities) if self.str_qualities else 0
        self.q10_str_quality = int(np.quantile(self.str_qualities, 0.1)) if self.str_qualities else 0

        # Build STR reference sequence from path
        self.reference = get_reference_sequence_from_path(self.path, self.locus)

        # Trim the anchors to get the STR reference sequence
        self.str_reference = self.reference[len(self.locus.left_anchor) : -len(self.locus.right_anchor)]

        # Trim with start and end from alignment
        self.reference = self.reference[self.path_start : self.path_end]

        # Get the error rate of the STR region
        str_cigar = "".join(self.str_cigar_synced)

        # Trim indels from ends of CIGAR string - these are often artefacts of flanking reads
        # Trim max bases/operations equal to the longest satellite
        longest_satellite = max(len(s.sequence) for s in self.locus.satellites)
        str_cigar = re.sub(rf"^[ID]{{1,{longest_satellite}}}|[ID]{{1,{longest_satellite}}}$", "", str_cigar)

        self.str_ref_divergence = compute_ref_divergence(str_cigar)

    def to_dict(self) -> dict:
        return {
            "query_name": self.name,
            "strand": self.strand,
            "read_str_sequence": self.str_sequence,
            "read_str_qualities": self.str_qualities,
            "alignment_type": self.type,
            "mean_str_qual": self.mean_str_quality,
            "q10_str_qual": self.q10_str_quality,
            "str_ref_divergence": self.str_ref_divergence,
        } | self.locus.to_dict()


def get_satellite_counts_from_path(path: list[str], locus: Locus) -> list[int]:
    # Filter all sub-satellites with j>0 i.e. satellite_i_1, satellite_i_2, ...

    # Count occurrences of each satellite
    return [len([node for node in path if node.startswith(f"satellite_{i}")]) for i in range(len(locus.satellites))]


# TODO: Implement skip connections when creating the graph for alignment of flanking readsm i.e. left flanking needs skip connection for all nodes to the right anchor
def create_repeat_graph_gfa_from_locus(locus: Locus) -> str:
    # TODO: Make this into a parameter so that you can check for anchor -> this mens the overlap is big enough

    left_anchor = locus.left_anchor[: -config.min_anchor_overlap]
    left_anchor_overlap = locus.left_anchor[-config.min_anchor_overlap :]

    right_anchor = locus.right_anchor[config.min_anchor_overlap :]
    right_anchor_overlap = locus.right_anchor[: config.min_anchor_overlap]

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
            edges.extend([f"L\t{prev_node}\t+\tbreak_{i}\t+\t0M\n" for prev_node in previous_nodes])
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
            edges.extend([f"L\t{prev_node}\t+\t{satellite_id}\t+\t0M\n" for prev_node in previous_nodes])
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
    left_anchor = locus.left_anchor[: -config.min_anchor_overlap]
    left_anchor_overlap = locus.left_anchor[-config.min_anchor_overlap :]

    right_anchor = locus.right_anchor[config.min_anchor_overlap :]
    right_anchor_overlap = locus.right_anchor[: config.min_anchor_overlap]

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
                # "-k",
                # "11",
                # "-w",
                # "9",
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
        logger.debug("minigraph stdout: %s", process.stdout)
        logger.debug("minigraph stderr: %s", process.stderr)

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

        graph_alignments.append(GraphAlignment.from_gaf_line(read=read, gaf_line=gaf_lines))

    return graph_alignments


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

    return "|".join(kmers)


def graph_align_reads_to_locus(
    reads: list[Read],
    locus: Locus,
) -> tuple[list[GraphAlignment], list[FilteredRead]]:
    # Initialize output lists
    alignments: list[GraphAlignment] = []
    flanking_alignments: list[GraphAlignment] = []
    unmapped_reads: list[FilteredRead] = []

    # Run the tool
    graph_alignments = get_graph_alignments(reads, locus)

    # Mark unmapped reads
    mapped_read_names = [aln.name for aln in graph_alignments]
    unmapped_reads.extend(
        [
            FilteredRead.from_read(
                read=r,
                error_flags="unmappable_read",
            )
            for r in reads
            if r.name not in mapped_read_names
        ],
    )

    # Process the results
    for aln in graph_alignments:
        # Filter out reads with errors
        if aln.type == AlignmentType.NO_ANCHORS:
            unmapped_reads.append(
                FilteredRead.from_read(
                    read=aln,
                    error_flags="no_anchors",
                ),
            )
            continue

        # Flanking reads
        if aln.type in [AlignmentType.LEFT_FLANKING, AlignmentType.RIGHT_FLANKING]:
            flanking_alignments.append(aln)
            continue

        # Spanning reads
        alignments.append(aln)

    # Remap flanking reads to locus
    remapped_flanking_alignments, unmapped_flanking_reads = remap_flanking_alignments_to_locus(flanking_alignments, locus)

    # Add the remapped flanking alignments to the lists
    alignments.extend(remapped_flanking_alignments)
    unmapped_reads.extend(unmapped_flanking_reads)

    # Remove flanking reads that do not visit the STR region
    non_overlapping_reads = [aln for aln in alignments if aln.str_sequence == "" and aln.type in [AlignmentType.LEFT_FLANKING, AlignmentType.RIGHT_FLANKING]]
    alignments = [aln for aln in alignments if aln not in non_overlapping_reads]

    # Mark reads that do not overlap the STR region
    unmapped_reads.extend([FilteredRead.from_read(read=aln, error_flags="not_overlapping_str") for aln in non_overlapping_reads])

    return alignments, unmapped_reads


def pad_with_right_anchor(seq: str, right_anchor: str) -> tuple[str, int]:
    # Check if seq needs trimmin (small over lap with anchor)
    max_overlap = min(len(seq), len(right_anchor), 50)
    min_overlap = 6
    best_error_rate = 1.0
    best_overlap = 0
    for i in range(max_overlap, min_overlap, -1):
        seq_overlap = seq[-i:]
        anchor_overlap = right_anchor[:i]
        error_rate = compute_levenshtein_rate(anchor_overlap, seq_overlap, indel_cost=0.25)
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_overlap = i

    if best_error_rate < 0.05 and best_overlap > 0:
        return seq + right_anchor[best_overlap:], best_overlap

    return seq + right_anchor, 0


def pad_with_left_anchor(seq: str, left_anchor: str) -> tuple[str, int]:
    # Check if seq needs trimming (small overlap with anchor)
    max_overlap = min(len(seq), len(left_anchor), 50)
    min_overlap = 6
    best_error_rate = 1.0
    best_overlap = 0
    for i in range(max_overlap, min_overlap, -1):
        seq_overlap = seq[:i]
        anchor_overlap = left_anchor[-i:]
        error_rate = compute_levenshtein_rate(anchor_overlap, seq_overlap)
        if error_rate < best_error_rate:
            best_error_rate = error_rate
            best_overlap = i

    if best_error_rate < 0.05 and best_overlap > 0:
        return left_anchor[:-best_overlap] + seq, best_overlap

    return left_anchor + seq, 0


def remap_flanking_alignments_to_locus(
    flanking_alignments: list[GraphAlignment],
    locus: Locus,
) -> tuple[list[GraphAlignment], list[FilteredRead]]:
    # Initialize lists
    synthetic_reads: list[Read] = []
    flanking_direction_map: dict[str, AlignmentType] = {}

    # Create synthetic reads by adding the anchor to the end where it is missing
    for aln in flanking_alignments:
        # Add anchor to the end where it is missing
        if aln.type == AlignmentType.LEFT_FLANKING:
            flanking_direction_map[aln.name] = AlignmentType.LEFT_FLANKING
            sequence, n_overlap = pad_with_right_anchor(aln.sequence, locus.right_anchor)
            added_bases = len(locus.right_anchor) - n_overlap
            qualities = aln.qualities + ([30] * added_bases)
            mod_5mc_probs = aln.mod_5mc_probs + ("!" * added_bases)
        else:
            sequence, n_overlap = pad_with_left_anchor(aln.sequence, locus.left_anchor)
            flanking_direction_map[aln.name] = AlignmentType.RIGHT_FLANKING
            added_bases = len(locus.left_anchor) - n_overlap
            qualities = ([30] * added_bases) + aln.qualities
            mod_5mc_probs = ("!" * added_bases) + aln.mod_5mc_probs

        # Create new "synthetic" spanning read
        synthetic_reads.append(
            Read(
                name=aln.name,
                sequence=sequence,
                qualities=qualities,
                mod_5mc_probs=mod_5mc_probs,
                strand=aln.strand,
                n_soft_clipped_left=0,
                n_soft_clipped_right=0,
                locus=aln.locus,
            ),
        )

    # Re-map the synthetic reads
    remapped_flanking_reads = get_graph_alignments(synthetic_reads, locus)

    # Initialize lists
    remapped_alignments: list[GraphAlignment] = []
    filtered_alignments: list[FilteredRead] = []

    # Mark unmapped reads
    unmapped_reads = [read for read in synthetic_reads if read.name not in [aln.name for aln in remapped_flanking_reads]]
    filtered_alignments.extend([FilteredRead.from_read(read=r, error_flags="unmappable_flanking_read") for r in unmapped_reads])

    for aln in remapped_flanking_reads:
        # Reads with errors
        if aln.type != AlignmentType.SPANNING:
            filtered_alignments.append(
                FilteredRead.from_read(
                    read=aln,
                    error_flags="unmappable_flanking_read",
                ),
            )
            continue

        # Flanking reads
        # Fix the alignment type
        aln.type = flanking_direction_map[aln.name]

        # Add to the list
        remapped_alignments.append(aln)

    return remapped_alignments, filtered_alignments


@dataclass
class ReadCall:
    locus: Locus
    alignment: GraphAlignment
    satellite_count: list[int]

    str_error_rate: float

    # Kmer strings (for visualization)
    obs_kmer_string: str
    ref_kmer_string: str
    mod_5mc_kmer_string: str
    qual_kmer_string: str

    # Grouped read call
    haplotype: Haplotype = Haplotype.NONE
    outlier_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.alignment.to_dict() | {
            "kmer_count": self.satellite_count,
            "kmer_count_str": "-".join(map(str, self.satellite_count)),
            "obs_kmer_string": self.obs_kmer_string,
            "ref_kmer_string": self.ref_kmer_string,
            "mod_5mc_kmer_string": self.mod_5mc_kmer_string,
            "qual_kmer_string": self.qual_kmer_string,
            "str_error_rate": self.str_error_rate,
            "haplotype": self.haplotype,
            "outlier_reasons": ";".join(self.outlier_reasons),
        }

    def add_outlier_reason(self, reason: str) -> ReadCall:
        self.add_outlier_reasons([reason])

        return self

    def add_outlier_reasons(self, reasons: list[str]) -> ReadCall:
        self.haplotype = Haplotype.OUTLIER
        self.outlier_reasons.extend(reasons)

        return self

    def set_haplotype(self, haplotype: Haplotype) -> ReadCall:
        self.haplotype = haplotype

        return self

    def is_spanning(self) -> bool:
        return self.alignment.type == AlignmentType.SPANNING


def get_read_calls(reads: list[Read], locus: Locus) -> tuple[list[ReadCall], list[FilteredRead]]:
    # Initialize lists
    read_calls: list[ReadCall] = []

    # Align reads to locus
    alignments, unmapped_reads = graph_align_reads_to_locus(reads, locus)

    for aln in alignments:
        # Count satellites from the mapping
        satellite_counts = get_satellite_counts_from_path(aln.path, locus)

        # Create kmer strings
        ref_kmer_string = get_kmer_string(
            locus=locus,
            synced_list=[*aln.str_reference],
            satellite_counts=satellite_counts,
        )
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
        qual_kmer_string = get_kmer_string(
            locus=locus,
            synced_list=[qual_to_char(qual) for sublist in aln.str_qualities_synced for qual in sublist],
            satellite_counts=satellite_counts,
        )

        # Estimate error rate
        other_alns = [x for x in alignments if x.name != aln.name]
        str_error_rate = estimate_error_rate(aln, other_alns)

        read_calls.append(
            ReadCall(
                locus=locus,
                alignment=aln,
                satellite_count=satellite_counts,
                obs_kmer_string=obs_kmer_string,
                ref_kmer_string=ref_kmer_string,
                mod_5mc_kmer_string=mod_5mc_kmer_string,
                qual_kmer_string=qual_kmer_string,
                str_error_rate=str_error_rate,
            ),
        )

    return read_calls, unmapped_reads


def estimate_error_rate(aln: GraphAlignment, other_alns: list[GraphAlignment]) -> float:
    # Helper function to extract k-mers
    def extract_kmers(sequences: list[str], k: int) -> list[str]:
        kmers: list[str] = []
        for seq in sequences:
            # Skip sequences that are shorter than k
            if len(seq) < k:
                continue
            # Extract k-mers
            kmers.extend(seq[i : i + k] for i in range(len(seq) - k + 1))
        return kmers

    read = aln.str_sequence
    background_reads = [aln.str_sequence for aln in other_alns]
    k = 11  # Length of k-mers

    # Step 0: Check if the read is empty
    if not read:
        return 0.0
    if not background_reads:
        return 0.0

    # Step 1: Get all k-mers from the read of interest
    read_kmers = extract_kmers([read], k)

    # Step 2: Build a k-mer count dictionary from the background reads
    background_kmers = extract_kmers(background_reads, k)
    unique_background_kmers = set(background_kmers)

    # Step 3: Count how many k-mers in the read are not found in the background
    error_kmers = [kmer for kmer in read_kmers if kmer not in unique_background_kmers]
    num_error_kmers = len(error_kmers)

    # Step 4: Estimate erroneous bases. Each base affects (up to) k k-mers
    n_errors = num_error_kmers / k

    # Step 5: Total bases in the read
    total_bases = len(read)

    # Step 6: Per-base error rate
    return n_errors / total_bases


def qual_to_char(qual: int) -> str:
    # Convert quality score to character
    return chr(qual + 33) if qual > 0 else "!"  # ASCII 33 is the lowest quality score
