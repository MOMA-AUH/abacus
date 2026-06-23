from __future__ import annotations

import itertools
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean

import networkx as nx
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


def get_reference_sequence_from_path(path: list[str], locus: Locus, graph: nx.DiGraph) -> str:
    """Reconstruct the reference sequence for an alignment path through the graph.

    Concatenates node sequences for all non-anchor nodes and wraps with the full anchors.
    Each node already stores its resolved sequence (ambiguous bases are individual nodes),
    so no post-hoc patching is needed.
    """
    anchor_nodes = {"left_anchor", "left_anchor_overlap", "right_anchor_overlap", "right_anchor"}
    str_region = "".join(graph.nodes[node]["sequence"] for node in path if node not in anchor_nodes)
    return locus.left_anchor + str_region + locus.right_anchor


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

    graph: nx.DiGraph = field(repr=False)

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
    def from_gaf_line(cls, read: Read, gaf_line: str, graph: nx.DiGraph) -> GraphAlignment:
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
            graph=graph,
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
        self.reference = get_reference_sequence_from_path(self.path, self.locus, self.graph)

        # Trim the anchors to get the STR reference sequence
        self.str_reference = self.reference[len(self.locus.left_anchor) : -len(self.locus.right_anchor)]

        # Trim with start and end from alignment
        self.reference = self.reference[self.path_start : self.path_end]

        # Get the error rate of the STR region
        str_cigar = "".join(self.str_cigar_synced)

        # Trim indels from ends of CIGAR string - these are often artefacts of flanking reads
        # Trim max bases/operations equal to the longest satellite
        longest_satellite = max(max(len(seq) for seq in s.sequences) for s in self.locus.satellites)
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


# TODO: Implement skip connections when creating the graph for alignment of flanking reads, i.e. left flanking needs skip connection for all nodes to the right anchor
def graph_to_gfa(graph: nx.DiGraph) -> str:
    """Convert a NetworkX DiGraph to a GFA-format string.

    Nodes must have a 'sequence' attribute. All edges use forward strand (+) and
    default overlap of '0M'.
    """
    node_lines = [f"S\t{node_id}\t{attrs['sequence']}" for node_id, attrs in graph.nodes(data=True)]
    edge_lines = [f"L\t{src}\t+\t{dst}\t+\t0M" for src, dst in graph.edges()]

    return "\n".join(node_lines) + "\n" + "\n".join(edge_lines)


def _parse_iupac_sequence(sequence: str) -> list[list[str]]:
    """Parse a satellite sequence into segments, expanding IUPAC ambiguous bases.

    Returns a list of segments. Each segment is a list of strings:
    - Single-element: unambiguous sequence chunk, e.g. ["ACGT"]
    - Multi-element: ambiguous position with all possible bases, e.g. ["A", "G"] for R
    """
    segments: list[list[str]] = []
    current = ""
    for base in sequence:
        if base in AMBIGUOUS_BASES_DICT:
            if current:
                segments.append([current])
                current = ""
            segments.append(AMBIGUOUS_BASES_DICT[base])
        else:
            current += base
    if current:
        segments.append([current])
    return segments


def _add_satellite_copy_to_graph(
    graph: nx.DiGraph,
    sub_satellites: list[list[str]],
    node_prefix: str,
    previous_nodes: list[str],
) -> list[str]:
    """Add one copy of a satellite to the graph, returning the frontier (last) node ids.

    For a simple single-segment satellite, one node is added with id = node_prefix.
    For multi-segment or ambiguous satellites, nodes get _k suffixes and possible base suffixes.
    The caller is responsible for adding self-loop edges if needed (repeat graph).
    """
    # Simple case: single unambiguous segment — use node_prefix directly (no _0 suffix)
    if len(sub_satellites) == 1 and len(sub_satellites[0]) == 1:
        graph.add_node(node_prefix, sequence=sub_satellites[0][0])
        for prev in previous_nodes:
            graph.add_edge(prev, node_prefix)
        return [node_prefix]

    # Multi-segment or ambiguous case: chain sub-nodes with _k suffixes
    current_nodes = list(previous_nodes)
    for k, segment in enumerate(sub_satellites):
        node_id_base = f"{'sub_' if k > 0 else ''}{node_prefix}_{k}"
        next_nodes = []

        if len(segment) > 1:
            # Ambiguous position: one node per possible base
            for base in segment:
                node_id = f"{node_id_base}_{base}"
                graph.add_node(node_id, sequence=base)
                for prev in current_nodes:
                    graph.add_edge(prev, node_id)
                next_nodes.append(node_id)
        else:
            graph.add_node(node_id_base, sequence=segment[0])
            for prev in current_nodes:
                graph.add_edge(prev, node_id_base)
            next_nodes.append(node_id_base)

        current_nodes = next_nodes

    return current_nodes


def _get_satellite_first_nodes(sub_satellites: list[list[str]], node_prefix: str) -> list[str]:
    """Return the first node ids of a satellite (used to construct self-loops in repeat graphs)."""
    # Simple case: single unambiguous segment
    if len(sub_satellites) == 1 and len(sub_satellites[0]) == 1:
        return [node_prefix]
    # First segment is ambiguous: one node per base
    if len(sub_satellites[0]) > 1:
        return [f"{node_prefix}_0_{base}" for base in sub_satellites[0]]
    # First segment is unambiguous
    return [f"{node_prefix}_0"]


def _add_break_to_graph(
    graph: nx.DiGraph,
    break_seq: str,
    node_prefix: str,
    previous_nodes: list[str],
) -> list[str]:
    """Add break nodes to the graph, expanding any IUPAC ambiguous bases.

    Reuses _parse_iupac_sequence and _add_satellite_copy_to_graph — no self-loop is
    added since breaks are traversed exactly once. Returns the new frontier node ids.
    """
    sub_segments = _parse_iupac_sequence(break_seq)
    return _add_satellite_copy_to_graph(graph, sub_segments, node_prefix, previous_nodes)


def create_repeat_graph(locus: Locus) -> nx.DiGraph:
    """Build a directed repeat graph for a locus using NetworkX.

    Each satellite is represented as a node (or sub-nodes for IUPAC ambiguity) with a
    self-loop to allow multiple copies. Breaks are single-pass nodes between satellites.
    """
    # TODO: Make this into a parameter so that you can check for anchor -> this means the overlap is big enough
    left_anchor = locus.left_anchor[: -config.min_anchor_overlap]
    left_anchor_overlap = locus.left_anchor[-config.min_anchor_overlap :]
    right_anchor = locus.right_anchor[config.min_anchor_overlap :]
    right_anchor_overlap = locus.right_anchor[: config.min_anchor_overlap]

    graph = nx.DiGraph()
    graph.add_node("left_anchor", sequence=left_anchor)
    graph.add_node("left_anchor_overlap", sequence=left_anchor_overlap)
    graph.add_edge("left_anchor", "left_anchor_overlap")
    previous_nodes = ["left_anchor_overlap"]

    for i, (satellite, pre_break) in enumerate(zip(locus.satellites, locus.breaks)):
        if pre_break:
            previous_nodes = _add_break_to_graph(graph, pre_break, f"break_{i}", previous_nodes)

        # Each alternative gets its own set of nodes, all branching from the same
        # previous_nodes and feeding into the same next frontier.
        # Cross-edges between alternatives are added so the aligner can switch
        # between alternatives on each new copy (e.g. CGG→CAA→CGG in one read).
        all_last_nodes: list[str] = []
        all_first_nodes: list[str] = []
        for alt_idx, alt_seq in enumerate(satellite.sequences):
            alt_prefix = f"satellite_{i}" if len(satellite.sequences) == 1 else f"satellite_{i}_alt{alt_idx}"
            sub_satellites = _parse_iupac_sequence(alt_seq)
            last_nodes = _add_satellite_copy_to_graph(graph, sub_satellites, alt_prefix, previous_nodes)
            first_nodes = _get_satellite_first_nodes(sub_satellites, alt_prefix)
            all_last_nodes.extend(last_nodes)
            all_first_nodes.extend(first_nodes)

        # Connect every last node to every first node across all alternatives.
        # This gives both self-loops (same alt → same alt) and cross-alternative
        # transitions (alt0 → alt1, alt1 → alt0), so the aligner can freely mix
        # alternatives within a single repeat run.
        for last, first in itertools.product(all_last_nodes, all_first_nodes):
            graph.add_edge(last, first)

        previous_nodes = previous_nodes + all_last_nodes if satellite.skippable else all_last_nodes

    if locus.breaks[-1]:
        previous_nodes = _add_break_to_graph(graph, locus.breaks[-1], f"break_{len(locus.breaks) - 1}", previous_nodes)

    graph.add_node("right_anchor_overlap", sequence=right_anchor_overlap)
    for prev_node in previous_nodes:
        graph.add_edge(prev_node, "right_anchor_overlap")
    graph.add_node("right_anchor", sequence=right_anchor)
    graph.add_edge("right_anchor_overlap", "right_anchor")

    return graph


def create_linear_graph(locus: Locus, satellite_counts: list[int]) -> nx.DiGraph:
    """Build a directed linear graph for a locus with a fixed number of satellite copies.

    Unlike create_repeat_graph, each satellite copy gets its own node (no self-loops),
    so the graph encodes a specific repeat length per satellite.
    """
    left_anchor = locus.left_anchor[: -config.min_anchor_overlap]
    left_anchor_overlap = locus.left_anchor[-config.min_anchor_overlap :]
    right_anchor = locus.right_anchor[config.min_anchor_overlap :]
    right_anchor_overlap = locus.right_anchor[: config.min_anchor_overlap]

    graph = nx.DiGraph()
    graph.add_node("left_anchor", sequence=left_anchor)
    graph.add_node("left_anchor_overlap", sequence=left_anchor_overlap)
    graph.add_edge("left_anchor", "left_anchor_overlap")
    previous_nodes = ["left_anchor_overlap"]

    for i, (satellite, pre_break) in enumerate(zip(locus.satellites, locus.breaks)):
        if pre_break:
            previous_nodes = _add_break_to_graph(graph, pre_break, f"break_{i}", previous_nodes)

        for j in range(satellite_counts[i]):
            # Each copy of each alternative gets its own node; chain copies sequentially
            all_last_nodes = []
            for alt_idx, alt_seq in enumerate(satellite.sequences):
                alt_prefix = f"satellite_{i}_{j}" if len(satellite.sequences) == 1 else f"satellite_{i}_{j}_alt{alt_idx}"
                sub_satellites = _parse_iupac_sequence(alt_seq)
                all_last_nodes.extend(_add_satellite_copy_to_graph(graph, sub_satellites, alt_prefix, previous_nodes))
            previous_nodes = all_last_nodes

    if locus.breaks[-1]:
        previous_nodes = _add_break_to_graph(graph, locus.breaks[-1], f"break_{len(locus.breaks) - 1}", previous_nodes)

    graph.add_node("right_anchor_overlap", sequence=right_anchor_overlap)
    for prev_node in previous_nodes:
        graph.add_edge(prev_node, "right_anchor_overlap")
    graph.add_node("right_anchor", sequence=right_anchor)
    graph.add_edge("right_anchor_overlap", "right_anchor")

    return graph


def get_graph_alignments(reads: list[Read], graph: nx.DiGraph) -> list[GraphAlignment]:
    graph_str = graph_to_gfa(graph)

    fastq_str = "".join(read.to_fastq() for read in reads)

    with tempfile.TemporaryDirectory() as _temp_dir:
        input_graph_gfa = Path(_temp_dir) / "graph.gfa"
        input_graph_gfa.write_text(graph_str)

        _t0 = time.perf_counter()
        process = subprocess.run(
            ["minigraph", "-c", "-j", "0.3", "-x", "lr", str(input_graph_gfa), "-"],
            input=fastq_str,
            capture_output=True,
            text=True,
            check=False,
        )
        logger.debug(f"[TIMING] minigraph alignment: {time.perf_counter() - _t0:.3f}s  ({len(reads)} reads)")
        if process.returncode != 0:
            logger.error("minigraph stdout:\n%s", process.stdout)
            logger.debug("minigraph stderr:\n%s", process.stderr)
            msg = f"minigraph failed with return code {process.returncode}"
            raise RuntimeError(msg)

        output_string = process.stdout

    graph_alignments: list[GraphAlignment] = []
    for read in reads:
        gaf_lines = next((line for line in output_string.split("\n") if line.startswith(read.name)), None)
        if gaf_lines is None:
            continue
        graph_alignments.append(GraphAlignment.from_gaf_line(read=read, gaf_line=gaf_lines, graph=graph))

    return graph_alignments


def get_kmer_string(locus: Locus, synced_list: list[str], satellite_counts: list[int]) -> str:
    # Get satellite sequences and counts
    satellite_seqs = [sat.sequences[0] for sat in locus.satellites]

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

    # Build graph once; reuse for both the initial alignment and flanking remap
    graph = create_repeat_graph(locus)
    graph_alignments = get_graph_alignments(reads, graph)

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
    remapped_flanking_alignments, unmapped_flanking_reads = remap_flanking_alignments_to_locus(flanking_alignments, locus, graph=graph)

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
    graph: nx.DiGraph,
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

    # Re-map the synthetic reads (reuse the same graph)
    remapped_flanking_reads = get_graph_alignments(synthetic_reads, graph)

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
            "str_bp_length": len(self.alignment.str_sequence),
            "obs_kmer_string": self.obs_kmer_string,
            "ref_kmer_string": self.ref_kmer_string,
            "mod_5mc_kmer_string": self.mod_5mc_kmer_string,
            "qual_kmer_string": self.qual_kmer_string,
            "str_error_rate": self.str_error_rate,
            "haplotype": self.haplotype,
            "filter_reasons": ";".join(self.outlier_reasons),
        }

    def add_outlier_reason(self, reason: str) -> ReadCall:
        self.add_outlier_reasons([reason])

        return self

    def add_outlier_reasons(self, reasons: list[str]) -> ReadCall:
        self.haplotype = Haplotype.OUTLIER
        self.outlier_reasons.extend(reasons)

        return self

    def add_qc_filter_reason(self, reason: str) -> ReadCall:
        self.haplotype = Haplotype.QC_FILTERED
        self.outlier_reasons.append(reason)

        return self

    def add_qc_filter_reasons(self, reasons: list[str]) -> ReadCall:
        self.haplotype = Haplotype.QC_FILTERED
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
