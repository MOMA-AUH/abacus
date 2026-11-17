import array

import pysam
import pytest

from abacus.graph import (
    GraphAlignment,
    Locus,
    Read,
    create_repeat_graph,
    get_graph_alignments,
    get_kmer_string,
    get_read_calls,
    get_reference_sequence_from_path,
    get_satellite_copy_lengths_from_path,
    remap_flanking_alignments_to_locus,
    split_into_copies,
)
from abacus.locus import Location
from abacus.utils import AlignmentType
from tests.helpers import make_locus, make_read, make_reads


def test_empty_reads_do_not_need_a_graph():
    graph = create_repeat_graph(make_locus("(CAG)+"))
    graph.clear()
    assert get_graph_alignments([], graph) == []


def test_synthetic_read_aligned_only_inside_left_anchor_has_no_repeat_evidence():
    locus = make_locus("(TTTAT)*")
    graph = create_repeat_graph(locus)
    read = make_read("anchor_only", locus.left_anchor, locus)
    read.n_left_synthetic_bases = len(locus.left_anchor) - 10
    anchor_length = len(graph.nodes["left_anchor"]["sequence"])
    aligned_length = anchor_length - 1
    # A valid local hit can end before reaching the inner anchor or repeat.
    gaf_line = "\t".join(map(str, [
        read.name, len(read.sequence), 0, aligned_length, "+", ">left_anchor",
        anchor_length, 0, aligned_length, aligned_length, aligned_length, 60,
        f"cg:Z:{aligned_length}M",
    ]))

    alignment = GraphAlignment.from_gaf_line(read, gaf_line, graph)

    assert alignment.type == AlignmentType.NO_ANCHORS
    assert alignment.str_sequence == ""
    assert alignment.str_qualities == []
    assert alignment.str_mod_5mc_synced == []
    assert alignment.str_cigar_synced == []


def test_read_ending_in_short_flank_is_spanning():
    locus = make_locus("(CTCC)+")
    repeat = "CTCC" * 3
    reads = [
        make_read("short_left", locus.left_anchor[-60:] + repeat + locus.right_anchor, locus),
        make_read("short_right", locus.left_anchor + repeat + locus.right_anchor[:60], locus),
    ]

    alignments = get_graph_alignments(reads, create_repeat_graph(locus))

    assert {a.name: a.type for a in alignments} == {
        "short_left": AlignmentType.SPANNING,
        "short_right": AlignmentType.SPANNING,
    }


def test_read_continuing_beyond_short_flank_is_flanking():
    locus = make_locus("(CTCC)+")
    repeat = "CTCC" * 3
    unaligned_length = 1000
    unaligned = "G" * unaligned_length
    reads = [
        # Genomic tail -> short left anchor -> repeat -> full right anchor; mirrored below.
        make_read("unaligned_left", unaligned + locus.left_anchor[-60:] + repeat + locus.right_anchor, locus),
        make_read("unaligned_right", locus.left_anchor + repeat + locus.right_anchor[:60] + unaligned, locus),
    ]

    alignments = get_graph_alignments(reads, create_repeat_graph(locus))

    assert {a.name: a.type for a in alignments} == {
        "unaligned_left": AlignmentType.RIGHT_FLANKING,
        "unaligned_right": AlignmentType.LEFT_FLANKING,
    }

    calls, filtered = get_read_calls(reads, locus)

    # Both inner anchors are mapped: keep the original count and flanking classification.
    assert {call.alignment.name: call.satellite_count for call in calls} == {
        "unaligned_left": [3],
        "unaligned_right": [3],
    }
    assert all(not call.is_spanning() for call in calls)
    assert not filtered


def test_read_reaching_outer_anchors_is_spanning_despite_unaligned_ends():
    locus = make_locus("(CTCC)+")
    repeat = "CTCC" * 3
    spanning = locus.left_anchor + repeat + locus.right_anchor
    unaligned_length = 1000
    unaligned = "G" * unaligned_length
    outer_right_flank = create_repeat_graph(locus).nodes["right_anchor"]["sequence"]
    reads = [
        make_read("unaligned_left", unaligned + spanning, locus, n_soft_clipped_left=unaligned_length),
        make_read("unaligned_right", spanning + unaligned, locus, n_soft_clipped_right=unaligned_length),
        # Left anchor -> 3 repeats -> right anchor -> spacer -> duplicate outer right flank.
        # Minigraph reports the spanning hit first, then a flank-only hit that must not replace it.
        make_read("extra_flank_hit", spanning + unaligned + outer_right_flank, locus),
    ]

    calls, filtered = get_read_calls(reads, locus)

    assert {call.alignment.name: call.satellite_count for call in calls} == {
        "unaligned_left": [3],
        "unaligned_right": [3],
        "extra_flank_hit": [3],
    }
    assert all(call.is_spanning() for call in calls)
    assert not filtered


@pytest.mark.parametrize(
    ("structure", "read_type", "repeat_str", "expected_counts", "expected_obs_kmers", "expected_ref_kmers"),
    [
        # One satellite
        pytest.param(
            "(GAA)*",
            "left",
            "GAA" * 5 + "GA",
            [6],
            ["GAA"] * 5 + ["GA"],
            ["GAA"] * 6,
            id="1-satellite-short-terminal-kmer",
        ),
        pytest.param(
            "(AAG)*",
            "right",
            "AG" + "AAG" * 5,
            [6],
            ["AG"] + ["AAG"] * 5,
            ["AAG"] * 6,
            id="1-satellite-right-short-terminal-kmer",
        ),
        pytest.param(
            "(GAA)*",
            "left",
            "GAA" * 5 + "GAAT",
            [6],
            ["GAA"] * 5 + ["GAAT"],
            ["GAA"] * 6,
            id="1-satellite-long-terminal-kmer",
        ),
        pytest.param(
            "(AAG)*",
            "right",
            "TAAG" + "AAG" * 5,
            [6],
            ["TAAG"] + ["AAG"] * 5,
            ["AAG"] * 6,
            id="1-satellite-right-long-terminal-kmer",
        ),
        # Two satellites
        pytest.param(
            "(A)*(GAA)*",
            "right",
            "A" + "GAA" * 5,
            [0, 5],
            ["GAA"] * 5,
            ["GAA"] * 5,
            id="2-satellites-right-false-a-tract",
        ),
        pytest.param(
            "(AAG)*(A)*",
            "left",
            "AAG" * 5 + "A",
            [5, 0],
            ["AAG"] * 5,
            ["AAG"] * 5,
            id="2-satellites-left-false-a-tract",
        ),
        pytest.param(
            "(A)*(GAA)*",
            "right",
            "A" * 2 + "GAA" * 5,
            [0, 5],
            ["GAA"] * 5,
            ["GAA"] * 5,
            id="2-satellites-right-false-2-a-tract",
        ),
        pytest.param(
            "(AAG)*(A)*",
            "left",
            "AAG" * 5 + "A" * 2,
            [5, 0],
            ["AAG"] * 5,
            ["AAG"] * 5,
            id="2-satellites-left-false-2-a-tract",
        ),
        pytest.param(
            "(A)*(GAA)*",
            "right",
            "A" * 3 + "GAA" * 5,
            [3, 5],
            ["A"] * 3 + ["GAA"] * 5,
            ["A"] * 3 + ["GAA"] * 5,
            id="2-satellites-right-true-3-a-tract",
        ),
        pytest.param(
            "(AAG)*(A)*",
            "left",
            "AAG" * 5 + "A" * 3,
            [5, 3],
            ["AAG"] * 5 + ["A"] * 3,
            ["AAG"] * 5 + ["A"] * 3,
            id="2-satellites-left-true-3-a-tract",
        ),
        pytest.param(
            "(C)*(GAA)*",
            "right",
            "C" + "GAA" * 5,
            [1, 5],
            ["C"] + ["GAA"] * 5,
            ["C"] * 1 + ["GAA"] * 5,
            id="2-satellites-right-true-c-tract",
        ),
        pytest.param(
            "(A)*(GAA)*",
            "spanning",
            "A" + "GAA" * 5,
            [1, 5],
            ["A"] + ["GAA"] * 5,
            ["A"] * 1 + ["GAA"] * 5,
            id="2-satellites-spanning-single-a",
        ),
        # Three satellites
        pytest.param(
            "(GAA)*(CAG)*(CGG)*",
            "left",
            "GAA" * 5 + "GA",
            [6, 0, 0],
            ["GAA"] * 5 + ["GA"],
            ["GAA"] * 6,
            id="3-satellites-ends-in-first",
        ),
        pytest.param(
            "(GGC)*(GAC)*(AAG)*",
            "right",
            "AG" + "AAG" * 5,
            [0, 0, 6],
            ["AG"] + ["AAG"] * 5,
            ["AAG"] * 6,
            id="3-satellites-starts-in-third",
        ),
        pytest.param(
            "(GAA)*(CAG)*(CGG)*",
            "left",
            "GAA" * 5 + "CAG" * 5 + "CA",
            [5, 6, 0],
            ["GAA"] * 5 + ["CAG"] * 5 + ["CA"],
            ["GAA"] * 5 + ["CAG"] * 6,
            id="3-satellites-ends-in-second",
        ),
        pytest.param(
            "(GGC)*(GAC)*(AAG)*",
            "right",
            "AC" + ("GAC" * 5 + "AAG" * 5),
            [0, 6, 5],
            ["AC"] + ["GAC"] * 5 + ["AAG"] * 5,
            ["GAC"] * 6 + ["AAG"] * 5,
            id="3-satellites-starts-in-second",
        ),
        pytest.param(
            "(GAA)*(CAG)*(CGG)*",
            "left",
            "GAA" * 5 + "CAG" * 5 + "CGG" * 5 + "CG",
            [5, 5, 6],
            ["GAA"] * 5 + ["CAG"] * 5 + ["CGG"] * 5 + ["CG"],
            ["GAA"] * 5 + ["CAG"] * 5 + ["CGG"] * 6,
            id="3-satellites-ends-in-third",
        ),
        pytest.param(
            "(GGC)*(GAC)*(AAG)*",
            "right",
            "GC" + "GGC" * 5 + "GAC" * 5 + "AAG" * 5,
            [6, 5, 5],
            ["GC"] + ["GGC"] * 5 + (["GAC"] * 5 + ["AAG"] * 5),
            ["GGC"] * 6 + ["GAC"] * 5 + ["AAG"] * 5,
            id="3-satellites-starts-in-first",
        ),
        # Breaks separate repeat runs or occupy the free end.
        pytest.param(
            "(AAG)*C(A)*",
            "left",
            "AAGCA",
            [1, 1],
            ["AAG", "C", "A"],
            ["AAG", "C", "A"],
            id="left-break-between-runs",
        ),
        pytest.param(
            "(AAG)*(A)*C",
            "left",
            "AAGAC",
            [1, 1],
            ["AAG", "A", "C"],
            ["AAG", "A", "C"],
            id="left-break-at-free-end",
        ),
        pytest.param(
            "(A)*C(GAA)*",
            "right",
            "ACGAA",
            [1, 1],
            ["A", "C", "GAA"],
            ["A", "C", "GAA"],
            id="right-break-between-runs",
        ),
        pytest.param(
            "C(A)*(GAA)*",
            "right",
            "CAGAA",
            [1, 1],
            ["C", "A", "GAA"],
            ["C", "A", "GAA"],
            id="right-break-at-free-end",
        ),
    ],
)
def test_read_calls_resolve_terminal_kmers(
    structure,
    read_type,
    repeat_str,
    expected_counts,
    expected_obs_kmers,
    expected_ref_kmers,
):
    locus = make_locus(structure)
    calls, filtered = get_read_calls(make_reads([repeat_str], locus, read_type), locus)
    assert not filtered
    assert [call.satellite_count for call in calls] == [expected_counts]
    assert calls[0].obs_kmer_string.split("|") == expected_obs_kmers
    assert calls[0].ref_kmer_string.split("|") == expected_ref_kmers
    assert calls[0].qual_kmer_string.split("|") == ["?" * len(kmer) for kmer in expected_obs_kmers]
    assert calls[0].mod_5mc_kmer_string.split("|") == ["!" * len(kmer) for kmer in expected_obs_kmers]
    assert calls[0].alignment.str_sequence == "".join(expected_obs_kmers)


@pytest.mark.parametrize(
    "matching_anchor",
    [
        pytest.param(True, id="matching-anchor"),
        pytest.param(False, id="contrasting-anchor"),
    ],
)
@pytest.mark.parametrize(
    ("read_type", "motif", "fragment"),
    [
        pytest.param("right", "AAG", "AG", id="right-two-bases"),
        pytest.param("right", "AAG", "G", id="right-one-base"),
        pytest.param("left", "GAA", "GA", id="left-two-bases"),
        pytest.param("left", "GAA", "G", id="left-one-base"),
    ],
)
def test_flanking_remap_preserves_repeat(read_type, motif, fragment, matching_anchor):
    locus = make_locus(f"({motif})*")
    missing = motif[: -len(fragment)] if read_type == "right" else motif[len(fragment) :]
    boundary = missing if matching_anchor else "C" * len(missing)
    # Control whether the synthetic anchor can complete the terminal motif.
    if read_type == "right":
        locus.left_anchor = locus.left_anchor[: -len(boundary)] + boundary
        repeat = fragment + motif * 5
        expected_type = AlignmentType.RIGHT_FLANKING
    else:
        locus.right_anchor = boundary + locus.right_anchor[len(boundary) :]
        repeat = motif * 5 + fragment
        expected_type = AlignmentType.LEFT_FLANKING

    graph = create_repeat_graph(locus)
    initial = get_graph_alignments(make_reads([repeat], locus, read_type), graph)
    assert len(initial) == 1
    assert initial[0].type == expected_type

    remapped, filtered = remap_flanking_alignments_to_locus(initial, locus, graph)

    assert not filtered
    assert len(remapped) == 1
    assert remapped[0].type == expected_type
    assert remapped[0].str_sequence == repeat


@pytest.mark.parametrize("read_type", ["left", "right"])
def test_flanking_remap_preserves_repeat_with_anchor_deletion(read_type):
    locus = make_locus("(CAG)*")
    repeat = "CAG" * 10
    overlap = locus.right_anchor[:10] if read_type == "left" else locus.left_anchor[-10:]
    overlap = overlap[:5] + overlap[6:]  # One deletion in the original anchor overlap.
    sequence = repeat + overlap if read_type == "left" else overlap + repeat
    graph = create_repeat_graph(locus)
    initial = get_graph_alignments(make_reads([sequence], locus, read_type), graph)

    remapped, filtered = remap_flanking_alignments_to_locus(initial, locus, graph)

    assert not filtered
    assert len(remapped) == 1
    assert remapped[0].str_sequence == repeat


@pytest.mark.parametrize(
    "structure, read, expected_satellite_counts, expected_str_reference",
    [
        # Single satellite tests
        pytest.param(
            "(AGA)+",
            "AGA",
            [1],
            "AGA",
            id="Single satellite x 1",
        ),
        pytest.param(
            "(CAG)+",
            "CAG" * 10,
            [10],
            "CAG" * 10,
            id="Single satellite x 10",
        ),
        pytest.param(
            "(TTA)+",
            "TTA" * 100,
            [100],
            "TTA" * 100,
            id="Single satellite x 100",
        ),
        # Multiple satellites tests
        pytest.param(
            "(AGA)+(CAG)+",
            "AGA" * 3 + "CAG" * 5,
            [3, 5],
            "AGA" * 3 + "CAG" * 5,
            id="Two satellites x (3, 5)",
        ),
        # Ambiguous bases
        # N = A or C or G or T
        # R = A or G
        # Y = C or T
        # W = A or T
        # K = G or T
        pytest.param(
            "(ANA)+",
            "AGA",
            [1],
            "AGA",
            id="Ambiguity, N: x (1)",
        ),
        pytest.param(
            "(AGA)+(ANA)+",
            "AGA" * 5 + "ATA" * 3,
            [5, 3],
            "AGA" * 5 + "ATA" * 3,
            id="Ambiguity, N: x (5 ,3)",
        ),
        pytest.param(
            "(N)+(AGA)+",
            "T" * 5 + "AGA",
            [5, 1],
            "T" * 5 + "AGA",
            id="Ambiguity, N: x (5, 1)",
        ),
        pytest.param(
            "(N)+",
            "T" * 19,
            [19],
            "T" * 19,
            id="Ambiguity, N: x (19)",
        ),
        pytest.param(
            "(ARA)+",
            "AGA",
            [1],
            "AGA",
            id="Ambiguity, R: x (1)",
        ),
        pytest.param(
            "(ATA)+(ARA)+",
            "ATA" * 5 + "AGA" * 3,
            [5, 3],
            "ATA" * 5 + "AGA" * 3,
            id="Ambiguity, R: x (5 ,3)",
        ),
        pytest.param(
            "(R)+(CAC)+(Y)+",
            "G" * 5 + "CAC" * 3 + "T" * 7,
            [5, 3, 7],
            "G" * 5 + "CAC" * 3 + "T" * 7,
            id="Ambiguity, R,Y: x (5, 3, 7)",
        ),
        # Skippability
        pytest.param(
            "(TTC)*",
            "",
            [0],
            "",
            id="Skippability x 0",
        ),
        pytest.param(
            "(TTC)*",
            "TTC" * 17,
            [17],
            "TTC" * 17,
            id="Skippability x 17",
        ),
        pytest.param(
            "(AGA)*(CAG)*",
            "",
            [0, 0],
            "",
            id="Skippability x (0, 0)",
        ),
        pytest.param(
            "(AGA)*(CAG)*",
            "AGA" * 11 + "CAG" * 13,
            [11, 13],
            "AGA" * 11 + "CAG" * 13,
            id="Skippability x (11, 13)",
        ),
        pytest.param(
            "(AGA)*(CAG)*",
            "CAG",
            [0, 1],
            "CAG",
            id="Skippability x (0, 1)",
        ),
        # Skippability with N
        pytest.param(
            "(N)*(AGA)*",
            "T" * 0 + "AGA" * 7,
            [0, 7],
            "AGA" * 7,
            id="Skippability with N x (0, 7)",
        ),
        # Breaks
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 0 + "TTTTT" + "CAG" * 0,
            [0, 0],
            "TTTTT",
            id="Breaks: Internal x (0, 0)",
        ),
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 7 + "TTTTT" + "CAG" * 0,
            [7, 0],
            "AGA" * 7 + "TTTTT",
            id="Breaks: Internal x (7, 0)",
        ),
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 0 + "TTTTT" + "CAG" * 5,
            [0, 5],
            "TTTTT" + "CAG" * 5,
            id="Breaks: Internal x (0, 5)",
        ),
        pytest.param(
            "(AGA)*TTTTT(CAG)*",
            "AGA" * 7 + "TTTTT" + "CAG" * 5,
            [7, 5],
            "AGA" * 7 + "TTTTT" + "CAG" * 5,
            id="Breaks: Internal x (7, 5)",
        ),
        pytest.param(
            "AAC(AAG)*(CAG)*AGG",
            "AAC" + "AAG" * 11 + "CAG" * 13 + "AGG",
            [11, 13],
            "AAC" + "AAG" * 11 + "CAG" * 13 + "AGG",
            id="Breaks: Pre and post x (11, 13)",
        ),
        pytest.param(
            "AAC(AAG)*TTT(CAG)*AGG",
            "AAC" + "AAG" * 3 + "TTT" + "CAG" * 2 + "AGG",
            [3, 2],
            "AAC" + "AAG" * 3 + "TTT" + "CAG" * 2 + "AGG",
            id="Breaks: Pre, internal and post x (3, 2)",
        ),
        # Edge cases
        pytest.param(
            "(AGA)+(CAG)*",
            "AGA" * 0 + "CAG" * 3,
            [1, 2],
            None,  # str_reference != read: forced AGA mismatch due to + constraint
            id="Edge case: Forced mismatch with +",
        ),
        pytest.param(
            "(TTTTTTTTTC)+",
            "AAAAAAAAAC" * 29,
            [29],
            None,  # str_reference != read: high-mismatch satellite
            id="Edge case: High percentage of mismatches in satellite",
        ),
        pytest.param(
            "(NTN)+(TNT)+",
            "ATG" * 3 + "TAT" * 17,
            [3, 17],
            "ATG" * 3 + "TAT" * 17,
            id="Edge case: High percentage of Ns in satellites",
        ),
        pytest.param(
            "(AGN)*",
            "AGT" * 100,
            [100],
            "AGT" * 100,
            id="Edge case: Many copies",
        ),
        # OR operator tests
        pytest.param(
            "(CAG|CAA)+",
            "CAG" * 5,
            [5],
            "CAG" * 5,
            id="OR operator: pure CAG x 5",
        ),
        pytest.param(
            "(CAG|CAA)+",
            "CAA" * 5,
            [5],
            "CAA" * 5,
            id="OR operator: pure CAA x 5",
        ),
        pytest.param(
            "(CAG|CAA)*",
            "",
            [0],
            "",
            id="OR operator: skippable x 0",
        ),
        pytest.param(
            "(CAG|CAA)*",
            "CAG" * 8,
            [8],
            "CAG" * 8,
            id="OR operator: skippable CAG x 8",
        ),
        pytest.param(
            "(CAG|CAA)+(CGG)+",
            "CAG" * 3 + "CGG" * 4,
            [3, 4],
            "CAG" * 3 + "CGG" * 4,
            id="OR operator: two satellites, CAG branch x (3, 4)",
        ),
        pytest.param(
            "(CAG|CAA)+(CGG)+",
            "CAA" * 3 + "CGG" * 4,
            [3, 4],
            "CAA" * 3 + "CGG" * 4,
            id="OR operator: two satellites, CAA branch x (3, 4)",
        ),
        # Mixed alternation: some copies use one alternative, some use the other
        pytest.param(
            "(CGG|CAA)+",
            "CGG" * 2 + "CAA" + "CGG" * 2,
            [5],
            "CGG" * 2 + "CAA" + "CGG" * 2,
            id="OR operator: mixed alts, 1 CAA singlet inside CGG repeats",
        ),
        pytest.param(
            "(CGG|CAA)+",
            "CGG" * 4 + "CAA" + "CGG" * 3 + "CAA" + "CGG" * 2,
            [11],
            "CGG" * 4 + "CAA" + "CGG" * 3 + "CAA" + "CGG" * 2,
            id="OR operator: mixed alts, 2 CAA singlets scattered in CGG repeats",
        ),
        pytest.param(
            "(CAG|CAA)+",
            "CAG" * 5 + "CAA" + "CAG" * 4,
            [10],
            "CAG" * 5 + "CAA" + "CAG" * 4,
            id="OR operator: mixed alts, CAA singlet inside CAG repeats",
        ),
    ],
)
def test_get_satellite_copy_lengths_from_path(structure, read, expected_satellite_counts, expected_str_reference):
    locus = make_locus(structure)
    reads = [make_read("read", locus.left_anchor + read + locus.right_anchor, locus)]

    graph_alignments = get_graph_alignments(reads, create_repeat_graph(locus))
    graph_alignment = next(a for a in graph_alignments if a.name == reads[0].name)
    path = graph_alignment.path
    satellite_copy_lengths = get_satellite_copy_lengths_from_path(locus=locus, path=path)
    satellite_counts = [len(lengths) for lengths in satellite_copy_lengths]

    assert satellite_counts == expected_satellite_counts
    if expected_str_reference is not None:
        assert graph_alignment.str_reference == expected_str_reference


@pytest.mark.parametrize(
    "structure, read, expected_expected_kmer_string, expected_observed_kmer_string",
    [
        pytest.param(
            "(CAG)*",
            "",
            "",
            "",
            id="Empty read",
        ),
        pytest.param(
            "(CAG)*",
            "CAG" * 10,
            "|".join(["CAG"] * 10),
            "|".join(["CAG"] * 10),
            id="Single satellite x 10",
        ),
        pytest.param(
            "(CAG)*",
            "CAG" * 4 + "TTT" + "CAG" * 5,
            "|".join(["CAG"] * 10),
            "|".join(["CAG"] * 4 + ["TTT"] + ["CAG"] * 5),
            id="Single w error satellite x 10",
        ),
        pytest.param(
            "(GAA|GA)+",
            "GAA" * 3 + "GA" * 2,
            "|".join(["GAA"] * 3 + ["GA"] * 2),
            "|".join(["GAA"] * 3 + ["GA"] * 2),
            id="OR operator: mixed alt lengths, GAA x3 then GA x2",
        ),
        pytest.param(
            "(CGG)+",
            "C" + "CGG" * 3,  # a base inserted right at the repeat's edge, not a chopping artifact
            "|".join(["CGG"] * 3),
            "CCGG|CGG|CGG",
            id="Insertion at the repeat's leading edge must not vanish into the trimmed anchor",
        ),
        pytest.param(
            "(CGG)+",
            "CGG" * 2 + "CG" * 4 + "CGG",
            "|".join(["CGG"] * 7),
            "|".join(["CGG"] * 2 + ["CG"] * 4 + ["CGG"]),
            id="A slipped CGG stretch is four short copies, not two copies with bases inserted",
        ),
        pytest.param(
            "(ACAAG|AAGAC|AACGG|ACGGA|AAAAG|AAGGC)+",
            "ACAAG" * 5 + "AC" + "ACAAG" * 5,
            "|".join(["ACAAG"] * 10),
            # Which neighbour absorbs the insertion is incidental; it must stay one over-long copy.
            "|".join(["ACAAG"] * 4 + ["ACAAGAC"] + ["ACAAG"] * 5),
            id="A genuine insertion stays one over-long copy, not split into fake short ones",
        ),
    ],
)
def test_get_satellite_strings(structure, read, expected_expected_kmer_string, expected_observed_kmer_string):
    locus = make_locus(structure)
    reads = [make_read("read", locus.left_anchor + read + locus.right_anchor, locus)]
    graph_alignment = get_graph_alignments(reads, create_repeat_graph(locus))[0]

    satellite_copy_lengths = get_satellite_copy_lengths_from_path(path=graph_alignment.path, locus=locus)

    # Create kmer strings
    expected_kmer_string = get_kmer_string(
        locus=locus,
        synced_list=[*graph_alignment.str_reference],
        satellite_copy_lengths=satellite_copy_lengths,
    )

    observed_kmer_string = get_kmer_string(
        locus=locus,
        synced_list=graph_alignment.str_sequence_synced,
        satellite_copy_lengths=satellite_copy_lengths,
    )

    assert expected_kmer_string == expected_expected_kmer_string
    assert observed_kmer_string == expected_observed_kmer_string


def test_get_kmer_string_mixed_alternative_lengths():
    # No minigraph involved here - isolates get_kmer_string's own chopping logic
    # from what path the aligner picks (that's covered separately in test_get_satellite_strings).
    locus = make_locus("(GAA|GA)+")

    # 3 copies of GAA followed by 2 copies of GA - a real mixed-length path
    synced_list = list("GAA" * 3 + "GA" * 2)
    satellite_copy_lengths = [[3, 3, 3, 2, 2]]

    kmer_string = get_kmer_string(
        locus=locus,
        synced_list=synced_list,
        satellite_copy_lengths=satellite_copy_lengths,
    )

    assert kmer_string == "|".join(["GAA"] * 3 + ["GA"] * 2)


@pytest.mark.parametrize(
    ("observed", "motif", "expected"),
    [
        pytest.param("CGCG", "CGG", ["CG", "CG"], id="One base short per copy"),
        pytest.param("GAAGAA", "GAA", ["GAA", "GAA"], id="Two whole copies packed into one slot"),
        pytest.param(
            "GAAAGAA",
            "GAA",
            None,
            id="No homogeneous tiling exists - mix of different pieces is rejected, not guessed",
        ),
        pytest.param(
            "A" * 20,
            "AAAAA",
            ["AAAAA"] * 4,
            id="Tie between 4 whole copies and 5 one-base-short copies favors fewer copies",
        ),
        pytest.param("GGGGGG", "CG", None, id="Motifs under 3 bases are never split - too ambiguous"),
        pytest.param("GAA", "GAA", None, id="Not over-long - nothing to split"),
    ],
)
def test_split_into_copies(observed: str, motif: str, expected: list[str] | None):
    assert split_into_copies(observed, motif) == expected


@pytest.mark.parametrize("flank", ["left", "right"])
def test_slipped_flanking_read_is_split(flank: str):
    locus = make_locus("(CGG)+")
    slipped = "CGG" * 6 + "CG" * 4 + "CGG" * 5
    sequence = locus.left_anchor + slipped if flank == "left" else slipped + locus.right_anchor
    read = make_read("read", sequence, locus)

    read_calls, _ = get_read_calls([read], locus)

    assert read_calls[0].obs_kmer_string == "|".join(["CGG"] * 6 + ["CG"] * 4 + ["CGG"] * 5)


@pytest.mark.parametrize(
    ("path", "reference_seq", "expected_reference"),
    [
        pytest.param(
            ["left_anchor", "satellite_0", "right_anchor"],
            "TCG",
            "TCG",
            id="Simple path with 1 satellite",
        ),
        pytest.param(
            ["left_anchor", "satellite_0_0_G", "sub_satellite_0_1", "right_anchor"],
            "NGC",
            "GGC",
            id="Ambiguous base in path (first base)",
        ),
        pytest.param(
            [
                "left_anchor",
                "satellite_0_0",
                "sub_satellite_0_1_G",
                "sub_satellite_0_2_A",
                "sub_satellite_0_3",
                "satellite_0_0",
                "sub_satellite_0_1_A",
                "sub_satellite_0_2_G",
                "sub_satellite_0_3",
                "right_anchor",
            ],
            "AARRG",
            "AAGAGAAAGG",
            id="Ambiguous bases (RFC1)",
        ),
    ],
)
def test_get_reference_sequence_from_path(path: list[str], reference_seq: str, expected_reference: str) -> None:
    # Create a dummy locus and the corresponding repeat graph
    locus = create_synthetic_simple_locus(reference_seq)
    graph = create_repeat_graph(locus)
    reference = get_reference_sequence_from_path(path, locus, graph)
    assert reference == expected_reference


def create_aligned_segment(query_name: str, query_sequence: str, mm_tag: str, ml_tag: list[int]) -> pysam.AlignedSegment:
    a = pysam.AlignedSegment()
    a.query_name = query_name
    a.query_sequence = query_sequence

    # Methylation tags
    a.set_tag("MM", mm_tag)
    a.set_tag("ML", array.array("B", ml_tag))
    return a


def create_synthetic_simple_locus(satellite_seq: str) -> Locus:
    locus = make_locus(f"({satellite_seq})+")
    locus.left_anchor = ""
    locus.right_anchor = ""
    return locus


# Note: Probs are represented with ASCII, i.e.:
# 0 = Not a C
# ! = 0
# \ = 0.1
# # = 0.2
# $ = 0.3
# % = 0.4
# & = 0.5
# ' = 0.6
# ( = 0.7
# ) = 0.8
# * = 0.9
# + = 1
@pytest.mark.parametrize(
    ("query_sequence", "mm_tag", "ml_tag", "expected_methylation"),
    [
        pytest.param(
            "A",
            "C+m?,;",
            [],
            "0",
            id="One base, no methylated C",
        ),
        pytest.param(
            "C",
            "C+m?,0;",
            [0],
            "!",
            id="One base, 1 unmethylated C",
        ),
        pytest.param(
            "C",
            "C+m?,0;",
            [255],
            "+",
            id="With one methylated C",
        ),
        pytest.param(
            "AAAA",
            "C+m?,;",
            [],
            "0000",
            id="Multiple bases, no methylated C",
        ),
        pytest.param(
            "ATCG",
            "C+m,0;",
            [128],
            "00&0",
            id="Multiple bases, 1 methylated C",
        ),
        pytest.param(
            "ATCGATCG",
            "C+m,0,0;",
            [128, 64],
            "00&000#0",
            id="With multiple methylation",
        ),
        pytest.param(
            "ACTTTTTCCAACCCTAACTCGTTCAGTTGCGTATTGCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCAACCCCCACCCTCACCCTCACCCTCACCCTCACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCCTAACCCCTAACCCCTAACCCTAACCCTAACCCCTAACCCCTAACCCCTAACCCCTAACCCTAACCCTAACCCTAACCCAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTCTAACCCTCTAACCCTAACCCTAACCCTCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTACCCTAACCCTACCCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTAACCCTACCCTAACCCCAACCCCAACCCCAACCCCAACCCCAACCCCAACCCTAACCCTAACCCTAACCCTAACCCTACCCTAACCCTAACCCTAACCCTAA",
            "C+h?,7,1;C+m?,7,1;",
            [32, 6, 56, 11],
            "0000000000000000000#000000000!000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            id="Case 1",
        ),
    ],
)
def test_methylation_from_alignment(
    query_sequence: str,
    mm_tag: str,
    ml_tag: list[int],
    expected_methylation: list[float],
) -> None:
    alignment = create_aligned_segment(
        query_name="query_name",
        query_sequence=query_sequence,
        mm_tag=mm_tag,
        ml_tag=ml_tag,
    )
    dummy_locus = Locus(
        id="dummy_locus",
        location=Location(chrom="chr1", start=0, end=0),
        left_anchor="",
        right_anchor="",
        structure="",
        satellites=[],
        breaks=[],
    )
    read = Read.from_alignment(alignment, locus=dummy_locus)

    assert read.mod_5mc_probs == expected_methylation


def observed_kmers(motif: str, repeat: str, read_type: str = "spanning") -> list[str]:
    """Return the kmers abacus observes for a read with `repeat` between the anchors; a flanking read has only its own anchor."""
    locus = make_locus(f"({motif})+")
    left_anchor = locus.left_anchor if read_type in ("spanning", "left") else ""
    right_anchor = locus.right_anchor if read_type in ("spanning", "right") else ""
    sequence = left_anchor + repeat + right_anchor
    read = make_read("read", sequence, locus)
    read_calls, _ = get_read_calls([read], locus)
    return read_calls[0].obs_kmer_string.split("|")


@pytest.mark.parametrize("read_type", ["spanning", "left", "right"])
def test_read_that_does_not_fit_the_motif_has_no_empty_kmers(read_type: str):
    """Minigraph inserts the spacers and deletes whole GT copies; realigned, every copy holds read bases."""
    spacer = "ATAAAATGGGAGTAAGCGTGAACAGACAGCAGAGGCCGCTATTACATATAC"
    repeat = ("GT" * 5 + spacer) * 5

    kmers = observed_kmers("GT", repeat, read_type)

    assert "" not in kmers
    assert "".join(kmers) == repeat
    assert kmers[:5] == ["GT"] * 5


def test_read_with_mismatches_in_every_copy_gets_one_unit_per_copy():
    """Minigraph deletes all 29 copies and inserts the read as one block; realigned, it's one unit per copy."""
    kmers = observed_kmers("TTTTTTTTTC", "AAAAAAAAAC" * 29)

    assert kmers == ["AAAAAAAAAC"] * 29
