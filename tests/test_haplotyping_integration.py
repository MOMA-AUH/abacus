from __future__ import annotations

import random
from typing import Literal

import pytest

from abacus.graph import (
    Read,
    get_read_calls,
)
from abacus.haplotyping import run_haplotyping
from abacus.locus import Location, Locus, Satellite
from abacus.utils import AlignmentType


def create_random_anchor(length: int = 1000) -> str:
    """Create a random anchor sequence."""
    return "".join(random.choices("ATCG", k=length))


def create_synthetic_locus(satellite_seqs: list[str], breaks: list[str] | None = None):
    """Create a synthetic locus with one or multiple satellites."""
    # Create random anchors
    left_anchor = create_random_anchor()
    right_anchor = create_random_anchor()

    # Set default breaks if not provided
    if breaks is None:
        breaks = [""] * (len(satellite_seqs) + 1)

    # Create satellites list
    satellites = []
    start_pos = 1000
    for i, seq in enumerate(satellite_seqs):
        end_pos = start_pos + len(seq)
        satellites.append(
            Satellite(
                id=f"test_{i}",
                sequence=seq,
                location=Location("chr1", start_pos, end_pos),
                skippable=(len(satellite_seqs) > 1),  # Only skippable in complex loci
            ),
        )
        start_pos = end_pos

    # Create locus
    locus = Locus(
        id="test",
        structure="test",
        location=Location(chrom="chr1", start=1000, end=start_pos),
        satellites=satellites,
        breaks=breaks,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )
    return locus


def create_sequence_from_counts(
    counts: int | tuple[int, ...],
    satellite_seqs: list[str],
    breaks: list[str] | None = None,
) -> str:
    """Create sequence from repeat counts for a single satellite or multiple satellites."""
    # Handle simple case (single satellite)
    if isinstance(counts, int):
        return satellite_seqs[0] * counts

    # Set default breaks if not provided
    if breaks is None:
        breaks = [""] * (len(satellite_seqs) + 1)

    # Handle complex case (multiple satellites)
    seq = ""
    for i, count in enumerate(counts):
        # Add break sequence if it exists
        if breaks[i]:
            seq += breaks[i]
        # Add satellite sequence
        seq += satellite_seqs[i] * count
    # Add final break sequence if it exists
    if len(breaks) > len(counts):
        seq += breaks[len(counts)]
    # Return the full sequence
    return seq


def create_synthetic_read(
    counts: int | tuple[int, ...],
    satellite_seqs: list[str],
    breaks: list[str],
    alignment_type: AlignmentType,
    locus: Locus,
    read_name: str,
) -> Read:
    """Create a synthetic read for testing."""
    left_anchor = locus.left_anchor
    right_anchor = locus.right_anchor

    # Create sequence from counts
    sequence = create_sequence_from_counts(counts, satellite_seqs, breaks)

    # Create full sequence based on alignment type
    if alignment_type == AlignmentType.SPANNING:
        full_sequence = left_anchor + sequence + right_anchor
    elif alignment_type == AlignmentType.LEFT_FLANKING:
        full_sequence = left_anchor + sequence
    else:  # RIGHT_FLANKING
        full_sequence = sequence + right_anchor

    # Create synthetic read
    return Read(
        name=read_name,
        sequence=full_sequence,
        qualities=[30] * len(full_sequence),
        mod_5mc_probs="!" * len(full_sequence),
        strand="+",
        n_soft_clipped_left=0,
        n_soft_clipped_right=0,
        locus=locus,
    )


@pytest.mark.parametrize(
    (
        "satellite_seqs",
        "breaks",
        "spanning_counts",
        "left_flanking_counts",
        "right_flanking_counts",
        "expected_group_sizes",
        "expected_means",
    ),
    [
        # Simple test cases (single satellite)
        pytest.param(
            ["CAG"],  # Single satellite as list[str]
            ["", ""],  # Start and end breaks
            [],  # No spanning reads
            [],  # No left flanking reads
            [],  # No right flanking reads
            # Expected group sizes
            {},
            # Expected heterozygous mean
            None,
            id="Simple-Empty",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [1],  # Single spanning read with 1 repeat
            [],
            [],
            # Expected group sizes
            {"hom": 1},
            # Expected heterozygous mean
            None,
            id="Simple-Single spanning",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [2, 2],  # Two spanning reads with 2 repeats each
            [1],  # One left flanking read with 1 repeat
            [1],  # One right flanking read with 1 repeat
            # Expected group sizes
            {"hom": 4},
            # Expected heterozygous mean
            None,
            id="Simple-Homozygous - Small",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [
                # Haplotype 1
                1,
                1,
                1,
                # Haplotype 2
                3,
                3,
                3,
                3,
            ],
            [],
            [],
            # Expected group sizes
            {"h1": 3, "h2": 4},
            # Expected heterozygous mean
            {"h1": [1.0], "h2": [3.0]},
            id="Simple-Heterozygous - Small",
        ),
        # Complex test cases (multiple satellites)
        pytest.param(
            ["CAG", "CTG"],  # Two different satellites
            ["", "", ""],  # Three breaks (start, middle, end)
            [
                # Haplotype 1
                (2, 3),
                (2, 3),
                # Haplotype 2
                (4, 2),
                (4, 2),
                (4, 2),
            ],
            [],
            [],
            {"h1": 2, "h2": 3},
            {"h1": [2.0, 3.0], "h2": [4.0, 2.0]},
            id="Complex-Two satellites - only spanning",
        ),
        pytest.param(
            ["CAG", "CTG"],
            ["", "", ""],
            [
                # Haplotype 1
                (2, 3),
                (2, 3),
                # Haplotype 2
                (4, 2),
                (4, 2),
                (4, 2),
            ],
            [
                (2, 1),
            ],
            [
                (0, 2),
            ],
            {"h1": 3, "h2": 4},
            {"h1": [2.0, 3.0], "h2": [4.0, 2.0]},
            id="Complex-Two satellites - with flanking",
        ),
        # Cases with 2 satellites and breaks (HTT is a good example)
        pytest.param(
            ["CAG", "CCG"],  # Two different satellites
            ["", "CAACAG", ""],  # Break in between
            [
                # Haplotype 1
                (2, 3),
                (2, 3),
                (2, 3),
                # Haplotype 2
                (5, 2),
                (5, 2),
                (5, 2),
                (5, 2),
            ],
            [],
            [],
            {"h1": 3, "h2": 4},
            {"h1": [2.0, 3.0], "h2": [5.0, 2.0]},
            id="Complex-Two satellites with break - only spanning",
        ),
        pytest.param(
            ["CAG", "CCG"],
            ["", "CAACAG", ""],
            [
                # Haplotype 1
                (2, 3),
                (2, 3),
                # Haplotype 2
                (4, 2),
                (4, 2),
                (4, 2),
            ],
            [
                (2, 1),
            ],
            [
                (0, 2),
            ],
            {"h1": 3, "h2": 4},
            {"h1": [2.0, 3.0], "h2": [4.0, 2.0]},
            id="Complex-Two satellites with break - with flanking",
        ),
        # Cases with 3 satellites
        pytest.param(
            ["CAG", "CCG", "CTG"],  # Three different satellites
            ["", "", "", ""],  # Four breaks
            [
                (2, 1, 3),  # H1: 2 CAG, 1 CCG, 3 CTG
                (2, 1, 3),  # H1: 2 CAG, 1 CCG, 3 CTG
                (3, 2, 1),  # H2: 3 CAG, 2 CCG, 1 CTG
                (3, 2, 1),  # H2: 3 CAG, 2 CCG, 1 CTG
            ],
            [(2, 1, 1)],  # One left flanking matching H1
            [(0, 1, 1)],  # One right flanking matching H2
            {"h1": 3, "h2": 3},
            {"h1": [2.0, 1.0, 3.0], "h2": [3.0, 2.0, 1.0]},
            id="Complex-Three satellites - heterozygous",
        ),
        # Test cases from data - simple loci
        pytest.param(
            ["AARRG"],
            ["", ""],
            [
                # Haplotype 1
                765,
                770,
                778,
                # Haplotype 2
            ],
            [
                # Haplotype 2
                863,
                864,
            ],
            [],
            # Expected group sizes
            {"h1": 3, "h2": 2},
            # Expected heterozygous mean
            {"h1": [771.0], "h2": [864.0]},
            id="Case 1: Long RCF1. Only one flanking read for H2",
        ),
        pytest.param(
            ["AARRG"],
            ["", ""],
            [
                # Haplotype 1
                *([12] * 10),
                # Haplotype 2
                927,
                940,
                947,
            ],
            [
                # Haplotype 1
                13,
                # Haplotype 2
                798,
                745,
                362,
            ],
            [],
            # Expected group sizes
            {"h1": 11, "h2": 6},
            # Expected heterozygous mean
            {"h1": [12.0], "h2": [938.0]},
            id="Case 2: RFC1. Short H1, Long H2",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                8,
                *([9] * 4),
                # Haplotype 2
                *([11] * 5),
            ],
            [
                10,
            ],
            [],
            # Expected group sizes
            {"h1": 5, "h2": 6},
            # Expected heterozygous mean
            {"h1": [9.0], "h2": [11.0]},
            id="Case 3: FGF14. Close haplotypes, 1 apart",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *([8] * 7),
                # Haplotype 2
                *([9] * 5),
            ],
            [],
            [],
            # Expected group sizes
            {"h1": 7, "h2": 5},
            # Expected heterozygous mean
            {"h1": [8.0], "h2": [9.0]},
            id="Case 4: FGF14. Close haplotypes 2, 0 apart",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *([5] * 10),
                # Haplotype 2
                12,
                *([13] * 5),
            ],
            [
                # Haplotype 2
                8,
                10,
            ],
            [],
            # Expected group sizes
            {"h1": 10, "h2": 8},
            # Expected heterozygous mean
            {"h1": [5.0], "h2": [13.0]},
            id="Case 5: DMPK",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *([8] * 9),
                # Haplotype 2
                107,
                *([108] * 2),
                *([109] * 7),
                *([110] * 3),
                111,
            ],
            [
                # Haplotype 2
                66,
                75,
                96,
                98,
            ],
            [],
            # Expected group sizes
            {"h1": 9, "h2": 18},
            # Expected heterozygous mean
            {"h1": [8.0], "h2": [109.0]},
            id="Case 6: RFC1",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *([28] * 8),
                # Haplotype 2
                *([30] * 5),
                31,
            ],
            [],
            [
                # Haplotype 1
                28,
            ],
            # Expected group sizes
            {"h1": 9, "h2": 6},
            # Expected heterozygous mean
            {"h1": [28.0], "h2": [30.0]},
            id="Case 7: ATXN1",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *([27] * 3),
                *([28] * 3),
                *([29] * 2),
                # Haplotype 2
                *([30] * 2),
                *([31] * 7),
                *([32] * 2),
                # Outlier
                35,
            ],
            [],
            [],
            # Expected group sizes
            {"h1": 8, "h2": 11, "outlier": 1},
            # Expected heterozygous mean
            {"h1": [28.0], "h2": [31.0]},
            id="Case 8: Simple outlier",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *([27] * 20),
                # Haplotype 2
                *([30] * 2),
            ],
            [],
            [],
            # Expected group sizes
            {"hom": 22},
            # Expected heterozygous mean
            None,
            id="Case 9: Skewed split. Should be called homozygous",
        ),
        pytest.param(
            ["GGCCCC"],
            ["", ""],
            [
                # Haplotype 1
                *([12] * 17),
                # Haplotype 2
                958,
            ],
            [],
            [
                54,
                298,
            ],
            # Expected group sizes
            {"h1": 17, "h2": 3},
            # Expected heterozygous mean
            {"h1": [12.0], "h2": [958.0]},
            id="Case 10: C9ORF72",
        ),
        pytest.param(
            ["NGC"],
            ["", ""],
            [
                # Haplotype 1
                *([12] * 17),
                # Haplotype 2
                *([15] * 21),
            ],
            [],
            [],
            # Expected group sizes
            {"h1": 17, "h2": 21},
            # Expected heterozygous mean
            {"h1": [12.0], "h2": [15.0]},
            id="Case 11: ARX_EIEE",
        ),
        pytest.param(
            ["GCC"],
            ["", ""],
            [],
            [7],
            [],
            # Expected group sizes
            {"hom": 1},
            # Expected heterozygous mean
            {"hom": [7.0]},
            id="Case 12: XYLT1 - Low coverage, no spanning reads",
        ),
        pytest.param(
            ["GGC"],
            ["", ""],
            [
                # Haplotype 1
                *([19] * 3),
                # Haplotype 2
                *([27] * 9),
                *([28] * 1),
            ],
            [],
            [],
            # Expected group sizes
            {"h1": 3, "h2": 10},
            # Expected heterozygous mean
            {"h1": [19.0], "h2": [27.0]},
            id="Case 13: NOTCH2NLC",
        ),
        pytest.param(
            ["CAG", "CCG", "CTG"],  # Three different satellites
            ["", "", "", ""],  # Four breaks
            [
                # Haplotype 1
                (13, 8, 18),
                (15, 8, 18),
                (15, 9, 18),
                (15, 9, 18),
                (15, 9, 18),
                (15, 9, 18),
                (15, 9, 18),
                (15, 9, 18),
                # Haplotype 2
                (15, 9, 20),
                (16, 8, 20),
                (16, 9, 19),
                (16, 9, 19),
                (16, 9, 20),
                (16, 9, 20),
                (16, 9, 21),
                (16, 9, 21),
                (16, 9, 21),
                (16, 9, 21),
                (16, 10, 20),
                (16, 9, 24),
            ],
            [],
            [],
            {"h1": 8, "h2": 12},
            {"h1": [15.0, 9.0, 18.0], "h2": [16.0, 9.0, 21.0]},
            id="Case 14: CNBP",
        ),
    ],
)
def test_haplotyping_integration(
    satellite_seqs: list[str],
    breaks: list[str],
    spanning_counts: list[int] | list[tuple[int, ...]],
    left_flanking_counts: list[int] | list[tuple[int, ...]],
    right_flanking_counts: list[int] | list[tuple[int, ...]],
    expected_group_sizes: dict[Literal["h1", "h2", "hom", "outlier"], int],
    expected_means: dict[str, list[float]] | None,
) -> None:
    """Test the haplotype grouping functionality for both simple and complex loci."""
    # Create synthetic locus
    locus = create_synthetic_locus(satellite_seqs, breaks)

    # Create synthetic reads
    reads: list[Read] = []

    # Create spanning reads
    reads.extend(
        [
            create_synthetic_read(
                counts=count,  # type: ignore
                satellite_seqs=satellite_seqs,
                breaks=breaks,
                alignment_type=AlignmentType.SPANNING,
                locus=locus,
                read_name=f"spanning_{i}",
            )
            for i, count in enumerate(spanning_counts)
        ],
    )

    # Create left flanking reads
    reads.extend(
        [
            create_synthetic_read(
                counts=count,  # type: ignore
                satellite_seqs=satellite_seqs,
                breaks=breaks,
                alignment_type=AlignmentType.LEFT_FLANKING,
                locus=locus,
                read_name=f"left_flanking_{i}",
            )
            for i, count in enumerate(left_flanking_counts)
        ],
    )

    # Create right flanking reads
    reads.extend(
        [
            create_synthetic_read(
                counts=count,  # type: ignore
                satellite_seqs=satellite_seqs,
                breaks=breaks,
                alignment_type=AlignmentType.RIGHT_FLANKING,
                locus=locus,
                read_name=f"right_flanking_{i}",
            )
            for i, count in enumerate(right_flanking_counts)
        ],
    )

    # Get read calls through the normal pipeline
    read_calls, _ = get_read_calls(reads, locus)

    # Run grouping
    grouped_reads, outlier_reads, heterozygous_parameters, homozygous_parameters, _ = run_haplotyping(read_calls, ploidy=2)

    # Count occurrences of each group
    group_counts: dict[str, int] = {}
    for read in grouped_reads + outlier_reads:
        group_counts[read.haplotype] = group_counts.get(read.haplotype, 0) + 1

    # Check if groups sizes match expected
    assert group_counts == expected_group_sizes, f"Test case failed: Expected {expected_group_sizes}, got {group_counts}"

    # Check if means match expected
    if expected_means is not None:
        # Heterozygous case
        if "h1" in expected_means and "h2" in expected_means:
            assert heterozygous_parameters.mean_h1.tolist() == expected_means["h1"], (
                f"Expected h1 mean {expected_means['h1']}, got {heterozygous_parameters.mean_h1[0]}"
            )
            assert heterozygous_parameters.mean_h2.tolist() == expected_means["h2"], (
                f"Expected h2 mean {expected_means['h2']}, got {heterozygous_parameters.mean_h2[0]}"
            )
        # Homozygous case
        if "hom" in expected_means:
            assert homozygous_parameters.mean.tolist() == expected_means["hom"], (
                f"Expected homozygous mean {expected_means['hom']}, got {homozygous_parameters.mean[0]}"
            )
