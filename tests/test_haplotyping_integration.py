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


def create_synthetic_simple_locus(satellite_seq: str):
    # Create random anchors
    left_anchor = create_random_anchor()
    right_anchor = create_random_anchor()

    # Create a simple locus with one satellite
    locus = Locus(
        id="test",
        structure="test",
        location=Location(chrom="chr1", start=1000, end=2000),
        satellites=[
            Satellite(
                id="test",
                sequence=satellite_seq,
                location=Location("chr1", 1000, 2000),
                skippable=False,
            ),
        ],
        breaks=["", ""],
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )
    return locus


def create_synthetic_complex_locus(satellite_seqs: list[str], breaks: list[str]):
    """Create a synthetic locus with multiple satellites."""
    # Create random anchors
    left_anchor = create_random_anchor()
    right_anchor = create_random_anchor()

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
                skippable=True,
            ),
        )
        start_pos = end_pos

    # Create complex locus
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


def create_synthetic_read(
    sequence: str,
    alignment_type: AlignmentType,
    locus: Locus,
    read_name: str,
) -> Read:
    """Create a synthetic read for testing."""
    left_anchor = locus.left_anchor
    right_anchor = locus.right_anchor

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
    ("satellite_seq", "spanning_counts", "left_flanking_counts", "right_flanking_counts", "expected_group_sizes", "expected_heterozygous_mean"),
    [
        # Simple test cases
        pytest.param(
            "CAG",
            [],  # No spanning reads
            [],  # No flanking reads
            [],  # No flanking reads
            # Expected group sizes
            {},
            # Expected heterozygous mean
            None,
            id="Empty",
        ),
        pytest.param(
            "CAG",
            [1],  # Single spanning read
            [],  # No flanking reads
            [],  # No flanking reads
            # Expected group sizes
            {"hom": 1},
            # Expected heterozygous mean
            None,
            id="Single spanning",
        ),
        pytest.param(
            "CAG",
            [2, 2],  # Two spanning reads
            [1],  # Single left flanking read
            [1],  # Single right flanking read
            # Expected group sizes
            {"hom": 4},
            # Expected heterozygous mean
            None,
            id="Homozygous - Small",
        ),
        pytest.param(
            "CAG",
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
            {"h1": 1.0, "h2": 3.0},
            id="Heterozygous - Small",
        ),
        # Test cases from data
        pytest.param(
            "AAGGG",
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
            {"h1": 771.0, "h2": 864.0},
            id="Case 1: Long RCF1. Only one flanking read for H2",
        ),
        pytest.param(
            "AAGGG",
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
            {"h1": 12.0, "h2": 938.0},
            id="Case 2: RFC1. Short H1, Long H2",
        ),
        pytest.param(
            "GAA",
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
            {"h1": 9.0, "h2": 11.0},
            id="Case 3: FGF14. Close haplotypes, 1 apart",
        ),
        pytest.param(
            "GAA",
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
            {"h1": 8.0, "h2": 9.0},
            id="Case 4: FGF14. Close haplotypes 2, 0 apart",
        ),
        pytest.param(
            "GAA",
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
            {"h1": 5.0, "h2": 13.0},
            id="Case 5: DMPK",
        ),
        pytest.param(
            "GAA",
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
            {"h1": 8.0, "h2": 109.0},
            id="Case 6: RFC1",
        ),
        pytest.param(
            "CTG",
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
            {"h1": 28.0, "h2": 30.0},
            id="Case 7: ATXN1",
        ),
        pytest.param(
            "CTG",
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
            {"h1": 28.0, "h2": 31.0},
            id="Case 8: Simple outlier",
        ),
        pytest.param(
            "CTG",
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
            "GGCCCC",
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
            {"h1": 12.0, "h2": 958.0},
            id="Case 10: C9ORF72",
        ),
    ],
)
def test_haplotyping_simple(
    satellite_seq: str,
    spanning_counts: list[int],
    left_flanking_counts: list[int],
    right_flanking_counts: list[int],
    expected_group_sizes: dict[Literal["h1", "h2", "hom", "outlier"], int],
    expected_heterozygous_mean: dict[str, float] | None,
) -> None:
    """Test the haplotype grouping functionality."""
    # Create synthetic locus
    locus = create_synthetic_simple_locus(satellite_seq)

    # Create synthetic reads
    reads: list[Read] = []

    # Create spanning reads
    reads.extend(
        [
            create_synthetic_read(
                sequence=satellite_seq * count,
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
                sequence=satellite_seq * count,
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
                sequence=satellite_seq * count,
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
    grouped_reads, outlier_reads, heterozygous_parameters, _, _ = run_haplotyping(read_calls, ploidy=2)

    # Count occurrences of each group
    group_counts: dict[str, int] = {}
    for read in grouped_reads + outlier_reads:
        group_counts[read.haplotype] = group_counts.get(read.haplotype, 0) + 1

    # Check if groups sizes match expected
    assert group_counts == expected_group_sizes, f"Test case failed: Expected {expected_group_sizes}, got {group_counts}"

    # Check if heterozygous mean matches expected
    if expected_heterozygous_mean:
        assert heterozygous_parameters.mean_h1[0] == expected_heterozygous_mean["h1"], (
            f"Expected {expected_heterozygous_mean['h1']}, got {heterozygous_parameters.mean_h1[0]}"
        )
        assert heterozygous_parameters.mean_h2[0] == expected_heterozygous_mean["h2"], (
            f"Expected {expected_heterozygous_mean['h2']}, got {heterozygous_parameters.mean_h2[0]}"
        )


@pytest.mark.parametrize(
    (
        "satellite_seqs",
        "breaks",
        "spanning_counts",
        "left_flanking_counts",
        "right_flanking_counts",
        "expected_groups",
    ),
    [
        # Cases with 2 satellites
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
            id="Two satellites - only spanning",
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
            id="Two satellites - with flanking",
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
            id="Two satellites with break - only spanning",
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
            id="Two satellites with break - with flanking",
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
            id="Three satellites - heterozygous",
        ),
    ],
    # TODO: Add Complex outlier case.
)
def test_haplotyping_complex(
    satellite_seqs: list[str],
    breaks: list[str],
    spanning_counts: list[tuple[int, ...]],
    left_flanking_counts: list[tuple[int, ...]],
    right_flanking_counts: list[tuple[int, ...]],
    expected_groups: dict[Literal["h1", "h2", "hom", "outlier"], int],
) -> None:
    """Test the haplotype grouping functionality with complex loci."""
    # Create synthetic locus
    locus = create_synthetic_complex_locus(satellite_seqs, breaks)

    # Create synthetic reads
    reads: list[Read] = []

    # Helper function to create sequence from repeat counts
    def create_sequence(counts: tuple[int, ...]) -> str:
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

    # Create spanning reads
    reads.extend(
        [
            create_synthetic_read(
                sequence=create_sequence(counts),
                alignment_type=AlignmentType.SPANNING,
                locus=locus,
                read_name=f"spanning_{i}",
            )
            for i, counts in enumerate(spanning_counts)
        ],
    )

    # Create left flanking reads
    reads.extend(
        [
            create_synthetic_read(
                sequence=create_sequence(counts),
                alignment_type=AlignmentType.LEFT_FLANKING,
                locus=locus,
                read_name=f"left_flanking_{i}",
            )
            for i, counts in enumerate(left_flanking_counts)
        ],
    )

    # Create right flanking reads
    reads.extend(
        [
            create_synthetic_read(
                sequence=create_sequence(counts),
                alignment_type=AlignmentType.RIGHT_FLANKING,
                locus=locus,
                read_name=f"right_flanking_{i}",
            )
            for i, counts in enumerate(right_flanking_counts)
        ],
    )

    # Get read calls through the normal pipeline
    read_calls, _ = get_read_calls(reads, locus)

    # Run grouping
    grouped_reads, outlier_reads, _, _, _ = run_haplotyping(read_calls, ploidy=2)

    # Count occurrences of each group
    group_counts: dict[str, int] = {}
    for read in grouped_reads + outlier_reads:
        group_counts[read.haplotype] = group_counts.get(read.haplotype, 0) + 1

    # Check if groups match expected
    assert group_counts == expected_groups, f"Test case failed: Expected {expected_groups}, got {group_counts}"
