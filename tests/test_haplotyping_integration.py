import random
from typing import Literal

import pytest

from abacus.graph import (
    AlignmentType,
    Read,
    get_read_calls,
)
from abacus.haplotyping import group_read_calls
from abacus.locus import Location, Locus, Satellite


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
    ("satellite_seq", "spanning_counts", "left_flanking_counts", "right_flanking_counts", "expected_groups"),
    [
        # Simple test cases
        pytest.param(
            "CAG",
            [],  # No spanning reads
            [],  # No flanking reads
            [],  # No flanking reads
            {},  # Should be called as homozygous
            id="Empty",
        ),
        pytest.param(
            "CAG",
            [1],  # Single spanning read
            [],  # No flanking reads
            [],  # No flanking reads
            {"hom": 1},  # Should be called as homozygous
            id="Single spanning",
        ),
        pytest.param(
            "CAG",
            [2, 2],  # Two spanning reads
            [1],  # Single left flanking read
            [1],  # Single right flanking read
            {"hom": 4},  # Should be called as homozygous
            id="Homozygous - Small",
        ),
        pytest.param(
            "CAG",
            [
                1,
                1,
                1,  # Haplotype 1
                3,
                3,
                3,
                3,  # Haplotype 2
            ],
            [],
            [],
            {"h1": 3, "h2": 4},
            id="Heterozygous - Small",
        ),
        # Test cases from data
        pytest.param(
            "AAGGG",
            [
                # Haplotype 1
                765,
                778,
                # Haplotype 2
            ],
            [
                # Haplotype 2
                863,
            ],
            [],
            {"h1": 2, "h2": 1},
            id="Case 1: Long RCF1. One flanking read for H2, 2 spanning reads for H1",
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
            {"h1": 11, "h2": 6},
            id="Case 2: RFC1. Short H1, Long H2",
        ),
        pytest.param(
            "GAA",
            [
                # Haplotype 1
                8,
                9,
                9,
                9,
                # Haplotype 2
                11,
                11,
                11,
            ],
            [],
            [],
            {"h1": 4, "h2": 3},
            id="Case 3: FGF14. Close haplotypes",
        ),
        pytest.param(
            "GAA",
            [
                # Haplotype 1
                *([8] * 25),
                # Haplotype 2
                *([9] * 25),
            ],
            [],
            [],
            {"h1": 25, "h2": 24},
            id="Case 4: FGF14. Close haplotypes 2",
        ),
    ],
)
def test_haplotype_grouping(
    satellite_seq: str,
    spanning_counts: list[int],
    left_flanking_counts: list[int],
    right_flanking_counts: list[int],
    expected_groups: dict[Literal["h1", "h2", "hom", "outlier"], int],
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
    grouped_reads, _, _ = group_read_calls(read_calls, ploidy=2)

    # Count occurrences of each group
    group_counts: dict[str, int] = {}
    for read in grouped_reads:
        group_counts[read.em_haplotype] = group_counts.get(read.em_haplotype, 0) + 1

    # Check if groups match expected
    assert group_counts == expected_groups, f"Test case failed: Expected {expected_groups}, got {group_counts}"
