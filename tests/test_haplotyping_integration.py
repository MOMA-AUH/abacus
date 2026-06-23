from __future__ import annotations

import random
from typing import Literal

import pytest

from abacus import config
from abacus.graph import (
    Read,
    get_read_calls,
)
from abacus.haplotyping import run_haplotyping
from abacus.locus import Location, Locus, Satellite
from abacus.utils import Haplotype


def create_random_anchor() -> str:
    """Create a random anchor sequence."""
    return "".join(random.choices("ATCG", k=config.Config.anchor_len))


def create_synthetic_locus(satellite_seqs: list[str], breaks: list[str] | None = None):
    """Create a synthetic locus with one or multiple satellites."""
    left_anchor = create_random_anchor()
    right_anchor = create_random_anchor()

    if breaks is None:
        breaks = [""] * (len(satellite_seqs) + 1)

    satellites = []
    start_pos = 1000
    for i, seq in enumerate(satellite_seqs):
        # Split on OR operator if present
        seqs = seq.split("|")
        end_pos = start_pos + len(seqs[0])
        satellites.append(
            Satellite(
                id=f"test_{i}",
                sequences=seqs,
                location=Location("chr1", start_pos, end_pos),
                skippable=(len(satellite_seqs) > 1),
            ),
        )
        start_pos = end_pos

    return Locus(
        id="test",
        structure="test",
        location=Location(chrom="chr1", start=1000, end=start_pos),
        satellites=satellites,
        breaks=breaks,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )


def make_read(name: str, sequence: str, locus: Locus) -> Read:
    """Create a synthetic read with the given full sequence (including anchors)."""
    return Read(
        name=name,
        sequence=sequence,
        qualities=[30] * len(sequence),
        mod_5mc_probs="!" * len(sequence),
        strand="+",
        n_soft_clipped_left=0,
        n_soft_clipped_right=0,
        locus=locus,
    )


@pytest.mark.parametrize(
    (
        "satellite_seqs",
        "breaks",
        "spanning_sequences",
        "left_flanking_sequences",
        "right_flanking_sequences",
        "expected_group_sizes",
        "expected_means",
    ),
    [
        # Simple test cases (single satellite)
        pytest.param(
            ["CAG"],
            ["", ""],
            [],
            [],
            [],
            {},
            None,
            id="Simple-Empty",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            ["CAG"],
            [],
            [],
            {"hom": 1},
            None,
            id="Simple-Single spanning",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            ["CAGCAG", "CAGCAG"],
            ["CAG"],
            ["CAG"],
            {"hom": 4},
            None,
            id="Simple-Homozygous - Small",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [
                # Haplotype 1
                *["CAG"] * 3,
                # Haplotype 2
                *["CAG" * 3] * 4,
            ],
            [],
            [],
            {"h1": 3, "h2": 4},
            {"h1": [1.0], "h2": [3.0]},
            id="Simple-Heterozygous - Small",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *["CTG" * 27] * 20,
                # Haplotype 2
                *["CTG" * 30] * 2,
            ],
            [],
            [],
            {"h1": 20, "h2": 2},
            {"h1": [27.0], "h2": [30.0]},
            id="Simple-Skewed split. Should be called heterozygous",
        ),
        # Complex test cases (multiple satellites)
        pytest.param(
            ["CAG", "CTG"],
            ["", "", ""],
            [
                # Haplotype 1
                *["CAG" * 2 + "CTG" * 3] * 2,
                # Haplotype 2
                *["CAG" * 4 + "CTG" * 2] * 3,
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
                *["CAG" * 2 + "CTG" * 3] * 2,
                # Haplotype 2
                *["CAG" * 4 + "CTG" * 2] * 3,
            ],
            ["CAG" * 2 + "CTG"],
            ["CTG" * 2],
            {"h1": 3, "h2": 4},
            {"h1": [2.0, 3.0], "h2": [4.0, 2.0]},
            id="Complex-Two satellites - with flanking",
        ),
        # Cases with 2 satellites and a break sequence (HTT is a good example)
        pytest.param(
            ["CAG", "CCG"],
            ["", "CAACAG", ""],
            [
                # Haplotype 1
                *["CAG" * 2 + "CAACAG" + "CCG" * 3] * 3,
                # Haplotype 2
                *["CAG" * 5 + "CAACAG" + "CCG" * 2] * 4,
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
                *["CAG" * 2 + "CAACAG" + "CCG" * 3] * 2,
                # Haplotype 2
                *["CAG" * 4 + "CAACAG" + "CCG" * 2] * 3,
            ],
            ["CAG" * 2 + "CAACAG" + "CCG"],
            ["CAACAG" + "CCG" * 2],
            {"h1": 3, "h2": 4},
            {"h1": [2.0, 3.0], "h2": [4.0, 2.0]},
            id="Complex-Two satellites with break - with flanking",
        ),
        # Cases with 3 satellites
        pytest.param(
            ["CAG", "CCG", "CTG"],
            ["", "", "", ""],
            [
                *["CAG" * 2 + "CCG" + "CTG" * 3] * 2,  # H1: 2 CAG, 1 CCG, 3 CTG
                *["CAG" * 3 + "CCG" * 2 + "CTG"] * 2,  # H2: 3 CAG, 2 CCG, 1 CTG
            ],
            ["CAG" * 2 + "CCG" + "CTG"],  # Left flanking matching H1
            ["CCG" + "CTG"],  # Right flanking matching H2
            {"h1": 3, "h2": 3},
            {"h1": [2.0, 1.0, 3.0], "h2": [3.0, 2.0, 1.0]},
            id="Complex-Three satellites - heterozygous",
        ),
        # OR operator test cases (satellite with multiple alternative sequences)
        pytest.param(
            ["CAG|CAA"],
            ["", ""],
            [
                *["CAG" * 10] * 5,
                *["CAA" * 15] * 7,
            ],
            [],
            [],
            {"h1": 5, "h2": 7},
            {"h1": [10.0], "h2": [15.0]},
            id="OR-Heterozygous by repeat count",
        ),
        pytest.param(
            ["CAG|CAA"],
            ["", ""],
            [
                *["CAG" * 10] * 7,
                *["CAA" * 10] * 5,
            ],
            [],
            [],
            {"h1": 7, "h2": 5},
            {"h1": [10.0], "h2": [10.0]},
            id="OR-Heterozygous by motif, but same length.",
        ),
        pytest.param(
            ["GGCCCC|GGCCCCC"],
            ["", ""],
            [
                *["GGCCCC" * 12] * 8,
                *["GGCCCCC" * 20] * 8,
            ],
            [],
            [],
            {"h1": 8, "h2": 8},
            {"h1": [12.0], "h2": [20.0]},
            id="OR-Heterozygous, different-length alternatives",
        ),
        # Haplotyping based on motif differences
        pytest.param(
            ["CAG"],
            ["", ""],
            [*["CAG" * 10] * 7, "CAG" * 9 + "CGG"],  # 8 reads: 7 normal + 1 with sequencing error, all same length
            [],
            [],
            {"hom": 8},
            {"hom": [10.0]},
            id="Sequence split-8 reads below min_haplotyping_depth-should not create singleton",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [*["CAG" * 10] * 12],
            [],
            [],
            {"hom": 12},
            {"hom": [10.0]},
            id="Sequence split-No differences should remain homozygous",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [
                # Haplotype 1
                *["CAGCAG"] * 7,
                # Haplotype 2
                *["CAGCGG"] * 5,
            ],
            [],
            [],
            {"h1": 7, "h2": 5},
            {"h1": [2.0], "h2": [2.0]},
            id="Sequence split-Short and simple",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [
                # Haplotype 1
                *["CAG" * 31] * 11,
                # Haplotype 2
                *["CAG" * 20 + "CGG" + "CAG" * 10] * 9,
            ],
            [],
            [],
            {"h1": 11, "h2": 9},
            {"h1": [31.0], "h2": [31.0]},
            id="Sequence split-Long and simple",
        ),
        pytest.param(
            ["CAG", "CCG"],
            ["", "CAACAG", ""],
            [
                # Haplotype 1
                *["CAG" * 7 + "CAACAG" + "CCGCCGCCG"] * 9,
                # Haplotype 2
                *["CAG" * 7 + "CAACAG" + "CCGCAGCCG"] * 7,
            ],
            [],
            [],
            {"h1": 9, "h2": 7},
            {"h1": [7.0, 3.0], "h2": [7.0, 3.0]},
            id="Sequence split-HTT-like case with interruption",
        ),
        # Outlier test cases
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *["CTG" * 27] * 1,
                *["CTG" * 28] * 6,
                *["CTG" * 29] * 2,
                # Haplotype 2
                *["CTG" * 30] * 1,
                *["CTG" * 31] * 8,
                *["CTG" * 32] * 2,
                # Outlier
                "CTG" * 35,
            ],
            [],
            [],
            {"h1": 9, "h2": 11, "outlier": 1},
            {"h1": [28.0], "h2": [31.0]},
            id="Outlier-Long outlier that should be separated from H2",
        ),
        # Test cases from data - simple loci
        pytest.param(
            ["AARRG"],
            ["", ""],
            [
                # Haplotype 1 (AARRG resolved: R -> A)
                "AAAAG" * 765,
                "AAAAG" * 770,
                "AAAAG" * 778,
            ],
            [
                # Haplotype 2
                "AAAAG" * 863,
                "AAAAG" * 864,
            ],
            [],
            {"h1": 3, "h2": 2},
            {"h1": [771.0], "h2": [864.0]},
            id="Case 1: Long RCF1. Only one flanking read for H2",
        ),
        pytest.param(
            ["AARRG"],
            ["", ""],
            [
                # Haplotype 1
                *["AAAAG" * 12] * 10,
                # Haplotype 2
                "AAAAG" * 927,
                "AAAAG" * 940,
                "AAAAG" * 947,
            ],
            [
                # Haplotype 1
                "AAAAG" * 13,
                # Haplotype 2
                "AAAAG" * 798,
                "AAAAG" * 745,
                "AAAAG" * 362,
            ],
            [],
            {"h1": 11, "h2": 6},
            {"h1": [12.0], "h2": [938.0]},
            id="Case 2: RFC1. Short H1, Long H2",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                "GAA" * 8,
                *["GAA" * 9] * 4,
                # Haplotype 2
                *["GAA" * 11] * 5,
            ],
            ["GAA" * 10],
            [],
            {"h1": 5, "h2": 6},
            {"h1": [9.0], "h2": [11.0]},
            id="Case 3: FGF14. Close haplotypes, 1 apart",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *["GAA" * 8] * 7,
                # Haplotype 2
                *["GAA" * 9] * 5,
            ],
            [],
            [],
            {"h1": 7, "h2": 5},
            {"h1": [8.0], "h2": [9.0]},
            id="Case 4: FGF14. Close haplotypes 2, 0 apart",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *["GAA" * 5] * 10,
                # Haplotype 2
                "GAA" * 12,
                *["GAA" * 13] * 5,
            ],
            [
                # Haplotype 2
                "GAA" * 8,
                "GAA" * 10,
            ],
            [],
            {"h1": 10, "h2": 8},
            {"h1": [5.0], "h2": [13.0]},
            id="Case 5: DMPK",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *["GAA" * 8] * 9,
                # Haplotype 2
                "GAA" * 107,
                *["GAA" * 108] * 2,
                *["GAA" * 109] * 7,
                *["GAA" * 110] * 3,
                "GAA" * 111,
            ],
            [
                # Haplotype 2
                "GAA" * 66,
                "GAA" * 75,
                "GAA" * 96,
                "GAA" * 98,
            ],
            [],
            {"h1": 9, "h2": 18},
            {"h1": [8.0], "h2": [109.0]},
            id="Case 6: RFC1",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *["CTG" * 28] * 8,
                # Haplotype 2
                *["CTG" * 30] * 5,
                "CTG" * 31,
            ],
            [],
            ["CTG" * 28],
            {"h1": 9, "h2": 6},
            {"h1": [28.0], "h2": [30.0]},
            id="Case 7: ATXN1",
        ),
        pytest.param(
            ["GGCCCC"],
            ["", ""],
            [
                # Haplotype 1
                *["GGCCCC" * 12] * 17,
                # Haplotype 2
                "GGCCCC" * 958,
            ],
            [],
            [
                "GGCCCC" * 54,
                "GGCCCC" * 298,
            ],
            {"h1": 17, "h2": 3},
            {"h1": [12.0], "h2": [958.0]},
            id="Case 9: C9ORF72",
        ),
        pytest.param(
            ["NGC"],
            ["", ""],
            [
                # Haplotype 1 (NGC resolved: N -> A)
                *["AGC" * 12] * 17,
                # Haplotype 2
                *["AGC" * 15] * 21,
            ],
            [],
            [],
            {"h1": 17, "h2": 21},
            {"h1": [12.0], "h2": [15.0]},
            id="Case 10: ARX_EIEE",
        ),
        pytest.param(
            ["GCC"],
            ["", ""],
            [],
            ["GCC" * 7],
            [],
            {"hom": 1},
            {"hom": [7.0]},
            id="Case 11: XYLT1 - Low coverage, no spanning reads",
        ),
        pytest.param(
            ["GGC"],
            ["", ""],
            [
                # Haplotype 1
                *["GGC" * 19] * 3,
                # Haplotype 2
                *["GGC" * 27] * 9,
                "GGC" * 28,
            ],
            [],
            [],
            {"h1": 3, "h2": 10},
            {"h1": [19.0], "h2": [27.0]},
            id="Case 12: NOTCH2NLC",
        ),
        pytest.param(
            ["CAG", "CCG", "CTG"],
            ["", "", "", ""],
            [
                # Haplotype 1
                "CAG" * 13 + "CCG" * 8 + "CTG" * 18,
                "CAG" * 15 + "CCG" * 8 + "CTG" * 18,
                *["CAG" * 15 + "CCG" * 9 + "CTG" * 18] * 6,
                # Haplotype 2
                "CAG" * 15 + "CCG" * 9 + "CTG" * 20,
                "CAG" * 16 + "CCG" * 8 + "CTG" * 20,
                *["CAG" * 16 + "CCG" * 9 + "CTG" * 19] * 2,
                *["CAG" * 16 + "CCG" * 9 + "CTG" * 20] * 2,
                *["CAG" * 16 + "CCG" * 9 + "CTG" * 21] * 4,
                "CAG" * 16 + "CCG" * 10 + "CTG" * 20,
                "CAG" * 16 + "CCG" * 9 + "CTG" * 24,
            ],
            [],
            [],
            {"h1": 8, "h2": 12},
            {"h1": [15.0, 9.0, 18.0], "h2": [16.0, 9.0, 21.0]},
            id="Case 13: CNBP",
        ),
        pytest.param(
            ["GCN"],
            ["", ""],
            [
                # Haplotype 1
                *["GCC" * 10] * 24,
                # Haplotype 2
                *["GCC" * 13] * 3,
                *["GCC" * 14] * 35,
                # Outlier
                *["GCC" * 1] * 1,
            ],
            [],
            [],
            {"h1": 24, "h2": 38, "outlier": 1},
            {"h1": [10.0], "h2": [14.0]},
            id="Case 14: PABPN1 - Outlier should not affect haplotype grouping (issue #18).",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1
                *["CTG" * 19] * 12,
                # Haplotype 2
                *["CTG" * 67] * 9,
                *["CTG" * 68] * 4,
            ],
            [],
            [
                # Outlier
                *["CTG" * 81] * 1,
            ],
            {"h1": 12, "h2": 13, "outlier": 1},
            {"h1": [19.0], "h2": [67.0]},
            id="Case 15: ATXN3 - Flanking outlier should be separated from H2",
        ),
        pytest.param(
            ["GAA"],
            ["", ""],
            [
                # Haplotype 1
                *["GAA" * 275] * 1,
                *["GAA" * 276] * 1,
                *["GAA" * 277] * 2,
                *["GAA" * 278] * 3,
                *["GAA" * 279] * 5,
                *["GAA" * 280] * 3,
                *["GAA" * 281] * 2,
                *["GAA" * 282] * 1,
                # Haplotype 2
                *["GAAGGA" * 137] * 1,
                *["GAAGGA" * 138] * 1,
                *["GAAGGA" * 138] * 2,
                *["GAAGGA" * 139] * 3,
                *["GAAGGA" * 139] * 5,
                *["GAAGGA" * 140] * 3,
                *["GAAGGA" * 140] * 2,
            ],
            [],
            [],
            {"h1": 18, "h2": 17},
            {"h1": [279.0], "h2": [278.0]},
            id="Case 16: FGF14 - Same varying length, differing motif",
        ),
        pytest.param(
            ["GGCCCC"],
            ["", ""],
            [
                # Haplotype 1
                *["GGCCCC" * 12] * 16,
                # Haplotype 2
                "GGCCCC" * 958,
                "GGCCCC" * 2349,
            ],
            [
                # Haplotype 2
                "GGCCCC" * 1440,
                "GGCCCC" * 298,
                "GGCCCC" * 54,
            ],
            [],
            {"h1": 16, "h2": 5},
            {"h1": [12.0], "h2": None},
            id="Case 17: C9ORF72 - High somatic mosaicism should not affect parameter estimation for shorter allele (H1).",
        ),
        pytest.param(
            ["CAG"],
            ["", ""],
            [
                # Haplotype 1 - normal allele
                *["CAG" * 5] * 14,
                # Haplotype 2 - expanded allele (DM1, somatic mosaicism)
                "CAG" * 935,
                "CAG" * 961,
                "CAG" * 1421,
                "CAG" * 1526,
                "CAG" * 1575,
                "CAG" * 1891,
                "CAG" * 1932,
                "CAG" * 1964,
                "CAG" * 1971,
                "CAG" * 2072,
                "CAG" * 2168,
                "CAG" * 2541,
                "CAG" * 3133,
            ],
            [],
            [
                # Haplotype 2 - right flanking
                "CAG" * 1066,
                "CAG" * 664,
                "CAG" * 378,
            ],
            {"h1": 14, "h2": 16},
            {"h1": [5.0], "h2": [1825.0]},
            id="Case 18: DMPK - High somatic mosaicism should not affect parameter estimation for shorter allele (H1).",
        ),
        pytest.param(
            ["CGG"],
            ["", ""],
            [
                # Haplotype 1 - normal allele
                "CGG" * 30,
                *["CGG" * 31] * 2,
                *["CGG" * 32] * 10,
                "CGG" * 33,
                # Haplotype 2 - full mutation (somatic mosaicism)
                "CGG" * 923,
                "CGG" * 933,
                "CGG" * 951,
                "CGG" * 965,
                "CGG" * 978,
                # Outliers from somatic mosaicism
                "CGG" * 201,
                "CGG" * 205,
            ],
            [],
            [],
            {"h1": 14, "h2": 5, "outlier": 2},
            {"h1": [32.0], "h2": [950.0]},
            id="Case 19: FMR1 - High somatic mosaicism should not affect parameter estimation for shorter allele (H1).",
        ),
        pytest.param(
            ["CGG"],
            ["", ""],
            [
                # Single premutation allele - all reads should be hom, none outliers
                "CGG" * 142,
                "CGG" * 145,
                "CGG" * 153,
                "CGG" * 158,
                "CGG" * 159,
                "CGG" * 173,
                "CGG" * 179,
                "CGG" * 180,
                "CGG" * 197,
                "CGG" * 197,
                "CGG" * 198,
                "CGG" * 206,
                "CGG" * 211,
                # Outlier
                "CGG" * 250,
            ],
            [],
            [],
            {"hom": 13, "outlier": 1},
            {"hom": [177.0]},
            id="Case 20: FMR1 - Single premutation allele should not be split into two haplotypes.",
        ),
        pytest.param(
            ["CTG"],
            ["", ""],
            [
                # Haplotype 1 - CTG with ATG interruptions at k-mer positions 13 and 15
                *["CTG" * 13 + "ATG" + "CTG" + "ATG" + "CTG" * 15] * 14,
                # Haplotype 2 - CTG with ATG interruptions at k-mer positions 14 and 16
                # Total: 14 + 1 + 2 + 1 + 14 = 32 k-mers (same length as H1)
                *["CTG" * 14 + "ATG" + "CTG" + "ATG" + "CTG" * 14] * 13,
            ],
            [],
            [],
            {"h1": 14, "h2": 13},
            {"h1": [31.0], "h2": [31.0]},
            id="Case 21: CTG - same length, ATG interruptions - sequence split should be detected",
        ),
    ],
)
def test_haplotyping_integration(
    satellite_seqs: list[str],
    breaks: list[str],
    spanning_sequences: list[str],
    left_flanking_sequences: list[str],
    right_flanking_sequences: list[str],
    expected_group_sizes: dict[Literal["h1", "h2", "hom", "outlier"], int],
    expected_means: dict[str, list[float]] | None,
) -> None:
    """Test the haplotype grouping functionality for both simple and complex loci."""
    locus = create_synthetic_locus(satellite_seqs, breaks)

    reads: list[Read] = (
        [make_read(f"spanning_{i}", locus.left_anchor + seq + locus.right_anchor, locus) for i, seq in enumerate(spanning_sequences)]
        + [make_read(f"left_flanking_{i}", locus.left_anchor + seq, locus) for i, seq in enumerate(left_flanking_sequences)]
        + [make_read(f"right_flanking_{i}", seq + locus.right_anchor, locus) for i, seq in enumerate(right_flanking_sequences)]
    )

    read_calls, _ = get_read_calls(reads, locus)
    grouped_reads, outlier_reads, _, _, final_params, _ = run_haplotyping(read_calls, ploidy=2)

    group_counts: dict[str, int] = {}
    for read in grouped_reads + outlier_reads:
        group_counts[read.haplotype] = group_counts.get(read.haplotype, 0) + 1

    assert group_counts == expected_group_sizes, f"Test case failed: Expected {expected_group_sizes}, got {group_counts}"

    for i, seq in enumerate(spanning_sequences):
        read_call = next((rc for rc in read_calls if rc.alignment.name == f"spanning_{i}"), None)
        assert read_call is not None, f"Read call for spanning_{i} not found"
        assert seq == read_call.alignment.str_sequence, f"Expected sequence in read {read_call.alignment.name} was {seq}, got {read_call.obs_kmer_string}"

    # Check value of means if expected_means is provided
    if expected_means is not None:
        if "h1" in expected_means and expected_means["h1"] is not None:
            assert final_params[Haplotype.H1].mean.tolist() == expected_means["h1"], (
                f"Expected h1 mean {expected_means['h1']}, got {final_params[Haplotype.H1].mean[0]}"
            )

        if "h2" in expected_means and expected_means["h2"] is not None:
            assert final_params[Haplotype.H2].mean.tolist() == expected_means["h2"], (
                f"Expected h2 mean {expected_means['h2']}, got {final_params[Haplotype.H2].mean[0]}"
            )

        if "hom" in expected_means and expected_means["hom"] is not None:
            assert final_params[Haplotype.HOM].mean.tolist() == expected_means["hom"], (
                f"Expected homozygous mean {expected_means['hom']}, got {final_params[Haplotype.HOM].mean[0]}"
            )
