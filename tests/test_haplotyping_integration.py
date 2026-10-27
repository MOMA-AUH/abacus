from __future__ import annotations

from collections import Counter
from typing import Literal

import pytest

from abacus.consensus import build_consensus_for_locus
from abacus.graph import (
    Read,
    get_read_calls,
)
from abacus.haplotyping import run_haplotyping
from abacus.utils import Haplotype
from tests.helpers import make_locus, make_read


@pytest.mark.parametrize(
    (
        "structure",
        "spanning_sequences",
        "left_flanking_sequences",
        "right_flanking_sequences",
        "expected_group_sizes",
        "expected_means",
        "expected_consensus",
    ),
    [
        # Simple test cases (single satellite)
        pytest.param(
            "(CAG)+",
            [],
            [],
            [],
            {},
            None,
            None,
            id="Simple-Empty",
        ),
        pytest.param(
            "(CAG)+",
            ["CAG"],
            [],
            [],
            {"hom": 1},
            None,
            {"hom": "CAG"},
            id="Simple-Single spanning",
        ),
        pytest.param(
            "(CAG)+",
            ["CAGCAG", "CAGCAG"],
            ["CAG"],
            ["CAG"],
            {"hom": 4},
            None,
            {"hom": "CAG" * 2},
            id="Simple-Homozygous - Small",
        ),
        pytest.param(
            "(CAG)+",
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
            {"h1": "CAG", "h2": "CAG" * 3},
            id="Simple-Heterozygous - Small",
        ),
        pytest.param(
            "(CTG)+",
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
            {"h1": "CTG" * 27, "h2": "CTG" * 30},
            id="Simple-Skewed split. Should be called heterozygous",
        ),
        # Complex test cases (multiple satellites)
        pytest.param(
            "(CAG)*(CTG)*",
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
            {"h1": "CAG" * 2 + "CTG" * 3, "h2": "CAG" * 4 + "CTG" * 2},
            id="Complex-Two satellites - only spanning",
        ),
        pytest.param(
            "(CAG)*(CTG)*",
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
            {"h1": "CAG" * 2 + "CTG" * 3, "h2": "CAG" * 4 + "CTG" * 2},
            id="Complex-Two satellites - with flanking",
        ),
        # Cases with 2 satellites and a break sequence (HTT is a good example)
        pytest.param(
            "(CAG)*CAACAG(CCG)*",
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
            {"h1": "CAG" * 2 + "CAACAG" + "CCG" * 3, "h2": "CAG" * 5 + "CAACAG" + "CCG" * 2},
            id="Complex-Two satellites with break - only spanning",
        ),
        pytest.param(
            "(CAG)*CAACAG(CCG)*",
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
            {"h1": "CAG" * 2 + "CAACAG" + "CCG" * 3, "h2": "CAG" * 4 + "CAACAG" + "CCG" * 2},
            id="Complex-Two satellites with break - with flanking",
        ),
        # Cases with 3 satellites
        pytest.param(
            "(CAG)*(CCG)*(CTG)*",
            [
                *["CAG" * 2 + "CCG" + "CTG" * 3] * 2,  # H1: 2 CAG, 1 CCG, 3 CTG
                *["CAG" * 3 + "CCG" * 2 + "CTG"] * 2,  # H2: 3 CAG, 2 CCG, 1 CTG
            ],
            ["CAG" * 2 + "CCG" + "CTG"],  # Left flanking matching H1
            ["CCG" + "CTG"],  # Right flanking matching H2
            {"h1": 3, "h2": 3},
            {"h1": [3.0, 2.0, 1.0], "h2": [2.0, 1.0, 3.0]},
            {"h1": "CAG" * 3 + "CCG" * 2 + "CTG", "h2": "CAG" * 2 + "CCG" + "CTG" * 3},
            id="Complex-Three satellites - heterozygous",
        ),
        # OR operator test cases (satellite with multiple alternative sequences)
        pytest.param(
            "(CAG|CAA)+",
            [
                *["CAG" * 10] * 5,
                *["CAA" * 15] * 7,
            ],
            [],
            [],
            {"h1": 5, "h2": 7},
            {"h1": [10.0], "h2": [15.0]},
            {"h1": "CAG" * 10, "h2": "CAA" * 15},
            id="OR-Heterozygous by repeat count",
        ),
        pytest.param(
            "(CAG|CAA)+",
            [
                *["CAG" * 10] * 7,
                *["CAA" * 10] * 5,
            ],
            [],
            [],
            {"h1": 7, "h2": 5},
            {"h1": [10.0], "h2": [10.0]},
            {"h1": "CAG" * 10, "h2": "CAA" * 10},
            id="OR-Heterozygous by motif, but same length.",
        ),
        pytest.param(
            "(GGCCCC|GGCCCCC)+",
            [
                *["GGCCCC" * 12] * 8,
                *["GGCCCCC" * 20] * 8,
            ],
            [],
            [],
            {"h1": 8, "h2": 8},
            {"h1": [12.0], "h2": [20.0]},
            {"h1": "GGCCCC" * 12, "h2": "GGCCCCC" * 20},
            id="OR-Heterozygous, different-length alternatives",
        ),
        # Haplotyping based on motif differences
        pytest.param(
            "(CAG)+",
            [*["CAG" * 10] * 7, "CAG" * 9 + "CGG"],  # 8 reads: 7 normal + 1 with sequencing error, all same length
            [],
            [],
            {"hom": 8},
            {"hom": [10.0]},
            {"hom": "CAG" * 10},
            id="Sequence split-8 reads below min_haplotyping_depth-should not create singleton",
        ),
        pytest.param(
            "(CAG)+",
            [*["CAG" * 10] * 12],
            [],
            [],
            {"hom": 12},
            {"hom": [10.0]},
            {"hom": "CAG" * 10},
            id="Sequence split-No differences should remain homozygous",
        ),
        pytest.param(
            "(CAG)+",
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
            {"h1": "CAGCAG", "h2": "CAGCGG"},
            id="Sequence split-Short and simple",
        ),
        pytest.param(
            "(CAG)+",
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
            {"h1": "CAG" * 31, "h2": "CAG" * 20 + "CGG" + "CAG" * 10},
            id="Sequence split-Long and simple",
        ),
        pytest.param(
            "(CAG)*CAACAG(CCG)*",
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
            {"h1": "CAG" * 7 + "CAACAG" + "CCGCCGCCG", "h2": "CAG" * 7 + "CAACAG" + "CCGCAGCCG"},
            id="Sequence split-HTT-like case with interruption",
        ),
        # Outlier test cases
        pytest.param(
            "(CTG)+",
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
            {"h1": "CTG" * 28, "h2": "CTG" * 31},
            id="Outlier-Long outlier that should be separated from H2",
        ),
        # Test cases from data - simple loci
        pytest.param(
            "(AARRG)+",
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
            {"h1": "AAAAG" * 771, "h2": "AAAAG" * 864},
            id="Case 1: Long RCF1. Only one flanking read for H2",
        ),
        pytest.param(
            "(AARRG)+",
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
            {"h1": "AAAAG" * 12, "h2": "AAAAG" * 938},
            id="Case 2: RFC1. Short H1, Long H2",
        ),
        pytest.param(
            "(GAA)+",
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
            {"h1": "GAA" * 9, "h2": "GAA" * 11},
            id="Case 3: FGF14. Close haplotypes, 1 apart",
        ),
        pytest.param(
            "(GAA)+",
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
            {"h1": "GAA" * 8, "h2": "GAA" * 9},
            id="Case 4: FGF14. Close haplotypes 2, 0 apart",
        ),
        pytest.param(
            "(GAA)+",
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
            {"h1": "GAA" * 5, "h2": "GAA" * 13},
            id="Case 5: DMPK",
        ),
        pytest.param(
            "(GAA)+",
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
            {"h1": "GAA" * 8, "h2": "GAA" * 109},
            id="Case 6: RFC1",
        ),
        pytest.param(
            "(CTG)+",
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
            {"h1": "CTG" * 28, "h2": "CTG" * 30},
            id="Case 7: ATXN1",
        ),
        pytest.param(
            "(GGCCCC)+",
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
            {"h1": "GGCCCC" * 12, "h2": "GGCCCC" * 958},
            id="Case 9: C9ORF72",
        ),
        pytest.param(
            "(NGC)+",
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
            {"h1": "AGC" * 12, "h2": "AGC" * 15},
            id="Case 10: ARX_EIEE",
        ),
        pytest.param(
            "(GCC)+",
            [],
            ["GCC" * 7],
            [],
            {"hom": 1},
            {"hom": [7.0]},
            {"hom": "GCC" * 7},
            id="Case 11: XYLT1 - Low coverage, no spanning reads",
        ),
        pytest.param(
            "(GGC)+",
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
            {"h1": "GGC" * 19, "h2": "GGC" * 27},
            id="Case 12: NOTCH2NLC",
        ),
        pytest.param(
            "(CAG)*(CCG)*(CTG)*",
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
            {"h1": "CAG" * 15 + "CCG" * 9 + "CTG" * 18, "h2": "CAG" * 16 + "CCG" * 9 + "CTG" * 21},
            id="Case 13: CNBP",
        ),
        pytest.param(
            "(GCN)+",
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
            {"h1": "GCC" * 10, "h2": "GCC" * 14},
            id="Case 14: PABPN1 - Outlier should not affect haplotype grouping (issue #18).",
        ),
        pytest.param(
            "(CTG)+",
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
            {"h1": "CTG" * 19, "h2": "CTG" * 67},
            id="Case 15: ATXN3 - Flanking outlier should be separated from H2",
        ),
        pytest.param(
            "(GAA)+",
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
            # H2's own motif is GAAGGA - a genuine, both-strand-agreeing rotation of GAA, not an error, so it
            # survives into the consensus as 139 clean GAAGGA units (278 3bp kmers), not 278 GAA units.
            {"h1": "GAA" * 279, "h2": "GAAGGA" * 139},
            id="Case 16: FGF14 - Same varying length, differing motif",
        ),
        pytest.param(
            "(GGCCCC)+",
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
            # H2's mean is None (too mosaic to estimate) - no target length, so no consensus to check either.
            {"h1": "GGCCCC" * 12},
            id="Case 17: C9ORF72 - High somatic mosaicism should not affect parameter estimation for shorter allele (H1).",
        ),
        pytest.param(
            "(CAG)+",
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
            # H2 is fit to widely spread mosaic reads, so its integer estimate shifts by one between numpy/scipy versions
            {"h1": [5.0], "h2": [pytest.approx(1825.0, abs=1)]},
            {"h1": "CAG" * 5, "h2": {"CAG" * n for n in range(1824, 1827)}},
            id="Case 18: DMPK - High somatic mosaicism should not affect parameter estimation for shorter allele (H1).",
        ),
        pytest.param(
            "(CGG)+",
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
            {"h1": "CGG" * 32, "h2": "CGG" * 950},
            id="Case 19: FMR1 - High somatic mosaicism should not affect parameter estimation for shorter allele (H1).",
        ),
        pytest.param(
            "(CGG)+",
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
            {"hom": "CGG" * 177},
            id="Case 20: FMR1 - Single premutation allele should not be split into two haplotypes.",
        ),
        pytest.param(
            "(CTG)+",
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
            {"h1": "CTG" * 13 + "ATG" + "CTG" + "ATG" + "CTG" * 15, "h2": "CTG" * 14 + "ATG" + "CTG" + "ATG" + "CTG" * 14},
            id="Case 21: CTG - same length, ATG interruptions - sequence split should be detected",
        ),
        pytest.param(
            "(AAAAG)+",
            [
                # Haplotype 1: short allele, tight cluster (~417-490 repeats)
                *["AAAAG" * 417] * 1,
                *["AAAAG" * 420] * 2,
                *["AAAAG" * 423] * 2,
                *["AAAAG" * 424] * 2,
                *["AAAAG" * 426] * 1,
                *["AAAAG" * 428] * 1,
                *["AAAAG" * 432] * 2,
                *["AAAAG" * 436] * 1,
                *["AAAAG" * 438] * 1,
                *["AAAAG" * 453] * 1,
                *["AAAAG" * 461] * 1,
                *["AAAAG" * 462] * 1,
                *["AAAAG" * 490] * 1,
                # Haplotype 2: long allele, only 2 supporting spanning reads (~761-765 repeats)
                "AAAAG" * 761,
                "AAAAG" * 765,
            ],
            [],
            [],
            {"h1": 16, "h2": 2, "outlier": 1},
            {"h1": [432.0], "h2": [763.0]},
            {"h1": "AAAAG" * 432, "h2": "AAAAG" * 763},
            id="Case 22: RFC1_complex - long allele with few spanning reads undetected due to biased initial split (regression test)",
        ),
        # Consensus segmentation: a middle satellite entirely unreached by any read (left-flanking
        # reads only ever reach satellite 1, right-flanking only ever reach satellite 3) - its
        # estimated count must come out as 0 (not confused with "unobserved"), and the consensus
        # must contain zero copies of it, not a fabricated or flat-position-voted value.
        pytest.param(
            "(CAG)*(CCG)*(CTG)*",
            [],
            ["CAG" * 17] * 3,
            ["CTG" * 12] * 3,
            {"hom": 6},
            {"hom": [17.0, 0.0, 12.0]},
            {"hom": "CAG" * 17 + "CTG" * 12},
            id="Consensus-Entirely unreached middle satellite (no spanning reads at all)",
        ),
        # Slipped reads: a stretch of short CG copies is cheaper for the aligner to pack into fewer
        # CGG copies with insertions, undercounting the read. Most reads below are slipped, each at
        # a different position with the same copy count as its allele - so every copy is still CGG
        # in most reads, and neither the means nor the consensus may shift.
        pytest.param(
            "(CGG)+",
            [
                *["CGG" * 30] * 3,
                *["CGG" * start + "CG" * 4 + "CGG" * (26 - start) for start in (4, 8, 12, 16, 20)],
                *["CGG" * 40] * 3,
                *["CGG" * start + "CG" * 4 + "CGG" * (36 - start) for start in (6, 12, 18, 24, 30)],
            ],
            [],
            [],
            {"h1": 8, "h2": 8},
            {"h1": [30.0], "h2": [40.0]},
            {"h1": "CGG" * 30, "h2": "CGG" * 40},
            id="Slipped CG copies in some reads are counted, not packed into fewer CGG copies",
        ),
        # Two reads far shorter than both alleles must not become a haplotype, leaving the 10-read
        # 51-copy allele as outlier
        pytest.param(
            "(TAGTCATAGTACCCCAA|TAGTCATAGTACCA)*(GVA)*",
            [
                # Haplotype 1
                *["GAA" * 37] * 2,
                *["GAA" * 38] * 10,
                *["GAA" * 39] * 3,
                "GAA" * 40,
                # Haplotype 2
                "GAA" * 49,
                *["GAA" * 50] * 2,
                *["GAA" * 51] * 5,
                *["GAA" * 52] * 2,
                # Outliers
                "GAA" * 8,
                "GAA" * 11,
            ],
            [],
            [],
            {"h1": 16, "h2": 10, "outlier": 2},
            {"h1": [0.0, 38.0], "h2": [0.0, 51.0]},
            {"h1": "GAA" * 38, "h2": "GAA" * 51},
            id="Case 24: FGF14_complex - Two short reads should not displace the 10-read long allele",
        ),
        # As Case 24, but the two short reads also carry the 17-mer. From a sample that looks ~3-5%
        # contaminated: these reads match neither phased haplotype and are most likely another person's DNA
        pytest.param(
            "(TAGTCATAGTACCCCAA|TAGTCATAGTACCA)*(GVA)*",
            [
                # Haplotype 1
                *["GAA" * 37] * 2,
                *["GAA" * 38] * 10,
                *["GAA" * 39] * 3,
                "GAA" * 40,
                # Haplotype 2
                "GAA" * 49,
                *["GAA" * 50] * 2,
                *["GAA" * 51] * 5,
                *["GAA" * 52] * 2,
                # Outliers: contamination
                "TAGTCATAGTACCCCAA" + "GAA" * 8,
                "TAGTCATAGTACCCCAA" + "GAA" * 11,
            ],
            [],
            [],
            {"h1": 16, "h2": 10, "outlier": 2},
            {"h1": [0.0, 38.0], "h2": [0.0, 51.0]},
            {"h1": "GAA" * 38, "h2": "GAA" * 51},
            id="Case 25: FGF14_complex - Two contaminating reads with the 17-mer should not displace the 10-read long allele",
        ),
        # Scattered short reads must not be split off as a haplotype in place of a tight minor allele of the same
        # size. Removing the minor allele leaves no more outliers than the scattered reads, so only the initial
        # split (median-based, not mean-based) gets this right.
        pytest.param(
            "(GAA)+",
            [
                # Haplotype 1
                *["GAA" * 20] * 10,
                *["GAA" * 21] * 4,
                # Haplotype 2
                *["GAA" * 32] * 4,
                # Outliers
                "GAA" * 3,
                *["GAA" * 7] * 2,
                "GAA" * 10,
            ],
            [],
            [],
            {"h1": 14, "h2": 4, "outlier": 4},
            {"h1": [20.0], "h2": [32.0]},
            {"h1": "GAA" * 20, "h2": "GAA" * 32},
            id="Case 26: Scattered short reads should not be split off in place of a tight minor allele",
        ),
    ],
)
def test_haplotyping_integration(
    structure: str,
    spanning_sequences: list[str],
    left_flanking_sequences: list[str],
    right_flanking_sequences: list[str],
    expected_group_sizes: dict[Literal["h1", "h2", "hom", "outlier"], int],
    expected_means: dict[str, list[float]] | None,
    expected_consensus: dict[str, str | set[str]] | None,
) -> None:
    """Test the haplotype grouping functionality for both simple and complex loci."""
    locus = make_locus(structure)

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

    # Build the real, final per-haplotype consensus the same way main.py does, and check it against
    # every haplotype expected_consensus names - always run so a case with no expectations still
    # acts as a smoke test that this doesn't crash.
    consensus_calls = build_consensus_for_locus(grouped_reads, final_params)
    observed_consensus = {consensus_call.haplotype.value: consensus_call.alignment.str_sequence for consensus_call in consensus_calls}
    if expected_consensus is not None:
        for haplotype, expected_sequence in expected_consensus.items():
            # A set lists every acceptable consensus, for cases whose estimate may differ by a unit
            allowed = expected_sequence if isinstance(expected_sequence, set) else {expected_sequence}
            assert observed_consensus.get(haplotype) in allowed, f"Expected {haplotype} consensus {expected_sequence}, got {observed_consensus.get(haplotype)}"


# Real GIPC1 flanks, around a (CCG)* repeat
GIPC1_LEFT_FLANK = "GCAGGCCCGCGACCCCCGAGGCCCCCGGAGCAGACACATCCTTCTCGCAGAGGCCACTCACCTGCTC"
GIPC1_RIGHT_FLANK = "CTGCCTCCGCCTCCCCGTGCGCACCCGGCTCGGCCCTCCGCAAACTCCAGACCGGGCCTCTCCCG"


@pytest.mark.parametrize(
    ("flanking_type", "noisy_anchor_parts"),
    [
        pytest.param(
            "left_flanking",
            # Each read ends 33-64 bp into the right anchor, with ~10% errors
            [
                "CTGCACCTCGCATCCTCCTGCGCACCCGACTCGG",
                "CTGCCTGCCGCCTCCCCATGCGCACCCGGCGCGGCCCCCGGAAAC",
                "CTTGCCTCCGCCTCCCCCGGCGCACCCTGCTCGGCCCTCCGCACAACTCCAGACCAGGGCCTC",
                "CTGCCTCGCGCCTCCCCGTGCGCACCCGGCTCGGCCACTCCGCAAACATCCAGACCGGGCCTCTCCC",
            ],
            id="Left-flanking reads ending in the right anchor",
        ),
        pytest.param(
            "right_flanking",
            # Each read starts 33-64 bp into the left anchor, with ~10% errors
            [
                "CACATCCTTCTGCGCAGAGGCCAATCCCTGCTC",
                "CCCCGGAGCAAGACACTCCTTCCTCGGAGAGCCACTCACCTGTC",
                "CGGACCCCCGAGGACCCCGGAGCAGACACATCCTTACGCGCAGAGGCCCACTCCACTCCTC",
                "GCGCCCGCAACCACCGAGGGCCCCCGGAGCAGACACATACCTTCTCGCAAGAAGGCCATCACCTGCTC",
            ],
            id="Right-flanking reads starting in the left anchor",
        ),
    ],
)
def test_flanking_reads_partly_in_noisy_anchor_are_not_counted_as_copies(flanking_type: str, noisy_anchor_parts: list[str]) -> None:
    """GIPC1: flanking reads that reach partway into the far anchor, with ONT errors, must not count it as copies.

    The read's part of the anchor must be found despite the errors, or it is aligned as extra CCG
    copies; the reads then form a false ~30-copy haplotype that pushes the 8 spanning 13-copy reads out.
    """
    locus = make_locus("(CCG)+")
    locus.left_anchor = locus.left_anchor[: -len(GIPC1_LEFT_FLANK)] + GIPC1_LEFT_FLANK
    locus.right_anchor = GIPC1_RIGHT_FLANK + locus.right_anchor[len(GIPC1_RIGHT_FLANK) :]

    h1_repeat = "CCG" * 10
    h2_repeat = "CCG" * 13

    # 22 spanning H1 reads, 8 spanning H2 reads, and 4 flanking H2 reads that reach into the far anchor
    spanning_sequences = [h1_repeat] * 22 + [h2_repeat] * 8
    if flanking_type == "left_flanking":
        flanking_sequences = [locus.left_anchor + h2_repeat + part for part in noisy_anchor_parts]
    else:
        flanking_sequences = [part + h2_repeat + locus.right_anchor for part in noisy_anchor_parts]

    reads = [make_read(f"spanning_{i}", locus.left_anchor + seq + locus.right_anchor, locus) for i, seq in enumerate(spanning_sequences)] + [
        make_read(f"flanking_{i}", seq, locus) for i, seq in enumerate(flanking_sequences)
    ]
    read_calls, _ = get_read_calls(reads, locus)

    # Every flanking read counts only its own 13 copies
    for read_call in read_calls:
        if read_call.alignment.name.startswith("flanking"):
            assert read_call.satellite_count == [13], f"{read_call.alignment.name}: {read_call.obs_kmer_string}"

    # So they join H2 instead of displacing it
    grouped_reads, outlier_reads, _, _, final_params, _ = run_haplotyping(read_calls, ploidy=2)
    assert Counter(read.haplotype for read in grouped_reads + outlier_reads) == {Haplotype.H1: 22, Haplotype.H2: 12}
    assert final_params[Haplotype.H1].mean.tolist() == [10.0]
    assert final_params[Haplotype.H2].mean.tolist() == [13.0]
