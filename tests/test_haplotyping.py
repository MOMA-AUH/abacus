from collections import Counter

import pytest

from abacus.graph import get_read_calls
from abacus.haplotyping import run_haplotyping
from abacus.utils import Haplotype
from tests.helpers import make_locus, make_read, make_reads


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

    reads = make_reads(spanning_sequences, locus) + [make_read(f"flanking_{i}", seq, locus) for i, seq in enumerate(flanking_sequences)]
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


def test_length_grouping_precedes_consensus_refinement():
    locus = make_locus("(CAG|CAA)+")
    # Length grouping puts the two 12-copy CAG reads with CAA; sequence supports CAG instead.
    sequences = ["CAG" * 10] * 2 + ["CAA" * 12] * 5 + ["CAG" * 12] * 2
    read_calls, _ = get_read_calls(make_reads(sequences, locus), locus)
    grouped, outliers, _, _, _, _ = run_haplotyping(read_calls, ploidy=2)
    caa_haplotype = grouped[2].haplotype
    misplaced = [read for read in grouped if read.alignment.name in ("spanning_7", "spanning_8")]
    assert not outliers
    assert all(read.is_spanning() and read.haplotype == caa_haplotype for read in misplaced)
    assert len(misplaced) == 2
    assert Counter(read.haplotype for read in grouped) == {Haplotype.H1: 2, Haplotype.H2: 7}


@pytest.mark.parametrize(
    ("sequences", "expected_groups"),
    [
        pytest.param(["CAG" * 10], {"outlier": 1}, id="singleton"),
        pytest.param(["CAG" * 10] * 2, {"hom": 2}, id="two_support_one_haplotype"),
        pytest.param(["CAG" * 10] * 2 + ["CAG" * 30], {"hom": 3}, id="three_cannot_support_two_haplotypes"),
        pytest.param(["CAG" * 10] * 2 + ["CAG" * 30] * 2, {"h1": 2, "h2": 2}, id="two_per_haplotype"),
        pytest.param(["CAG" * 10] * 3 + ["CAG" * 30], {"hom": 3, "outlier": 1}, id="singleton_removed_then_homozygous"),
        pytest.param(["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"], {"hom": 8}, id="below_sequence_split_depth"),
    ],
)
def test_haplotyping_support(sequences, expected_groups):
    locus = make_locus("(CAG)+")
    read_calls, _ = get_read_calls(make_reads(sequences, locus), locus)

    grouped, removed, _, _, _, _ = run_haplotyping(read_calls, ploidy=2)

    assert Counter(read.haplotype.value for read in grouped + removed) == expected_groups
