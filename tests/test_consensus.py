from __future__ import annotations

import random

import pytest

from abacus import config
from abacus.consensus import create_consensus_calls
from abacus.graph import Read, get_read_calls
from abacus.locus import Location, Locus, Satellite
from abacus.utils import Haplotype


def create_random_anchor() -> str:
    """Create a random anchor sequence."""
    return "".join(random.choices("ATCG", k=config.Config.anchor_len))


def create_synthetic_locus(satellite_seqs: list[str], breaks: list[str] | None = None) -> Locus:
    """Create a synthetic locus with one or multiple satellites."""
    left_anchor = create_random_anchor()
    right_anchor = create_random_anchor()

    if breaks is None:
        breaks = [""] * (len(satellite_seqs) + 1)

    satellites = []
    start_pos = 1000
    for i, seq in enumerate(satellite_seqs):
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


def make_read(name: str, sequence: str, locus: Locus, strand: str = "+") -> Read:
    """Create a synthetic read with the given full sequence (including anchors)."""
    return Read(
        name=name,
        sequence=sequence,
        qualities=[30] * len(sequence),
        mod_5mc_probs="!" * len(sequence),
        strand=strand,
        n_soft_clipped_left=0,
        n_soft_clipped_right=0,
        locus=locus,
    )


@pytest.mark.parametrize(
    ("satellite_seqs", "spanning_sequences", "left_flanking_sequences", "right_flanking_sequences", "expected_consensus"),
    [
        pytest.param(
            ["CAG"],
            ["CAG" * 4],
            [],
            [],
            "CAG" * 4,
            id="A single spanning read is its own consensus",
        ),
        pytest.param(
            ["CAG"],
            ["CAG" * 4] * 3,
            [],
            [],
            "CAG" * 4,
            id="Spanning reads that all agree produce that consensus",
        ),
        pytest.param(
            ["CAG"],
            ["CAG" * 3, "CAG" * 4, "CAG" * 5],
            [],
            [],
            "CAG" * 4,
            id="Spanning reads - equal motif, differing length",
        ),
        pytest.param(
            ["CAG"],
            ["CAG" * 2 + "AAA" + "CAG" * 2] * 3,
            [],
            [],
            "CAG" * 2 + "AAA" + "CAG" * 2,
            id="Spanning reads - equal motif, with interruption",
        ),
        pytest.param(
            ["CAG"],
            [
                "CAG" * 1 + "AAA" + "CAG" * 3,
                "CAG" * 2 + "AAA" + "CAG" * 2,
                "CAG" * 3 + "AAA" + "CAG" * 1,
            ],
            [],
            [],
            "CAG" * 2 + "AAA" + "CAG" * 2,
            id="Spanning reads - obvious interuption in the middle of the motif, but in different positions.",
        ),
        pytest.param(
            ["CAG"],
            [
                "CAG" * 1 + "AAA" + "CAG" * 2,
                "CAG" * 2 + "AAA" + "CAG" * 1,
            ],
            [],
            [],
            "CAG" * 1 + "AAA" + "CAG" * 2,
            id="Spanning reads - obvious interuption in the middle of the motif, but in different positions - with tie.",
        ),
        pytest.param(
            ["CAG"],
            [
                "CAG" * 1 + "CCG" + "CAG" * 3,
                *["CAG" * 2 + "CCG" + "CAG" * 2] * 2,
                "CAG" * 3 + "CCG" + "CAG" * 1,
            ],
            [],
            [],
            "CAG" * 2 + "CCG" + "CAG" * 2,
            id="Spanning reads - slight interuption in the middle of the motif, different positions.",
        ),
        pytest.param(
            ["CAG"],
            [
                "CAG" * 2 + "CCG" + "CAG" * 2,
                "CAG" * 1 + "CCG" + "CAG" * 2,
                "CAG" * 1 + "CCG" + "CAG" * 1,
            ],
            [],
            [],
            "CAG" * 1 + "CCG" + "CAG" * 2,
            id="Spanning reads - slight interuption in the middle of the motif, different positions, differing length.",
        ),
        pytest.param(
            ["CAG"],
            ["CAG" * 4 + "CCG"],
            [],
            ["CAG" * 3] * 2,
            "CAG" * 5,
            id="Right flanks should fix error in spanning read.",
        ),
        pytest.param(
            ["CAG"],
            ["CCG" + "CAG" * 4],
            ["CAG" * 3] * 2,
            [],
            "CAG" * 5,
            id="Left flanks should fix error in spanning read.",
        ),
        pytest.param(
            ["CAG"],
            ["CAG" * 4] * 2,
            [],
            ["CAG" * 3] * 2,
            "CAG" * 4,
            id="Right-flanking reads that agree with spanning reads don't change the consensus",
        ),
        pytest.param(
            ["CAG"],
            ["CAG" * 4],
            [],
            ["CAG" * 6] * 2,
            "CAG" * 6,
            id="Right-flanking reads that agree with each other extend the consensus past spanning reads",
        ),
        pytest.param(
            ["CAG|AAA"],
            ["CAG" * 4],
            [],
            ["AAA" * 3] * 2,
            "CAG" + "AAA" * 3,
            id="Right-flanking reads from a different allele pull the consensus tail towards their own motif",
        ),
        pytest.param(
            ["CAG|AAA"],
            ["CAG" * 4],
            ["AAA" * 3] * 2,
            [],
            "AAA" * 3 + "CAG",
            id="Left-flanking reads from a different allele pull the consensus head towards their own motif",
        ),
        pytest.param(
            ["CAG|AAA"],
            ["CAG" * 4],
            ["CAG" * 3 + "CA"] * 2,
            [],
            "CAG" * 4,
            id="Left-flanking reads ending mid-unit shouldn't out-vote a full unit with a partial one",
        ),
        pytest.param(
            ["CAG|AAA"],
            ["CAG" * 4],
            [],
            ["AG" + "CAG" * 3] * 2,
            "CAG" * 4,
            id="Right-flanking reads starting mid-unit shouldn't out-vote a full unit with a partial one",
        ),
    ],
)
def test_create_consensus_calls_respects_flanking_read_coverage(
    satellite_seqs: list[str],
    spanning_sequences: list[str],
    left_flanking_sequences: list[str],
    right_flanking_sequences: list[str],
    expected_consensus: str,
):
    """Flanking reads should only outvote spanning reads at positions they actually cover, with real content."""
    # Anchor content affects how the aligner reports a boundary kmer, so fix the seed for determinism.
    random.seed(0)
    locus = create_synthetic_locus(satellite_seqs, ["", ""])

    reads = (
        [make_read(f"spanning_{i}", locus.left_anchor + seq + locus.right_anchor, locus) for i, seq in enumerate(spanning_sequences)]
        + [make_read(f"left_flanking_{i}", locus.left_anchor + seq, locus) for i, seq in enumerate(left_flanking_sequences)]
        + [make_read(f"right_flanking_{i}", seq + locus.right_anchor, locus) for i, seq in enumerate(right_flanking_sequences)]
    )
    read_calls, _ = get_read_calls(reads, locus)
    for read_call in read_calls:
        read_call.haplotype = Haplotype.H1

    consensus_calls = create_consensus_calls(read_calls=read_calls, haplotype=Haplotype.H1)

    assert len(consensus_calls) == 1
    assert consensus_calls[0].alignment.str_sequence == expected_consensus


# Each entry: (unit sequence repeated to build the read, alignment type, strand, read count).
StrandedReadSpec = tuple[str, str, str, int]


def make_stranded_reads(specs: list[StrandedReadSpec], locus: Locus) -> list[Read]:
    """Build reads from (sequence, alignment type, strand, count) specs - see StrandedReadSpec."""
    full_sequence_for_type = {
        "spanning": lambda seq: locus.left_anchor + seq + locus.right_anchor,
        "left": lambda seq: locus.left_anchor + seq,
        "right": lambda seq: seq + locus.right_anchor,
    }
    reads = []
    for spec_i, (seq, read_type, strand, count) in enumerate(specs):
        full_sequence = full_sequence_for_type[read_type](seq)
        reads += [make_read(f"{read_type}_{strand}_{spec_i}_{j}", full_sequence, locus, strand=strand) for j in range(count)]
    return reads


@pytest.mark.parametrize(
    ("satellite_seqs", "read_specs", "expected_consensus"),
    [
        pytest.param(
            ["CGG|GGG"],
            [
                ("CGG" * 4, "spanning", "+", 1),
                ("GGG" * 4, "spanning", "+", 10),
                ("CGG" * 4, "spanning", "-", 3),
            ],
            "CGG" * 4,
            id="Plus-strand-only error is the pooled majority, but has no minus-strand support",
        ),
        pytest.param(
            ["CGG|GGG"],
            [
                ("CGG" * 4, "spanning", "+", 1),
                ("GGG" * 4, "spanning", "+", 10),
                ("CGG" * 4, "spanning", "-", 1),
                ("CGG" * 3, "left", "-", 1),
                ("CGG" * 3, "right", "-", 1),
            ],
            "CGG" * 4,
            id="Minus-strand support for the correct motif comes from flanking reads, not just spanning ones",
        ),
        pytest.param(
            ["CAG|AAA"],
            [
                ("CAG" + "AAA" + "CAG" + "AAA" + "CAG", "spanning", "+", 10),
                ("CAG" + "CAG" + "CAG" + "AAA" + "CAG", "spanning", "+", 1),
                ("CAG" + "CAG" + "CAG" + "AAA" + "CAG", "spanning", "-", 3),
            ],
            "CAG" + "CAG" + "CAG" + "AAA" + "CAG",
            id="AAA is a plus-strand-only error at unit 1 but a genuine both-strand interruption at unit 3 - eligibility must be positional, not global",
        ),
    ],
)
def test_create_consensus_calls_resist_strand_biased_errors(satellite_seqs: list[str], read_specs: list[StrandedReadSpec], expected_consensus: str):
    """A motif that is common on one strand but rare/wrong-directioned should not out-vote a
    minority motif that is actually seen on both strands (a plus-strand-only ONT error should not
    be allowed to overrule a motif also confirmed by minus-strand reads)."""
    random.seed(0)
    locus = create_synthetic_locus(satellite_seqs, ["", ""])
    reads = make_stranded_reads(read_specs, locus)

    read_calls, _ = get_read_calls(reads, locus)
    for read_call in read_calls:
        read_call.haplotype = Haplotype.H1

    consensus_calls = create_consensus_calls(read_calls=read_calls, haplotype=Haplotype.H1)

    assert len(consensus_calls) == 1
    assert consensus_calls[0].alignment.str_sequence == expected_consensus
