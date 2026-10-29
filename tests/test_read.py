"""Read conversion tests: synthetic records shaped like reads in a SAM/BAM file, read as expected."""

from __future__ import annotations

import array

import pysam
import pytest

from abacus.read import Read, get_5mc_modification_probs
from tests.helpers import make_locus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_alignment(sequence: str | None, qual_string: str | None = None) -> pysam.AlignedSegment:
    alignment = pysam.AlignedSegment()
    alignment.query_name = "read_0001"
    if sequence is not None:
        alignment.query_sequence = sequence
    if qual_string is not None:
        alignment.query_qualities = array.array("B", [ord(c) - 33 for c in qual_string])
    return alignment


# ---------------------------------------------------------------------------
# QUAL: SAM QUAL to the Phred+33 FASTQ quality string written for minigraph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sequence", "qual_string", "expected_fastq"),
    [
        pytest.param(
            "AACGTTACGCATTACGCAAGGTTACGCAA",
            ",590+36:,+76<921B1.,441357:4)",
            "@read_0001\nAACGTTACGCATTACGCAAGGTTACGCAA\n+\n,590+36:,+76<921B1.,441357:4)\n",
            id="ONT read record",
        ),
        pytest.param(
            "AA",
            "!~",
            "@read_0001\nAA\n+\n!~\n",
            id="SAM quality range endpoints",
        ),
        pytest.param(
            "ACGTACGT",
            "5<8I+2E9",
            "@read_0001\nACGTACGT\n+\n5<8I+2E9\n",
            id="Phred scores",
        ),
        pytest.param(
            "ACGT",
            None,
            "@read_0001\nACGT\n+\n\n",
            id="record without qualities",
        ),
    ],
)
def test_from_alignment_qualities(sequence: str, qual_string: str | None, expected_fastq: str) -> None:
    """A SAM record's QUAL becomes Phred scores, its Phred+33 quality string, and the FASTQ written for minigraph."""
    read = Read.from_alignment(make_alignment(sequence, qual_string), locus=make_locus("(CAG)+"))

    assert read.qualities == [ord(c) - 33 for c in (qual_string or "")]
    assert read.quality_string() == (qual_string or "")
    assert read.to_fastq() == expected_fastq


# ---------------------------------------------------------------------------
# MM: 5mC modification probabilities from MM/ML tags
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sequence", "mm_tag", "ml_tag", "is_reverse", "expected_mod_probs"),
    [
        # Modkit-style tag: a 5hmC call on the first C and 5mC calls on later cytosines.
        # 5mC bins: 255 -> '+', 128 -> '&', 0 -> '!'; the 5hmC call is ignored.
        pytest.param(
            "AACGTTACGCATTACGCAAGGTTACGCAA",
            "C+h?,0;C+m?,1,2,0;",
            [200, 255, 128, 0],
            False,
            "0000000+00000000&0000000!0000",
            id="forward strand",
        ),
        # pysam parses MM strand-aware: the tag describes the original read and
        # positions come back in stored-sequence coordinates.
        pytest.param(
            "AACGTTACGCATTACGCAAGGTTACGCAA",
            "C+h?,0;C+m?,1,2,0;",
            [200, 255, 128, 0],
            True,
            "000!0000&00000000000+00000000",
            id="reverse strand",
        ),
        pytest.param(
            "ATGC",
            "C+m?,0;",
            [128],
            False,
            "000&",
            id="modification on the last base",
        ),
        # Callers that emit MM without ML leave the probability unknown; pysam reports -1, which bins to '!'.
        pytest.param(
            "ACGT",
            "C+m?,0;",
            None,
            False,
            "0!00",
            id="MM tag without ML tag",
        ),
        pytest.param(
            "ACGTACGT",
            None,
            None,
            False,
            "0" * 8,
            id="no MM tag",
        ),
    ],
)
def test_5mc_probs_from_mm_tags(
    sequence: str,
    mm_tag: str | None,
    ml_tag: list[int] | None,
    is_reverse: bool,
    expected_mod_probs: str,
) -> None:
    alignment = make_alignment(sequence)
    alignment.is_reverse = is_reverse
    if mm_tag is not None:
        alignment.set_tag("MM", mm_tag)
        if ml_tag is not None:
            alignment.set_tag("ML", array.array("B", ml_tag))

    assert get_5mc_modification_probs(alignment) == expected_mod_probs
