"""Unit tests for the consensus hill-climb internals in abacus.consensus_search.

Separate from test_consensus.py, which tests create_consensus_calls end-to-end through the real
graph-alignment pipeline.
"""

import pytest

from abacus import consensus_search


def units(s: str) -> list[str]:
    return [s[i : i + 3] for i in range(0, len(s), 3)]


def test_slippage_growth_skips_lone_interruptions():
    """Slippage lengthens a stretch of identical units; it does not duplicate a one-off
    interruption into a false repeat. Shrinking has no such restriction.
    """
    seed = "AABA"  # runs: AA, B, A
    assert set(consensus_search.slippage_candidates(seed, grow=True)) == {"AAABA"}
    assert set(consensus_search.slippage_candidates(seed, grow=False)) == {"ABA", "AAA", "AAB"}


def test_repair_length_extends_the_repeat_not_the_interruption():
    """A seed one unit short must regain a repeat unit, never a second copy of the interruption."""
    reads = [(units("CAG" * 4 + "AAA" + "CAG" * 5), "spanning", "+")] * 5
    encoded, char_to_kmer = consensus_search.encode_reads(reads)
    target = consensus_search.median_length(encoded, spanning_only=True)
    short_seed = encoded[0][0][:-1]

    repaired = consensus_search.repair_length(short_seed, encoded, target)

    assert len(repaired) == target
    assert [char_to_kmer[c] for c in repaired].count("AAA") == 1


def test_starting_seed_prefers_spanning_reads_over_more_numerous_flanking_reads():
    """A single spanning read sees the whole locus; two flanking reads of a different allele only
    ever see their own side. The flanking reads' raw kmer count (6) outnumbers the spanning read's
    (4), but the search must still start from what the spanning read actually shows.
    """
    reads = [
        (["CAG", "CAG", "CAG", "CAG"], "spanning", "+"),
        (["AAA", "AAA", "AAA"], "right", "+"),
        (["AAA", "AAA", "AAA"], "right", "+"),
    ]
    encoded, char_to_kmer = consensus_search.encode_reads(reads)

    seed = consensus_search.starting_seed(encoded)

    assert [char_to_kmer[c] for c in seed] == ["CAG"] * 4


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [(["CAG", "CAG", "CAG", "CAG"], "spanning", "+")] + [(["AAA", "AAA", "AAA"], "right", "+")] * 2,
            ["CAG", "AAA", "AAA", "AAA"],
            id="right",
        ),
        pytest.param(
            [(["CAG", "CAG", "CAG", "CAG"], "spanning", "+")] + [(["AAA", "AAA", "AAA"], "left", "+")] * 2,
            ["AAA", "AAA", "AAA", "CAG"],
            id="left",
        ),
    ],
)
def test_hill_climb_does_not_let_flanking_reads_corrupt_a_lone_spanning_read(reads, expected):
    """Regression test: a majority-vote starting seed built without spanning priority picks the
    flanking reads' motif outright, corrupting the whole consensus rather than just the tail/head
    the flanking reads actually cover.
    """
    assert consensus_search.hill_climb(reads) == expected


def test_trim_partial_boundary_kmers_drops_uncorroborated_boundary():
    """A left-flanking read's last kmer, or a right-flanking read's first kmer, is dropped when it
    never appears anywhere else as an interior kmer - the signature of a read simply running out
    mid-unit, not a real short motif.
    """
    reads = [
        (["CAG", "CAG", "CAG"], "spanning", "+"),
        (["CAG", "CAG", "CA"], "left", "+"),
        (["AG", "CAG", "CAG"], "right", "+"),
    ]

    trimmed = consensus_search.trim_partial_boundary_kmers(reads)

    assert trimmed == [
        (["CAG", "CAG", "CAG"], "spanning", "+"),
        (["CAG", "CAG"], "left", "+"),
        (["CAG", "CAG"], "right", "+"),
    ]


def test_trim_partial_boundary_kmers_keeps_corroborated_boundary():
    """A boundary kmer that also shows up as an interior kmer elsewhere is real vocabulary, not a
    truncation artifact, and must not be dropped.
    """
    reads = [
        (["CAG", "AAA", "CAG"], "spanning", "+"),
        (["CAG", "AAA"], "left", "+"),  # boundary kmer "AAA" is corroborated by the spanning read
    ]

    assert consensus_search.trim_partial_boundary_kmers(reads) == reads


def test_trim_partial_boundary_kmers_does_not_let_two_truncated_reads_corroborate_each_other():
    """Two reads independently truncated at the same point still share a boundary kmer, but
    neither occurrence is interior - read support alone must not be enough to keep it.
    """
    reads = [(["AG", "CAG", "CAG"], "right", "+")] * 2

    trimmed = consensus_search.trim_partial_boundary_kmers(reads)

    assert trimmed == [(["CAG", "CAG"], "right", "+")] * 2


def test_trim_partial_boundary_kmers_drops_a_flanking_read_left_with_nothing():
    """A flanking read that is only ever its own boundary kmer trims down to empty, not a crash."""
    reads = [(["CAG", "CAG"], "spanning", "+"), (["AG"], "right", "+")]

    trimmed = consensus_search.trim_partial_boundary_kmers(reads)

    assert trimmed == [(["CAG", "CAG"], "spanning", "+")]


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [(["CAG", "CAG", "CAG", "CAG"], "spanning", "+")] + [(["CAG", "CAG", "CAG", "CA"], "left", "+")] * 2,
            ["CAG", "CAG", "CAG", "CAG"],
            id="left-partial",
        ),
        pytest.param(
            [(["CAG", "CAG", "CAG", "CAG"], "spanning", "+")] + [(["AG", "CAG", "CAG", "CAG"], "right", "+")] * 2,
            ["CAG", "CAG", "CAG", "CAG"],
            id="right-partial",
        ),
    ],
)
def test_hill_climb_does_not_let_a_partial_boundary_kmer_outvote_a_full_unit(reads, expected):
    """Regression test for the two mid-unit-boundary cases in test_consensus.py."""
    assert consensus_search.hill_climb(reads) == expected


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [(["CGG"], "spanning", "+")] * 1 + [(["GGG"], "spanning", "+")] * 10 + [(["CGG"], "spanning", "-")] * 3,
            ["CGG"],
            id="spanning only",
        ),
        pytest.param(
            [(["CGG"], "spanning", "+")] * 1
            + [(["GGG"], "spanning", "+")] * 10
            + [(["CGG"], "spanning", "-")] * 1
            + [(["CGG"], "left", "-")] * 1
            + [(["CGG"], "right", "-")] * 1,
            ["CGG"],
            id="minus-strand support split across spanning and flanking reads",
        ),
    ],
)
def test_hill_climb_requires_cross_strand_support_before_trusting_the_majority_token(reads, expected):
    """GGG is the pooled majority (10 reads) but plus-strand-only; CGG is the minority yet the
    only motif seen on both strands. Zero support on one strand must lose to both-strand support,
    regardless of vote count.
    """
    assert consensus_search.hill_climb(reads) == expected


def test_hill_climb_disqualifies_a_kmer_by_position_not_just_globally():
    """AAA is a plus-strand-only error at unit 1, but a genuine both-strand-confirmed interruption
    at unit 3. Eligibility elsewhere must not make it eligible at unit 1 too.
    """
    error_read: consensus_search.KmerRead = (["CAG", "AAA", "CAG", "AAA", "CAG"], "spanning", "+")
    correct_read_plus: consensus_search.KmerRead = (["CAG", "CAG", "CAG", "AAA", "CAG"], "spanning", "+")
    correct_read_minus: consensus_search.KmerRead = (["CAG", "CAG", "CAG", "AAA", "CAG"], "spanning", "-")
    reads = [error_read for _ in range(10)] + [correct_read_plus] + [correct_read_minus for _ in range(3)]

    assert consensus_search.hill_climb(reads) == ["CAG", "CAG", "CAG", "AAA", "CAG"]


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [
                (["CAG", "CAG", "AAA", "CAG", "CAG", "CAG"], "spanning", "+"),
                (["CAG", "CAG", "CAG", "AAA", "CAG", "CAG"], "spanning", "-"),
            ],
            ["CAG", "CAG", "AAA", "CAG", "CAG", "CAG"],
            id="same length, interruption one column apart",
        ),
        pytest.param(
            [
                (["CAG", "CAG", "AAA", "CAG", "CAG", "CAG"], "spanning", "+"),
                (["CAG", "CAG", "CAG", "AAA", "CAG", "CAG", "CAG"], "spanning", "-"),
            ],
            ["CAG", "CAG", "CAG", "AAA", "CAG", "CAG", "CAG"],
            id="differing total length, interruption one column apart",
        ),
    ],
)
def test_hill_climb_tolerates_a_one_column_registration_offset_between_strands(reads, expected):
    """Upstream length variation can shift a real interruption one column between strands; exact
    cross-strand agreement would wrongly drop it. A one-column window recognizes these as the
    same interruption without bridging genuinely distinct positions (contrast the disqualification
    test above, two units apart).
    """
    assert consensus_search.hill_climb(reads) == expected


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [
                (["CAG", "CAG", "AAA", "CAG", "CAG", "CAG"], "spanning", "+"),
                (["CAG", "CAG", "CAG", "AAA", "CAG", "CAG"], "spanning", "+"),
                (["CAG", "CAG", "CAG", "CAG", "AAA", "CAG"], "spanning", "+"),
            ],
            ["CAG", "CAG", "CAG", "AAA", "CAG", "CAG"],
            id="same strand, out of phase by one each",
        ),
        pytest.param(
            [
                (["CAG", "CAG", "AAA", "CAG", "CAG", "CAG"], "spanning", "+"),
                (["CAG", "CAG", "CAG", "AAA", "CAG", "CAG"], "spanning", "-"),
                (["CAG", "CAG", "CAG", "CAG", "AAA", "CAG"], "spanning", "+"),
            ],
            ["CAG", "CAG", "CAG", "AAA", "CAG", "CAG"],
            id="cross strand, minus-strand read is the middle one",
        ),
    ],
)
def test_hill_climb_resolves_out_of_phase_interruptions_to_the_shared_middle_position(reads, expected):
    """Three reads place the same interruption one column apart (idx 2, 3, 4); none agree exactly,
    but distance minimization already resolves this to the middle position without any
    strand-eligibility logic. The cross-strand variant checks that eligibility doesn't interfere:
    the lone minus-strand read at the middle shouldn't need a second minus-strand read to count.
    """
    assert consensus_search.hill_climb(reads) == expected
