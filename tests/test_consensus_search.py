"""Unit tests for the consensus hill-climb internals in abacus.consensus_search.

Separate from test_consensus.py, which tests create_consensus_calls end-to-end through the real
graph-alignment pipeline.
"""

import pytest

from abacus import consensus_search

# Each character labels one distinct repeat unit; its DNA sequence is irrelevant here.


ReadSpec = tuple[str, str, str, int]


def repeat_reads_by_coverage(reads: list[ReadSpec]) -> list[consensus_search.EncodedRead]:
    """Repeat each (tokens, type, strand) read by its specified coverage."""
    repeated: list[consensus_search.EncodedRead] = []
    for tokens, read_type, strand, coverage in reads:
        repeated.extend([(tokens, read_type, strand)] * coverage)
    return repeated


# Boundary trimming and flanking reads


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [
                ("ABA", "spanning", "+"),
                ("AB", "left", "+"),
            ],
            [
                ("ABA", "spanning", "+"),
                ("AB", "left", "+"),
            ],
            id="keep-interior-corroborated-boundary",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("BAA", "right", "+", 2),
                ],
            ),
            repeat_reads_by_coverage(
                [
                    ("AA", "right", "+", 2),
                ],
            ),
            id="truncated-boundaries-do-not-corroborate-each-other",
        ),
        pytest.param(
            [
                ("AA", "spanning", "+"),
                ("B", "right", "+"),
            ],
            [("AA", "spanning", "+")],
            id="drop-read-trimmed-to-empty",
        ),
        pytest.param(
            [
                ("A", "spanning", "+"),
                ("A", "left", "+"),
                ("B", "right", "-"),
            ],
            [
                ("A", "spanning", "+"),
                ("A", "left", "+"),
            ],
            id="single-spanning-token-corroborates-complete-break-only",
        ),
        pytest.param(
            [("A", "left", "+"), ("B", "right", "-")],
            [("A", "left", "+"), ("B", "right", "-")],
            id="no-interior-evidence-keeps-flanking-tokens",
        ),
        pytest.param(
            [("", "left", "+")],
            [],
            id="no-interior-evidence-still-drops-empty-reads",
        ),
    ],
)
def test_trim_partial_boundary_kmers(
    reads: list[consensus_search.EncodedRead],
    expected: list[consensus_search.EncodedRead],
) -> None:
    """Use interior evidence to trim boundaries when available."""
    kmer_reads = [(list(tokens), read_type, strand) for tokens, read_type, strand in reads]
    expected_reads = [(list(tokens), read_type, strand) for tokens, read_type, strand in expected]
    assert consensus_search._trim_partial_boundary_kmers(kmer_reads) == expected_reads


# Final consensus: input reads, target length, and expected output


@pytest.mark.parametrize(
    ("reads", "length", "expected"),
    [
        # Empty output: no usable reads or a zero target length.
        pytest.param(
            [],
            5,
            "",
            id="no-reads-positive-target",
        ),
        pytest.param(
            [
                ("", "spanning", "+"),
                ("", "left", "+"),
                ("", "right", "-"),
            ],
            5,
            "",
            id="all-empty-reads-positive-target",
        ),
        pytest.param(
            [
                ("AB", "spanning", "+"),
            ],
            0,
            "",
            id="nonempty-read-zero-target",
        ),
        pytest.param(
            [],
            0,
            "",
            id="no-reads-zero-target",
        ),
        # Target length: fit pure repeats and preserve distinct end tokens when shortening.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAAA", "spanning", "+", 1),
                    ("AAAAAA", "spanning", "+", 1),
                    ("AAAAAAA", "spanning", "+", 2),
                    ("AAAAAAAA", "spanning", "+", 1),
                ],
            ),
            7,
            "AAAAAAA",
            id="pure-repeat-target-7",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAAA", "spanning", "+", 1),
                    ("AAAAAA", "spanning", "+", 1),
                    ("AAAAAAA", "spanning", "+", 2),
                    ("AAAAAAAA", "spanning", "+", 1),
                ],
            ),
            4,
            "AAAA",
            id="pure-repeat-target-4",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("BAAACC", "spanning", "+", 1),
                ],
            ),
            5,
            "BAACC",
            id="longer-read-preserves-endpoints",
        ),
        # Boundary trimming: partial flanking tokens must not outvote complete units.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAA", "spanning", "+", 1),
                    ("AAAB", "left", "+", 2),
                ],
            ),
            4,
            "AAAA",
            id="partial-boundary-left-partial",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAA", "spanning", "+", 1),
                    ("BAAA", "right", "+", 2),
                ],
            ),
            4,
            "AAAA",
            id="partial-boundary-right-partial",
        ),
        # Flanking coverage: preserve uncovered ends and combine reads anchored at opposite ends.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAA", "spanning", "+", 1),
                    ("BBB", "right", "+", 2),
                ],
            ),
            4,
            "ABBB",
            id="flanking-coverage-right",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAA", "spanning", "+", 1),
                    ("BBB", "left", "+", 2),
                ],
            ),
            4,
            "BBBA",
            id="flanking-coverage-left",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("BAAA", "left", "+", 1),
                    ("AAAC", "right", "+", 1),
                ],
            ),
            4,
            "BAAC",
            id="flanking-only-anchored-ends-right-strand-+",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("BAAA", "left", "+", 1),
                    ("AAAC", "right", "-", 1),
                ],
            ),
            4,
            "BAAC",
            id="flanking-only-anchored-ends-right-strand--",
        ),
        # Strand bias: require support across strands at the disputed position.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("A", "spanning", "+", 1),
                    ("B", "spanning", "+", 10),
                    ("A", "spanning", "-", 3),
                ],
            ),
            1,
            "A",
            id="strand-bias-spanning only",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("A", "spanning", "+", 1),
                    ("B", "spanning", "+", 10),
                    ("A", "spanning", "-", 1),
                    ("A", "left", "-", 1),
                    ("A", "right", "-", 1),
                ],
            ),
            1,
            "A",
            id="strand-bias-minus-strand support split across spanning and flanking reads",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("ABABA", "spanning", "+", 10),
                    ("AAABA", "spanning", "+", 1),
                    ("AAABA", "spanning", "-", 3),
                ],
            ),
            5,
            "AAABA",
            id="strand-bias-is-position-specific",
        ),
        # Error suppression: reject isolated tokens and tolerate an opposite-strand outlier.
        pytest.param(
            [
                ("AAAAAAAAAA", "spanning", "+"),
                ("AAAAABAAAA", "spanning", "+"),
            ],
            10,
            "AAAAAAAAAA",
            id="isolated-error-depth 2, one strand",
        ),
        pytest.param(
            [
                ("AAAAAAAAAA", "spanning", "+"),
                ("AAAAABAAAA", "spanning", "-"),
            ],
            10,
            "AAAAAAAAAA",
            id="isolated-error-depth 2, both strands",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAABAAAA", "spanning", "+", 2),
                    ("AAAABAAAA", "spanning", "+", 1),
                    ("CCCAAAAA", "spanning", "-", 1),
                ],
            ),
            8,
            "AAABAAAA",
            id="outlier-does-not-block-shared-interruption",
        ),
        # Shifted interruptions: preserve shared tokens and resolve their positions.
        pytest.param(
            [
                ("AABAAA", "spanning", "+"),
                ("AAABAA", "spanning", "-"),
            ],
            6,
            # The all-A seed has equally scoring B substitutions at indices 2 and 3.
            # Weak moves break score ties by leftmost position, so B belongs at index 2.
            "AABAAA",
            id="one-column-shift-same length, interruption one column apart",
        ),
        pytest.param(
            [
                ("AABAAA", "spanning", "+"),
                ("AAABAAA", "spanning", "-"),
            ],
            7,
            "AAABAAA",
            id="one-column-shift-differing total length, interruption one column apart",
        ),
        pytest.param(
            [
                ("AABAAA", "spanning", "+"),
                ("AAABAA", "spanning", "+"),
                ("AAAABA", "spanning", "+"),
            ],
            6,
            "AAABAA",
            id="shared-middle-same strand, out of phase by one each",
        ),
        pytest.param(
            [
                ("AABAAA", "spanning", "+"),
                ("AAABAA", "spanning", "-"),
                ("AAAABA", "spanning", "+"),
            ],
            6,
            "AAABAA",
            id="shared-middle-cross strand, minus-strand read is the middle one",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("BAA", "spanning", "+", 1),
                    ("AAB", "spanning", "+", 1),
                ],
            ),
            3,
            "ABA",
            id="two-column-separation-depth-1",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("BAA", "spanning", "+", 2),
                    ("AAB", "spanning", "+", 2),
                ],
            ),
            3,
            "ABA",
            id="two-column-separation-depth-2",
        ),
        # Unit indels: preserve shared interruptions and distinct runs across unequal read lengths.
        pytest.param(
            [
                ("AAAABAAAAAAA", "spanning", "+"),
                ("AAAAAABAAAAAAA", "spanning", "+"),
                ("AABAAAAAAA", "spanning", "+"),
            ],
            12,
            "AAAABAAAAAAA",
            id="shared-interruptions-two-unit-indels-length-12-same-strand",
        ),
        pytest.param(
            [
                ("AAAABAAAAAAA", "spanning", "+"),
                ("AAAAAABAAAAAAA", "spanning", "+"),
                ("AABAAAAAAA", "spanning", "-"),
            ],
            12,
            "AAAABAAAAAAA",
            id="shared-interruptions-two-unit-indels-length-12-both-strands",
        ),
        pytest.param(
            [
                ("AAAA" + "BAAAAAA" * 25 + "A", "spanning", "+"),
                ("AAAAAA" + "BAAAAAA" * 25 + "A", "spanning", "+"),
                ("AA" + "BAAAAAA" * 25 + "A", "spanning", "+"),
            ],
            180,
            "AAAA" + "BAAAAAA" * 25 + "A",
            id="shared-interruptions-two-unit-indels-length-180-same-strand",
        ),
        pytest.param(
            [
                ("AAAA" + "BAAAAAA" * 25 + "A", "spanning", "+"),
                ("AAAAAA" + "BAAAAAA" * 25 + "A", "spanning", "+"),
                ("AA" + "BAAAAAA" * 25 + "A", "spanning", "-"),
            ],
            180,
            "AAAA" + "BAAAAAA" * 25 + "A",
            id="shared-interruptions-two-unit-indels-length-180-both-strands",
        ),
        pytest.param(
            [
                ("AABABAAA", "spanning", "+"),
                ("AAAABABAAA", "spanning", "+"),
                ("BABAAA", "spanning", "-"),
            ],
            8,
            "AABABAAA",
            id="nearby-interruptions-remain-distinct",
        ),
        # Missing C/B runs must not make genuine BAB/ACA look like errors.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("ADABCABC", "spanning", "+", 1),
                    ("AABABC", "spanning", "-", 1),
                    ("AABCAC", "spanning", "+", 1),
                ],
            ),
            7,
            "AABCABC",
            id="missing-runs-must-not-be-merged-as-interruptions",
        ),
        # End interruptions: preserve the first or last token across unequal read lengths.
        pytest.param(
            [
                ("BAAAA", "spanning", "+"),
                ("BAAAAAA", "spanning", "+"),
                ("BAAA", "spanning", "-"),
            ],
            5,
            "BAAAA",
            id="shared-interruption-at-first-unit",
        ),
        pytest.param(
            [
                ("AAAAB", "spanning", "+"),
                ("AAAAAAB", "spanning", "+"),
                ("AAAB", "spanning", "-"),
            ],
            5,
            "AAAAB",
            id="shared-interruption-at-last-unit",
        ),
        # Majority interruptions: recover a shared token absent from the longer opposite-strand read.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAABAAAAA", "spanning", "+", 2),
                    ("AAAAAAAAAAAAA", "spanning", "-", 1),
                ],
            ),
            10,
            "AAAABAAAAA",
            id="majority-interruption-target-10",
        ),
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("AAAABAAAAA", "spanning", "+", 2),
                    ("AAAAAAAAAAAAA", "spanning", "-", 1),
                ],
            ),
            11,
            # The all-A seed has equally scoring B substitutions at indices 4 and 5.
            # Weak moves break score ties by leftmost position, so B belongs at index 4.
            "AAAABAAAAAA",
            id="majority-interruption-target-11",
        ),
        # CA -> AC reduces distance to ABC by two; pruning must allow that gain.
        # Shared padding keeps the ABC read below the outlier threshold.
        pytest.param(
            repeat_reads_by_coverage(
                [
                    ("CA" + "D" * 20, "spanning", "+", 2),
                    ("ABC" + "D" * 20, "spanning", "+", 1),
                ],
            ),
            22,
            "AC" + "D" * 20,
            id="swap-pruning-allows-two-unit-distance-improvement",
        ),
    ],
)
def test_find_consensus(
    reads: list[consensus_search.EncodedRead],
    length: int,
    expected: str,
) -> None:
    kmer_reads = [(list(tokens), read_type, strand) for tokens, read_type, strand in reads]
    result = consensus_search.find_consensus(kmer_reads, length)
    assert result == list(expected)


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        ([], {}),
        ([("BAAA", "left", "+"), ("AAAC", "right", "-")], {"A": 3, "B": 1, "C": 1}),
        # A noisy expansion and singleton token must not set the side's median.
        ([("AB", "left", "+"), ("AB", "left", "-"), ("AAAAABX", "left", "+")], {"A": 1, "B": 1}),
        ([("AAAC", "right", "-")], {"A": 3, "C": 1}),
        # Spanning evidence takes precedence even over numerous longer flanks.
        ([("AB", "spanning", "+")] + [("AAAAAC", "left", "-")] * 3, {"A": 1, "B": 1}),
        # Preserve the existing upper-median convention for even spanning depth.
        ([("A", "spanning", "+"), ("AAB", "spanning", "-")], {"A": 2, "B": 1}),
    ],
)
def test_expected_composition(reads: list[consensus_search.EncodedRead], expected: dict[str, int]) -> None:
    assert consensus_search._expected_composition(reads) == expected


# Weak proposal eligibility


@pytest.mark.parametrize(
    ("consensus", "pos", "read_tokens", "expected"),
    [
        pytest.param(
            "a",
            0,
            [
                ("a", "+"),
                ("b", "-"),
            ],
            True,
            id="equal-reads-and-strand-count",
        ),
        pytest.param(
            "a",
            0,
            [("a", "+")] * 3 + [("b", "-")],
            False,
            id="fewer-reads-equal-strand-count",
        ),
        pytest.param(
            "a",
            0,
            [
                ("a", "+"),
                ("b", "-"),
                ("b", "-"),
            ],
            True,
            id="more-reads-equal-strand-count",
        ),
        pytest.param(
            "a",
            0,
            [
                ("a", "+"),
                ("b", "+"),
                ("b", "-"),
            ],
            True,
            id="one-strand-to-both-strands",
        ),
        pytest.param(
            "aaa",
            1,
            [
                ("aba", "+"),
                ("aba", "+"),
                ("aaa", "-"),
            ],
            True,
            id="neighbours-do-not-upgrade-current-strand-support",
        ),
        pytest.param(
            "aaa",
            1,
            [
                ("aba", "+"),
                ("aaa", "+"),
                ("baa", "-"),
            ],
            True,
            id="proposal-can-gain-opposite-strand-from-neighbour",
        ),
        pytest.param(
            "aaa",
            1,
            [
                ("aba", "+"),
                ("aba", "+"),
                ("aaa", "+"),
                ("aaa", "-"),
            ],
            False,
            id="protect-current-with-exact-both-strand-support",
        ),
        pytest.param(
            "a",
            0,
            [("a", "+")] * 3 + [("b", "+")],
            True,
            id="strand-supported-single-read-proposal",
        ),
        pytest.param(
            "a",
            0,
            [
                ("a", "+"),
                ("a", "-"),
            ]
            + [("b", "+")] * 3,
            False,
            id="more-reads-cannot-lose-strand-support",
        ),
        pytest.param(
            "a",
            0,
            [("a", "+")] + [("b", "-")] * 2 + [("c", "-")] * 97,
            True,
            id="relative-support-bypasses-percentage-cutoff",
        ),
        pytest.param(
            "a",
            0,
            [("b", "+")] + [("c", "-")] * 99,
            True,
            id="unsupported-current-needs-no-two-read-floor",
        ),
        pytest.param(
            "a",
            0,
            [("a", "+")] * 99 + [("bba", "+")],
            True,
            id="insertion-only-proposal-bypasses-observed-read-cutoff",
        ),
        # Three nearby occurrences in one read cannot outweigh two supporting reads.
        pytest.param(
            "aaa",
            1,
            [("aaa", "+"), ("aaa", "+"), ("bbb", "-")],
            False,
            id="nearby-token-occurrences-count-each-read-once",
        ),
        pytest.param(
            "aaa",
            1,
            [("aaa", "+"), ("aaa", "+"), ("bbb", "-"), ("bbb", "-")],
            True,
            id="nearby-token-occurrences-in-two-reads-meet-relative-support",
        ),
    ],
)
def test_weak_substitution_proposal_eligibility(
    consensus: str,
    pos: int,
    read_tokens: list[tuple[str, str]],
    expected: bool,
) -> None:
    """Allow strand-filtered proposals or equal-or-better relative read and strand support."""
    reads: list[consensus_search.EncodedRead] = [(tokens, "spanning", strand) for tokens, strand in read_tokens]
    votes = consensus_search._count_column_votes(consensus, consensus_search._align_reads(consensus, reads))

    candidates = consensus_search._candidate_moves(consensus, votes)

    assert (consensus_search.Move("sub", pos, "b") in candidates) == expected


@pytest.mark.parametrize("max_iters", [1, 2, 100])
def test_regression_weak_batch_must_improve_full_objective(max_iters: int) -> None:
    """Distant substitutions still interact through the composition penalty."""
    reads: list[consensus_search.EncodedRead] = [
        ("B" + "A" * 99, "spanning", "+"),
        ("A" * 99 + "B", "spanning", "+"),
    ]
    kmer_reads = [(list(tokens), kind, strand) for tokens, kind, strand in reads]
    result = "".join(consensus_search.find_consensus(kmer_reads, 100, max_iters=max_iters))
    distances = [consensus_search.OSA.distance(result, tokens) for tokens, _, _ in reads]
    score = (
        sum(distances),
        consensus_search._composition_penalty(result, consensus_search._expected_composition(reads)),
        sum(distance * distance for distance in distances),
    )

    # Initially both reads have distance 1, and composition is off by one A and one B.
    # Either substitution alone improves composition; applying both cancels that gain.
    assert score < (2, 2, 2)
    assert result == "B" + "A" * 99


def test_regression_strong_insertions_require_a_majority_for_the_same_block() -> None:
    """Different inserted blocks must not pool their votes into a false majority."""
    consensus = "ABC"
    reads = repeat_reads_by_coverage(
        [
            ("AXYBC", "spanning", "+", 2),
            ("AZWBC", "spanning", "+", 2),
            ("AUVBC", "spanning", "+", 2),
            (consensus, "spanning", "+", 4),
        ]
    )
    votes = consensus_search._count_column_votes(consensus, consensus_search._align_reads(consensus, reads))

    # Six of ten reads insert something, but each particular block has only two votes.
    # All original tokens have unanimous support, so there is no strong edit to apply.
    assert consensus_search._apply_strong_changes(consensus, votes, length=3) == consensus
