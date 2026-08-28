"""Unit tests for the consensus hill-climb internals in abacus.consensus_search.

Separate from test_consensus.py, which tests create_consensus_calls end-to-end through the real
graph-alignment pipeline.
"""

import random

import pytest

from abacus import consensus_search

ERROR_UNITS = ["AAA", "GGT", "TTA", "CCG", "GCA"]


def units(s: str) -> list[str]:
    return [s[i : i + 3] for i in range(0, len(s), 3)]


def drifted_reads(rng_seed: int, n_units: int = 60, n_reads: int = 25) -> tuple[list[tuple[list[str], str]], list[str]]:
    """Reads whose only indels sit in the first 10 positions.

    Everything downstream is then length-shifted relative to the seed, which is precisely the
    state where slicing a read at *seed* coordinates picks up the wrong content.
    """
    rng = random.Random(rng_seed)
    truth = ["CAG"] * n_units
    truth[45] = truth[46] = "GGT"
    reads = []
    for _ in range(n_reads):
        read = list(truth)
        for _ in range(rng.randrange(4)):
            pos = rng.randrange(0, 10)
            if rng.random() < 0.5:
                del read[pos]
            else:
                read.insert(pos, "CAG")
        read[rng.randrange(20, len(read))] = rng.choice(ERROR_UNITS)
        reads.append((read, "spanning"))
    return reads, truth


def sub_candidates(seed, aligned, encoded):
    vocab = set(consensus_search.build_vocab(encoded))
    _, sub_votes, _, _ = consensus_search.collect_votes(seed, aligned, vocab)
    for (pos, kmer), votes in sub_votes.items():
        if votes >= consensus_search.MIN_READ_SUPPORT and pos < len(seed) and seed[pos] != kmer:
            yield ("sub", pos, kmer)


# A window still cannot see *genuine* non-local slack: when a read carries surplus units
# elsewhere, the global aligner can absorb a substitution the window has to pay for. That is
# inherent and rare. Systematic drift is not - scoring at raw seed coordinates puts the read
# window over the wrong content and was exact on only 24% of these candidates, wrong by up to 10.
MIN_EXACT_FRACTION = 0.9


def windowed_vs_exact(rng_seed: int):
    reads, _ = drifted_reads(rng_seed)
    encoded, _ = consensus_search.encode_reads(reads)
    seed = consensus_search.starting_seed(encoded)
    aligned = consensus_search.align_reads(seed, encoded)
    baseline = consensus_search.total_dist(seed, encoded)
    for op in sub_candidates(seed, aligned, encoded):
        exact = baseline - consensus_search.total_dist(consensus_search.apply_op(seed, op), encoded)
        yield op, consensus_search.windowed_delta(seed, op, op[1], op[1], aligned), exact


def all_windowed_vs_exact():
    results = [r for rng_seed in range(1, 25) for r in windowed_vs_exact(rng_seed)]
    assert len(results) > 20, "no substitution candidates - test builds the wrong state"
    return results


def test_windowed_delta_is_exact_for_substitutions():
    """A substitution preserves length, so its effect is local and a windowed score should equal
    a full rescore. Move ranking consumes the magnitude, not just the sign, so drift here
    silently reorders which moves get applied."""
    results = all_windowed_vs_exact()
    exact = [(op, w, e) for op, w, e in results if w == e]
    assert len(exact) / len(results) >= MIN_EXACT_FRACTION, (
        f"only {len(exact)}/{len(results)} scored exactly; worst: {max(results, key=lambda r: abs(r[1] - r[2]))}"
    )


def test_windowed_delta_never_flips_the_sign_for_substitutions():
    """Whatever slack remains must not turn a losing move into a winning one, or vice versa."""
    for op, windowed, exact in all_windowed_vs_exact():
        assert (windowed > 0) == (exact > 0), f"{op}: windowed={windowed} exact={exact}"


def test_slippage_growth_skips_lone_interruptions():
    """Slippage lengthens a stretch of identical units; it does not duplicate a one-off
    interruption into a false repeat. Shrinking has no such restriction."""
    seed = "AABA"  # runs: AA, B, A
    assert set(consensus_search.slippage_candidates(seed, grow=True)) == {"AAABA"}
    assert set(consensus_search.slippage_candidates(seed, grow=False)) == {"ABA", "AAA", "AAB"}


def test_repair_length_extends_the_repeat_not_the_interruption():
    """A seed one unit short must regain a repeat unit, never a second copy of the interruption."""
    reads = [(units("CAG" * 4 + "AAA" + "CAG" * 5), "spanning")] * 5
    encoded, char_to_kmer = consensus_search.encode_reads(reads)
    target = consensus_search.median_length(encoded, spanning_only=True)
    short_seed = encoded[0][0][:-1]

    repaired = consensus_search.repair_length(short_seed, encoded, target)

    assert len(repaired) == target
    assert [char_to_kmer[c] for c in repaired].count("AAA") == 1


def test_starting_seed_prefers_spanning_reads_over_more_numerous_flanking_reads():
    """A single spanning read sees the whole locus; two flanking reads of a different allele only
    ever see their own side. The flanking reads' raw kmer count (6) outnumbers the spanning read's
    (4), but the search must still start from what the spanning read actually shows."""
    reads = [(["CAG", "CAG", "CAG", "CAG"], "spanning"), (["AAA", "AAA", "AAA"], "right"), (["AAA", "AAA", "AAA"], "right")]
    encoded, char_to_kmer = consensus_search.encode_reads(reads)

    seed = consensus_search.starting_seed(encoded)

    assert [char_to_kmer[c] for c in seed] == ["CAG"] * 4


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param([(["CAG", "CAG", "CAG", "CAG"], "spanning")] + [(["AAA", "AAA", "AAA"], "right")] * 2, ["CAG", "AAA", "AAA", "AAA"], id="right"),
        pytest.param([(["CAG", "CAG", "CAG", "CAG"], "spanning")] + [(["AAA", "AAA", "AAA"], "left")] * 2, ["AAA", "AAA", "AAA", "CAG"], id="left"),
    ],
)
def test_hill_climb_does_not_let_flanking_reads_corrupt_a_lone_spanning_read(reads, expected):
    """Regression test: a majority-vote starting seed built without spanning priority picks the
    flanking reads' motif outright, corrupting the whole consensus rather than just the tail/head
    the flanking reads actually cover."""
    assert consensus_search.hill_climb(reads) == expected


def test_trim_partial_boundary_kmers_drops_uncorroborated_boundary():
    """A left-flanking read's last kmer, or a right-flanking read's first kmer, is dropped when it
    never appears anywhere else as an interior kmer - the signature of a read simply running out
    mid-unit, not a real short motif."""
    reads = [
        (["CAG", "CAG", "CAG"], "spanning"),
        (["CAG", "CAG", "CA"], "left"),
        (["AG", "CAG", "CAG"], "right"),
    ]

    trimmed = consensus_search.trim_partial_boundary_kmers(reads)

    assert trimmed == [
        (["CAG", "CAG", "CAG"], "spanning"),
        (["CAG", "CAG"], "left"),
        (["CAG", "CAG"], "right"),
    ]


def test_trim_partial_boundary_kmers_keeps_corroborated_boundary():
    """A boundary kmer that also shows up as an interior kmer elsewhere is real vocabulary, not a
    truncation artifact, and must not be dropped."""
    reads = [
        (["CAG", "AAA", "CAG"], "spanning"),
        (["CAG", "AAA"], "left"),  # boundary kmer "AAA" is corroborated by the spanning read
    ]

    assert consensus_search.trim_partial_boundary_kmers(reads) == reads


def test_trim_partial_boundary_kmers_does_not_let_two_truncated_reads_corroborate_each_other():
    """Two reads independently truncated at the same point still share a boundary kmer, but
    neither occurrence is interior - read support alone must not be enough to keep it."""
    reads = [(["AG", "CAG", "CAG"], "right")] * 2

    trimmed = consensus_search.trim_partial_boundary_kmers(reads)

    assert trimmed == [(["CAG", "CAG"], "right")] * 2


def test_trim_partial_boundary_kmers_drops_a_flanking_read_left_with_nothing():
    """A flanking read that is only ever its own boundary kmer trims down to empty, not a crash."""
    reads = [(["CAG", "CAG"], "spanning"), (["AG"], "right")]

    trimmed = consensus_search.trim_partial_boundary_kmers(reads)

    assert trimmed == [(["CAG", "CAG"], "spanning")]


@pytest.mark.parametrize(
    ("reads", "expected"),
    [
        pytest.param(
            [(["CAG", "CAG", "CAG", "CAG"], "spanning")] + [(["CAG", "CAG", "CAG", "CA"], "left")] * 2,
            ["CAG", "CAG", "CAG", "CAG"],
            id="left-partial",
        ),
        pytest.param(
            [(["CAG", "CAG", "CAG", "CAG"], "spanning")] + [(["AG", "CAG", "CAG", "CAG"], "right")] * 2,
            ["CAG", "CAG", "CAG", "CAG"],
            id="right-partial",
        ),
    ],
)
def test_hill_climb_does_not_let_a_partial_boundary_kmer_outvote_a_full_unit(reads, expected):
    """Regression test for the two mid-unit-boundary cases in test_consensus.py."""
    assert consensus_search.hill_climb(reads) == expected
