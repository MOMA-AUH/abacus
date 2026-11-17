import pytest

from abacus.config import config
from abacus.equal_length_split import run_equal_length_backup_test
from abacus.graph import get_read_calls
from abacus.locus import Locus
from abacus.utils import Haplotype
from tests.helpers import make_locus, make_reads


def _read_calls(kmer_strings: list[str], locus: Locus, flanking_sequences: dict[str, list[str]] | None = None):
    reads = make_reads(kmer_strings, locus)
    for read_type, sequences in (flanking_sequences or {}).items():
        reads += make_reads(sequences, locus, read_type=read_type)
    read_calls, unmapped = get_read_calls(reads, locus)
    assert not unmapped
    for rc in read_calls:
        rc.set_haplotype(Haplotype.HOM)
    return read_calls


@pytest.mark.parametrize(
    ("sequences", "flanking_sequences", "expected_run", "expected_significant"),
    [
        pytest.param(
            ["CAG" * 10] * 12,
            {},
            True,
            True,
            id="identical_sequences_no_split",
        ),
        pytest.param(
            ["CAG" * 10] * 6 + ["CAG" * 14] * 6,
            {},
            True,
            True,
            id="terminal_repeat_gaps_do_not_split",
        ),
        pytest.param(
            ["CCG" * 4 + "CAG" * 10] * 6 + ["CAG" * 10] * 6,
            {},
            True,
            True,
            id="leading_spanning_deletions_do_not_split",
        ),
        pytest.param(
            ["CAG" * 10 + "CCG" * 4] * 6 + ["CAG" * 10] * 6,
            {},
            True,
            True,
            id="trailing_spanning_deletions_do_not_split",
        ),
        pytest.param(
            ["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"] * 5,
            {},
            True,
            False,
            id="different_sequences_split",
        ),
        pytest.param(
            ["CAG" * 5 + "CCG" * 4 + "CAG" * 5] * 5 + ["CAG" * 10] * 5,
            {
                "left": ["CAG" * 7, "CAG" * 5 + "CCG" * 4 + "CAG" * 2],
                "right": ["CAG" * 7, "CAG" * 2 + "CCG" * 4 + "CAG" * 5],
            },
            True,
            False,
            id="internal_deletions_split",
        ),
        pytest.param(
            ["CAG" * 10] * 10,
            {"left": ["CAG" * 5] * 5},
            True,
            True,
            id="uncovered_right_ends_do_not_split",
        ),
        pytest.param(
            ["CAG" * 10] * 10,
            {"right": ["CAG" * 5] * 5},
            True,
            True,
            id="uncovered_left_ends_do_not_split",
        ),
        pytest.param(
            ["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"],
            {},
            False,
            False,
            id="below_min_haplotyping_depth",
        ),
    ],
)
def test_run_equal_length_backup_test(sequences, flanking_sequences, expected_run, expected_significant):
    """Run only at sufficient depth, and test whether the sequences support a split."""
    locus = make_locus("(CAG)+")
    read_calls = _read_calls(sequences, locus, flanking_sequences)

    grouped, excluded, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert bool(summary["backup_test_run"].iloc[0]) == expected_run
    assert bool(summary["backup_test_significant"].iloc[0]) == expected_significant
    if not expected_run or expected_significant:
        assert excluded == []
        assert grouped == read_calls
        assert all(rc.haplotype == Haplotype.HOM for rc in grouped)
    else:
        assert excluded == []
        assert {rc.haplotype for rc in grouped} == {Haplotype.H1, Haplotype.H2}


@pytest.mark.parametrize(("n_short", "split_expected"), [(3, True), (4, False)])
def test_sequence_split_requires_80_percent_usable_covering_reads(n_short, split_expected):
    locus = make_locus("(CAG)+")
    sequences = ["CAG" * 9 + "CGG"] * 6 + ["CAG" * 9 + "CTG"] * 6 + ["CAG" * 8] * n_short
    # Short spanning reads count in coverage; these flanking reads do not reach the marker.
    read_calls = _read_calls(sequences, locus, {"left": ["CAG" * 4] * 5})

    grouped, excluded, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    if split_expected:
        assert {rc.haplotype for rc in grouped} == {Haplotype.H1, Haplotype.H2}
        assert len(grouped) == 12
        assert len(excluded) == n_short + 5
        assert summary["backup_test_n_h1"].iloc[0] == 6
        assert summary["backup_test_n_h2"].iloc[0] == 6
    else:
        assert grouped == read_calls
        assert excluded == []
        assert all(rc.haplotype == Haplotype.HOM for rc in grouped)


@pytest.mark.parametrize(
    ("n_h1", "n_h2", "min_support", "split_expected"),
    [(3, 1, 2, False), (2, 2, 3, False), (3, 2, 3, False), (2, 2, 2, True)],
)
def test_run_equal_length_backup_test_only_discards_reads_for_supported_split(monkeypatch, n_h1, n_h2, min_support, split_expected):
    """An unsupported split leaves every original read in the HOM group."""
    monkeypatch.setattr(config, "min_reads_per_haplotype", min_support)
    monkeypatch.setattr(config, "min_haplotyping_depth", 2)
    locus = make_locus("(CAG)+")
    # The 10 flanking reads do not reach either marker and cannot support the split.
    sequences = ["CAG" * 5 + "CTG" + "CAG" * 5] * n_h1 + ["CAG" * 5 + "CCG" + "CAG" * 5] * n_h2
    read_calls = _read_calls(sequences, locus, {"left": ["CAG" * 3] * 10})

    grouped, new_outliers, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert summary["backup_test_n_h1"].iloc[0] == n_h1
    assert summary["backup_test_n_h2"].iloc[0] == n_h2
    if split_expected:
        assert len(new_outliers) == 10
        assert all(r.outlier_reasons == ["not_split_kmer"] for r in new_outliers)
        assert len(grouped) == n_h1 + n_h2
        assert {r.haplotype for r in grouped} == {Haplotype.H1, Haplotype.H2}
    else:
        assert new_outliers == []
        assert grouped == read_calls
        assert all(r.haplotype == Haplotype.HOM for r in grouped)
    assert all(not r.outlier_reasons for r in grouped)
