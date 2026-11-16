import pytest

from abacus.config import config
from abacus.equal_length_split import run_equal_length_backup_test
from abacus.graph import get_read_calls
from abacus.locus import Locus
from abacus.utils import Haplotype
from tests.helpers import make_locus, make_reads


def _spanning_read_calls(kmer_strings: list[str], locus: Locus):
    reads = make_reads(kmer_strings, locus)
    read_calls, _ = get_read_calls(reads, locus)
    for rc in read_calls:
        rc.set_haplotype(Haplotype.HOM)
    return read_calls


@pytest.mark.parametrize(
    ("sequences", "expected_significant"),
    [
        pytest.param(
            ["CAG" * 10] * 12,
            True,
            id="backup_test-identical_sequences_no_split",
        ),
        pytest.param(
            ["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"] * 5,
            False,
            id="backup_test-different_sequences_split",
        ),
    ],
)
def test_run_equal_length_backup_test_significance(sequences: list[str], expected_significant: bool):
    """Backup test should be significant (no split) when there are no sequence differences."""
    locus = make_locus("(CAG)+")
    read_calls = _spanning_read_calls(sequences, locus)

    _, _, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert bool(summary["backup_test_run"].iloc[0]) is True
    assert bool(summary["backup_test_significant"].iloc[0]) == expected_significant


def test_run_equal_length_backup_test_skips_below_min_haplotyping_depth():
    """Backup test should not run when spanning reads < min_haplotyping_depth (default 10)."""
    locus = make_locus("(CAG)+")
    # 8 reads: 7 normal + 1 with a sequencing error — same length, different sequence
    sequences = ["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"]
    read_calls = _spanning_read_calls(sequences, locus)

    _, _, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert not bool(summary["backup_test_run"].iloc[0])


@pytest.mark.parametrize(
    ("n_h1", "n_h2", "min_support", "split_expected"),
    [(1, 1, 2, False), (2, 2, 3, False), (3, 2, 3, False), (2, 2, 2, True)],
)
def test_run_equal_length_backup_test_only_discards_reads_for_supported_split(monkeypatch, n_h1, n_h2, min_support, split_expected):
    """An unsupported split leaves every original read in the HOM group."""
    monkeypatch.setattr(config, "min_reads_per_haplotype", min_support)
    locus = make_locus("(CAG)+")
    # The split column contains CTG or CCG; the other 10 reads have a gap there.
    sequences = ["CAG" * 10] * 10 + ["CAG" * 5 + "CTG" + "CAG" * 5] * n_h1 + ["CAG" * 5 + "CCG" + "CAG" * 5] * n_h2
    read_calls = _spanning_read_calls(sequences, locus)

    grouped, new_outliers, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert summary["backup_test_n_h1"].iloc[0] == n_h1
    assert summary["backup_test_n_h2"].iloc[0] == n_h2
    if split_expected:
        assert len(new_outliers) == 10
        assert all(r.outlier_reasons == ["not_split_base"] for r in new_outliers)
        assert len(grouped) == n_h1 + n_h2
        assert {r.haplotype for r in grouped} == {Haplotype.H1, Haplotype.H2}
    else:
        assert new_outliers == []
        assert grouped == read_calls
        assert all(r.haplotype == Haplotype.HOM for r in grouped)
    assert all(not r.outlier_reasons for r in grouped)
