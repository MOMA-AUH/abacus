import random

import pytest

from abacus import config
from abacus.equal_length_split import run_equal_length_backup_test
from abacus.graph import Read, get_read_calls
from abacus.locus import Location, Locus, Satellite
from abacus.utils import Haplotype


def _make_locus(satellite_seq: str) -> Locus:
    random.seed(0)
    left_anchor = "".join(random.choices("ATCG", k=config.Config.anchor_len))
    right_anchor = "".join(random.choices("ATCG", k=config.Config.anchor_len))
    sat = Satellite(
        id="test_0",
        sequences=[satellite_seq],
        location=Location("chr1", 1000, 1000 + len(satellite_seq)),
        skippable=False,
    )
    return Locus(
        id="test",
        structure="test",
        location=Location("chr1", 1000, 1000 + len(satellite_seq)),
        satellites=[sat],
        breaks=["", ""],
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )


def _spanning_read_calls(kmer_strings: list[str], locus: Locus):
    reads = [
        Read(
            name=f"read_{i}",
            sequence=locus.left_anchor + seq + locus.right_anchor,
            qualities=[30] * (len(locus.left_anchor) + len(seq) + len(locus.right_anchor)),
            mod_5mc_probs="!" * (len(locus.left_anchor) + len(seq) + len(locus.right_anchor)),
            strand="+",
            n_soft_clipped_left=0,
            n_soft_clipped_right=0,
            locus=locus,
        )
        for i, seq in enumerate(kmer_strings)
    ]
    read_calls, _ = get_read_calls(reads, locus)
    for rc in read_calls:
        rc.set_haplotype(Haplotype.HOM)
    return read_calls


@pytest.mark.parametrize(
    ("sequences", "expected_significant", "description"),
    [
        pytest.param(
            ["CAG" * 10] * 12,
            True,
            "identical sequences — no differences, no split possible",
            id="backup_test-identical_sequences_no_split",
        ),
        pytest.param(
            ["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"] * 5,
            False,
            "different sequences — split is consistent with 50/50",
            id="backup_test-different_sequences_split",
        ),
    ],
)
def test_run_equal_length_backup_test_significance(sequences: list[str], expected_significant: bool, description: str):
    """Backup test should be significant (no split) when there are no sequence differences."""
    locus = _make_locus("CAG")
    read_calls = _spanning_read_calls(sequences, locus)

    _, _, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert bool(summary["backup_test_run"].iloc[0]) is True, f"Expected test to run ({description})"
    assert bool(summary["backup_test_significant"].iloc[0]) == expected_significant, (
        f"Expected backup_test_significant={expected_significant} ({description}), "
        f"got {summary['backup_test_significant'].iloc[0]}"
    )


def test_run_equal_length_backup_test_skips_below_min_haplotyping_depth():
    """Backup test should not run when spanning reads < min_haplotyping_depth (default 10)."""
    locus = _make_locus("CAG")
    # 8 reads: 7 normal + 1 with a sequencing error — same length, different sequence
    sequences = ["CAG" * 10] * 7 + ["CAG" * 9 + "CGG"]
    read_calls = _spanning_read_calls(sequences, locus)

    _, _, summary = run_equal_length_backup_test(read_calls, alpha=0.05)

    assert not bool(summary["backup_test_run"].iloc[0]), (
        "Backup test should not run with 8 reads when min_haplotyping_depth=10"
    )
