import random

import numpy as np
import pytest

from abacus import config
from abacus.graph import Read, get_read_calls
from abacus.haplotyping import run_equal_length_backup_test
from abacus.locus import Location, Locus, Satellite
from abacus.parameter_estimation import (
    discrete_multivariate_normal_logpdf,
    estimate_mean_flanking,
    flanking_logpdf,
)
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


@pytest.mark.parametrize(
    ("x", "mean", "var", "expected"),
    [
        pytest.param(
            np.array([]),
            np.array([0]),
            np.array([1]),
            np.array([]),
            id="No counts",
        ),
        pytest.param(
            np.array([[1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([0]),
            id="One count",
        ),
        pytest.param(
            np.array([[1], [1], [1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([0, 0, 0]),
            id="Multiple counts",
        ),
        pytest.param(
            np.array([[1, 2, 3]]),
            np.array([1, 2, 3]),
            np.array([0.001, 0.001, 0.001]),
            np.array([0]),
            id="One count, multiple dimensions",
        ),
        pytest.param(
            np.array([[1, 2, 3], [1, 2, 3]]),
            np.array([1, 2, 3]),
            np.array([0.001, 0.001, 0.001]),
            np.array([0, 0]),
            id="Multiple counts, multiple dimensions",
        ),
    ],
)
def test_discrete_multivariate_normal_logpdf(x: np.ndarray, mean: np.ndarray, var: np.ndarray, expected: np.ndarray):
    logpdf = discrete_multivariate_normal_logpdf(x, mean, var)

    assert logpdf.shape[0] == x.shape[0]
    assert np.all(logpdf <= 0)
    assert np.all(logpdf >= -np.inf)
    assert np.allclose(logpdf, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("x", "mean", "unit_var", "is_left_flanking", "expected"),
    [
        pytest.param(
            np.array([]),
            np.array([0]),
            np.array([1]),
            np.array([]),
            np.array([]),
            id="No counts",
        ),
        pytest.param(
            np.array([[1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([True]),
            np.array([0]),
            id="One count",
        ),
        pytest.param(
            np.array([[1, 1, 1]]),
            np.array([1, 1, 1]),
            np.array([0.01, 0.01, 0.01]),
            np.array([True]),
            np.array([0]),
            id="One count, multiple dimensions",
        ),
        pytest.param(
            np.array([[1], [1], [1]]),
            np.array([1]),
            np.array([0.01]),
            np.array([True, True, True]),
            np.array([0, 0, 0]),
            id="Multiple counts",
        ),
        pytest.param(
            np.array([[1, 1, 1], [1, 1, 1]]),
            np.array([1, 1, 1]),
            np.array([0.01, 0.01, 0.01]),
            np.array([True, True]),
            np.array([0, 0]),
            id="Multiple counts, multiple dimensions",
        ),
    ],
)
def test_flanking_logpdf(x: np.ndarray, mean: np.ndarray, unit_var: np.ndarray, is_left_flanking: list[bool], expected: np.ndarray):
    logpdf = flanking_logpdf(x, mean, unit_var, is_left_flanking)

    assert logpdf.shape[0] == x.shape[0]
    assert np.all(logpdf <= 0)
    assert np.all(logpdf >= -np.inf)
    assert np.allclose(logpdf, expected, atol=1e-5)


@pytest.mark.parametrize(
    ("x", "gamma", "is_left_flank", "expected"),
    [
        pytest.param(
            np.array([]),
            np.array([]),
            [],
            np.array([]),
            id="Empty",
        ),
        pytest.param(
            np.array([[0, 0, 0]]),
            np.array([1]),
            [True],
            np.array([0, 0, 0]),
            id="All zeros, left flank",
        ),
        pytest.param(
            np.array([[0, 1, 1]]),
            np.array([1]),
            [True],
            np.array([0, 1, 1]),
            id="One sample, left flank",
        ),
        pytest.param(
            np.array([[1, 1, 0]]),
            np.array([1]),
            [False],
            np.array([1, 1, 0]),
            id="One sample, right flank",
        ),
        pytest.param(
            np.array([[1, 1, 0], [1, 1, 0]]),
            np.array([1, 1]),
            [False, False],
            np.array([1, 1, 0]),
            id="Two sample, right flank",
        ),
        pytest.param(
            np.array([[1, 2], [1, 2], [2, 2]]),
            np.array([0, 0.5, 0.5]),
            [True, True, True],
            np.array([1.25, 1.5]),
            id="Multiple samples, left flank",
        ),
    ],
)
def test_estimate_mean_flanking(x: np.ndarray, gamma: np.ndarray, is_left_flank: list[bool], expected: np.ndarray):
    mean_flanking = estimate_mean_flanking(x, gamma, is_left_flank)

    assert np.allclose(mean_flanking, expected, atol=1e-5)
