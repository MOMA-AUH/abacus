from __future__ import annotations

import pytest

from abacus.filtering import filter_low_qual_read_calls, mark_qc_outliers
from abacus.graph import get_read_calls
from tests.helpers import make_locus, make_read


def test_zero_copy_read_passes_str_quality_filters():
    """A read that skips the repeat has no STR bases, so there is no STR quality to filter on."""
    locus = make_locus("(CAG)+")
    read_calls, _ = get_read_calls([make_read("read", locus.left_anchor + locus.right_anchor, locus)], locus)

    good_read_calls, removed_read_calls = filter_low_qual_read_calls(read_calls)

    assert len(good_read_calls) == 1
    assert removed_read_calls == []


@pytest.mark.parametrize(
    ("divergences", "expected_outliers"),
    [
        pytest.param([0.0] * 7 + [0.5], [7], id="one bad read in a clean allele"),
        pytest.param([0.76, 0.78, 0.79, 0.8, 0.8, 0.81, 0.82, 0.84], [], id="a whole allele that differs from the catalog"),
        pytest.param([0.0] * 7 + [0.15], [], id="an outlier below the tolerance"),
    ],
)
def test_ref_divergence_outliers_are_relative_to_the_haplotype(divergences: list[float], expected_outliers: list[int]):
    locus = make_locus("(CAG)+")
    reads = [make_read(f"read_{i}", locus.left_anchor + "CAG" * 10 + locus.right_anchor, locus) for i in range(len(divergences))]
    read_calls, _ = get_read_calls(reads, locus)
    for read_call, divergence in zip(read_calls, divergences, strict=True):
        read_call.alignment.str_ref_divergence = divergence

    mark_qc_outliers(read_calls)

    assert [i for i, rc in enumerate(read_calls) if "outlier_ref_divergence" in rc.outlier_reasons] == expected_outliers
