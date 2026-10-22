from __future__ import annotations

import random

from abacus.filtering import filter_low_qual_read_calls
from abacus.graph import get_read_calls
from tests.test_consensus import create_synthetic_locus, make_read


def test_zero_copy_read_passes_str_quality_filters():
    """A read that skips the repeat has no STR bases, so there is no STR quality to filter on."""
    random.seed(0)
    locus = create_synthetic_locus(["CAG"])
    read_calls, _ = get_read_calls([make_read("read", locus.left_anchor + locus.right_anchor, locus)], locus)

    good_read_calls, removed_read_calls = filter_low_qual_read_calls(read_calls)

    assert len(good_read_calls) == 1
    assert removed_read_calls == []
