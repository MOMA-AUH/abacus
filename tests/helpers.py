"""Shared builders for synthetic loci and reads."""

import random
from collections.abc import Iterable
from typing import Literal

from abacus.config import config
from abacus.locus import Location, Locus, Satellite, parse_str_pattern
from abacus.read import Read


def make_locus(structure: str) -> Locus:
    rng = random.Random(42)
    left_anchor = "".join(rng.choices("ATCG", k=config.anchor_len))
    right_anchor = "".join(rng.choices("ATCG", k=config.anchor_len))
    sequences, skippable, breaks = parse_str_pattern(structure)
    satellites = []
    start = 1000
    for i, (alternatives, skip) in enumerate(zip(sequences, skippable)):
        end = start + len(alternatives[0])
        satellites.append(Satellite(
            id=f"test_{i}",
            sequences=alternatives,
            location=Location("chr1", start, end),
            skippable=skip,
        ))
        start = end
    return Locus(
        id="test",
        structure=structure,
        location=Location("chr1", 1000, start),
        satellites=satellites,
        breaks=breaks,
        left_anchor=left_anchor,
        right_anchor=right_anchor,
    )


def make_read(
    name: str, sequence: str, locus: Locus, strand: str = "+", *,
    n_soft_clipped_left: int = 0, n_soft_clipped_right: int = 0,
    qualities: list[int] | None = None,
) -> Read:
    return Read(
        name=name,
        sequence=sequence,
        qualities=qualities if qualities is not None else [30] * len(sequence),
        mod_5mc_probs="!" * len(sequence),
        strand=strand,
        locus=locus,
        n_soft_clipped_left=n_soft_clipped_left,
        n_soft_clipped_right=n_soft_clipped_right,
    )


def make_reads(
    sequences: Iterable[str], locus: Locus, read_type: Literal["spanning", "left", "right"] = "spanning",
    strand: str = "+", *, name_prefix: str | None = None,
) -> list[Read]:
    """Add the appropriate anchors to repeat sequences and build named reads."""
    anchors = {
        "spanning": (locus.left_anchor, locus.right_anchor),
        "left": (locus.left_anchor, ""),
        "right": ("", locus.right_anchor),
    }
    left, right = anchors[read_type]
    prefix = name_prefix if name_prefix is not None else read_type
    return [make_read(f"{prefix}_{i}", left + seq + right, locus, strand) for i, seq in enumerate(sequences)]
