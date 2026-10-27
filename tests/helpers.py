"""Shared builders for synthetic loci and reads."""

import random

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
) -> Read:
    return Read(
        name=name,
        sequence=sequence,
        qualities=[30] * len(sequence),
        mod_5mc_probs="!" * len(sequence),
        strand=strand,
        locus=locus,
        n_soft_clipped_left=n_soft_clipped_left,
        n_soft_clipped_right=n_soft_clipped_right,
    )
