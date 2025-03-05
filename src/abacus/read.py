from dataclasses import dataclass

import pysam

from abacus.locus import Locus
from abacus.logging import logger


@dataclass
class Read:
    name: str
    sequence: str
    qualities: list[int]
    mod_5mc_probs: str

    strand: str

    n_soft_clipped_left: int
    n_soft_clipped_right: int

    locus: Locus

    def quality_string(self) -> str:
        return "".join([chr(q + 33) for q in self.qualities])

    def to_fastq(self) -> str:
        return f"@{self.name}\n{self.sequence}\n+\n{self.quality_string()}\n"

    @classmethod
    def from_aligment(cls, alignment: pysam.AlignedSegment, locus: Locus) -> "Read":
        # Get read information
        name = alignment.query_name or ""
        sequence = alignment.query_sequence or ""
        qualities = [int(q) for q in alignment.query_qualities or []]
        strand = "-" if alignment.is_reverse else "+"

        # Get soft-clipped bases
        soft_clip_op_index = 4
        n_soft_clipped_left = alignment.cigartuples[0][1] if alignment.cigartuples and alignment.cigartuples[0][0] == soft_clip_op_index else 0
        n_soft_clipped_right = alignment.cigartuples[-1][1] if alignment.cigartuples and alignment.cigartuples[-1][0] == soft_clip_op_index else 0

        # Get 5mC modification probabilities
        mod_5mc_probs = get_5mc_modification_probs(alignment)

        return cls(
            name=name,
            sequence=sequence,
            qualities=qualities,
            mod_5mc_probs=mod_5mc_probs,
            strand=strand,
            locus=locus,
            n_soft_clipped_left=n_soft_clipped_left,
            n_soft_clipped_right=n_soft_clipped_right,
        )

    def is_reverse(self) -> bool:
        return self.strand == "-"


@dataclass
class FilteredRead(Read):
    error_flags: str

    def to_dict(self) -> dict:
        return {
            "query_name": self.name,
            "strand": self.strand,
            "error_flags": self.error_flags,
        } | self.locus.to_dict()

    @classmethod
    def from_read(cls, read: Read, error_flags: str) -> "FilteredRead":
        return cls(
            name=read.name,
            sequence=read.sequence,
            qualities=read.qualities,
            mod_5mc_probs=read.mod_5mc_probs,
            strand=read.strand,
            locus=read.locus,
            n_soft_clipped_left=read.n_soft_clipped_left,
            n_soft_clipped_right=read.n_soft_clipped_right,
            error_flags=error_flags,
        )


def get_5mc_modification_probs(alignment: pysam.AlignedSegment) -> str:
    sequence = alignment.query_sequence if alignment.query_sequence else ""
    mod_5mc_probs = "0" * len(sequence)

    # Check if alignment has modified bases
    if alignment.modified_bases is None:
        return mod_5mc_probs

    # Check if alignment has 5mC modifications
    mod_5mc_lists = [mod_list for mod_key, mod_list in alignment.modified_bases.items() if mod_key[0] == "C" and mod_key[2] == "m"]
    if not mod_5mc_lists:
        return mod_5mc_probs

    if len(mod_5mc_lists) > 1:
        messege = f"Multiple 5mC modifications found in alignment {alignment.query_name}"
        logger.warning(messege)
        raise ValueError(messege)

    # Get 5mC modifications
    mod_5mc = mod_5mc_lists[0]

    for pos, scaled_prob in mod_5mc:
        # Scale the probability to 0-1 (0-255 -> 0-1)
        prob = scaled_prob / 255
        # Convert methylation to string using 10 bins (printable ASCII characters, 33 and up)
        prob_char = chr(int(prob * 10) + 33)
        # Update the modification probabilities
        mod_5mc_probs = mod_5mc_probs[:pos] + prob_char + mod_5mc_probs[pos + 1 :]

    return mod_5mc_probs
