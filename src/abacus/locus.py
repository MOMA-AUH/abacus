import json
import re
from dataclasses import dataclass
from pathlib import Path

from pyfaidx import Fasta

from abacus.config import config


@dataclass
class Satellite:
    """Class for keeping data for a satellite"""

    sequence: str
    skippable: bool


@dataclass
class Locus:
    """Class for keeping data for a STR locus"""

    id: str
    chrom: str
    start: int
    end: int
    left_anchor: str
    right_anchor: str
    structure: str
    satellites: list[Satellite]
    breaks: list[str]

    def to_dict(self):
        return {
            "locus_id": self.id,
            "locus_chrom": self.chrom,
            "locus_start": self.start,
            "locus_end": self.end,
            "satellites_str": "-".join([sat.sequence for sat in self.satellites]),
            "structure": self.structure,
        }


def load_loci_from_json(json_path: Path, ref: Fasta) -> list[Locus]:
    with open(json_path, encoding="utf-8") as f:
        loci_json = json.load(f)

    loci = []
    for item in loci_json:
        chrom, start, end = process_region(item["ReferenceRegion"])
        satellites, breaks = process_str_pattern(item["LocusStructure"])

        # Get region around STR
        left_anchor = str(ref[chrom][(start - config.anchor_len) : start])
        right_anchor = str(ref[chrom][end : (end + config.anchor_len)])

        loci.append(
            Locus(
                id=item["LocusId"],
                chrom=chrom,
                start=start,
                end=end,
                structure=item["LocusStructure"],
                left_anchor=left_anchor,
                right_anchor=right_anchor,
                satellites=satellites,
                breaks=breaks,
            ),
        )
    return loci


def process_str_pattern(str_pattern: str) -> tuple[list[Satellite], list[str]]:
    # Extract satellites and satellite operators
    satellite_pattern = r"(?<=\()(?:[GCATNRYSWKMBDHV]+)(?=\)[\*\+])"
    satellite_operator_pattern = r"(?<=\))([\*\+])"

    satellites = re.findall(satellite_pattern, str_pattern)
    satellite_operators = re.findall(satellite_operator_pattern, str_pattern)
    satellites_skippable = [op == "*" for op in satellite_operators]

    satellites = [Satellite(sequence=sat, skippable=skip) for sat, skip in zip(satellites, satellites_skippable)]

    # Extract breaks
    break_pattern = r"(?<=\)[\*\+])(?:[GCATN]*)(?=\()"
    internal_breaks = re.findall(break_pattern, str_pattern)
    pre_break = str_pattern[: str_pattern.find("(")]
    post_break = str_pattern[str_pattern.rfind(")") + 2 :]

    return satellites, [pre_break] + internal_breaks + [post_break]


def process_region(reference_region) -> tuple[str, int, int]:
    if not isinstance(reference_region, list):
        reference_region = [reference_region]

    chroms, starts, ends = [], [], []

    # Extract chrom, start and end from reference region
    for region in reference_region:
        matches = re.match(r"(.+)\:(\d+)-(\d+)", region)

        if not matches:
            raise ValueError(f"Invalid reference region format: {region}")

        chroms.append(matches[1])
        starts.append(int(matches[2]))
        ends.append(int(matches[3]))

    chrom = chroms[0]
    start = min(starts)
    end = max(ends)

    return chrom, start, end
