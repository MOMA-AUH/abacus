import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from pyfaidx import Fasta

from abacus.constants import ANCHOR_LEN


def process_str_pattern(str_pattern):
    satellite_PATTERN = r"(?<=\()(?:[GCATN]+)(?=\)[\*\+])"
    BREAK_PATTERN = r"(?<=\)[\*\+])(?:[GCATN]*)(?=\()"

    satellites = re.findall(satellite_PATTERN, str_pattern)
    breaks = re.findall(BREAK_PATTERN, str_pattern)
    prefix = str_pattern[: str_pattern.find("(")]
    suffix = str_pattern[str_pattern.rfind(")") + 2 :]

    return satellites, [prefix] + breaks + [suffix]


def process_region(reference_region) -> Tuple[str, int, int]:
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
    satellites: list[str]
    breaks: list[str]

    def to_dict(self):
        return {
            "locus_id": self.id,
            "locus_chrom": self.chrom,
            "locus_start": self.start,
            "locus_end": self.end,
            "satellites_str": "-".join(self.satellites),
            "structure": self.structure,
        }


def load_loci_from_json(json_path: Path, ref: Fasta) -> List[Locus]:

    with open(json_path, "r", encoding="utf-8") as f:
        loci_json = json.load(f)

    loci = []
    for item in loci_json:
        chrom, start, end = process_region(item["ReferenceRegion"])
        satellites, breaks = process_str_pattern(item["LocusStructure"])

        # Get region around STR
        left_anchor = str(ref[chrom][(start - ANCHOR_LEN) : start])
        right_anchor = str(ref[chrom][end : (end + ANCHOR_LEN)])

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
            )
        )
    return loci
