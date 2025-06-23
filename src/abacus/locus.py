import json
import re
from dataclasses import dataclass
from pathlib import Path

from pyfaidx import Fasta

from abacus.config import config
from abacus.logging import logger


@dataclass
class Location:
    """Class for keeping data for a position"""

    chrom: str
    start: int
    end: int


@dataclass
class Satellite:
    """Class for keeping data for a satellite"""

    id: str
    sequence: str
    location: Location
    skippable: bool


@dataclass
class Locus:
    """Class for keeping data for a STR locus"""

    id: str
    location: Location
    left_anchor: str
    right_anchor: str
    structure: str
    satellites: list[Satellite]
    breaks: list[str]

    def to_dict(self):
        return {
            "locus_id": self.id,
            "locus_chrom": self.location.chrom,
            "locus_start": self.location.start,
            "locus_end": self.location.end,
            "satellites_str": "-".join([sat.sequence for sat in self.satellites]),
            "structure": self.structure,
        }


def load_loci_from_json(json_path: Path, ref_path: Path) -> list[Locus]:
    # Load reference
    ref = Fasta(ref_path)

    # Load loci from json
    with json_path.open("r") as f:
        loci_json = json.load(f)

    required_fields = ["LocusId", "ReferenceRegion", "LocusStructure"]

    loci = []
    for item in loci_json:
        # Check if all required fields are present
        missing_fields = [field for field in required_fields if field not in item]
        if missing_fields:
            error_message = f"Incomplete locus data: {missing_fields}"
            raise ValueError(error_message)

        # Extract LocusId
        locus_id = item["LocusId"]

        # Process STR pattern
        satellite_seqs, satellites_skippable, breaks = process_str_pattern(item["LocusStructure"])

        if not satellite_seqs:
            error_message = f"No valid satellite pattern found in LocusStructure: {item['LocusStructure']}, LocusId: {locus_id}"
            logger.error(error_message)
            continue

        # Process reference region and ensure it's a list
        reference_region = item["ReferenceRegion"] if isinstance(item["ReferenceRegion"], list) else [item["ReferenceRegion"]]
        locus_location, satellite_locations = process_region(reference_region)

        # Get satellite ids
        if "VariantId" in item:
            satellite_ids = item["VariantId"]
        elif len(satellite_seqs) == 1:
            satellite_ids = [f"{locus_id}"]
        else:
            satellite_ids = [f"{locus_id}.{i + 1}" for i in range(len(satellite_seqs))]
        
        # Create satellites
        satellites = create_satellites(satellite_seqs, satellites_skippable, satellite_locations, satellite_ids, locus_id)

        # Get region around STR
        left_anchor = str(ref[locus_location.chrom][(locus_location.start - config.anchor_len) : locus_location.start])
        right_anchor = str(ref[locus_location.chrom][locus_location.end : (locus_location.end + config.anchor_len)])

        loci.append(
            Locus(
                id=locus_id,
                location=locus_location,
                structure=item["LocusStructure"],
                left_anchor=left_anchor,
                right_anchor=right_anchor,
                satellites=satellites,
                breaks=breaks,
            ),
        )
    return loci


def create_satellites(
    satellites_seq: list[str],
    satellites_skippable: list[bool],
    locations: list[Location],
    satelitte_ids: list[str],
    locus_id: str,
) -> list[Satellite]:
    # Check if correct number of satellite ids
    if len(satellites_seq) != len(satelitte_ids):
        error_message = f"Locus {locus_id}: Mismatch between number of satellites ({len(satellites_seq)}) and satellite IDs ({len(satelitte_ids)})"
        raise ValueError(error_message)

    # Check if correct number of satellites and locations
    if len(satellites_seq) != len(locations) and len(locations) != 1:
        error_message = f"Locus {locus_id}: Number of locations ({len(locations)}) must match the number of satellites ({len(satellites_seq)}) or be exactly 1"
        raise ValueError(error_message)

    # If only one location is provided, assign it to all satellites
    if len(locations) == 1:
        locations = locations * len(satellites_seq)

    return [
        Satellite(
            id=idx,
            sequence=sat,
            skippable=skip,
            location=location,
        )
        for sat, skip, location, idx in zip(satellites_seq, satellites_skippable, locations, satelitte_ids)
    ]


def process_str_pattern(str_pattern: str) -> tuple[list[str], list[bool], list[str]]:
    # Extract satellites and satellite operators
    satellite_pattern = r"(?<=\()(?:[GCATNRYSWKMBDHV]+)(?=\)[\*\+])"
    satellite_operator_pattern = r"(?<=\))([\*\+])"

    satellites = re.findall(satellite_pattern, str_pattern)
    satellite_operators = re.findall(satellite_operator_pattern, str_pattern)
    satellites_skippable = [op == "*" for op in satellite_operators]

    # Extract breaks
    break_pattern = r"(?<=\)[\*\+])(?:[GCATN]*)(?=\()"
    internal_breaks = re.findall(break_pattern, str_pattern)
    pre_break = str_pattern[: str_pattern.find("(")]
    post_break = str_pattern[str_pattern.rfind(")") + 2 :]

    return satellites, satellites_skippable, [pre_break, *internal_breaks, post_break]


def process_region(reference_region: list[str]) -> tuple[Location, list[Location]]:
    chroms, starts, ends = [], [], []

    # Extract chrom, start and end from reference region
    for region in reference_region:
        matches = re.match(r"(.+)\:(\d+)-(\d+)", region)

        if not matches:
            error_message = f"Invalid reference region format: {region}"
            raise ValueError(error_message)

        chroms.append(matches[1])
        starts.append(int(matches[2]))
        ends.append(int(matches[3]))

    # Check if all regions are on the same chromosome
    if len(set(chroms)) != 1:
        error_message = "All reference regions must be on the same chromosome"
        raise ValueError(error_message)

    # Create locus and satellite locations
    chrom = chroms[0]
    locus_location = Location(chrom=chrom, start=min(starts), end=max(ends))
    satellite_locations = [Location(chrom=chrom, start=start, end=end) for start, end in zip(starts, ends)]

    return locus_location, satellite_locations
