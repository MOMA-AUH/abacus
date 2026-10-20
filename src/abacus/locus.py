import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pyfaidx import Fasta

from abacus.config import config


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
    sequences: list[str]
    location: Location
    skippable: bool
    benign_max: int | None = None
    pathogenic_min: int | None = None

    def status(self, count: float) -> str | None:
        """Classify a repeat unit count against the catalog thresholds; None when the satellite has none."""
        if self.benign_max is None and self.pathogenic_min is None:
            return None
        if self.pathogenic_min is not None and count >= self.pathogenic_min:
            return "pathogenic"
        if self.benign_max is not None and count > self.benign_max:
            return "intermediate"
        return "normal"


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

    def to_dict(self) -> dict[str, str | int]:
        return {
            "locus_id": self.id,
            "locus_chrom": self.location.chrom,
            "locus_start": self.location.start,
            "locus_end": self.location.end,
            "satellites_str": "-".join(["|".join(sat.sequences) for sat in self.satellites]),
            "structure": self.structure,
        }


def load_loci_from_json(json_path: Path, ref_path: Path) -> list[Locus]:
    # Load reference
    ref = Fasta(ref_path)

    # Load loci from json
    with json_path.open("r") as f:
        loci_json = json.load(f)

    loci = []
    for item in loci_json:
        check_required_fields(item)

        # Required fields
        locus_id = item["LocusId"]
        locus_structure = item["LocusStructure"]
        reference_region = item["ReferenceRegion"]

        # Optional fields (None when missing)
        variant_ids = item.get("VariantId")
        benign_max = item.get("BenignMax")
        pathogenic_min = item.get("PathogenicMin")

        # Split the structure into satellites and the breaks between them
        satellite_seqs, satellites_skippable, breaks = parse_str_pattern(locus_structure)

        # Locus location spans all satellite locations
        locus_location, satellite_locations = parse_region(as_list(reference_region))

        # Satellite ids from VariantId, or derived from LocusId
        satellite_ids = parse_satellite_ids(variant_ids, locus_id, len(satellite_seqs))

        # Create satellites
        satellites = create_satellites(
            satellite_seqs,
            satellites_skippable,
            satellite_locations,
            satellite_ids,
            locus_id,
            benign_max,
            pathogenic_min,
        )

        # Get region around STR
        left_anchor, right_anchor = get_anchors(ref, locus_location)

        loci.append(
            Locus(
                id=locus_id,
                location=locus_location,
                structure=locus_structure,
                left_anchor=left_anchor,
                right_anchor=right_anchor,
                satellites=satellites,
                breaks=breaks,
            ),
        )
    return loci


def as_list(value: Any) -> list[Any]:
    """Return a catalog field that can be a single value or a list as a list."""
    return value if isinstance(value, list) else [value]


def check_required_fields(item: dict) -> None:
    required_fields = ["LocusId", "ReferenceRegion", "LocusStructure"]
    missing_fields = [field for field in required_fields if field not in item]
    if missing_fields:
        error_message = f"Incomplete locus data: {missing_fields}"
        raise ValueError(error_message)


def parse_satellite_ids(variant_ids: str | list[str] | None, locus_id: str, n_satellites: int) -> list[str]:
    if variant_ids is not None:
        return as_list(variant_ids)
    if n_satellites == 1:
        return [locus_id]
    return [f"{locus_id}.{i + 1}" for i in range(n_satellites)]


def parse_thresholds(value: int | list[int | None] | None, field: str, n_satellites: int, locus_id: str) -> list[int | None]:
    if value is None:
        return [None] * n_satellites

    # A scalar is only unambiguous for a single-satellite locus, so it fails the length check otherwise
    values = as_list(value)
    if len(values) != n_satellites:
        error_message = f"Locus {locus_id}: {field} must be a list with one value per satellite ({n_satellites})"
        raise ValueError(error_message)

    # bool is a subclass of int, so it is excluded explicitly
    if any(v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 0) for v in values):
        error_message = f"Locus {locus_id}: {field} values must be non-negative integers or null: {values}"
        raise ValueError(error_message)

    return values


def get_anchors(ref: Fasta, location: Location) -> tuple[str, str]:
    """Get the reference sequence flanking the locus on each side."""
    chrom = ref[location.chrom]
    left_anchor = str(chrom[(location.start - config.anchor_len) : location.start])
    right_anchor = str(chrom[location.end : (location.end + config.anchor_len)])
    return left_anchor, right_anchor


def create_satellites(
    satellites_seq: list[list[str]],
    satellites_skippable: list[bool],
    locations: list[Location],
    satelitte_ids: list[str],
    locus_id: str,
    benign_max: int | list[int | None] | None = None,
    pathogenic_min: int | list[int | None] | None = None,
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

    # Optional thresholds: a scalar for single-satellite loci, otherwise one value (or null) per satellite
    benign_maxs = parse_thresholds(benign_max, "BenignMax", len(satellites_seq), locus_id)
    pathogenic_mins = parse_thresholds(pathogenic_min, "PathogenicMin", len(satellites_seq), locus_id)
    for benign, pathogenic in zip(benign_maxs, pathogenic_mins, strict=True):
        if benign is not None and pathogenic is not None and benign >= pathogenic:
            error_message = f"Locus {locus_id}: BenignMax must be smaller than PathogenicMin ({benign} >= {pathogenic})"
            raise ValueError(error_message)

    return [
        Satellite(
            id=idx,
            sequences=sat,
            skippable=skip,
            location=location,
            benign_max=benign,
            pathogenic_min=pathogenic,
        )
        for sat, skip, location, idx, benign, pathogenic in zip(
            satellites_seq,
            satellites_skippable,
            locations,
            satelitte_ids,
            benign_maxs,
            pathogenic_mins,
            strict=True,
        )
    ]


def parse_str_pattern(str_pattern: str) -> tuple[list[list[str]], list[bool], list[str]]:
    # Extract satellites and satellite operators
    # Allow | inside () for alternation syntax, e.g. (CAG|CAA)+
    satellite_pattern = r"(?<=\()(?:[GCATNRYSWKMBDHV|]+)(?=\)[\*\+])"
    satellite_operator_pattern = r"(?<=\))([\*\+])"

    # Split each match on | to get a list of alternatives per satellite
    satellites = [seq.split("|") for seq in re.findall(satellite_pattern, str_pattern)]
    if not satellites:
        error_message = f"No valid satellite pattern found in LocusStructure: {str_pattern}"
        raise ValueError(error_message)
    satellite_operators = re.findall(satellite_operator_pattern, str_pattern)
    satellites_skippable = [op == "*" for op in satellite_operators]

    # Extract breaks
    break_pattern = r"(?<=\)[\*\+])(?:[GCATNRYSWKMBDHV]*)(?=\()"
    internal_breaks = re.findall(break_pattern, str_pattern)
    pre_break = str_pattern[: str_pattern.find("(")]
    post_break = str_pattern[str_pattern.rfind(")") + 2 :]

    return satellites, satellites_skippable, [pre_break, *internal_breaks, post_break]


def parse_region(reference_region: list[str]) -> tuple[Location, list[Location]]:
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
