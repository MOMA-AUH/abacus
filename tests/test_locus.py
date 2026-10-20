import json
from unittest.mock import MagicMock, patch

import pytest

from abacus.locus import Location, Satellite, create_satellites, load_loci_from_json, parse_region, parse_str_pattern

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fasta_mock(anchor_len=20):
    """Return a mock pyfaidx.Fasta that yields empty anchor strings."""
    seq_mock = MagicMock()
    seq_mock.__getitem__ = MagicMock(return_value="N" * anchor_len)
    seq_mock.__str__ = MagicMock(return_value="N" * anchor_len)

    chrom_mock = MagicMock()
    chrom_mock.__getitem__ = MagicMock(return_value=seq_mock)

    fasta_mock = MagicMock()
    fasta_mock.__getitem__ = MagicMock(return_value=chrom_mock)
    return fasta_mock


def _write_json(tmp_path, entries):
    p = tmp_path / "catalog.json"
    p.write_text(json.dumps(entries))
    return p


# ---------------------------------------------------------------------------
# parse_str_pattern
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "structure, expected_n_satellites, expected_skippable",
    [
        ("(GT)*", 1, [True]),
        ("(CAG)+", 1, [False]),
        ("(CAG)*(GCG)*", 2, [True, True]),
        ("(CAG)*GGGG(GCG)*", 2, [True, True]),
        ("(CAG)+(GCG)+", 2, [False, False]),
        ("(CAG|CAA)+", 1, [False]),
    ],
)
def test_parse_str_pattern_satellite_count(structure, expected_n_satellites, expected_skippable):
    satellites, skippable, _ = parse_str_pattern(structure)
    assert len(satellites) == expected_n_satellites
    assert skippable == expected_skippable


def test_parse_str_pattern_breaks_single():
    _, _, breaks = parse_str_pattern("(CAG)*GGGG(GCG)*")
    assert breaks == ["", "GGGG", ""]


def test_parse_str_pattern_no_satellite_raises():
    with pytest.raises(ValueError, match="No valid satellite pattern"):
        parse_str_pattern("GGGG")


# ---------------------------------------------------------------------------
# parse_region
# ---------------------------------------------------------------------------


def test_parse_region_single():
    locus_loc, sat_locs = parse_region(["chr10:100-200"])
    assert locus_loc == Location(chrom="chr10", start=100, end=200)
    assert sat_locs == [Location(chrom="chr10", start=100, end=200)]


def test_parse_region_multi():
    locus_loc, sat_locs = parse_region(["chr10:100-150", "chr10:160-200"])
    assert locus_loc == Location(chrom="chr10", start=100, end=200)
    assert len(sat_locs) == 2


def test_parse_region_cross_chrom_raises():
    with pytest.raises(ValueError, match="same chromosome"):
        parse_region(["chr10:100-200", "chr11:100-200"])


def test_parse_region_invalid_format_raises():
    with pytest.raises(ValueError, match="Invalid reference region"):
        parse_region(["chr10_100_200"])


# ---------------------------------------------------------------------------
# create_satellites
# ---------------------------------------------------------------------------


def test_create_satellites_id_count_mismatch_raises():
    sat_seqs = [["GT"]]
    skippable = [True]
    locations = [Location("chr10", 100, 200)]
    with pytest.raises(ValueError, match="Mismatch between number of satellites"):
        create_satellites(sat_seqs, skippable, locations, ["id1", "id2"], "locus1")


def test_create_satellites_location_count_mismatch_raises():
    sat_seqs = [["CAG"], ["GCG"]]
    skippable = [True, True]
    locations = [Location("chr10", 100, 150), Location("chr10", 160, 200), Location("chr10", 210, 250)]
    with pytest.raises(ValueError, match="Number of locations"):
        create_satellites(sat_seqs, skippable, locations, ["id1", "id2"], "locus1")


def test_create_satellites_single_location_broadcast():
    sat_seqs = [["CAG"], ["GCG"]]
    skippable = [True, True]
    locations = [Location("chr10", 100, 200)]
    satellites = create_satellites(sat_seqs, skippable, locations, ["id1", "id2"], "locus1")
    assert len(satellites) == 2
    assert all(s.location == Location("chr10", 100, 200) for s in satellites)


# ---------------------------------------------------------------------------
# load_loci_from_json — VariantId handling
# ---------------------------------------------------------------------------


@patch("abacus.locus.Fasta")
def test_load_loci_string_variant_id_single_satellite(mock_fasta_cls, tmp_path):
    """String VariantId must be treated as one ID for a single-satellite locus."""
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "chr10_10000703_10000721",
        "LocusStructure": "(GT)*",
        "ReferenceRegion": "chr10:10000703-10000721",
        "VariantId": "chr10_10000703_10000721",
        "VariantType": "Repeat",
    }
    json_path = _write_json(tmp_path, [entry])
    loci = load_loci_from_json(json_path, tmp_path / "ref.fa")
    assert len(loci) == 1
    assert len(loci[0].satellites) == 1
    assert loci[0].satellites[0].id == "chr10_10000703_10000721"


@patch("abacus.locus.Fasta")
def test_load_loci_list_variant_id_single_satellite(mock_fasta_cls, tmp_path):
    """List VariantId with one element works for single-satellite locus."""
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "locus1",
        "LocusStructure": "(CAG)*",
        "ReferenceRegion": "chr1:100-130",
        "VariantId": ["my_id"],
    }
    loci = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert loci[0].satellites[0].id == "my_id"


@patch("abacus.locus.Fasta")
def test_load_loci_list_variant_id_multi_satellite(mock_fasta_cls, tmp_path):
    """List VariantId with two elements works for two-satellite locus."""
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "locus_multi",
        "LocusStructure": "(CAG)*GGGG(GCG)*",
        "ReferenceRegion": ["chr1:100-130", "chr1:134-160"],
        "VariantId": ["id_cag", "id_gcg"],
    }
    loci = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert len(loci[0].satellites) == 2
    assert [s.id for s in loci[0].satellites] == ["id_cag", "id_gcg"]


@patch("abacus.locus.Fasta")
def test_load_loci_no_variant_id_single_satellite(mock_fasta_cls, tmp_path):
    """Without VariantId, single-satellite locus uses LocusId as satellite ID."""
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "locus_solo",
        "LocusStructure": "(GT)*",
        "ReferenceRegion": "chr1:100-120",
    }
    loci = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert loci[0].satellites[0].id == "locus_solo"


@patch("abacus.locus.Fasta")
def test_load_loci_no_variant_id_multi_satellite(mock_fasta_cls, tmp_path):
    """Without VariantId, multi-satellite locus generates locus_id.1, locus_id.2, ..."""
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "locus_multi",
        "LocusStructure": "(CAG)*(GCG)*",
        "ReferenceRegion": ["chr1:100-130", "chr1:130-160"],
    }
    loci = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert [s.id for s in loci[0].satellites] == ["locus_multi.1", "locus_multi.2"]


@patch("abacus.locus.Fasta")
def test_load_loci_variant_id_count_mismatch_raises(mock_fasta_cls, tmp_path):
    """VariantId list length mismatching satellite count must raise ValueError."""
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "locus_multi",
        "LocusStructure": "(CAG)*(GCG)*",
        "ReferenceRegion": ["chr1:100-130", "chr1:130-160"],
        "VariantId": ["only_one_id"],
    }
    with pytest.raises(ValueError, match="Mismatch between number of satellites"):
        load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")


# ---------------------------------------------------------------------------
# load_loci_from_json — validation
# ---------------------------------------------------------------------------


@patch("abacus.locus.Fasta")
def test_load_loci_missing_required_field_raises(mock_fasta_cls, tmp_path):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {"LocusId": "x", "LocusStructure": "(GT)*"}  # missing ReferenceRegion
    with pytest.raises(ValueError, match="Incomplete locus data"):
        load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")


@patch("abacus.locus.Fasta")
def test_load_loci_multiple_entries(mock_fasta_cls, tmp_path):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entries = [
        {"LocusId": "loc1", "LocusStructure": "(GT)*", "ReferenceRegion": "chr1:100-120"},
        {"LocusId": "loc2", "LocusStructure": "(CAG)+", "ReferenceRegion": "chr2:200-230"},
    ]
    loci = load_loci_from_json(_write_json(tmp_path, entries), tmp_path / "ref.fa")
    assert len(loci) == 2
    assert loci[0].id == "loc1"
    assert loci[1].id == "loc2"


# ---------------------------------------------------------------------------
# load_loci_from_json — BenignMax / PathogenicMin thresholds
# ---------------------------------------------------------------------------


@patch("abacus.locus.Fasta")
def test_load_loci_without_thresholds(mock_fasta_cls, tmp_path):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {"LocusId": "loc1", "LocusStructure": "(CAG)*", "ReferenceRegion": "chr1:100-130"}
    (locus,) = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert (locus.satellites[0].benign_max, locus.satellites[0].pathogenic_min) == (None, None)


@patch("abacus.locus.Fasta")
def test_load_loci_scalar_thresholds_single_satellite(mock_fasta_cls, tmp_path):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {"LocusId": "loc1", "LocusStructure": "(CAG)*", "ReferenceRegion": "chr1:100-130", "BenignMax": 26, "PathogenicMin": 36}
    (locus,) = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert (locus.satellites[0].benign_max, locus.satellites[0].pathogenic_min) == (26, 36)


@patch("abacus.locus.Fasta")
def test_load_loci_list_thresholds_multi_satellite(mock_fasta_cls, tmp_path):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {
        "LocusId": "HTT",
        "LocusStructure": "(CAG)*CAACAG(CCG)*",
        "ReferenceRegion": ["chr4:3074876-3074933", "chr4:3074939-3074966"],
        "BenignMax": [26, None],
        "PathogenicMin": [36, None],
    }
    (locus,) = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert [(s.benign_max, s.pathogenic_min) for s in locus.satellites] == [(26, 36), (None, None)]


@patch("abacus.locus.Fasta")
def test_load_loci_only_one_threshold_key(mock_fasta_cls, tmp_path):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {"LocusId": "loc1", "LocusStructure": "(CAG)*", "ReferenceRegion": "chr1:100-130", "PathogenicMin": 36}
    (locus,) = load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")
    assert (locus.satellites[0].benign_max, locus.satellites[0].pathogenic_min) == (None, 36)


@pytest.mark.parametrize(
    "thresholds, match",
    [
        ({"BenignMax": 26}, "must be a list with one value per satellite"),
        ({"BenignMax": [26]}, "must be a list with one value per satellite"),
        ({"PathogenicMin": [36, None, None]}, "must be a list with one value per satellite"),
        ({"BenignMax": [-1, None]}, "must be non-negative integers or null"),
        ({"BenignMax": [26.5, None]}, "must be non-negative integers or null"),
        ({"BenignMax": ["26", None]}, "must be non-negative integers or null"),
        ({"BenignMax": [36, None], "PathogenicMin": [36, None]}, "BenignMax must be smaller than PathogenicMin"),
    ],
)
@patch("abacus.locus.Fasta")
def test_load_loci_invalid_thresholds_raise(mock_fasta_cls, tmp_path, thresholds, match):
    mock_fasta_cls.return_value = _make_fasta_mock()
    entry = {"LocusId": "loc1", "LocusStructure": "(CAG)*(CCG)*", "ReferenceRegion": ["chr1:100-130", "chr1:130-160"], **thresholds}
    with pytest.raises(ValueError, match=match):
        load_loci_from_json(_write_json(tmp_path, [entry]), tmp_path / "ref.fa")


# ---------------------------------------------------------------------------
# Satellite.status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "benign_max, pathogenic_min, count, expected",
    [
        (None, None, 100, None),
        (26, 36, 26, "normal"),
        (26, 36, 26.5, "intermediate"),
        (26, 36, 35.9, "intermediate"),
        (26, 36, 36, "pathogenic"),
        (None, 36, 35.9, "normal"),
        (None, 36, 36, "pathogenic"),
        (26, None, 26, "normal"),
        (26, None, 1000, "intermediate"),
    ],
)
def test_satellite_status(benign_max, pathogenic_min, count, expected):
    satellite = Satellite(id="s", sequences=["CAG"], location=Location("chr1", 0, 3), skippable=True, benign_max=benign_max, pathogenic_min=pathogenic_min)
    assert satellite.status(count) == expected
