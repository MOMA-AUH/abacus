from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pyfaidx import Fasta

from abacus.config import config
from abacus.consensus import ConsensusCall, get_consensus_read_call
from abacus.graph import AlignmentType
from abacus.locus import Location
from abacus.parameter_estimation import HomozygousParameters
from abacus.str_vcf import create_vcf_records, generate_vcf_header
from abacus.utils import Haplotype
from tests.test_consensus import create_synthetic_locus


@pytest.fixture
def ref_path(tmp_path: Path) -> Path:
    path = tmp_path / "ref.fa"
    path.write_text(">chr1\n" + "A" * 999 + "T" + "C" * 1000 + "\n")
    Fasta(path)  # writes the .fai index
    return path


def make_consensus_call(locus, sequence: str, haplotype: Haplotype, spanning: int = 10, flanking: int = 0) -> ConsensusCall:
    read_call = get_consensus_read_call(locus, sequence, AlignmentType.SPANNING, haplotype)
    read_call.set_haplotype(haplotype)
    return ConsensusCall.from_read_call(read_call, spanning_reads=spanning, flanking_reads=flanking)


def params(*n_units: float) -> HomozygousParameters:
    mean = np.array(n_units, dtype=float)
    return HomozygousParameters(mean=mean, unit_var=np.full(len(mean), 0.1), mean_ci_low=mean - 1, mean_ci_high=mean + 2)


def info_values(record: str) -> dict[str, str]:
    return dict(item.split("=", 1) for item in record.split("\t")[7].split(";"))


def format_values(record: str) -> dict[str, str]:
    fields = record.split("\t")
    return dict(zip(fields[8].split(":"), fields[9].split(":")))


def test_header_follows_vcf_4_5(ref_path: Path):
    header = generate_vcf_header(ref_path, "sample")

    assert header.startswith("##fileformat=VCFv4.5\n")
    assert f"##reference=file://{ref_path}\n" in header
    assert '##ALT=<ID=CNV:TR,Description="Tandem repeat">' in header
    for key in ["SVLEN", "RN", "RUS", "RUC", "LPSRUS", "LPSRUC", "TRUC", "CITRUC", "LOCUSID", "REPID", "CONS"]:
        assert f"##INFO=<ID={key}," in header
    assert '##FORMAT=<ID=AD,Number=R,Type=Integer,' in header
    assert '##FORMAT=<ID=ADSP,Number=R,Type=Integer,' in header
    assert '##FORMAT=<ID=ADFL,Number=R,Type=Integer,' in header
    assert "##FORMAT=<ID=DP,Number=1,Type=Integer," in header
    for removed in ["ID=END,", "ID=RU,", "ID=REF,", "ID=LowQual,", "ID=LC,", "ID=REPCN,", "ID=REPCI,", "ID=CONTRCONS,", "##ALT=<ID=STR"]:
        assert removed not in header
    assert header.endswith("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample")


def test_het_record_uses_cnv_tr_with_padding_base(ref_path: Path):
    locus = create_synthetic_locus(["CAG", "CCG"], breaks=["", "CAACAG", ""])
    final_params = {Haplotype.H1: params(5, 3), Haplotype.H2: params(8, 4)}
    calls = [
        make_consensus_call(locus, "CAG" * 5 + "CAACAG" + "CCG" * 3, Haplotype.H1, spanning=11, flanking=2),
        make_consensus_call(locus, "CAG" * 8 + "CAACAG" + "CCG" * 4, Haplotype.H2, spanning=5, flanking=3),
    ]

    cag, ccg = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    chrom, pos, id_, ref, alt, qual, filt, _, _, _ = cag.split("\t")
    # Satellite starts at 0-based 1000, so the padding base is 1-based 1000 (0-based 999, a "T")
    assert (chrom, pos, id_, ref, alt, qual, filt) == ("chr1", "1000", "Abacus.test.1", "T", "<CNV:TR>,<CNV:TR>", ".", "PASS")
    info = info_values(cag)
    assert info["SVLEN"] == "3,3"
    assert info["RN"] == "1,1"
    assert info["RUS"] == "CAG,CAG"
    assert info["RUC"] == "5,8"
    assert info["TRUC"] == "5.0,8.0"
    assert info["CITRUC"] == "-1.0,2.0,-1.0,2.0"
    assert info["LOCUSID"] == "test"
    assert info["REPID"] == "test_0"
    assert not {"END", "RU", "CONS"} & info.keys()

    fmt = format_values(cag)
    assert fmt == {"GT": "1|2", "PS": "1000", "DP": "21", "AD": "0,13,8", "ADSP": "0,11,5", "ADFL": "0,2,3"}

    assert ccg.split("\t")[2] == "Abacus.test.2"
    ccg_info = info_values(ccg)
    assert ccg_info["RUS"] == "CCG,CCG"
    assert ccg_info["RUC"] == "3,4"
    assert format_values(ccg)["PS"] == "1000"


def test_interrupted_allele_is_split_into_runs(ref_path: Path):
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(18), Haplotype.H2: params(12)}
    calls = [
        make_consensus_call(locus, "CAG" * 9 + "CAA" + "CAG" * 8, Haplotype.H1),
        make_consensus_call(locus, "CAG" * 12, Haplotype.H2),
    ]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    info = info_values(record)
    assert info["RN"] == "3,1"
    assert info["RUS"] == "CAG,CAA,CAG,CAG"
    assert info["RUC"] == "9,1,8,12"
    assert info["LPSRUS"] == "CAG,CAG"
    assert info["LPSRUC"] == "9,12"


def test_multi_motif_satellite_reports_motif_runs(ref_path: Path):
    locus = create_synthetic_locus(["AAAAG|AAGGG"])
    final_params = {Haplotype.HOM: params(7)}
    calls = [make_consensus_call(locus, "AAAAG" * 3 + "AAGGG" * 4, Haplotype.HOM)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=False)

    info = info_values(record)
    assert info["RN"] == "2"
    assert info["RUS"] == "AAAAG,AAGGG"
    assert info["RUC"] == "3,4"
    assert info["TRUC"] == "7.0"
    # Any motif can be the longest pure segment, not just the first catalog unit
    assert info["LPSRUS"] == "AAGGG"
    assert info["LPSRUC"] == "4"


def test_longest_pure_segment_tie_keeps_first_run(ref_path: Path):
    locus = create_synthetic_locus(["CAG|CAA"])
    final_params = {Haplotype.HOM: params(10)}
    calls = [make_consensus_call(locus, "CAA" * 5 + "CAG" * 5, Haplotype.HOM)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=False)

    info = info_values(record)
    assert info["LPSRUS"] == "CAA"
    assert info["LPSRUC"] == "5"


def test_homozygous_record_has_one_alt_and_no_phase_set(ref_path: Path):
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.HOM: params(5)}
    calls = [make_consensus_call(locus, "CAG" * 5, Haplotype.HOM, spanning=7, flanking=1)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=False)

    assert record.split("\t")[4] == "<CNV:TR>"
    assert info_values(record)["SVLEN"] == "3"
    assert format_values(record) == {"GT": "1/1", "DP": "8", "AD": "0,8", "ADSP": "0,7", "ADFL": "0,1"}


def test_allele_depths_follow_alt_order_regardless_of_call_order(ref_path: Path):
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(5), Haplotype.H2: params(10)}
    calls = [
        make_consensus_call(locus, "CAG" * 10, Haplotype.H2, spanning=5, flanking=3),
        make_consensus_call(locus, "CAG" * 5, Haplotype.H1, spanning=11, flanking=0),
    ]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    assert info_values(record)["RUC"] == "5,10"
    values = format_values(record)
    assert values["AD"] == "0,11,8"
    assert values["ADSP"] == "0,11,5"
    assert values["ADFL"] == "0,0,3"


def test_haplotype_without_consensus_has_no_repeat_sequences(ref_path: Path):
    # A zero-length haplotype yields an empty consensus, so create_consensus_calls returns no call for it
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(5), Haplotype.H2: params(0)}
    calls = [make_consensus_call(locus, "CAG" * 5, Haplotype.H1, spanning=11, flanking=2)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    info = info_values(record)
    assert info["RN"] == "1,0"
    assert info["RUS"] == "CAG"
    assert info["RUC"] == "5"
    assert info["TRUC"] == "5.0,0.0"
    assert info["LPSRUS"] == "CAG,."
    assert info["LPSRUC"] == "5,."
    values = format_values(record)
    assert values["AD"] == "0,13,0"
    assert values["ADSP"] == "0,11,0"
    assert values["ADFL"] == "0,2,0"


def test_consensus_is_opt_in_info_field(ref_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "add_consensus_to_vcf", True)
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(0), Haplotype.H2: params(5)}
    calls = [make_consensus_call(locus, "CAG" * 5, Haplotype.H2, spanning=11, flanking=2)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    assert info_values(record)["CONS"] == ".,CAGCAGCAGCAGCAG"
    assert "CONS" not in format_values(record)


def test_consensus_is_missing_for_allele_without_units_in_satellite(ref_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "add_consensus_to_vcf", True)
    locus = create_synthetic_locus(["CAG", "CCG"])
    final_params = {Haplotype.H1: params(5, 0), Haplotype.H2: params(8, 4)}
    calls = [
        make_consensus_call(locus, "CAG" * 5, Haplotype.H1),
        make_consensus_call(locus, "CAG" * 8 + "CCG" * 4, Haplotype.H2),
    ]

    _, ccg = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    info = info_values(ccg)
    assert info["RN"] == "0,1"
    assert info["CONS"] == ".,CCGCCGCCGCCG"


def test_repeat_absent_from_reference_has_svlen_one(ref_path: Path):
    locus = create_synthetic_locus(["CAG"])
    locus.satellites[0].location = Location("chr1", 1000, 1000)
    final_params = {Haplotype.HOM: params(5)}
    calls = [make_consensus_call(locus, "CAG" * 5, Haplotype.HOM)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=False)

    fields = record.split("\t")
    assert (fields[1], fields[3]) == ("1000", "T")
    assert info_values(record)["SVLEN"] == "1"


def test_header_declares_threshold_fields(ref_path: Path):
    header = generate_vcf_header(ref_path, "sample")

    assert "##INFO=<ID=STRSTATUS,Number=A,Type=String," in header
    assert "##INFO=<ID=BENIGNMAX,Number=1,Type=Integer," in header
    assert "##INFO=<ID=PATHOGENICMIN,Number=1,Type=Integer," in header


def test_thresholds_add_status_per_allele(ref_path: Path):
    locus = create_synthetic_locus(["CAG", "CCG"], breaks=["", "CAACAG", ""])
    locus.satellites[0].benign_max = 26
    locus.satellites[0].pathogenic_min = 36
    final_params = {Haplotype.H1: params(19, 7), Haplotype.H2: params(45, 10)}
    calls = [
        make_consensus_call(locus, "CAG" * 19 + "CAACAG" + "CCG" * 7, Haplotype.H1),
        make_consensus_call(locus, "CAG" * 45 + "CAACAG" + "CCG" * 10, Haplotype.H2),
    ]

    cag, ccg = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    info = info_values(cag)
    assert (info["STRSTATUS"], info["BENIGNMAX"], info["PATHOGENICMIN"]) == ("normal,pathogenic", "26", "36")
    assert not {"STRSTATUS", "BENIGNMAX", "PATHOGENICMIN"} & info_values(ccg).keys()


def test_single_threshold_omits_the_other(ref_path: Path):
    locus = create_synthetic_locus(["CAG"])
    locus.satellites[0].pathogenic_min = 36
    final_params = {Haplotype.HOM: params(30)}
    calls = [make_consensus_call(locus, "CAG" * 30, Haplotype.HOM)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=False)

    info = info_values(record)
    assert (info["STRSTATUS"], info["PATHOGENICMIN"]) == ("normal", "36")
    assert "BENIGNMAX" not in info
