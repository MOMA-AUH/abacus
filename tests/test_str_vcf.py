from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from abacus.config import config
from abacus.consensus import ConsensusCall, get_consensus_read_call
from abacus.graph import AlignmentType
from abacus.parameter_estimation import HomozygousParameters
from abacus.str_vcf import create_vcf_records
from abacus.utils import Haplotype
from tests.test_consensus import create_synthetic_locus


@pytest.fixture
def ref_path(tmp_path: Path) -> Path:
    path = tmp_path / "ref.fa"
    path.write_text(">chr1\n" + "A" * 2000 + "\n")
    return path


def make_consensus_call(locus, n_units: int, haplotype: Haplotype, spanning: int, flanking: int) -> ConsensusCall:
    read_call = get_consensus_read_call(locus, "CAG" * n_units, AlignmentType.SPANNING, haplotype)
    read_call.set_haplotype(haplotype)
    return ConsensusCall.from_read_call(read_call, spanning_reads=spanning, flanking_reads=flanking)


def params(n_units: float) -> HomozygousParameters:
    return HomozygousParameters(mean=np.array([n_units]), unit_var=np.array([0.1]), mean_ci_low=np.array([n_units - 1]), mean_ci_high=np.array([n_units + 1]))


def format_values(record: str) -> dict[str, str]:
    fields = record.split("\t")
    return dict(zip(fields[8].split(":"), fields[9].split(":")))


def test_allele_depths_follow_alt_order_regardless_of_call_order(ref_path: Path):
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(5), Haplotype.H2: params(10)}
    calls = [
        make_consensus_call(locus, 10, Haplotype.H2, spanning=5, flanking=3),
        make_consensus_call(locus, 5, Haplotype.H1, spanning=11, flanking=0),
    ]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    assert record.split("\t")[4] == "<STR5>,<STR10>"
    values = format_values(record)
    assert values["AD"] == "11,8"
    assert values["ADSP"] == "11,5"
    assert values["ADFL"] == "0,3"


def test_allele_depths_are_zero_for_haplotype_without_consensus(ref_path: Path):
    # A zero-length haplotype yields an empty consensus, so create_consensus_calls returns no call for it
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(5), Haplotype.H2: params(0)}
    calls = [make_consensus_call(locus, 5, Haplotype.H1, spanning=11, flanking=2)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    assert record.split("\t")[4] == "<STR5>,<STR0>"
    values = format_values(record)
    assert values["AD"] == "13,0"
    assert values["ADSP"] == "11,0"
    assert values["ADFL"] == "2,0"


def test_consensus_is_missing_for_haplotype_without_consensus(ref_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(config, "add_consensus_to_vcf", True)
    monkeypatch.setattr(config, "add_contracted_consensus_to_vcf", True)
    locus = create_synthetic_locus(["CAG"])
    final_params = {Haplotype.H1: params(0), Haplotype.H2: params(5)}
    calls = [make_consensus_call(locus, 5, Haplotype.H2, spanning=11, flanking=2)]

    (record,) = create_vcf_records(calls, ref_path, final_params, locus_is_het=True)

    values = format_values(record)
    assert values["CONS"] == ".,CAGCAGCAGCAGCAG"
    assert values["CONTRCONS"] == ".,5(CAG)"
